#include <erl_nif.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include "c_api/Index_c.h"
#include "c_api/index_factory_c.h"
#include "c_api/index_io_c.h"
#include "c_api/clone_index_c.h"
#include "c_api/Clustering_c.h"
#include "c_api/error_c.h"
#include "c_api/faiss_c.h"

#ifdef FAISS_GPU_ENABLED
#include "c_api/gpu/StandardGpuResources_c.h"
#include "c_api/gpu/GpuAutoTune_c.h"
#endif

/* FaissEx.Index passes metric as a bare integer; these pin the values it
 * hardcodes to FaissMetricType so a FAISS bump that renumbers the enum
 * fails to compile instead of silently searching the wrong metric. */
_Static_assert(METRIC_INNER_PRODUCT == 0, "FaissMetricType renumbered: update FaissEx.Index");
_Static_assert(METRIC_L2 == 1, "FaissMetricType renumbered: update FaissEx.Index");

/* ========== Resource Types ==========
 *
 * Locking convention: FAISS objects are not thread-safe when a mutation
 * (add/train/reset) races any other operation on the same object. Each
 * resource carries an ErlNifRWLock: mutations take the write lock,
 * read-only operations (search, reconstruct, serialization, getters) take
 * the read lock, so concurrent searches still run in parallel.
 *
 * nif_train_clustering is the only function locking two resources; the
 * fixed order is quantizer index first, then clustering. Keep that order
 * everywhere to avoid deadlock.
 *
 * Every NIF that can block on a lock must run on a dirty scheduler — a
 * writer may hold the lock for seconds (train), and blocking a normal
 * scheduler thread stalls the VM.
 */

static ErlNifResourceType *INDEX_RESOURCE_TYPE;
static ErlNifResourceType *CLUSTERING_RESOURCE_TYPE;
#ifdef FAISS_GPU_ENABLED
static ErlNifResourceType *GPU_RESOURCES_RESOURCE_TYPE;
#endif

typedef struct {
    FaissIndex *index;
    ErlNifRWLock *lock;
    /* Cached at wrap time. FAISS sets Index::d at construction and nothing
     * reachable through this NIF mutates it (prepend_transform/addIndex are
     * not exposed), so reads need no lock. Re-verify when bumping
     * FAISS_GIT_REV. */
    int dim;
} IndexResource;

typedef struct {
    FaissClustering *clustering;
    ErlNifRWLock *lock;
} ClusteringResource;

#ifdef FAISS_GPU_ENABLED
typedef struct {
    FaissStandardGpuResources *resources;
} GpuResourcesResource;
#endif

/* ========== Destructors ========== */

static void index_resource_destructor(ErlNifEnv *env, void *obj) {
    (void)env;
    IndexResource *res = (IndexResource *)obj;
    if (res->index) {
        faiss_Index_free(res->index);
        res->index = NULL;
    }
    if (res->lock) {
        enif_rwlock_destroy(res->lock);
        res->lock = NULL;
    }
}

static void clustering_resource_destructor(ErlNifEnv *env, void *obj) {
    (void)env;
    ClusteringResource *res = (ClusteringResource *)obj;
    if (res->clustering) {
        faiss_Clustering_free(res->clustering);
        res->clustering = NULL;
    }
    if (res->lock) {
        enif_rwlock_destroy(res->lock);
        res->lock = NULL;
    }
}

#ifdef FAISS_GPU_ENABLED
static void gpu_resources_destructor(ErlNifEnv *env, void *obj) {
    (void)env;
    GpuResourcesResource *res = (GpuResourcesResource *)obj;
    if (res->resources) {
        faiss_StandardGpuResources_free(res->resources);
        res->resources = NULL;
    }
}
#endif

/* ========== Helpers ========== */

static ERL_NIF_TERM atom_ok;
static ERL_NIF_TERM atom_error;
static ERL_NIF_TERM atom_true;
static ERL_NIF_TERM atom_false;

static ERL_NIF_TERM make_ok(ErlNifEnv *env, ERL_NIF_TERM term) {
    return enif_make_tuple2(env, atom_ok, term);
}

static ERL_NIF_TERM make_ok_atom(ErlNifEnv *env) {
    (void)env;
    return atom_ok;
}

static ERL_NIF_TERM make_error_msg(ErlNifEnv *env, const char *msg) {
    ERL_NIF_TERM bin;
    size_t len = strlen(msg);
    unsigned char *buf = enif_make_new_binary(env, len, &bin);
    memcpy(buf, msg, len);
    return enif_make_tuple2(env, atom_error, bin);
}

static ERL_NIF_TERM make_faiss_error(ErlNifEnv *env, const char *fallback) {
    /* Relies on faiss_get_last_error() being thread-local (verified for
     * v1.14.3: c_api/error_impl.cpp) so concurrent NIF calls on different
     * dirty-scheduler threads cannot cross-contaminate error messages.
     * Re-verify when bumping FAISS_GIT_REV. */
    const char *err = faiss_get_last_error();
    if (err && strlen(err) > 0) {
        return make_error_msg(env, err);
    }
    return make_error_msg(env, fallback);
}

/* Overflow-safe multiplication: returns 0 on overflow, 1 on success */
static int safe_mul(size_t a, size_t b, size_t *result) {
    if (a != 0 && b > SIZE_MAX / a) return 0;
    *result = a * b;
    return 1;
}

/* Rejects strings (paths, factory descriptions) with interior NUL bytes,
 * which would silently truncate after null-termination. */
static int contains_nul(const ErlNifBinary *bin) {
    return bin->size > 0 && memchr(bin->data, 0, bin->size) != NULL;
}

/* Wraps a FaissIndex in a resource with its own rwlock.
 * Takes ownership: frees the index and returns 0 on failure. */
static int wrap_index_resource(ErlNifEnv *env, FaissIndex *index, ERL_NIF_TERM *ref_out) {
    ErlNifRWLock *lock = enif_rwlock_create("faiss_ex_index");
    if (!lock) {
        faiss_Index_free(index);
        return 0;
    }

    IndexResource *res = enif_alloc_resource(INDEX_RESOURCE_TYPE, sizeof(IndexResource));
    if (!res) {
        enif_rwlock_destroy(lock);
        faiss_Index_free(index);
        return 0;
    }
    res->index = index;
    res->lock = lock;
    res->dim = faiss_Index_d(index);
    *ref_out = enif_make_resource(env, res);
    enif_release_resource(res);
    return 1;
}

/* ========== NIF: Index ========== */

/* nif_new_index(dim, description_binary, metric_int) */
static ERL_NIF_TERM nif_new_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    int dim;
    ErlNifBinary desc_bin;
    int metric;

    if (!enif_get_int(env, argv[0], &dim) ||
        !enif_inspect_binary(env, argv[1], &desc_bin) ||
        !enif_get_int(env, argv[2], &metric)) {
        return make_error_msg(env, "invalid arguments");
    }

    if (dim <= 0) return make_error_msg(env, "dim must be positive");
    if (contains_nul(&desc_bin)) return make_error_msg(env, "string contains NUL byte");

    /* Null-terminate the description string */
    char *desc = (char *)enif_alloc(desc_bin.size + 1);
    if (!desc) return make_error_msg(env, "out of memory");
    memcpy(desc, desc_bin.data, desc_bin.size);
    desc[desc_bin.size] = '\0';

    FaissIndex *index = NULL;
    int ret = faiss_index_factory(&index, dim, desc, (FaissMetricType)metric);
    enif_free(desc);

    if (ret != 0) {
        return make_faiss_error(env, "failed to create index");
    }

    ERL_NIF_TERM ref;
    if (!wrap_index_resource(env, index, &ref)) {
        return make_error_msg(env, "failed to allocate resource");
    }

    return make_ok(env, ref);
}

/* nif_clone_index(ref) */
static ERL_NIF_TERM nif_clone_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    IndexResource *res;
    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res)) {
        return make_error_msg(env, "invalid index reference");
    }

    FaissIndex *cloned = NULL;
    enif_rwlock_rlock(res->lock);
    int ret = faiss_clone_index(res->index, &cloned);
    enif_rwlock_runlock(res->lock);
    if (ret != 0) {
        return make_faiss_error(env, "failed to clone index");
    }

    ERL_NIF_TERM ref;
    if (!wrap_index_resource(env, cloned, &ref)) {
        return make_error_msg(env, "failed to allocate resource");
    }

    return make_ok(env, ref);
}

/* nif_add_to_index(ref, n, data_binary) */
static ERL_NIF_TERM nif_add_to_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    IndexResource *res;
    ErlNifSInt64 n;
    ErlNifBinary data_bin;

    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res) ||
        !enif_get_int64(env, argv[1], &n) ||
        !enif_inspect_binary(env, argv[2], &data_bin)) {
        return make_error_msg(env, "invalid arguments");
    }

    if (n < 0) return make_error_msg(env, "n must be non-negative");
    if (n == 0) {
        if (data_bin.size != 0) return make_error_msg(env, "data binary size mismatch");
        return make_ok_atom(env);
    }

    enif_rwlock_rwlock(res->lock);

    int dim = res->dim;
    size_t nd, expected;
    if (!safe_mul((size_t)n, (size_t)dim, &nd) || !safe_mul(nd, sizeof(float), &expected)) {
        enif_rwlock_rwunlock(res->lock);
        return make_error_msg(env, "size overflow");
    }
    if (data_bin.size != expected) {
        enif_rwlock_rwunlock(res->lock);
        return make_error_msg(env, "data binary size mismatch");
    }

    int ret = faiss_Index_add(res->index, (idx_t)n, (const float *)data_bin.data);
    enif_rwlock_rwunlock(res->lock);
    if (ret != 0) {
        return make_faiss_error(env, "failed to add vectors");
    }

    return make_ok_atom(env);
}

/* nif_add_with_ids_to_index(ref, n, data_binary, ids_binary) */
static ERL_NIF_TERM nif_add_with_ids_to_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    IndexResource *res;
    ErlNifSInt64 n;
    ErlNifBinary data_bin, ids_bin;

    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res) ||
        !enif_get_int64(env, argv[1], &n) ||
        !enif_inspect_binary(env, argv[2], &data_bin) ||
        !enif_inspect_binary(env, argv[3], &ids_bin)) {
        return make_error_msg(env, "invalid arguments");
    }

    if (n < 0) return make_error_msg(env, "n must be non-negative");
    if (n == 0) {
        if (data_bin.size != 0 || ids_bin.size != 0) {
            return make_error_msg(env, "binary size mismatch");
        }
        return make_ok_atom(env);
    }

    enif_rwlock_rwlock(res->lock);

    int dim = res->dim;
    size_t nd, expected_data, expected_ids;
    if (!safe_mul((size_t)n, (size_t)dim, &nd) || !safe_mul(nd, sizeof(float), &expected_data) ||
        !safe_mul((size_t)n, sizeof(int64_t), &expected_ids)) {
        enif_rwlock_rwunlock(res->lock);
        return make_error_msg(env, "size overflow");
    }

    if (data_bin.size != expected_data || ids_bin.size != expected_ids) {
        enif_rwlock_rwunlock(res->lock);
        return make_error_msg(env, "binary size mismatch");
    }

    int ret = faiss_Index_add_with_ids(res->index, (idx_t)n,
                                        (const float *)data_bin.data,
                                        (const idx_t *)ids_bin.data);
    enif_rwlock_rwunlock(res->lock);
    if (ret != 0) {
        return make_faiss_error(env, "failed to add vectors with ids");
    }

    return make_ok_atom(env);
}

/* nif_search_index(ref, n, data_binary, k) */
static ERL_NIF_TERM nif_search_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    IndexResource *res;
    ErlNifSInt64 n, k;
    ErlNifBinary data_bin;

    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res) ||
        !enif_get_int64(env, argv[1], &n) ||
        !enif_inspect_binary(env, argv[2], &data_bin) ||
        !enif_get_int64(env, argv[3], &k)) {
        return make_error_msg(env, "invalid arguments");
    }

    if (n < 0) return make_error_msg(env, "n must be non-negative");
    if (k <= 0) return make_error_msg(env, "k must be positive");

    if (n == 0) {
        if (data_bin.size != 0) {
            return make_error_msg(env, "data binary size mismatch");
        }
        ERL_NIF_TERM empty_d, empty_l;
        enif_make_new_binary(env, 0, &empty_d);
        enif_make_new_binary(env, 0, &empty_l);
        return make_ok(env, enif_make_tuple2(env, empty_d, empty_l));
    }

    /* res->dim needs no lock, so all size checks and result allocations
     * happen before the read lock — the critical section is just the
     * faiss_Index_search call. */
    int dim = res->dim;
    size_t nd, expected;
    if (!safe_mul((size_t)n, (size_t)dim, &nd) || !safe_mul(nd, sizeof(float), &expected)) {
        return make_error_msg(env, "size overflow");
    }
    if (data_bin.size != expected) {
        return make_error_msg(env, "data binary size mismatch");
    }

    size_t nk, dist_size, label_size;
    if (!safe_mul((size_t)n, (size_t)k, &nk) ||
        !safe_mul(nk, sizeof(float), &dist_size) ||
        !safe_mul(nk, sizeof(int64_t), &label_size)) {
        return make_error_msg(env, "size overflow");
    }

    ErlNifBinary distances_bin, labels_bin;
    if (!enif_alloc_binary(dist_size, &distances_bin)) {
        return make_error_msg(env, "failed to allocate result binaries");
    }
    if (!enif_alloc_binary(label_size, &labels_bin)) {
        enif_release_binary(&distances_bin);
        return make_error_msg(env, "failed to allocate result binaries");
    }

    enif_rwlock_rlock(res->lock);
    int ret = faiss_Index_search(res->index, (idx_t)n,
                                  (const float *)data_bin.data, (idx_t)k,
                                  (float *)distances_bin.data,
                                  (idx_t *)labels_bin.data);
    enif_rwlock_runlock(res->lock);
    if (ret != 0) {
        enif_release_binary(&distances_bin);
        enif_release_binary(&labels_bin);
        return make_faiss_error(env, "search failed");
    }

    ERL_NIF_TERM distances_term = enif_make_binary(env, &distances_bin);
    ERL_NIF_TERM labels_term = enif_make_binary(env, &labels_bin);

    return make_ok(env, enif_make_tuple2(env, distances_term, labels_term));
}

/* nif_train_index(ref, n, data_binary) */
static ERL_NIF_TERM nif_train_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    IndexResource *res;
    ErlNifSInt64 n;
    ErlNifBinary data_bin;

    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res) ||
        !enif_get_int64(env, argv[1], &n) ||
        !enif_inspect_binary(env, argv[2], &data_bin)) {
        return make_error_msg(env, "invalid arguments");
    }

    if (n < 0) return make_error_msg(env, "n must be non-negative");
    if (n == 0) {
        if (data_bin.size != 0) return make_error_msg(env, "data binary size mismatch");
        return make_ok_atom(env);
    }

    enif_rwlock_rwlock(res->lock);

    int dim = res->dim;
    size_t nd, expected;
    if (!safe_mul((size_t)n, (size_t)dim, &nd) || !safe_mul(nd, sizeof(float), &expected)) {
        enif_rwlock_rwunlock(res->lock);
        return make_error_msg(env, "size overflow");
    }
    if (data_bin.size != expected) {
        enif_rwlock_rwunlock(res->lock);
        return make_error_msg(env, "data binary size mismatch");
    }

    int ret = faiss_Index_train(res->index, (idx_t)n, (const float *)data_bin.data);
    enif_rwlock_rwunlock(res->lock);
    if (ret != 0) {
        return make_faiss_error(env, "training failed");
    }

    return make_ok_atom(env);
}

/* nif_reset_index(ref) */
static ERL_NIF_TERM nif_reset_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    IndexResource *res;
    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res)) {
        return make_error_msg(env, "invalid index reference");
    }

    enif_rwlock_rwlock(res->lock);
    int ret = faiss_Index_reset(res->index);
    enif_rwlock_rwunlock(res->lock);
    if (ret != 0) {
        return make_faiss_error(env, "reset failed");
    }

    return make_ok_atom(env);
}

/* nif_reconstruct_batch(ref, n, keys_binary) */
static ERL_NIF_TERM nif_reconstruct_batch(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    IndexResource *res;
    ErlNifSInt64 n;
    ErlNifBinary keys_bin;

    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res) ||
        !enif_get_int64(env, argv[1], &n) ||
        !enif_inspect_binary(env, argv[2], &keys_bin)) {
        return make_error_msg(env, "invalid arguments");
    }

    if (n < 0) return make_error_msg(env, "n must be non-negative");

    size_t expected_keys;
    if (!safe_mul((size_t)n, sizeof(int64_t), &expected_keys)) {
        return make_error_msg(env, "size overflow");
    }
    if (keys_bin.size != expected_keys) {
        return make_error_msg(env, "keys binary size mismatch");
    }

    if (n == 0) {
        ERL_NIF_TERM empty;
        enif_make_new_binary(env, 0, &empty);
        return make_ok(env, empty);
    }

    const idx_t *keys = (const idx_t *)keys_bin.data;

    /* Contiguous ascending keys (the common case) can use the batched
     * faiss_Index_reconstruct_n instead of one call per key. Unsigned
     * arithmetic: keys are caller-controlled and keys[i-1] + 1 would be
     * signed-overflow UB at INT64_MAX. */
    int contiguous = 1;
    for (ErlNifSInt64 i = 1; i < n; i++) {
        if ((uint64_t)keys[i] != (uint64_t)keys[i - 1] + 1u) {
            contiguous = 0;
            break;
        }
    }

    enif_rwlock_rlock(res->lock);

    int dim = res->dim;
    size_t nd, result_size;
    if (!safe_mul((size_t)n, (size_t)dim, &nd) || !safe_mul(nd, sizeof(float), &result_size)) {
        enif_rwlock_runlock(res->lock);
        return make_error_msg(env, "size overflow");
    }

    ErlNifBinary result_bin;
    if (!enif_alloc_binary(result_size, &result_bin)) {
        enif_rwlock_runlock(res->lock);
        return make_error_msg(env, "failed to allocate result binary");
    }

    float *result = (float *)result_bin.data;
    int ret = 0;

    if (contiguous) {
        ret = faiss_Index_reconstruct_n(res->index, keys[0], (idx_t)n, result);
    } else {
        for (ErlNifSInt64 i = 0; i < n; i++) {
            ret = faiss_Index_reconstruct(res->index, keys[i], result + i * dim);
            if (ret != 0) break;
        }
    }
    enif_rwlock_runlock(res->lock);

    if (ret != 0) {
        enif_release_binary(&result_bin);
        return make_faiss_error(env, "reconstruct failed");
    }

    return make_ok(env, enif_make_binary(env, &result_bin));
}

/* nif_compute_residuals(ref, n, data_binary, keys_binary) */
static ERL_NIF_TERM nif_compute_residuals(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    IndexResource *res;
    ErlNifSInt64 n;
    ErlNifBinary data_bin, keys_bin;

    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res) ||
        !enif_get_int64(env, argv[1], &n) ||
        !enif_inspect_binary(env, argv[2], &data_bin) ||
        !enif_inspect_binary(env, argv[3], &keys_bin)) {
        return make_error_msg(env, "invalid arguments");
    }

    if (n < 0) return make_error_msg(env, "n must be non-negative");

    if (n == 0) {
        if (data_bin.size != 0 || keys_bin.size != 0) {
            return make_error_msg(env, "binary size mismatch");
        }
        ERL_NIF_TERM empty;
        enif_make_new_binary(env, 0, &empty);
        return make_ok(env, empty);
    }

    enif_rwlock_rlock(res->lock);

    int dim = res->dim;
    size_t nd, expected_data, expected_keys;
    if (!safe_mul((size_t)n, (size_t)dim, &nd) || !safe_mul(nd, sizeof(float), &expected_data) ||
        !safe_mul((size_t)n, sizeof(int64_t), &expected_keys)) {
        enif_rwlock_runlock(res->lock);
        return make_error_msg(env, "size overflow");
    }

    if (data_bin.size != expected_data || keys_bin.size != expected_keys) {
        enif_rwlock_runlock(res->lock);
        return make_error_msg(env, "binary size mismatch");
    }

    ErlNifBinary result_bin;
    if (!enif_alloc_binary(expected_data, &result_bin)) {
        enif_rwlock_runlock(res->lock);
        return make_error_msg(env, "failed to allocate result binary");
    }

    const float *data = (const float *)data_bin.data;
    const idx_t *keys = (const idx_t *)keys_bin.data;
    float *result = (float *)result_bin.data;

    int ret = faiss_Index_compute_residual_n(res->index, (idx_t)n, data, result, keys);
    enif_rwlock_runlock(res->lock);
    if (ret != 0) {
        enif_release_binary(&result_bin);
        return make_faiss_error(env, "compute_residual failed");
    }

    return make_ok(env, enif_make_binary(env, &result_bin));
}

/* nif_write_index(ref, path_binary) */
static ERL_NIF_TERM nif_write_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    IndexResource *res;
    ErlNifBinary path_bin;

    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res) ||
        !enif_inspect_binary(env, argv[1], &path_bin)) {
        return make_error_msg(env, "invalid arguments");
    }

    if (contains_nul(&path_bin)) return make_error_msg(env, "string contains NUL byte");

    char *path = (char *)enif_alloc(path_bin.size + 1);
    if (!path) return make_error_msg(env, "out of memory");
    memcpy(path, path_bin.data, path_bin.size);
    path[path_bin.size] = '\0';

    enif_rwlock_rlock(res->lock);
    int ret = faiss_write_index_fname(res->index, path);
    enif_rwlock_runlock(res->lock);
    enif_free(path);

    if (ret != 0) {
        return make_faiss_error(env, "failed to write index");
    }

    return make_ok_atom(env);
}

/* nif_read_index(path_binary, io_flags) */
static ERL_NIF_TERM nif_read_index(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    ErlNifBinary path_bin;
    int io_flags;

    if (!enif_inspect_binary(env, argv[0], &path_bin) ||
        !enif_get_int(env, argv[1], &io_flags)) {
        return make_error_msg(env, "invalid arguments");
    }

    if (contains_nul(&path_bin)) return make_error_msg(env, "string contains NUL byte");

    char *path = (char *)enif_alloc(path_bin.size + 1);
    if (!path) return make_error_msg(env, "out of memory");
    memcpy(path, path_bin.data, path_bin.size);
    path[path_bin.size] = '\0';

    FaissIndex *index = NULL;
    int ret = faiss_read_index_fname(path, io_flags, &index);
    enif_free(path);

    if (ret != 0) {
        return make_faiss_error(env, "failed to read index");
    }

    ERL_NIF_TERM ref;
    if (!wrap_index_resource(env, index, &ref)) {
        return make_error_msg(env, "failed to allocate resource");
    }

    return make_ok(env, ref);
}

/* nif_get_index_dim(ref) — reads the immutable cached dim, no lock, so it
 * stays on a normal scheduler. */
static ERL_NIF_TERM nif_get_index_dim(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    IndexResource *res;
    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res)) {
        return make_error_msg(env, "invalid index reference");
    }
    return make_ok(env, enif_make_int(env, res->dim));
}

/* nif_get_index_ntotal(ref) */
static ERL_NIF_TERM nif_get_index_ntotal(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    IndexResource *res;
    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res)) {
        return make_error_msg(env, "invalid index reference");
    }
    enif_rwlock_rlock(res->lock);
    int64_t ntotal = (int64_t)faiss_Index_ntotal(res->index);
    enif_rwlock_runlock(res->lock);
    return make_ok(env, enif_make_int64(env, ntotal));
}

/* nif_get_index_is_trained(ref) */
static ERL_NIF_TERM nif_get_index_is_trained(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    IndexResource *res;
    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res)) {
        return make_error_msg(env, "invalid index reference");
    }
    enif_rwlock_rlock(res->lock);
    int trained = faiss_Index_is_trained(res->index);
    enif_rwlock_runlock(res->lock);
    return make_ok(env, trained ? atom_true : atom_false);
}

/* ========== NIF: GPU ========== */

#ifdef FAISS_GPU_ENABLED

/* nif_index_cpu_to_gpu(ref, device) - returns {gpu_resources_ref, gpu_index_ref} */
static ERL_NIF_TERM nif_index_cpu_to_gpu(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    IndexResource *res;
    int device;

    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res) ||
        !enif_get_int(env, argv[1], &device)) {
        return make_error_msg(env, "invalid arguments");
    }

    FaissStandardGpuResources *gpu_res = NULL;
    int ret = faiss_StandardGpuResources_new(&gpu_res);
    if (ret != 0) {
        return make_faiss_error(env, "failed to create GPU resources");
    }

    FaissIndex *gpu_index = NULL;
    enif_rwlock_rlock(res->lock);
    ret = faiss_index_cpu_to_gpu((FaissGpuResourcesProvider *)gpu_res, device, res->index, &gpu_index);
    enif_rwlock_runlock(res->lock);
    if (ret != 0) {
        faiss_StandardGpuResources_free(gpu_res);
        return make_faiss_error(env, "failed to move index to GPU");
    }

    GpuResourcesResource *gpu_res_resource = enif_alloc_resource(
        GPU_RESOURCES_RESOURCE_TYPE, sizeof(GpuResourcesResource));
    if (!gpu_res_resource) {
        faiss_Index_free(gpu_index);
        faiss_StandardGpuResources_free(gpu_res);
        return make_error_msg(env, "failed to allocate resource");
    }
    gpu_res_resource->resources = gpu_res;
    ERL_NIF_TERM gpu_res_ref = enif_make_resource(env, gpu_res_resource);
    enif_release_resource(gpu_res_resource);

    /* gpu_res is owned by gpu_res_resource; env cleanup frees it on failure */
    ERL_NIF_TERM gpu_idx_ref;
    if (!wrap_index_resource(env, gpu_index, &gpu_idx_ref)) {
        return make_error_msg(env, "failed to allocate resource");
    }

    return make_ok(env, enif_make_tuple2(env, gpu_res_ref, gpu_idx_ref));
}

/* nif_index_gpu_to_cpu(ref) */
static ERL_NIF_TERM nif_index_gpu_to_cpu(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    IndexResource *res;
    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res)) {
        return make_error_msg(env, "invalid index reference");
    }

    FaissIndex *cpu_index = NULL;
    enif_rwlock_rlock(res->lock);
    int ret = faiss_index_gpu_to_cpu(res->index, &cpu_index);
    enif_rwlock_runlock(res->lock);
    if (ret != 0) {
        return make_faiss_error(env, "failed to move index to CPU");
    }

    ERL_NIF_TERM ref;
    if (!wrap_index_resource(env, cpu_index, &ref)) {
        return make_error_msg(env, "failed to allocate resource");
    }

    return make_ok(env, ref);
}

/* nif_get_num_gpus() */
static ERL_NIF_TERM nif_get_num_gpus(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    (void)argv;
    return make_ok(env, enif_make_int(env, faiss_get_num_gpus()));
}

#else /* !FAISS_GPU_ENABLED */

static ERL_NIF_TERM nif_index_cpu_to_gpu(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    (void)argv;
    return make_error_msg(env, "GPU support not compiled");
}

static ERL_NIF_TERM nif_index_gpu_to_cpu(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    (void)argv;
    return make_error_msg(env, "GPU support not compiled");
}

static ERL_NIF_TERM nif_get_num_gpus(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    (void)argv;
    return make_ok(env, enif_make_int(env, 0));
}

#endif /* FAISS_GPU_ENABLED */

/* ========== NIF: Clustering ========== */

/* nif_new_clustering(d, k) */
static ERL_NIF_TERM nif_new_clustering(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    int d, k;
    if (!enif_get_int(env, argv[0], &d) ||
        !enif_get_int(env, argv[1], &k)) {
        return make_error_msg(env, "invalid arguments");
    }

    if (d <= 0) return make_error_msg(env, "d must be positive");
    if (k <= 0) return make_error_msg(env, "k must be positive");

    FaissClustering *clustering = NULL;
    int ret = faiss_Clustering_new(&clustering, d, k);
    if (ret != 0) {
        return make_faiss_error(env, "failed to create clustering");
    }

    ErlNifRWLock *lock = enif_rwlock_create("faiss_ex_clustering");
    if (!lock) {
        faiss_Clustering_free(clustering);
        return make_error_msg(env, "failed to allocate resource");
    }

    ClusteringResource *res = enif_alloc_resource(CLUSTERING_RESOURCE_TYPE, sizeof(ClusteringResource));
    if (!res) {
        enif_rwlock_destroy(lock);
        faiss_Clustering_free(clustering);
        return make_error_msg(env, "failed to allocate resource");
    }
    res->clustering = clustering;
    res->lock = lock;
    ERL_NIF_TERM ref = enif_make_resource(env, res);
    enif_release_resource(res);

    return make_ok(env, ref);
}

/* nif_train_clustering(clust_ref, n, data_binary, idx_ref) */
static ERL_NIF_TERM nif_train_clustering(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    ClusteringResource *clust_res;
    ErlNifSInt64 n;
    ErlNifBinary data_bin;
    IndexResource *idx_res;

    if (!enif_get_resource(env, argv[0], CLUSTERING_RESOURCE_TYPE, (void **)&clust_res) ||
        !enif_get_int64(env, argv[1], &n) ||
        !enif_inspect_binary(env, argv[2], &data_bin) ||
        !enif_get_resource(env, argv[3], INDEX_RESOURCE_TYPE, (void **)&idx_res)) {
        return make_error_msg(env, "invalid arguments");
    }

    if (n <= 0) return make_error_msg(env, "n must be positive");

    /* Training mutates both the clustering state and the quantizer index.
     * Lock order: index first, then clustering. */
    enif_rwlock_rwlock(idx_res->lock);
    enif_rwlock_rwlock(clust_res->lock);

    int d = faiss_Clustering_d(clust_res->clustering);
    size_t nd, expected;
    if (!safe_mul((size_t)n, (size_t)d, &nd) || !safe_mul(nd, sizeof(float), &expected)) {
        enif_rwlock_rwunlock(clust_res->lock);
        enif_rwlock_rwunlock(idx_res->lock);
        return make_error_msg(env, "size overflow");
    }
    if (data_bin.size != expected) {
        enif_rwlock_rwunlock(clust_res->lock);
        enif_rwlock_rwunlock(idx_res->lock);
        return make_error_msg(env, "data binary size mismatch");
    }

    int ret = faiss_Clustering_train(clust_res->clustering, (idx_t)n,
                                      (const float *)data_bin.data, idx_res->index);
    enif_rwlock_rwunlock(clust_res->lock);
    enif_rwlock_rwunlock(idx_res->lock);
    if (ret != 0) {
        return make_faiss_error(env, "clustering training failed");
    }

    return make_ok_atom(env);
}

/* nif_get_clustering_centroids(clust_ref) - returns {k, d, centroids_binary} */
static ERL_NIF_TERM nif_get_clustering_centroids(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    ClusteringResource *res;
    if (!enif_get_resource(env, argv[0], CLUSTERING_RESOURCE_TYPE, (void **)&res)) {
        return make_error_msg(env, "invalid clustering reference");
    }

    enif_rwlock_rlock(res->lock);

    size_t k = faiss_Clustering_k(res->clustering);
    size_t d = faiss_Clustering_d(res->clustering);

    float *centroids = NULL;
    size_t centroids_size = 0;
    faiss_Clustering_centroids(res->clustering, &centroids, &centroids_size);

    if (!centroids || centroids_size == 0) {
        enif_rwlock_runlock(res->lock);
        return make_error_msg(env, "no centroids available");
    }

    size_t bin_size;
    if (!safe_mul(centroids_size, sizeof(float), &bin_size)) {
        enif_rwlock_runlock(res->lock);
        return make_error_msg(env, "size overflow");
    }

    ErlNifBinary bin;
    if (!enif_alloc_binary(bin_size, &bin)) {
        enif_rwlock_runlock(res->lock);
        return make_error_msg(env, "failed to allocate centroids binary");
    }
    memcpy(bin.data, centroids, bin_size);
    enif_rwlock_runlock(res->lock);

    ERL_NIF_TERM result = enif_make_tuple3(env,
        enif_make_int64(env, (int64_t)k),
        enif_make_int64(env, (int64_t)d),
        enif_make_binary(env, &bin));

    return make_ok(env, result);
}

/* ========== NIF Init ========== */

static int register_resource_types(ErlNifEnv *env, ErlNifResourceFlags flags) {
    INDEX_RESOURCE_TYPE = enif_open_resource_type(
        env, NULL, "FaissIndex", index_resource_destructor, flags, NULL);
    if (!INDEX_RESOURCE_TYPE) return -1;

    CLUSTERING_RESOURCE_TYPE = enif_open_resource_type(
        env, NULL, "FaissClustering", clustering_resource_destructor, flags, NULL);
    if (!CLUSTERING_RESOURCE_TYPE) return -1;

#ifdef FAISS_GPU_ENABLED
    GPU_RESOURCES_RESOURCE_TYPE = enif_open_resource_type(
        env, NULL, "FaissGpuResources", gpu_resources_destructor, flags, NULL);
    if (!GPU_RESOURCES_RESOURCE_TYPE) return -1;
#endif

    return 0;
}

static void init_atoms(ErlNifEnv *env) {
    atom_ok = enif_make_atom(env, "ok");
    atom_error = enif_make_atom(env, "error");
    atom_true = enif_make_atom(env, "true");
    atom_false = enif_make_atom(env, "false");
}

static int on_load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
    (void)priv_data;
    (void)load_info;
    init_atoms(env);
    return register_resource_types(env, ERL_NIF_RT_CREATE);
}

static int on_upgrade(ErlNifEnv *env, void **priv_data, void **old_priv_data, ERL_NIF_TERM load_info) {
    (void)priv_data;
    (void)old_priv_data;
    (void)load_info;
    init_atoms(env);
    return register_resource_types(env, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER);
}

/* Getters reading mutable state (ntotal/is_trained) run dirty too: they
 * block on the rwlock, and a writer (train/add) may hold it for seconds.
 * The dim getter reads the immutable cached field without the lock, so it
 * stays on a normal scheduler. */
static ErlNifFunc nif_funcs[] = {
    /* Index */
    {"nif_new_index", 3, nif_new_index, 0},
    {"nif_clone_index", 1, nif_clone_index, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_add_to_index", 3, nif_add_to_index, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_add_with_ids_to_index", 4, nif_add_with_ids_to_index, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_search_index", 4, nif_search_index, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_train_index", 3, nif_train_index, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_reset_index", 1, nif_reset_index, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_reconstruct_batch", 3, nif_reconstruct_batch, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_compute_residuals", 4, nif_compute_residuals, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_write_index", 2, nif_write_index, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"nif_read_index", 2, nif_read_index, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"nif_get_index_dim", 1, nif_get_index_dim, 0},
    {"nif_get_index_ntotal", 1, nif_get_index_ntotal, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_get_index_is_trained", 1, nif_get_index_is_trained, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    /* GPU */
    {"nif_index_cpu_to_gpu", 2, nif_index_cpu_to_gpu, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"nif_index_gpu_to_cpu", 1, nif_index_gpu_to_cpu, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"nif_get_num_gpus", 0, nif_get_num_gpus, 0},
    /* Clustering */
    {"nif_new_clustering", 2, nif_new_clustering, 0},
    {"nif_train_clustering", 4, nif_train_clustering, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_get_clustering_centroids", 1, nif_get_clustering_centroids, ERL_NIF_DIRTY_JOB_CPU_BOUND},
};

ERL_NIF_INIT(Elixir.FaissEx.NIF, nif_funcs, on_load, NULL, on_upgrade, NULL)
