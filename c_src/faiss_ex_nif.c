#include <erl_nif.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>

#ifndef FAISS_EX_NO_OMP
#include <omp.h>
#endif

#include "c_api/Index_c.h"
#include "c_api/index_factory_c.h"
#include "c_api/index_io_c.h"
#include "c_api/clone_index_c.h"
#include "c_api/Clustering_c.h"
#include "c_api/IndexFlat_c.h"
#include "c_api/IndexIVF_c.h"
#include "c_api/error_c.h"
#include "c_api/faiss_c.h"

#ifdef FAISS_GPU_ENABLED
#include "c_api/gpu/StandardGpuResources_c.h"
#include "c_api/gpu/GpuAutoTune_c.h"
#include "c_api/gpu/DeviceUtils_c.h"
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
 * Unwinding the lock: validate before acquiring it wherever the inputs
 * allow. Argument size checks read res->dim (IndexResource) or res->d/res->k
 * (ClusteringResource), cached immutable fields, so those checks, input
 * alignment copies, and the result enif_alloc_binary belong above the lock
 * — the critical section then holds nothing but the faiss_* call and its
 * one unlock. Where validation must read locked state (the centroids
 * pointer owned by the clustering object), the critical section records
 * only a status (an int, or a const char * message), releases once, and
 * every ERL_NIF_TERM is built after the release — a lock must never cover
 * term construction or error formatting. Never hand-write an unlock per
 * error branch: 16 such branches accumulated once already, and one
 * omission wedges the resource forever — nothing in the build or the test
 * suite can catch it.
 *
 * Every NIF that can block on a lock must run on a dirty scheduler — a
 * writer may hold the lock for seconds (train), and blocking a normal
 * scheduler thread stalls the VM.
 *
 * Waiting happens on the dirty-IO pool, never the dirty-CPU pool. A dirty
 * NIF that is merely blocked on a lock still occupies its scheduler thread,
 * so N waiters on one busy index would hold every dirty-CPU thread and
 * starve work on unrelated indexes. Each locked NIF therefore has two
 * entries sharing one implementation (LOCKED_NIF below): the registered
 * dirty-CPU entry validates, allocates, then only *tries* the lock; on
 * contention it releases everything it allocated and reschedules itself as
 * the <name>_io entry (dirty IO), which re-runs the same code and blocks.
 * No lock is ever held across a reschedule.
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
    /* Cached at creation; FAISS never changes a Clustering's d or k. Size
     * checks and result allocation read these without the lock. */
    int d;
    int k;
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

static ERL_NIF_TERM make_error_msg(ErlNifEnv *env, const char *msg) {
    ERL_NIF_TERM bin;
    size_t len = strlen(msg);
    unsigned char *buf = enif_make_new_binary(env, len, &bin);
    memcpy(buf, msg, len);
    return enif_make_tuple2(env, atom_error, bin);
}

static ERL_NIF_TERM make_faiss_error(ErlNifEnv *env, const char *fallback) {
    /* Relies on faiss_get_last_error() being thread-local (verified for
     * v1.15.1: c_api/error_impl.cpp) so concurrent NIF calls on different
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

/* Byte size of n rows of `cols` values of `elem` bytes each.
 * Returns 0 on overflow, 1 on success. n must be non-negative. */
static int rows_bytes(ErlNifSInt64 n, size_t cols, size_t elem, size_t *out) {
    size_t cells;
    return safe_mul((size_t)n, cols, &cells) && safe_mul(cells, elem, out);
}

/* Size validation shared by every NIF taking a row-major input binary.
 * Returns NULL when b holds exactly n rows of `cols` values of `elem`
 * bytes, otherwise the error message: "size overflow", or `mismatch` (the
 * caller's existing message — tests assert these strings byte for byte).
 * n must be non-negative. */
static const char *check_rows(ErlNifSInt64 n, size_t cols, size_t elem,
                              const ErlNifBinary *b, const char *mismatch) {
    size_t expected;
    if (!rows_bytes(n, cols, elem, &expected)) return "size overflow";
    return b->size == expected ? NULL : mismatch;
}

/* Stand-in for the data pointer of an empty input: FAISS gets a valid,
 * maximally aligned address even for n == 0 and never dereferences it. */
static const int64_t empty_input[1] = {0};

/* Returns, via *out, b's bytes at an address aligned for `align`.
 * Sub-binaries (binary_part/3, pattern-match tails) can start at any byte
 * offset, and FAISS dereferences float and idx_t pointers directly, so a
 * misaligned pointer is undefined behavior. Aligned input passes through
 * with *owned = NULL; misaligned input is copied into *owned (enif_alloc
 * memory is aligned for any scalar type), which the caller releases with
 * free_owned once FAISS is done with it — after the unlock. Returns 0 when
 * out of memory. */
static int aligned_input(const ErlNifBinary *b, size_t align, const void **out, void **owned) {
    *owned = NULL;
    if (b->size == 0) {
        *out = empty_input;
        return 1;
    }
    if ((uintptr_t)b->data % align == 0) {
        *out = b->data;
        return 1;
    }
    void *copy = enif_alloc(b->size);
    if (!copy) return 0;
    memcpy(copy, b->data, b->size);
    *owned = copy;
    *out = copy;
    return 1;
}

static void free_owned(void *owned) {
    if (owned) enif_free(owned);
}

/* ========== Lock acquisition and rescheduling ========== */

typedef enum { LOCK_TRY, LOCK_BLOCK } LockMode;

/* Returns 1 with the lock held, 0 when LOCK_TRY met contention. */
static int acquire_read(ErlNifRWLock *lock, LockMode mode) {
    if (mode == LOCK_BLOCK) {
        enif_rwlock_rlock(lock);
        return 1;
    }
    return enif_rwlock_tryrlock(lock) == 0;
}

static int acquire_write(ErlNifRWLock *lock, LockMode mode) {
    if (mode == LOCK_BLOCK) {
        enif_rwlock_rwlock(lock);
        return 1;
    }
    return enif_rwlock_tryrwlock(lock) == 0;
}

/* Declares name##_impl and defines the two entries described in the
 * header comment: `name` (registered dirty CPU, LOCK_TRY) and `name##_io`
 * (dirty IO, LOCK_BLOCK). */
#define LOCKED_NIF(name)                                                              \
    static ERL_NIF_TERM name##_impl(ErlNifEnv *, int, const ERL_NIF_TERM[], LockMode); \
    static ERL_NIF_TERM name##_io(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) { \
        return name##_impl(env, argc, argv, LOCK_BLOCK);                              \
    }                                                                                 \
    static ERL_NIF_TERM name(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {    \
        return name##_impl(env, argc, argv, LOCK_TRY);                                \
    }

/* The contention path of a LOCK_TRY entry. The caller must already have
 * released every allocation and holds no lock. */
#define RESCHEDULE_BLOCKING(name, env, argc, argv) \
    enif_schedule_nif((env), #name, ERL_NIF_DIRTY_JOB_IO_BOUND, name##_io, (argc), (argv))

/* ========== OpenMP thread count ==========
 *
 * FaissEx.set_omp_threads/1 stores the count here; 0 means the runtime
 * default captured at load (OMP_NUM_THREADS or the core count).
 * omp_set_num_threads sets a per-thread ICV and dirty scheduler threads
 * are reused across calls, so the value is applied on the calling thread
 * before every FAISS compute call rather than once. Built with
 * FAISS_EX_NO_OMP (libfaiss linked against a runtime this NIF cannot
 * share, e.g. Intel iomp5), the setting is accepted and ignored. */
static _Atomic int g_omp_threads = 0;
static int g_omp_default = 1;

static void apply_omp_threads(void) {
#ifndef FAISS_EX_NO_OMP
    int threads = atomic_load_explicit(&g_omp_threads, memory_order_relaxed);
    omp_set_num_threads(threads > 0 ? threads : g_omp_default);
#endif
}

/* Rejects strings (paths, factory descriptions) with interior NUL bytes,
 * which would silently truncate after null-termination. */
static int contains_nul(const ErlNifBinary *bin) {
    return bin->size > 0 && memchr(bin->data, 0, bin->size) != NULL;
}

/* Copies a binary into a NUL-terminated C string for the FAISS C API.
 * Returns NULL when out of memory; the caller owns the result and frees it
 * with enif_free. Guard with contains_nul first — this does not check. */
static char *dup_cstr(const ErlNifBinary *bin) {
    char *s = (char *)enif_alloc(bin->size + 1);
    if (!s) return NULL;
    memcpy(s, bin->data, bin->size);
    s[bin->size] = '\0';
    return s;
}

/* Rejects reconstruction keys that FAISS will not bounds-check safely.
 * IndexFlat::reconstruct tests only `key < ntotal`, which every negative key
 * passes, so a negative key reads before the codes array and returns heap
 * contents as vector data. Verified against v1.14.3 (reconstruct([-1, -5])
 * yields garbage floats with no error); the check is unchanged in v1.15.1.
 * INT64_MAX is rejected too: IndexFlatCodes::reconstruct(key) is
 * reconstruct_n(key, 1), whose `i0 + ni <= ntotal` signed-overflows there
 * and passes. The upper bound stays FAISS's job -- it needs ntotal, which
 * needs the lock. */
static int keys_in_range(const idx_t *keys, ErlNifSInt64 n) {
    for (ErlNifSInt64 i = 0; i < n; i++) {
        if (keys[i] < 0 || keys[i] == INT64_MAX) return 0;
    }
    return 1;
}

/* FAISS exceptions must never be thrown inside an OpenMP parallel region:
 * they cannot cross the region boundary to the C API's try/catch, so the
 * runtime calls std::terminate and the whole VM aborts. This holds even
 * for a region serialized by an `if` clause. Index::reconstruct_n and
 * Index::compute_residual_n (the base implementations) call reconstruct
 * inside such a region, and reconstruct throws for a key past ntotal or an
 * index type without it (IDMap). Verified v1.15.1: Flat
 * compute_residuals([v], [5]) and IDMap reconstruct([0]) aborted the VM.
 *
 * Only IndexFlat (IndexFlatCodes) and IndexIVF override reconstruct_n with
 * an up-front range check and no parallel region, so only they get the
 * batched call, and only after the caller has checked the range against
 * ntotal under the lock. Re-verify both overrides when bumping FAISS. */
static int has_safe_reconstruct_n(FaissIndex *index) {
    return faiss_IndexFlat_cast(index) != NULL || faiss_IndexIVF_cast(index) != NULL;
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

/* Factory descriptions are short ("IVF4096,PQ64x8", "IDMap,HNSW32"); the cap
 * bounds the parser's input. */
#define MAX_DESCRIPTION_BYTES 256

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
    if (desc_bin.size > MAX_DESCRIPTION_BYTES) return make_error_msg(env, "description too long");

    char *desc = dup_cstr(&desc_bin);
    if (!desc) return make_error_msg(env, "out of memory");

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
LOCKED_NIF(nif_clone_index)
static ERL_NIF_TERM nif_clone_index_impl(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[],
                                         LockMode mode) {
    IndexResource *res;
    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res)) {
        return make_error_msg(env, "invalid index reference");
    }

    FaissIndex *cloned = NULL;
    if (!acquire_read(res->lock, mode)) {
        return RESCHEDULE_BLOCKING(nif_clone_index, env, argc, argv);
    }
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
LOCKED_NIF(nif_add_to_index)
static ERL_NIF_TERM nif_add_to_index_impl(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[],
                                          LockMode mode) {
    IndexResource *res;
    ErlNifSInt64 n;
    ErlNifBinary data_bin;

    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res) ||
        !enif_get_int64(env, argv[1], &n) ||
        !enif_inspect_binary(env, argv[2], &data_bin)) {
        return make_error_msg(env, "invalid arguments");
    }

    if (n < 0) return make_error_msg(env, "n must be non-negative");
    const char *err = check_rows(n, (size_t)res->dim, sizeof(float), &data_bin,
                                 "data binary size mismatch");
    if (err) return make_error_msg(env, err);
    if (n == 0) return atom_ok;

    const void *data;
    void *owned;
    if (!aligned_input(&data_bin, _Alignof(float), &data, &owned)) {
        return make_error_msg(env, "out of memory");
    }

    apply_omp_threads();
    if (!acquire_write(res->lock, mode)) {
        free_owned(owned);
        return RESCHEDULE_BLOCKING(nif_add_to_index, env, argc, argv);
    }
    int ret = faiss_Index_add(res->index, (idx_t)n, (const float *)data);
    enif_rwlock_rwunlock(res->lock);
    free_owned(owned);

    if (ret != 0) return make_faiss_error(env, "failed to add vectors");
    return atom_ok;
}

/* nif_add_with_ids_to_index(ref, n, data_binary, ids_binary) */
LOCKED_NIF(nif_add_with_ids_to_index)
static ERL_NIF_TERM nif_add_with_ids_to_index_impl(ErlNifEnv *env, int argc,
                                                   const ERL_NIF_TERM argv[], LockMode mode) {
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
    const char *err = check_rows(n, (size_t)res->dim, sizeof(float), &data_bin,
                                 "binary size mismatch");
    if (!err) err = check_rows(n, 1, sizeof(idx_t), &ids_bin, "binary size mismatch");
    if (err) return make_error_msg(env, err);
    if (n == 0) return atom_ok;

    const void *data, *ids;
    void *owned_data, *owned_ids = NULL;
    if (!aligned_input(&data_bin, _Alignof(float), &data, &owned_data) ||
        !aligned_input(&ids_bin, _Alignof(idx_t), &ids, &owned_ids)) {
        free_owned(owned_data);
        return make_error_msg(env, "out of memory");
    }

    apply_omp_threads();
    if (!acquire_write(res->lock, mode)) {
        free_owned(owned_data);
        free_owned(owned_ids);
        return RESCHEDULE_BLOCKING(nif_add_with_ids_to_index, env, argc, argv);
    }
    int ret = faiss_Index_add_with_ids(res->index, (idx_t)n, (const float *)data,
                                       (const idx_t *)ids);
    enif_rwlock_rwunlock(res->lock);
    free_owned(owned_data);
    free_owned(owned_ids);

    if (ret != 0) return make_faiss_error(env, "failed to add vectors with ids");
    return atom_ok;
}

/* nif_search_index(ref, n, data_binary, k) */
LOCKED_NIF(nif_search_index)
static ERL_NIF_TERM nif_search_index_impl(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[],
                                          LockMode mode) {
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

    const char *err = check_rows(n, (size_t)res->dim, sizeof(float), &data_bin,
                                 "data binary size mismatch");
    if (err) return make_error_msg(env, err);

    if (n == 0) {
        ERL_NIF_TERM empty_d, empty_l;
        enif_make_new_binary(env, 0, &empty_d);
        enif_make_new_binary(env, 0, &empty_l);
        return make_ok(env, enif_make_tuple2(env, empty_d, empty_l));
    }

    size_t dist_size, label_size;
    if (!rows_bytes(n, (size_t)k, sizeof(float), &dist_size) ||
        !rows_bytes(n, (size_t)k, sizeof(idx_t), &label_size)) {
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

    const void *data;
    void *owned;
    if (!aligned_input(&data_bin, _Alignof(float), &data, &owned)) {
        enif_release_binary(&distances_bin);
        enif_release_binary(&labels_bin);
        return make_error_msg(env, "out of memory");
    }

    apply_omp_threads();
    if (!acquire_read(res->lock, mode)) {
        free_owned(owned);
        enif_release_binary(&distances_bin);
        enif_release_binary(&labels_bin);
        return RESCHEDULE_BLOCKING(nif_search_index, env, argc, argv);
    }
    int ret = faiss_Index_search(res->index, (idx_t)n, (const float *)data, (idx_t)k,
                                 (float *)distances_bin.data, (idx_t *)labels_bin.data);
    enif_rwlock_runlock(res->lock);
    free_owned(owned);

    if (ret != 0) {
        enif_release_binary(&distances_bin);
        enif_release_binary(&labels_bin);
        return make_faiss_error(env, "search failed");
    }

    ERL_NIF_TERM distances_term = enif_make_binary(env, &distances_bin);
    ERL_NIF_TERM labels_term = enif_make_binary(env, &labels_bin);
    return make_ok(env, enif_make_tuple2(env, distances_term, labels_term));
}

/* nif_train_index(ref, n, data_binary)
 *
 * n == 0 goes to FAISS too: it decides whether an empty training set is an
 * error (IVF: fewer points than clusters) or a no-op (Flat, which needs no
 * training). aligned_input supplies a valid pointer for the empty input. */
LOCKED_NIF(nif_train_index)
static ERL_NIF_TERM nif_train_index_impl(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[],
                                         LockMode mode) {
    IndexResource *res;
    ErlNifSInt64 n;
    ErlNifBinary data_bin;

    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res) ||
        !enif_get_int64(env, argv[1], &n) ||
        !enif_inspect_binary(env, argv[2], &data_bin)) {
        return make_error_msg(env, "invalid arguments");
    }

    if (n < 0) return make_error_msg(env, "n must be non-negative");
    const char *err = check_rows(n, (size_t)res->dim, sizeof(float), &data_bin,
                                 "data binary size mismatch");
    if (err) return make_error_msg(env, err);

    const void *data;
    void *owned;
    if (!aligned_input(&data_bin, _Alignof(float), &data, &owned)) {
        return make_error_msg(env, "out of memory");
    }

    apply_omp_threads();
    if (!acquire_write(res->lock, mode)) {
        free_owned(owned);
        return RESCHEDULE_BLOCKING(nif_train_index, env, argc, argv);
    }
    int ret = faiss_Index_train(res->index, (idx_t)n, (const float *)data);
    enif_rwlock_rwunlock(res->lock);
    free_owned(owned);

    if (ret != 0) return make_faiss_error(env, "training failed");
    return atom_ok;
}

/* nif_reset_index(ref) */
LOCKED_NIF(nif_reset_index)
static ERL_NIF_TERM nif_reset_index_impl(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[],
                                         LockMode mode) {
    IndexResource *res;
    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res)) {
        return make_error_msg(env, "invalid index reference");
    }

    if (!acquire_write(res->lock, mode)) {
        return RESCHEDULE_BLOCKING(nif_reset_index, env, argc, argv);
    }
    int ret = faiss_Index_reset(res->index);
    enif_rwlock_rwunlock(res->lock);
    if (ret != 0) {
        return make_faiss_error(env, "reset failed");
    }

    return atom_ok;
}

/* nif_reconstruct_batch(ref, n, keys_binary) */
LOCKED_NIF(nif_reconstruct_batch)
static ERL_NIF_TERM nif_reconstruct_batch_impl(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[],
                                               LockMode mode) {
    IndexResource *res;
    ErlNifSInt64 n;
    ErlNifBinary keys_bin;

    if (!enif_get_resource(env, argv[0], INDEX_RESOURCE_TYPE, (void **)&res) ||
        !enif_get_int64(env, argv[1], &n) ||
        !enif_inspect_binary(env, argv[2], &keys_bin)) {
        return make_error_msg(env, "invalid arguments");
    }

    if (n < 0) return make_error_msg(env, "n must be non-negative");
    const char *err = check_rows(n, 1, sizeof(idx_t), &keys_bin, "keys binary size mismatch");
    if (err) return make_error_msg(env, err);

    if (n == 0) {
        ERL_NIF_TERM empty;
        enif_make_new_binary(env, 0, &empty);
        return make_ok(env, empty);
    }

    size_t result_size;
    if (!rows_bytes(n, (size_t)res->dim, sizeof(float), &result_size)) {
        return make_error_msg(env, "size overflow");
    }

    /* The keys are read here, before FAISS sees them, so align first. */
    const void *keys_ptr;
    void *owned;
    if (!aligned_input(&keys_bin, _Alignof(idx_t), &keys_ptr, &owned)) {
        return make_error_msg(env, "out of memory");
    }
    const idx_t *keys = (const idx_t *)keys_ptr;

    /* Contiguous ascending keys (the common case) can use the batched
     * faiss_Index_reconstruct_n instead of one call per key; see
     * has_safe_reconstruct_n for when that is allowed. Unsigned arithmetic:
     * keys are caller-controlled and keys[i-1] + 1 would be signed-overflow
     * UB at INT64_MAX. */
    int contiguous = 1;
    for (ErlNifSInt64 i = 1; i < n; i++) {
        if ((uint64_t)keys[i] != (uint64_t)keys[i - 1] + 1u) {
            contiguous = 0;
            break;
        }
    }

    /* reconstruct_n guards with `i0 + ni <= ntotal`, which signed-overflows
     * to a negative value -- and therefore passes -- once keys[0] nears
     * INT64_MAX. No real index can hold such a range, so reject it rather
     * than hand FAISS a bound it computes as UB. */
    ErlNifBinary result_bin;
    if (!keys_in_range(keys, n) || (contiguous && keys[0] > INT64_MAX - n)) {
        err = "key out of range";
    } else if (!enif_alloc_binary(result_size, &result_bin)) {
        err = "failed to allocate result binary";
    }
    if (err) {
        free_owned(owned);
        return make_error_msg(env, err);
    }

    float *result = (float *)result_bin.data;
    int dim = res->dim;
    int batched = contiguous && has_safe_reconstruct_n(res->index);
    int ret = 0;

    apply_omp_threads();
    if (!acquire_read(res->lock, mode)) {
        free_owned(owned);
        enif_release_binary(&result_bin);
        return RESCHEDULE_BLOCKING(nif_reconstruct_batch, env, argc, argv);
    }
    if (batched && keys[0] + n <= faiss_Index_ntotal(res->index)) {
        ret = faiss_Index_reconstruct_n(res->index, keys[0], (idx_t)n, result);
    } else {
        /* One guarded call per key, outside any parallel region: a key
         * FAISS rejects becomes an error return, never an abort. */
        for (ErlNifSInt64 i = 0; i < n; i++) {
            ret = faiss_Index_reconstruct(res->index, keys[i], result + i * dim);
            if (ret != 0) break;
        }
    }
    enif_rwlock_runlock(res->lock);
    free_owned(owned);

    if (ret != 0) {
        enif_release_binary(&result_bin);
        return make_faiss_error(env, "reconstruct failed");
    }

    return make_ok(env, enif_make_binary(env, &result_bin));
}

/* nif_compute_residuals(ref, n, data_binary, keys_binary) */
LOCKED_NIF(nif_compute_residuals)
static ERL_NIF_TERM nif_compute_residuals_impl(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[],
                                               LockMode mode) {
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
    const char *err = check_rows(n, (size_t)res->dim, sizeof(float), &data_bin,
                                 "binary size mismatch");
    if (!err) err = check_rows(n, 1, sizeof(idx_t), &keys_bin, "binary size mismatch");
    if (err) return make_error_msg(env, err);

    if (n == 0) {
        ERL_NIF_TERM empty;
        enif_make_new_binary(env, 0, &empty);
        return make_ok(env, empty);
    }

    /* The keys are read here, before FAISS sees them, so align first. */
    const void *data, *keys_ptr;
    void *owned_data, *owned_keys = NULL;
    if (!aligned_input(&data_bin, _Alignof(float), &data, &owned_data) ||
        !aligned_input(&keys_bin, _Alignof(idx_t), &keys_ptr, &owned_keys)) {
        free_owned(owned_data);
        return make_error_msg(env, "out of memory");
    }
    const idx_t *keys = (const idx_t *)keys_ptr;

    ErlNifBinary result_bin;
    if (!keys_in_range(keys, n)) {
        err = "key out of range";
    } else if (!enif_alloc_binary(data_bin.size, &result_bin)) {
        err = "failed to allocate result binary";
    }
    if (err) {
        free_owned(owned_data);
        free_owned(owned_keys);
        return make_error_msg(env, err);
    }

    apply_omp_threads();
    if (!acquire_read(res->lock, mode)) {
        free_owned(owned_data);
        free_owned(owned_keys);
        enif_release_binary(&result_bin);
        return RESCHEDULE_BLOCKING(nif_compute_residuals, env, argc, argv);
    }
    /* Not faiss_Index_compute_residual_n: Index::compute_residual_n runs in
     * an OpenMP region and aborts the VM on any key FAISS rejects (see
     * has_safe_reconstruct_n). One guarded call per vector instead. */
    const float *xs = (const float *)data;
    float *residuals = (float *)result_bin.data;
    int dim = res->dim;
    int ret = 0;
    for (ErlNifSInt64 i = 0; i < n; i++) {
        ret = faiss_Index_compute_residual(res->index, xs + i * dim, residuals + i * dim, keys[i]);
        if (ret != 0) break;
    }
    enif_rwlock_runlock(res->lock);
    free_owned(owned_data);
    free_owned(owned_keys);

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

    char *path = dup_cstr(&path_bin);
    if (!path) return make_error_msg(env, "out of memory");

    enif_rwlock_rlock(res->lock);
    int ret = faiss_write_index_fname(res->index, path);
    enif_rwlock_runlock(res->lock);
    enif_free(path);

    if (ret != 0) {
        return make_faiss_error(env, "failed to write index");
    }

    return atom_ok;
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

    char *path = dup_cstr(&path_bin);
    if (!path) return make_error_msg(env, "out of memory");

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
    /* FAISS indexes its device table with this value; a negative device
     * aborts the process inside CUDA instead of returning an error. */
    if (device < 0) return make_error_msg(env, "device must be non-negative");

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
    /* The C API returns a status and writes the count through the pointer. */
    int num_gpus = 0;
    if (faiss_get_num_gpus(&num_gpus) != 0) {
        return make_faiss_error(env, "failed to query GPU count");
    }
    return make_ok(env, enif_make_int(env, num_gpus));
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
    res->d = d;
    res->k = k;
    ERL_NIF_TERM ref = enif_make_resource(env, res);
    enif_release_resource(res);

    return make_ok(env, ref);
}

/* nif_train_clustering(clust_ref, n, data_binary, idx_ref) */
LOCKED_NIF(nif_train_clustering)
static ERL_NIF_TERM nif_train_clustering_impl(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[],
                                              LockMode mode) {
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

    /* d is the cached immutable clustering dimension, so every check and
     * the alignment copy happen before either lock. */
    const char *err = check_rows(n, (size_t)clust_res->d, sizeof(float), &data_bin,
                                 "data binary size mismatch");
    if (!err && idx_res->dim != clust_res->d) err = "quantizer dim mismatch";
    if (err) return make_error_msg(env, err);

    const void *data;
    void *owned;
    if (!aligned_input(&data_bin, _Alignof(float), &data, &owned)) {
        return make_error_msg(env, "out of memory");
    }

    /* Training mutates both the clustering state and the quantizer index.
     * Lock order: index first, then clustering; release in the mirror
     * order. Under LOCK_TRY, failing the second lock releases the first
     * before rescheduling. */
    apply_omp_threads();
    if (!acquire_write(idx_res->lock, mode)) {
        free_owned(owned);
        return RESCHEDULE_BLOCKING(nif_train_clustering, env, argc, argv);
    }
    if (!acquire_write(clust_res->lock, mode)) {
        enif_rwlock_rwunlock(idx_res->lock);
        free_owned(owned);
        return RESCHEDULE_BLOCKING(nif_train_clustering, env, argc, argv);
    }
    int ret = faiss_Clustering_train(clust_res->clustering, (idx_t)n, (const float *)data,
                                     idx_res->index);
    enif_rwlock_rwunlock(clust_res->lock);
    enif_rwlock_rwunlock(idx_res->lock);
    free_owned(owned);

    if (ret != 0) return make_faiss_error(env, "clustering training failed");
    return atom_ok;
}

/* nif_get_clustering_centroids(clust_ref) - returns {k, d, centroids_binary} */
LOCKED_NIF(nif_get_clustering_centroids)
static ERL_NIF_TERM nif_get_clustering_centroids_impl(ErlNifEnv *env, int argc,
                                                      const ERL_NIF_TERM argv[], LockMode mode) {
    ClusteringResource *res;
    if (!enif_get_resource(env, argv[0], CLUSTERING_RESOURCE_TYPE, (void **)&res)) {
        return make_error_msg(env, "invalid clustering reference");
    }

    /* k and d are cached, so the result binary is sized and allocated
     * before the lock. The centroids pointer belongs to the locked
     * clustering object, so the check and the copy stay inside; the
     * critical section records only a status message. */
    size_t bin_size;
    if (!rows_bytes(res->k, (size_t)res->d, sizeof(float), &bin_size)) {
        return make_error_msg(env, "size overflow");
    }
    ErlNifBinary bin;
    if (!enif_alloc_binary(bin_size, &bin)) {
        return make_error_msg(env, "failed to allocate centroids binary");
    }

    const char *err = NULL;
    float *centroids = NULL;
    size_t centroids_size = 0;

    if (!acquire_read(res->lock, mode)) {
        enif_release_binary(&bin);
        return RESCHEDULE_BLOCKING(nif_get_clustering_centroids, env, argc, argv);
    }
    faiss_Clustering_centroids(res->clustering, &centroids, &centroids_size);
    if (!centroids || centroids_size == 0) {
        err = "no centroids available";
    } else if (centroids_size * sizeof(float) != bin_size) {
        err = "centroids size mismatch";
    } else {
        memcpy(bin.data, centroids, bin_size);
    }
    enif_rwlock_runlock(res->lock);

    if (err) {
        enif_release_binary(&bin);
        return make_error_msg(env, err);
    }

    return make_ok(env, enif_make_tuple3(env,
        enif_make_int64(env, (int64_t)res->k),
        enif_make_int64(env, (int64_t)res->d),
        enif_make_binary(env, &bin)));
}

/* ========== NIF: Runtime ========== */

/* nif_set_omp_threads(n) — an atomic store, so a normal scheduler. */
static ERL_NIF_TERM nif_set_omp_threads(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    int threads;
    if (!enif_get_int(env, argv[0], &threads) || threads < 0) {
        return make_error_msg(env, "invalid arguments");
    }
    atomic_store(&g_omp_threads, threads);
    return atom_ok;
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

/* load_info is the initial OpenMP thread count (FaissEx.NIF reads it from
 * `config :faiss_ex, omp_threads: n`); 0 keeps the runtime default. */
static int init_omp(ErlNifEnv *env, ERL_NIF_TERM load_info) {
    int threads;
    if (!enif_get_int(env, load_info, &threads) || threads < 0) return -1;
#ifndef FAISS_EX_NO_OMP
    g_omp_default = omp_get_max_threads();
#endif
    atomic_store(&g_omp_threads, threads);
    return 0;
}

static int on_load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
    (void)priv_data;
    init_atoms(env);
    if (init_omp(env, load_info) != 0) return -1;
    return register_resource_types(env, ERL_NIF_RT_CREATE);
}

static int on_upgrade(ErlNifEnv *env, void **priv_data, void **old_priv_data, ERL_NIF_TERM load_info) {
    (void)priv_data;
    (void)old_priv_data;
    init_atoms(env);
    if (init_omp(env, load_info) != 0) return -1;
    return register_resource_types(env, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER);
}

/* Every entry that may wait on a lock is dirty. The locked compute/mutation
 * NIFs are LOCKED_NIF pairs: registered dirty CPU, waiting (if any) on
 * their <name>_io twin, which enif_schedule_nif reaches without a table
 * entry. The ntotal/is_trained getters do no CPU work, so they are dirty IO
 * and simply block there. The dim getter reads the immutable cached field
 * without the lock, so it stays on a normal scheduler. nif_new_index runs
 * the factory parser and allocates the index (an "IVF<n>" quantizer can be
 * large); nif_get_num_gpus initializes the CUDA runtime on first call. */
static ErlNifFunc nif_funcs[] = {
    /* Index */
    {"nif_new_index", 3, nif_new_index, ERL_NIF_DIRTY_JOB_CPU_BOUND},
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
    {"nif_get_index_ntotal", 1, nif_get_index_ntotal, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"nif_get_index_is_trained", 1, nif_get_index_is_trained, ERL_NIF_DIRTY_JOB_IO_BOUND},
    /* GPU */
    {"nif_index_cpu_to_gpu", 2, nif_index_cpu_to_gpu, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"nif_index_gpu_to_cpu", 1, nif_index_gpu_to_cpu, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"nif_get_num_gpus", 0, nif_get_num_gpus, ERL_NIF_DIRTY_JOB_IO_BOUND},
    /* Clustering */
    {"nif_new_clustering", 2, nif_new_clustering, 0},
    {"nif_train_clustering", 4, nif_train_clustering, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nif_get_clustering_centroids", 1, nif_get_clustering_centroids, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    /* Runtime */
    {"nif_set_omp_threads", 1, nif_set_omp_threads, 0},
};

ERL_NIF_INIT(Elixir.FaissEx.NIF, nif_funcs, on_load, NULL, on_upgrade, NULL)
