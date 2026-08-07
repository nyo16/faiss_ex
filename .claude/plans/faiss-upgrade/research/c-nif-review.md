# C NIF Layer Review — `c_src/faiss_ex_nif.c`

Reviewed: `c_src/faiss_ex_nif.c` (846 lines), `lib/faiss_ex/nif.ex`, `Makefile`, plus
verification against the pinned FAISS v1.10.0 source at `~/.cache/faiss_ex/faiss-v1.10.0/`
and the current FAISS `main` C API headers (latest release v1.14.3, 2026-06-12).

Severity scale: **critical** (VM crash / memory unsafety reachable from Elixir),
**high** (whole-VM degradation), **medium** (correctness/perf gaps to fix during the
upgrade), **low** (nits, hardening, style).

---

## Critical

### C1. No locking — concurrent index mutation + read is undefined behavior [critical]

There is no `ErlNifRWLock`/`ErlNifMutex` anywhere in `c_src/faiss_ex_nif.c`. The
`IndexResource` struct (c_src/faiss_ex_nif.c:27-29) wraps a bare `FaissIndex*`, and NIF
resources are freely shareable across BEAM processes. With multiple dirty CPU schedulers,
the following can run simultaneously on the *same* `FaissIndex*`:

- `faiss_Index_add` (c_src/faiss_ex_nif.c:205), `faiss_Index_add_with_ids` (:243),
  `faiss_Index_train` (:349), `faiss_Index_reset` (:365) — mutators
- `faiss_Index_search` (:308), `faiss_Index_reconstruct` (:419),
  `faiss_Index_compute_residual_n` (:475), `faiss_write_index_fname` (:500),
  `faiss_clone_index` (:163), `faiss_index_cpu_to_gpu` (:599) — readers

FAISS indexes are safe for **concurrent reads only**. Concurrent add+search is undefined
behavior: `IndexFlat` stores vectors in a `std::vector<float>` that reallocates on add
(readers dereference freed memory), IVF inverted lists grow, HNSW mutates the graph.
This means two plain Elixir processes calling `Index.add/2` and `Index.search/3` on the
same index can segfault the entire VM — a NIF must never allow pure Elixir code to do
that.

Same problem for clustering: `nif_train_clustering` (c_src/faiss_ex_nif.c:739) mutates
the clustering object (and the quantizer index) while `nif_get_clustering_centroids`
(:761-776) takes a raw pointer into the centroids `std::vector` and `memcpy`s from it —
a concurrent train makes that a read of freed memory.

**Fix:** embed a lock in each resource:

```c
typedef struct {
    FaissIndex *index;
    ErlNifRWLock *lock;   /* enif_rwlock_create in each alloc site,
                             enif_rwlock_destroy in destructor (line 43) */
} IndexResource;
```

- `enif_rwlock_rwlock` (write): add, add_with_ids, train, reset, train_clustering
  (on both the clustering resource and the quantizer index resource).
- `enif_rwlock_rlock` (read): search, reconstruct_batch, compute_residuals, write_index,
  clone (source), cpu_to_gpu (source), gpu_to_cpu (source), get_clustering_centroids.
- The trivial getters (`nif_get_index_dim`/`ntotal`/`is_trained`,
  c_src/faiss_ex_nif.c:547-575) should take the read lock too; `ntotal` changes under add
  (a torn/racy read is benign on 64-bit but formally a data race).
- Destructor needs no lock: the resource is only destructed when no term references it,
  so no NIF call can be in flight.

A rwlock preserves concurrent searches (FAISS search is itself OpenMP-parallel), so this
costs nothing in the common read-heavy workload.

---

## High

### H1. `nif_reset_index` runs on a normal scheduler [high]

`{"nif_reset_index", 1, nif_reset_index, 0}` (c_src/faiss_ex_nif.c:827).
`faiss_Index_reset` frees all stored vectors — O(ntotal) work plus potentially gigabytes
of deallocation (and for a GPU index, device synchronization). On a large index this
blocks a normal scheduler far beyond the ~1 ms budget and degrades the whole VM
(scheduler collapse). Flag it `ERL_NIF_DIRTY_JOB_CPU_BOUND` like the other index
operations.

---

## Medium

### M1. Every `cpu_to_gpu` allocates a private `StandardGpuResources` [medium]

`nif_index_cpu_to_gpu` (c_src/faiss_ex_nif.c:592-599) unconditionally calls
`faiss_StandardGpuResources_new` per transfer, and the NIF API offers no way to pass an
existing GPU-resources ref (nif.ex:27). Each `StandardGpuResources` owns cuBLAS handles,
streams, and a temp-memory reservation, so N indexes on one device multiply GPU memory
N-fold. FAISS's intended pattern is one `StandardGpuResources` shared across indexes on
a device. During the upgrade, add an arity that accepts an existing
`GPU_RESOURCES_RESOURCE_TYPE` ref (the Elixir struct already carries
`gpu_resources_ref`, lib/faiss_ex/index.ex:43).

### M2. `nif_reconstruct_batch` loops single-key `faiss_Index_reconstruct` [medium]

c_src/faiss_ex_nif.c:418-424 calls `faiss_Index_reconstruct` once per key. Each call
crosses the C API `try/catch` wrapper plus a virtual dispatch — for large n this is much
slower than a batch call. Verified against FAISS `main`: the C API still does **not**
expose C++ `reconstruct_batch`, so the loop is the only fully general option, but a fast
path via `faiss_Index_reconstruct_n` when the keys are a contiguous ascending range
(the common case after plain `add`) would remove the per-key overhead. At minimum this is
correctly on a dirty CPU scheduler (c_src/faiss_ex_nif.c:828).

### M3. Embedded NUL bytes silently truncate paths and factory strings [medium]

`nif_new_index` (c_src/faiss_ex_nif.c:129-132), `nif_write_index` (:495-498), and
`nif_read_index` (:521-524) copy the binary and append `'\0'`, but never reject interior
NULs. A path binary `"/data/a\0.index"` silently writes `/data/a`. Add
`if (memchr(bin.data, 0, bin.size)) return make_error_msg(env, "string contains NUL");`.

### M4. `nif_get_clustering_centroids` on a normal scheduler with unbounded memcpy [medium]

c_src/faiss_ex_nif.c:842 (flag 0) — the `memcpy` at :776 is `k*d*4` bytes. Typical
k-means output is small, but k and d are caller-controlled and unbounded (e.g.
k=1M, d=128 → 512 MB memcpy on a normal scheduler). Either move to dirty CPU or accept
and document. (The data race with concurrent `train` is covered by C1.)

---

## Low

### L1. Type errors return `{:error, "invalid arguments"}` instead of badarg [low]

Every argument-decode failure (e.g. c_src/faiss_ex_nif.c:122-126, :187-191) returns an
error tuple. Idiomatic NIFs raise `enif_make_badarg(env)` for malformed arguments —
programmer errors then crash with a stack trace instead of being conflated with runtime
FAISS errors that callers legitimately match on.

### L2. `make_atom` does a name lookup on every call [low]

c_src/faiss_ex_nif.c:74-80 — `ok`/`error`/`true`/`false` are looked up by string on every
NIF return. Cache the four `ERL_NIF_TERM` atoms in statics during `on_load` (atoms are
valid process-independently). Micro-optimization; measurable only on hot small calls.

### L3. Alignment assumption on input binary payloads [low]

Casts like `(const float *)data_bin.data` (c_src/faiss_ex_nif.c:205, :309-311) and
`(const idx_t *)keys_bin.data` (:415, :472) assume 4/8-byte alignment. Erlang
sub-binaries (`binary_part/3`) can point at arbitrary byte offsets, making these loads
formally UB (harmless on x86-64/ARMv8 for these types). `FaissEx.Shared` always builds
fresh aligned binaries, so this is unreachable through the public API today — note it if
the NIF module ever becomes public.

### L4. `new_clustering` / `new_index` accept non-positive d/k/dim [low]

`nif_new_clustering` (c_src/faiss_ex_nif.c:687-696) passes d/k straight to
`faiss_Clustering_new`, which stores them unchecked. Negative d later surfaces as a
confusing `"size overflow"` from `safe_mul` in train (:732); negative k as
`"no centroids available"`. No memory unsafety (verified the safe_mul guards catch the
`(size_t)` wraparound), but validate `d > 0 && k > 0` up front for sane errors. Same for
`dim` in `nif_new_index` (:122) — FAISS catches it, but the error is opaque.

### L5. Makefile links `-lstdc++`; `-lc++` is correct on Darwin [low]

Makefile:45. Verified it links on this toolchain (modern ld64 aliases it to libc++), but
`-lc++` is the honest flag on macOS. The NIF itself is pure C — the C++ runtime
dependency arrives transitively via `libfaiss_c`, so the flag is likely droppable
entirely; keep it only if static-linking scenarios need it.

### L6. `k > ntotal` semantics are pass-through — document [low]

`nif_search_index` correctly rejects `k <= 0` (c_src/faiss_ex_nif.c:268) and gracefully
handles absurd k (an `n*k` allocation failure at :300 returns an error rather than
crashing — good). When `k > ntotal`, FAISS pads labels with `-1` and distances with
sentinel values; the NIF returns the full `n*k` buffers and lib/faiss_ex/index.ex:144-145
chunks them as-is. Correct behavior, but callers see `-1` labels — worth documenting in
`FaissEx.Index.search/3`.

### L7. `load_nif` doesn't handle `:code.priv_dir` failure [low]

lib/faiss_ex/nif.ex:6 — `:code.priv_dir(:faiss_ex)` returns `{:error, :bad_name}` when
the app isn't loaded; `:filename.join` then raises an opaque error. Cosmetic.

### L8. `n == 0` search path skips input validation [low]

c_src/faiss_ex_nif.c:270-281 returns empty result binaries before checking
`data_bin.size` (any garbage binary is accepted when n=0), and allocates two 0-byte
binaries where `enif_make_new_binary(env, 0, &term)` would be simpler. Cosmetic;
`n == 0` handling is otherwise consistent across add/train/reconstruct/residuals.

---

## Verified sound (no action needed)

- **Error-path frees**: every FAISS object created before a failed
  `enif_alloc_resource` is freed — c_src/faiss_ex_nif.c:142-146 (new_index), :168-171
  (clone), :534-538 (read_index), :643-647 (gpu_to_cpu), :701-704 (new_clustering), and
  the two-stage GPU path :605-624 (on the second alloc failure, `gpu_res` is already
  owned by the first resource whose term is garbage in `env`, so GC frees it — the
  comment at :619 is accurate). No leaks found.
- **Binary ownership**: `enif_alloc_binary` results are either passed to
  `enif_make_binary` (ownership transferred) or `enif_release_binary`d on every error
  path (:300-316, :410-426, :459-481, :772-781). `enif_make_new_binary` is used correctly
  for env-owned error strings (:90-96). No double free, no leak.
- **Overflow**: all `n * d * sizeof(T)` computations go through `safe_mul`
  (c_src/faiss_ex_nif.c:107-111) after an `n >= 0` check; input binary sizes are compared
  against the exact expected size before FAISS sees the pointers. Solid.
- **`enif_get_*` returns**: all checked.
- **Error strings**: `faiss_get_last_error` is backed by a `thread_local
  std::exception_ptr` in v1.10.0 (c_api/error_impl.cpp:15) — no cross-thread race with
  dirty schedulers, and the returned pointer (valid only until the next FAISS call on
  that thread) is copied into a binary immediately (c_src/faiss_ex_nif.c:98-104).
  Correct. Re-verify the thread_local at upgrade time.
- **GPU destructor ordering is safe**: `faiss::gpu::GpuIndex` holds
  `std::shared_ptr<GpuResources> resources_` (faiss/gpu/GpuIndex.h:179), and
  `StandardGpuResources` is just a provider wrapping
  `shared_ptr<StandardGpuResourcesImpl>`. If the `FaissGpuResources` NIF resource is
  GC'd before the GPU index resource, the impl survives via the index's shared_ptr — no
  use-after-free. The Elixir struct retaining `gpu_resources_ref`
  (lib/faiss_ex/index.ex:296) is belt-and-braces, not load-bearing. One note: the GPU
  index destructor (`faiss_Index_free`, :47) can run on a normal scheduler thread during
  GC and performs CUDA synchronization — unavoidable with resource destructors,
  acceptable.
- **Dirty flags otherwise correct** (c_src/faiss_ex_nif.c:819-843): add/search/train/
  clone/reconstruct/residuals/train_clustering → DIRTY_CPU; write/read index and GPU
  transfers → DIRTY_IO; getters and constructors normal. Only reset (H1) and
  centroids (M4) are misassigned; nothing trivially fast is needlessly dirty
  (index_factory in `nif_new_index` on a normal scheduler is fine — construction is
  allocation-bound and cheap; keep an eye on exotic factory strings).
- **erl_nif modernity**: nothing deprecated. `ERL_NIF_INIT` with NULL reload slot (:845),
  `enif_open_resource_type` with `ERL_NIF_RT_CREATE|ERL_NIF_RT_TAKEOVER` on upgrade
  (:816), destructors preserved across takeover. All current practice.
- **io_flags** pass through untouched (:517→:527, default 0 from
  lib/faiss_ex/index.ex:248).

## FAISS v1.10.0 → v1.14.3 upgrade notes

- **No breaking C API changes** between 1.10.0 and 1.14.3 (checked CHANGELOG for
  1.11.0/1.12.0/1.13.x/1.14.x). Changes are additive: IndexBinaryIVF exposed to the
  C API in 1.12.0 (with new "Ex"-suffixed extended entry points), IndexBinaryFlat in
  1.14.1.
- All signatures this NIF uses were verified identical on current `main`:
  `faiss_Index_{add,add_with_ids,search,train,reset,reconstruct,reconstruct_n,
  compute_residual_n}`, the `d`/`ntotal`/`is_trained` getters, and the GPU cloner cast
  `(FaissGpuResourcesProvider *)` sanctioned by `FAISS_DECLARE_CLASS_INHERITED`
  (c_api/gpu/StandardGpuResources_c.h:23).
- `faiss_Index_search_with_params` exists (already in 1.10) — worth exposing during the
  upgrade for per-query nprobe/efSearch.
- GPU: CUDA 11 support deprecated as of 1.12.0 — CUDA 12 required. The RAFT→cuVS
  migration (1.11.0) renames cmake flags (`FAISS_ENABLE_RAFT` → `FAISS_ENABLE_CUVS`);
  the Makefile doesn't set either, so no change needed unless cuVS is wanted.
- Makefile: bump `FAISS_GIT_REV ?= v1.10.0` (Makefile:11) to `v1.14.3`; the cache dir
  scheme (`faiss-<rev>`) already isolates builds per revision.
