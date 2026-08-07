# C NIF Review: perf-binary-fast-path

**Scope:** `git diff master..perf-binary-fast-path -- c_src/faiss_ex_nif.c` plus full read of the branch version of `c_src/faiss_ex_nif.c` (1003 lines). Line numbers below refer to the branch version.

**Changes reviewed:**
1. `int dim` cached in `IndexResource` at wrap time (`c_src/faiss_ex_nif.c:48`, `:167`)
2. Six under-lock `faiss_Index_d()` calls replaced with `res->dim` reads
3. `nif_search_index` restructured — size checks and result allocations hoisted above `enif_rwlock_rlock`
4. `nif_get_index_dim` lock removed, moved from `ERL_NIF_DIRTY_JOB_CPU_BOUND` to normal scheduler (`:990`)

---

## (a) Lock/unlock pairing — every return path walked

Every function that takes a lock was audited return-by-return:

| Function | Lock | Returns inside critical section | Result |
|---|---|---|---|
| `nif_clone_index` (:222–224) | rlock | none | PASS |
| `nif_add_to_index` (:256–270) | rwlock | :261, :265 unlock before return; :270 unlocks after FAISS call | PASS |
| `nif_add_with_ids_to_index` (:300–318) | rwlock | :306, :311 unlock; :318 unlocks | PASS |
| `nif_search_index` (:381–386) | rlock | none — critical section is only `faiss_Index_search` | PASS |
| `nif_train_index` (:418–432) | rwlock | :423, :427 unlock; :432 unlocks | PASS |
| `nif_reset_index` (:448–450) | rwlock | none | PASS |
| `nif_reconstruct_batch` (:501–527) | rlock | :506, :512 unlock; loop break falls through to :527 unlock | PASS |
| `nif_compute_residuals` (:562–588) | rlock | :568, :573, :579 unlock; :588 unlocks | PASS |
| `nif_write_index` (:615–617) | rlock | none | PASS |
| `nif_get_index_ntotal` (:679–681) | rlock | none | PASS |
| `nif_get_index_is_trained` (:692–694) | rlock | none | PASS |
| `nif_index_cpu_to_gpu` (:720–722) | rlock | none | PASS |
| `nif_index_gpu_to_cpu` (:757–759) | rlock | none | PASS |
| `nif_train_clustering` (:860–879) | 2× rwlock (index, then clustering) | :866–867, :871–872 unlock both in reverse order; :878–879 unlock both | PASS |
| `nif_get_clustering_centroids` (:895–921) | rlock | :905, :911, :917 unlock; :921 unlocks | PASS |

Also verified the inverse hazard: the search error paths that no longer unlock (`:359`, `:362`, `:369`, `:374`, `:378`) all execute strictly before `enif_rwlock_rlock` at `:381` — no stray unlock of an unheld lock remains.

**No missing unlocks. No unlock-without-lock.**

## (b) Binary lifecycle — every `enif_alloc_binary` traced

- `nif_search_index` `distances_bin` (:373): labels alloc failure → released (:377); search failure → released (:388); success → `enif_make_binary` (:393). PASS
- `nif_search_index` `labels_bin` (:376): search failure → released (:389); success → `enif_make_binary` (:394). PASS
- `nif_reconstruct_batch` `result_bin` (:511): failure → released (:530); success → `enif_make_binary` (:534). PASS
- `nif_compute_residuals` `result_bin` (:578): failure → released (:590); success → `enif_make_binary` (:594). PASS
- `nif_get_clustering_centroids` `bin` (:916): no failure path after alloc; `enif_make_binary` (:926). PASS

Note: `enif_rwlock_rlock` at `:381` cannot fail-return (ERTS lock ops abort on error), so there is no path between the two allocations and the search call that could skip both the `make_binary` and `release_binary`.

**No leaks.**

## (c) Race analysis: allocations/checks hoisted out of the search lock

**No race.** The reasoning:

- Output buffer sizes depend only on caller-supplied `n` and `k` (`dist_size = n*k*4`, `label_size = n*k*8`, `:365–370`). They do not depend on `ntotal` or any index state. Verified against the FAISS 1.14.3 C API contract (`~/.cache/faiss_ex/faiss-v1.14.3/c_api/Index_c.h:105–106`): `labels`/`distances` are "size n*k" unconditionally — when fewer than `k` neighbors exist, FAISS pads labels with `-1` rather than writing fewer/more entries. So `ntotal` changing between allocation and search cannot cause over- or under-allocation.
- Input validation uses `res->dim`, which is immutable (see (d)); `data_bin.size == n*dim*4` checked at `:361` guarantees `faiss_Index_search` reads exactly the caller's binary. `data_bin` is a term owned by the calling process and valid for the whole NIF call.
- The actual index access (`faiss_Index_search`, `:382`) still runs under the read lock, so `add`/`train`/`reset` cannot mutate index internals mid-search. That is the only invariant the lock ever protected here.

**LOW — behavioral note, not a bug:** result buffers (`n*k*12` bytes) are now allocated *before* blocking on the rlock. If a writer holds the lock for seconds (`train`), concurrent searchers now hold their allocated result binaries while waiting, where on master they blocked first and allocated after acquiring the lock. Peak memory under writer contention rises by the sum of in-flight searchers' result buffers. Bounded by caller-supplied `n*k`; acceptable, but worth knowing.

## (d) Lock-free `dim` read vs. the BEAM memory model

**Safe.** The happens-before chain:

1. **Compiler ordering:** `res->dim = faiss_Index_d(index)` (`:167`) precedes `enif_make_resource(env, res)` (`:168`). `enif_make_resource` is an opaque external call through which `res` escapes, so the compiler must complete the store before the call.
2. **Publication:** for any *other* scheduler thread to call `nif_get_index_dim` with this resource, the term must first travel from the creating process to that thread's current process — via NIF return + message send (message-queue enqueue/dequeue use atomics/locks with release/acquire semantics), ETS (table rwlocks), or persistent_term (release store). Even same-process access from a different scheduler thread requires process migration, which goes through run-queue locks. Every such channel is a synchronization edge, so the reader's load of `res->dim` (`:669`) observes the writer's store.
3. **No post-publication writes:** `dim` is written exactly once, in `wrap_index_resource`, before `enif_make_resource`. The destructor (`:64–75`) never touches it. All five `IndexResource` creation sites go through `wrap_index_resource` (`:206`, `:230`, `:654`, `:741`, `:765`), so no resource can exist with an uninitialized `dim`. With no concurrent writes there is no data race under the C11 model.
4. **Precedent:** this exact publication guarantee was *already* load-bearing on master — `res->lock` and `res->index` are read without synchronization by every NIF (you must read `res->lock` unsynchronized to take the lock at all). Caching `dim` adds no new assumption.

**Immutability claim verified against FAISS v1.14.3 source:** the only post-construction mutators of `Index::d` are `IndexPreTransform::prepend_transform` (`faiss/IndexPreTransform.cpp:50`), `IndexReplicas` (`faiss/IndexReplicas.cpp:188`), and `IndexShards` (`faiss/IndexShards.cpp:96`) — none reachable through this NIF's exposed surface (add/add_with_ids/train/search/reset/reconstruct/compute_residuals/write/clone/GPU transfer). `Clustering::train` only calls `reset`/`train`/`add`/`search` on the quantizer and asserts `index.d == d` (`faiss/Clustering.cpp:81`, `:142–143`) — it never writes `d`. Note that `faiss_read_index` can deserialize an `IndexPreTransform`/`IndexIDMap` etc., but since `prepend_transform`/`addIndex` are not exposed, `d` is frozen once the object exists; `wrap_index_resource` caches it after construction completes in all five call sites.

Type check: `faiss_Index_d` returns `int` (`c_api/Index_c.h:53`, `FAISS_DECLARE_GETTER(Index, int, d)`), matching the `int dim` field — no truncation.

## (e) Normal-scheduler flag on `nif_get_index_dim`

**Safe.** The body (`:663–670`) is `enif_get_resource` (constant-time type/tag check), `enif_make_int`, `enif_make_tuple2` — no locks, no FAISS calls, no I/O, no allocation beyond small heap terms. Sub-microsecond worst case; well under the 1 ms budget. The prior dirty flag existed solely because the function blocked on the rwlock; with the lock gone the flag is correctly dropped. The comment block at `:973–976` accurately documents why ntotal/is_trained stay dirty.

Pre-existing observation (out of scope for this diff, unchanged from master): `nif_get_num_gpus` (`:996`, flag 0) calls `faiss_get_num_gpus()` → `cudaGetDeviceCount()` in GPU builds, whose first call can initialize the CUDA driver and exceed 1 ms. Worth revisiting separately if GPU builds matter.

---

## Findings

| # | Severity | Location | Finding |
|---|---|---|---|
| 1 | LOW | `c_src/faiss_ex_nif.c:373–381` | Search result binaries are now allocated before blocking on the rlock; under a long-held write lock (train), waiting searchers hold `n*k*12` bytes each. Bounded and acceptable; note only. |
| 2 | LOW | `c_src/faiss_ex_nif.c:501–514`, `:562–581` | `nif_reconstruct_batch` and `nif_compute_residuals` still do the `res->dim` read, `safe_mul` checks, and `enif_alloc_binary` inside the rlock. Not a bug (read lock, readers don't block each other), but the same hoist applied to search would shrink these critical sections for consistency. |
| 3 | LOW | `c_src/faiss_ex_nif.c:45–48` | The dim-immutability comment is version-dependent (verified for v1.14.3). Consider adding "re-verify when bumping FAISS_GIT_REV," matching the convention already used for the thread-local error comment at `:126–129`. |
| 4 | INFO (pre-existing) | `c_src/faiss_ex_nif.c:996` | `nif_get_num_gpus` on a normal scheduler calls `cudaGetDeviceCount` in GPU builds; first call may exceed 1 ms. Unchanged by this branch. |

## Verdict

**APPROVE.** No CRITICAL, HIGH, or MEDIUM issues. Lock discipline is intact on every return path, all allocated binaries are consumed or released on every path, the hoisted search allocations introduce no race (output sizes depend only on caller args; the FAISS contract writes exactly `n*k` entries), the lock-free `dim` read is sound under the BEAM/C11 memory model (single pre-publication write, publication via synchronized term transfer, precedent already set by `res->lock`/`res->index`), and the normal-scheduler flag on `nif_get_index_dim` is correct for its constant-time, non-blocking body. Findings are advisory only.
