# Performance Analysis: c_src/faiss_ex_nif.c

Scope: C-side waste only. Locking design (per-resource ErlNifRWLock) and dirty-scheduler
placement for mutating/blocking ops are correct and out of scope for removal.

Baseline: add 10k×128 f32 ~0.4ms; search 100×k=10 ~2.2ms end-to-end (Elixir encode
dominates). All findings below are evaluated against that scale — most are sub-microsecond
and not worth chasing further; flagged mainly for completeness/future-proofing at larger n.

## Findings

### 1. `nif_get_index_dim` forces a dirty-scheduler round-trip for an immutable value
**Impact: Medium (only if called per-op from Elixir) | Effort: Low**
`c_src/faiss_ex_nif.c:660-670`

`d` (vector dimensionality) is fixed at index construction (`faiss_index_factory`,
`faiss_clone_index`, `faiss_read_index_fname`, GPU transfer all produce an index whose `d`
never changes for its lifetime). Currently every `nif_get_index_dim` call:
- dispatches to a dirty CPU scheduler (queue/context-switch overhead, dozens of µs under load)
- takes `res->lock` (rlock) even though no other thread can ever change `d`

Current:
```c
enif_rwlock_rlock(res->lock);
int dim = faiss_Index_d(res->index);
enif_rwlock_runlock(res->lock);
return make_ok(env, enif_make_int(env, dim));
```

Proposed: add `int dim;` to `IndexResource`, populate it once in `wrap_index_resource`
(call `faiss_Index_d` there — safe, single owner, no lock needed yet since the resource
term hasn't been published). Then:
```c
static ERL_NIF_TERM nif_get_index_dim(...) {
    IndexResource *res;
    if (!enif_get_resource(...)) return make_error_msg(...);
    return make_ok(env, enif_make_int(env, res->dim)); /* no lock, no dirty sched needed */
}
```
Register with flags `0` (normal scheduler) instead of `ERL_NIF_DIRTY_JOB_CPU_BOUND`.
This also lets every other function (`add`, `search`, `train`, `reconstruct_batch`,
`compute_residuals`) read `res->dim` instead of calling `faiss_Index_d(res->index)` under
lock — trades a virtual C++ call for a struct read, negligible but free.

Note: `ntotal` and `is_trained` genuinely mutate (add/reset, train) and must stay
lock-guarded + dirty; do not apply this to them.

Estimated gain: saves one dirty-scheduler dispatch (typically several µs, more under
scheduler contention) per `dim` call. Not visible at the 100-query baseline but matters if
`Index.dim/1` is called per-request in a hot path (e.g. to size a buffer before every
search) rather than cached Elixir-side.

### 2. Search: result-binary allocation done while holding the rlock
**Impact: Low at current scale | Effort: Low**
`c_src/faiss_ex_nif.c:348-383`

`enif_alloc_binary` for `distances_bin` (n·k·4 bytes) and `labels_bin` (n·k·8 bytes) happens
inside the `rlock`/`runlock` window, alongside the size-overflow arithmetic. At baseline
scale (100×10 = 4000B + 8000B ≈ 12KB) this is sub-microsecond and not measurable against
2.2ms. It is a genuine (if small) unnecessary lock-hold extension: rlock is shared-safe for
concurrent searches, but a **writer** (add/train) waiting to acquire the write lock is
blocked for the duration every reader holds it, so extra time under rlock (even allocation
time) delays writers proportionally. Once dim is cached per finding #1, `nd`/`dist_size`/
`label_size` computation and both `enif_alloc_binary` calls can move entirely before
`enif_rwlock_rlock`, shrinking the locked section to just the `faiss_Index_search` call
itself.

Current: alloc happens between `rlock`/`runlock`.
Proposed: compute sizes and allocate both binaries before `enif_rwlock_rlock`; only
`faiss_Index_search` (and the mismatch check against `res->dim`, which no longer needs the
lock either) run inside the lock.

Estimated gain: negligible at n=100 (allocation is a few µs out of 2.2ms); becomes
meaningful only under sustained writer contention (frequent concurrent `add`/`train`) at
much larger batch sizes where malloc of multi-MB result buffers could stretch into
milliseconds. Low priority — do only opportunistically alongside #1 since it shares the
dim-caching prerequisite.

### 3. Result-binary handling (search/reconstruct/residuals): no double copy — confirmed correct, no change
**Impact: N/A (verification only)**
`c_src/faiss_ex_nif.c:369-394, 508-532, 575-593`

Pattern used everywhere: `enif_alloc_binary` → FAISS writes directly into `bin.data` → on
success `enif_make_binary(env, &bin)` transfers ownership into the term without copying; on
failure `enif_release_binary(&bin)` frees it. This is the *correct* pattern for FFI calls
that can fail after allocation — `enif_make_new_binary` returns a term-bound buffer
immediately and offers no clean release path if the subsequent FAISS call errors (the
unused term would just await normal GC, no crash, but the code would lose the explicit
error-path symmetry it has now). **Do not "simplify" to `enif_make_new_binary` here** —
current code is already the lower-overhead, correct choice given the fallible call after
allocation. No avoidable copy exists in any of these three functions.

### 4. `get_clustering_centroids` memcpy under rlock — confirmed necessary, no change
**Impact: N/A (verification only)**
`c_src/faiss_ex_nif.c:895-921`

`faiss_Clustering_centroids` returns a pointer into the clustering object's own buffer, not
a copy. A concurrent writer (`nif_train_clustering`) could reallocate/mutate that buffer the
instant the lock is released, so the `memcpy` into the newly allocated result binary must
stay inside the `rlock`/`runlock` window. No safe shrink is possible here — confirmed as
already optimal.

### 5. `reconstruct_batch` contiguity pre-scan — confirmed negligible, no change
**Impact: N/A (verification only)**
`c_src/faiss_ex_nif.c:491-497`

The O(n) scan over `keys` happens before the lock is taken, using only caller-supplied
binary data (no shared-state risk). Its cost (a tight loop of n int64 comparisons) is
orders of magnitude cheaper than either path it selects between (`reconstruct_n` batched
call, or the O(n) per-key `faiss_Index_reconstruct` loop it lets you avoid). No action
needed — the win from choosing the batched path dominates the scan cost even for very large
n.

### 6. Repeated `faiss_Index_d(res->index)` calls under lock across add/train/search/reconstruct/residuals
**Impact: Low | Effort: Low (bundled with #1)**
`c_src/faiss_ex_nif.c:253, 297, 350, 418, 501, 562`

Each of these 6 call sites re-fetches `dim` via a C API call every invocation, while holding
the lock. `faiss_Index_d` is a trivial getter (single virtual dispatch, no computation), so
this is not itself expensive — but combined with #1's resource-struct caching, all 6 sites
can read `res->dim` directly, removing 6 virtual-call sites from the locked critical section
for free once the field exists. Bundle this change with #1 rather than doing it standalone.

## Non-findings (checked, no waste found)

- No per-call heap allocations exist outside the necessary result buffers and the
  cold-path C-string null-termination buffers (`nif_new_index`, `nif_write_index`,
  `nif_read_index` — these run once per index lifetime / once per I/O op, not in the hot
  add/search loop).
- `safe_mul` overflow checks are O(1) branch-only, no measurable cost.
- GPU transfer functions are correctly IO_BOUND dirty jobs; not in the measured hot path.
- No evidence of redundant binary copies anywhere in the file — every `enif_make_binary`
  correctly consumes a previously `enif_alloc_binary`'d buffer with no intermediate copy.

## Summary Table

| # | Location | Impact | Effort | Action |
|---|----------|--------|--------|--------|
| 1 | `nif_get_index_dim` (:660) + `IndexResource` struct (:42) | Medium (usage-dependent) | Low | Cache `dim` at construction, drop lock + dirty flag for the getter |
| 2 | `nif_search_index` (:348-383) | Low | Low | Move size calc + alloc before rlock (do alongside #1) |
| 3 | search/reconstruct/residuals binary handling | N/A | — | Confirmed correct, no change |
| 4 | `get_clustering_centroids` (:895) | N/A | — | Confirmed necessary, no change |
| 5 | `reconstruct_batch` pre-scan (:491) | N/A | — | Confirmed negligible, no change |
| 6 | Repeated `faiss_Index_d` calls (6 sites) | Low | Low | Free side-effect of #1 |
