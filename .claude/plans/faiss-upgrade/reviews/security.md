# Security Audit: faiss_ex NIF layer (HEAD~4..HEAD)

## Executive Summary

The security-relevant boundary is `c_src/faiss_ex_nif.c`, which turns caller-supplied
binaries, sizes, and file paths into raw pointers handed to FAISS. The new code is
notably careful: every `n*d` product goes through `safe_mul`, binary sizes are
validated against the claimed `n` before any pointer cast, interior NUL bytes are
rejected on all three string inputs, and lock/error paths were audited branch by
branch. No BLOCKER-level memory-safety or locking defects found. One WARNING
(signed-overflow UB in a hot loop) and a few SUGGESTIONs below.

## Findings

### WARNING — signed integer overflow (UB) in reconstruct contiguity check
- **Location**: `c_src/faiss_ex_nif.c:476`
- **Issue**: `keys[i] != keys[i - 1] + 1` where `keys` is `idx_t` (`int64_t`,
  signed). A caller passing `keys[i-1] == INT64_MAX` makes `+ 1` signed overflow,
  which is undefined behavior. Keys are attacker/caller-controlled (arbitrary
  `int64` from a client binary), so this is reachable, not theoretical.
- **Impact**: Low in practice — with wrap-around the branch just falls through to
  the safe per-key path — but UB permits the optimizer to do anything, and it sits
  in a loop over untrusted data.
- **Fix**: Compare without overflowing, e.g.
  `if ((uint64_t)keys[i] != (uint64_t)keys[i - 1] + 1u)` or
  `if (keys[i] - 1 != keys[i - 1])` (also UB on INT64_MIN; prefer the unsigned form).

### SUGGESTION — contiguous reconstruct passes unvalidated key0 to FAISS
- **Location**: `c_src/faiss_ex_nif.c:501`
- **Issue**: `faiss_Index_reconstruct_n(res->index, keys[0], (idx_t)n, result)` forwards
  a caller-controlled `keys[0]` (may be negative or `>= ntotal`) plus `n` as a range.
  Bounds enforcement is delegated to FAISS; the non-contiguous path (line 504) checks
  each `ret` per key, but the batched path trusts FAISS to reject a bad start+range.
  Recommend confirming `faiss_Index_reconstruct_n` bounds-checks `[key0, key0+n)`
  against `ntotal` for the index types you expose (Flat/IVF); if not, add
  `keys[0] >= 0 && keys[0] + n <= ntotal` before taking the batched path.

### SUGGESTION — faiss_get_last_error() thread-locality assumption
- **Location**: `c_src/faiss_ex_nif.c:121-127` (`make_faiss_error`)
- **Issue**: Correctness relies on `faiss_get_last_error()` being thread-local so the
  message read here matches the failing call on the same dirty-scheduler thread. This
  holds for current FAISS C API (thread_local `g_last_error`). Worth a one-line comment
  pinning the assumption so a future FAISS bump doesn't silently cross-contaminate error
  strings between concurrent NIF calls. No memory-safety impact.

## Verified Clean

- **enif_alloc_binary release on all failure branches**: `nif_search_index`
  (:358/:362 second-alloc failure releases the first; :373 search failure releases
  both), `nif_reconstruct_batch:492/511`, `nif_compute_residuals:559/571`,
  `nif_get_clustering_centroids:899` — all correct.
- **enif_alloc'd strings freed on every path**: `desc` (:183-194), `path`
  (write :591-599, read :621-628) freed before every return, including error returns.
- **enif_make_binary use-after-free**: none — binaries are only made into terms after
  the FAISS call succeeds; on error they are released and not referenced.
- **wrap_index_resource ownership**: takes ownership and frees the index on both
  lock-create (:146-149) and alloc-resource (:151-156) failure. All callers
  (new/clone/read/gpu) rely on this correctly.
- **rwlock destroyed exactly once**: only in the two destructors (:67-70, :80-83),
  guarded by non-NULL + reset to NULL; create-time failure paths destroy the lock
  before the resource exists (:153, :812). No double-destroy.
- **GPU tuple partial-failure (nif_index_cpu_to_gpu:711-728)**: on
  `wrap_index_resource` failure the gpu_index is freed by wrap; `gpu_res` is already
  owned by an env-referenced resource whose destructor frees it when the error term
  is returned — no leak, comment at :722 is accurate.
- **Locking — no return-while-holding / double-unlock / order violation**: every
  `rlock`/`rwlock` has a matching unlock on all success and error branches
  (add, search, train, reset, reconstruct, residuals, write, getters). Two-lock path
  `nif_train_clustering:843-862` follows the documented order (index then clustering)
  and unlocks on every early return.
- **Input validation**: `safe_mul` guards every `n*d`, `n*d*4`, `n*k`, `n*8`
  computation; each binary size is checked against the claimed `n` before the pointer
  cast; `n<0`, `n==0`, `k<=0`, `dim<=0`, `d<=0` all handled; empty-input fast paths
  validate `size==0`.
- **NUL-byte rejection coverage**: all three string inputs covered — `nif_new_index`
  desc (:180), `nif_write_index` path (:589), `nif_read_index` path (:619). No other
  string inputs exist in the surface.
- **Integer conversions**: `ErlNifSInt64 n` cast to `size_t`/`idx_t` only after the
  `n >= 0` guard; `result + i * dim` (:504) cannot overflow because `n*dim*4 <= SIZE_MAX`
  was already enforced, bounding `i*dim` below `INT64_MAX`.

## Path Handling

`nif_write_index`/`nif_read_index` accept arbitrary filesystem paths by design (this
is a library-level FAISS I/O API, not a web boundary). NUL-byte rejection is present;
nothing beyond the documented contract is violated. No action needed.

## Web / Auth Checklist

N/A — no web, auth, SQL, template, or session surface in this project. Iron Laws
1-6 do not apply to a NIF binding.

## Tools to Recommend (agent has no Bash access)

- `clang --analyze c_src/faiss_ex_nif.c` (or `scan-build mix compile`) to
  machine-confirm the error-path allocation audit.
- Compile the NIF with `-fsanitize=address,undefined` and run
  `mix test` — the UBSan build would flag the line 476 signed overflow directly.
- `mix test` under ASan while exercising failure paths (bad sizes, huge `n`,
  non-contiguous keys) to confirm no leak/UAF regressions.
