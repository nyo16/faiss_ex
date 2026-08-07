# Performance Analysis: faiss_ex (2026-07-12)

Baseline (measured, Apple M4 Max, FAISS v1.14.3): NIF-side add of 10k×128
vectors ≈ 0.4ms; end-to-end `Index.add` of the same batch ≈ 14.5ms — the gap
is list→binary encode + BEAM major-GC pauses scanning the caller's ~20MB list
corpus. Search 100 queries k=10 ≈ 2.2ms end-to-end.

Tracks: Elixir data path + C NIF layer (research/ has full reports).
No Ecto/LiveView/OTP-process surface exists in this project.

## Priority matrix

| Priority | Finding | Impact | Effort |
|---|---|---|---|
| PLAN | 1. Binary-in/binary-out fast-path API | High | High |
| DO FIRST | 2. Flat-accumulator batch encode | Medium | Low |
| QUICK WIN | 3. Cache `dim` in IndexResource (C) | Medium-low | Low |
| QUICK WIN | 4. Single-pass decode (kill decode-then-chunk) | Low-medium | Low |
| QUICK WIN | 5. Fold `length(keys)` into key encode | Low | Low |
| SKIP | Pre-lock result alloc in search (fold into #3 if done) | Low | Low |

## Findings

### 1. Binary-in/binary-out fast path — PLAN
**Impact**: the ONLY lever that removes the measured ~10–14ms gap on
`add`/`train` with large batches; the gap is major-GC scanning the caller's
list corpus during traversal, so no encode tweak can eliminate it.
**Location**: new API surface (deferred-features list from the v0.2.0 plan).
**Fix**: accept/return `{n, f32_binary}` directly on `add/2`, `train/2` (and
optionally `search/3` results). Pairs with the deferred
`search_with_params`/`remove_ids` follow-up plan.
**Estimated gain**: `Index.add` 10k×128: 14.5ms → ~0.4ms for callers that
already hold binaries (embeddings from HTTP/DB arrive as binaries anyway).

### 2. Flat-accumulator batch encode — DO FIRST
**Impact**: removes N intermediate per-row binaries + `Enum.reverse` + final
`IO.iodata_to_binary` full copy per batch call.
**Location**: `lib/faiss_ex/shared.ex:39-67` (`encode_batch`/`encode_row!`).
**Current**: per-row binary, accumulate list, reverse, iodata flatten.
**Fix**: thread one growing binary accumulator across the whole batch (same
append-optimization pattern `encode_row` already uses), keeping the per-row
count for the row-indexed ArgumentError.
**Estimated gain**: halves encode-side allocations; shaves the non-GC part of
the encode cost (~1.3ms fast-path for 1000×128).

### 3. Cache `dim` in IndexResource — QUICK WIN (C)
**Impact**: `nif_get_index_dim` becomes lock-free on a normal scheduler (no
dirty round-trip); removes 6 redundant `faiss_Index_d` calls under lock in
add/train/search/reconstruct/residuals; enables pre-lock result allocation in
search (shrinks the read critical section to just `faiss_Index_search`).
**Location**: `c_src/faiss_ex_nif.c:148` (`wrap_index_resource`), getter :660.
**Fix**: store `int dim` in the resource struct at creation (immutable after
construction), read it without the lock.
**Estimated gain**: getter latency µs→ns and no dirty hop; marginal add/search
gain; matters under sustained writer contention.

### 4. Single-pass decode — QUICK WIN
**Impact**: decode builds a flat n*k list then `Enum.chunk_every/2` rebuilds
it as sublists — 2× list allocation on every search/reconstruct/residuals/
centroids result.
**Location**: `lib/faiss_ex/index.ex:189-190, 249, 269`,
`lib/faiss_ex/clustering.ex:95`.
**Fix**: `Shared.binary_to_float_rows(bin, row_len)` decoding by binary
splitting per row in one pass.
**Estimated gain**: negligible at 100×k=10; meaningful for large
`reconstruct`/`get_centroids` batches (halves decode allocations).

### 5. Fold length+encode for keys — QUICK WIN (tiny)
**Location**: `lib/faiss_ex/index.ex:244-245` (`reconstruct/2`).
**Fix**: one traversal returning `{n, keys_bin}` like `encode_vectors!`.
**Estimated gain**: one O(n) pass saved; small.

## Verified NOT waste (don't "fix")

- `enif_alloc_binary` → `enif_make_binary` transfers ownership (no copy) and
  is required for the `enif_release_binary` error path.
- Centroids memcpy under rlock is mandatory (pointer belongs to locked object).
- Contiguity pre-scan in reconstruct is negligible vs the batched-call win.
- Clustering does no per-call rework (quantizer reuse already landed in 0.2.0).
