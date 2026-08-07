# Elixir Data-Path Performance Analysis

Scope: `lib/faiss_ex/shared.ex`, `lib/faiss_ex/index.ex`, `lib/faiss_ex/clustering.ex`,
`bench/faiss_ex_bench.exs`. Baseline: NIF-side `add` 10k×128 f32 ≈0.4ms vs end-to-end
`Index.add` ≈14.5ms (gap = encode + major-GC on the caller's ~20MB list corpus);
`search` 100×k=10 end-to-end ≈2.2ms.

## Finding 1 — `encode_batch` builds N intermediate row-binaries + reverse + final flatten copy
- **Impact**: medium (allocations/GC pressure per `add`/`train`/`search` call; not the
  dominant 10ms term, but adds avoidable garbage on top of it)
- **Effort**: low (single-file, `shared.ex` only)
- **Location**: `lib/faiss_ex/shared.ex:39-49` (`encode_batch`) + `:51-67` (`encode_row!`)
- **Current pattern**: for a batch of N rows, each row is encoded into its own small
  binary via a tail-recursive append accumulator (`encode_row!`, correctly triggers
  the BEAM single-owner binary-append optimization — that part is fine). The N row
  binaries are consed into a list, `Enum.reverse/1`'d, then passed to
  `IO.iodata_to_binary/1`, which does one full-size copy (e.g. 5.12MB for
  10k×128×4B) to flatten them into the final contiguous binary. Net: N extra
  heap binary objects + a list of N cons cells + one reverse pass + one full copy.
- **Proposed pattern**: drop the per-row binary/list nesting; thread a single
  growing accumulator binary across the *entire* batch (flat float counter reset
  at `dim`, same append-optimization semantics already used in `encode_row`),
  e.g. `encode_batch(rows, dim, n, row_index, count, acc)` appending directly with
  `<<acc::binary, f::float-32-native>>`. This removes the N-binary list, the
  reverse, and the final `iodata_to_binary` copy — one accumulator, amortized O(1)
  appends via the append optimization.
- **Estimated gain**: removes ~N small binary allocations (10k × ~512B headers) and
  the terminal 5MB flatten copy; likely shaves low-single-digit ms off the 14.5ms
  `add` figure and reduces bytes-copied/GC-trigger frequency proportionally to N.
  Does **not** touch the ~10ms major-GC term (see Finding 4) — that's inherent to
  reading the caller's list, not to how the binary is assembled.
- Note: `IO.iodata_to_binary` vs `:erlang.list_to_binary` vs a binary comprehension
  over a pre-flattened list were considered — a binary comprehension would require
  flattening the nested lists first (an *extra* full-size list pass before encoding),
  strictly worse. `iodata_to_binary` is the right choice *if* iodata is kept at all;
  the real win is skipping the iodata step entirely as above.

## Finding 2 — decode-then-chunk double allocation (`binary_to_floats` |> `Enum.chunk_every/2`)
- **Impact**: low-medium, scales with n*k (biggest for `get_centroids`/`reconstruct`
  with large batches; negligible for the measured 100×k=10 search case)
- **Effort**: low (single-file, `shared.ex`)
- **Locations**: `lib/faiss_ex/index.ex:189-190` (search), `:249` (reconstruct),
  `:269` (compute_residuals); `lib/faiss_ex/clustering.ex:95` (get_centroids)
- **Current pattern**: `binary_to_floats/1` walks the binary and builds one flat
  list of n*k cons cells; `Enum.chunk_every(k)` then walks that flat list and
  builds n new k-length lists, discarding the flat list's spine immediately.
  Total garbage: ~2×(n*k) cons cells instead of ~1×(n*k).
- **Proposed pattern**: single-pass binary-to-nested-list, splitting on
  `k*4`-byte (or `dim*4`) boundaries directly, e.g.:
  ```elixir
  defp decode_rows(<<>>, _row_bytes, acc), do: Enum.reverse(acc)
  defp decode_rows(bin, row_bytes, acc) do
    <<row::binary-size(row_bytes), rest::binary>> = bin
    decode_rows(rest, row_bytes, [for(<<f::float-32-native <- row>>, do: f) | acc])
  end
  ```
  builds only the final n*k cons cells once, no intermediate flat list.
- **Estimated gain**: halves decode-side list allocations. Given search end-to-end
  is already ≈2.2ms (dominated by NIF dispatch + 100×128 encode, not this decode of
  100×10=1000 floats), the visible win there is sub-millisecond. Worth doing for
  `reconstruct`/`get_centroids` call sites with larger n*d, where it's a cleaner
  proportional win. Not worth doing in isolation for search alone.

## Finding 3 — `reconstruct/2` traverses `keys` twice
- **Impact**: low (keys lists are typically small — IDs, not full vectors)
- **Effort**: low
- **Location**: `lib/faiss_ex/index.ex:244-245`
  ```elixir
  n = length(keys)
  keys_bin = Shared.int64s_to_binary(keys)
  ```
- **Current pattern**: `length/1` is one O(n) pass, `int64s_to_binary` is a second
  O(n) pass over the same list.
- **Proposed pattern**: fold count + encode in one traversal (mirrors how
  `encode_vectors!`'s batch branch already tracks `n` while encoding), e.g. a
  tail-recursive `encode_keys(keys, 0, <<>>)` returning `{n, bin}`.
- **Estimated gain**: negligible at typical key-list sizes (tens to low thousands
  of int64 ids); flag as a "consider" only if `reconstruct`/`compute_residuals` are
  ever called with very large key batches. Not worth the churn otherwise.

## Finding 4 — binary-in/out fast path is the only way to remove the encode/GC gap itself
- **Impact**: high, but only for callers who can produce a raw f32/i64 binary
  directly (e.g. from `Nx` tensors, memory-mapped/file-sourced data, or a
  producer that never materializes an Elixir list)
- **Effort**: high (API addition, not a rewrite — needs new binary-accepting
  clauses alongside the existing list API)
- **Analysis**: per the measured baseline, the ~10ms gap between NIF-side add
  (0.4ms) and end-to-end `Index.add` (14.5ms) is attributed to major-GC pauses
  from the BEAM promoting/scanning the caller's ~20MB list-of-lists corpus during
  traversal, not the binary construction itself. Findings 1-3 reduce *encoding*
  allocations but cannot remove that cost — it's inherent to the corpus existing
  as ~1.28M list cons cells on the caller's heap at all. The only way to eliminate
  it is for the caller to never build that list: accept a pre-packed binary
  (`<<f::float-32-native, ...>>`, row-major, with explicit `n`/`dim`) and skip
  `encode_vectors!` entirely for that code path.
- **Where it pays off most**: `add/2` and `train/2` are called with the largest
  batches in practice (bench uses 10k×128) and show the full 14ms gap — highest
  ROI. `search/3` is comparatively cheap already (100 queries ≈2.2ms) — lower
  priority. `add_with_ids/3` benefits proportionally to `add/2`.
  `reconstruct`/`compute_residuals`/`get_centroids` are decode-heavy, not
  encode-heavy, so a binary-out (skip decode-to-list) variant would help more
  there than binary-in.
- Confirmed deferred/future feature — no action needed now beyond noting it's the
  correct next lever if `add`/`train` latency becomes a bottleneck for callers who
  already hold binary-shaped data upstream.

## Clustering (`lib/faiss_ex/clustering.ex`)
- `train/2` (`:77-86`) creates a fresh quantizer `Index.new(d, "Flat")` every call —
  this is required (a clustering object needs its own quantizer), not redundant
  per-call rework.
- `get_cluster_assignment/2` (`:116-119`) correctly reuses the already-trained
  quantizer index for search — no re-clustering or re-training per call, as the
  module doc states. Nothing to flag here.
- `get_centroids/1` (`:92-100`) shares the Finding 2 decode-then-chunk pattern;
  no additional per-call rework beyond that.

## Summary Table

| # | Location | Impact | Effort |
|---|----------|--------|--------|
| 1 | shared.ex:39-67 `encode_batch`/`encode_row!` | medium | low |
| 2 | shared.ex decode + `Enum.chunk_every` call sites | low-medium | low |
| 3 | index.ex:244-245 `reconstruct/2` double traversal | low | low |
| 4 | binary-in/out API (deferred) | high (for binary-native callers) | high |
