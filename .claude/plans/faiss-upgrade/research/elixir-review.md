# Elixir Review: faiss_ex

Scope: `lib/faiss_ex.ex`, `lib/faiss_ex/index.ex`, `lib/faiss_ex/clustering.ex`,
`lib/faiss_ex/shared.ex`, `lib/faiss_ex/nif.ex`, tests, README.md.

## Summary

- **Status**: Changes Requested
- **Issues Found**: 19

## Critical Issues

1. **lib/faiss_ex/clustering.ex:98-104 — `get_cluster_assignment/2` rebuilds and
   re-populates a Flat quantizer index on every single call.** This is O(k*d)
   work (NIF index create + add) per assignment call, on top of decoding
   centroids from binary. For repeated calls (e.g. streaming assignment of new
   points) this is a serious, avoidable performance bug and also silently
   ignores the `index` field already stored on the struct from `train/2`
   (`clustering.index`, set at clustering.ex:69, is never read again). Fix:
   build the quantizer once in `train/2`, add centroids to it there, and store
   it in `clustering.index`; `get_cluster_assignment/2` should just
   `Index.search(clustering.index, data, 1)`. Severity: **high** (perf +
   dead-field correctness).

2. **lib/faiss_ex/index.ex:288 — GPU device_id silently ignored when re-entering
   `cpu_to_gpu` from `new/3`.** `new/3` (index.ex:84-87) calls
   `cpu_to_gpu(index, device_id)` only when `device: {:cuda, device_id}` is
   passed — this part is fine — but `cpu_to_gpu/2` pattern-matches
   `%__MODULE__{device: :host}` only; if a caller passes an already-GPU index
   it raises `FunctionClauseError` instead of a tagged error tuple. Minor but
   inconsistent with the rest of the tagged-tuple API. Severity: **medium**.

## Warnings

3. **lib/faiss_ex/shared.ex:35-41 — `to_flat_floats/1` uses `List.flatten/1`
   plus a separate `length/1` call, and callers immediately call
   `length(flat)` again in `validate_length!/3` (index.ex:323-330).** That's
   three full traversals of a potentially large (millions-of-floats) list
   before any binary conversion happens: `List.flatten` (index.ex chain),
   `length(list)` in `to_flat_floats`, and `length(flat)` in
   `validate_length!`. For large batches (e.g. 10k x 128 = 1.28M floats, per
   the benchmark table) this is wasted CPU on every `add/2`/`search/3`/`train/2`
   call. Prefer computing `n` and validating shape while flattening in one
   pass (e.g. track count via `Enum.reduce` while building the flat list, or
   validate using `Enum.count` once and reuse it instead of two `length/1`
   calls on the same list). Severity: **medium**.

4. **lib/faiss_ex/shared.ex:15-17,25-27 — `floats_to_binary`/`int64s_to_binary`
   use `for ... into: <<>>`.** This is fine in modern OTP (binary
   comprehension into `<<>>` is optimized to avoid quadratic rebuilding), so
   not a bug, but worth a comment noting the reliance on this optimization
   since it's non-obvious from reading the code — a future refactor to
   `Enum.into` or `Enum.reduce` with `<>` would silently reintroduce O(n²)
   behavior. Severity: **low** (documentation/footgun note only).

5. **lib/faiss_ex/index.ex:144-145 — `search/3` decodes the full binary result
   to a flat list via `binary_to_floats/1` then immediately re-chunks with
   `Enum.chunk_every/2`.** For large `n * k` result sets this allocates two
   full-size lists. Same pattern at index.ex:202 (`reconstruct/2`) and
   index.ex:223 (`compute_residuals/2`), and clustering.ex:82
   (`get_centroids/1`). Not wrong, but a hot path per the benchmarks (search
   is exercised at up to 100 queries). Consider a combined
   decode-and-chunk helper in `Shared` to avoid the double list build, or
   document this as an accepted tradeoff for simplicity. Severity: **low**.

6. **lib/faiss_ex/index.ex:73 — `Map.fetch!(@metric_map, metric)` raises
   `KeyError` for an invalid `:metric` option**, which is inconsistent with
   the rest of the API's `{:error, reason}` convention (e.g.
   `add/2`, `search/3` return tagged tuples on NIF errors). A caller passing
   `metric: :cosine` gets a raw `KeyError: key :cosine not found in: %{...}`
   instead of a descriptive `{:error, "invalid metric: ..."}`. Severity:
   **medium** (API consistency / DX).

7. **lib/faiss_ex/clustering.ex:65 — `{:ok, quantizer} = Index.new(d, "Flat")`
   in `train/2` uses a bare match instead of handling `{:error, _}`.** If
   `Index.new/3` ever fails (e.g. OOM, invalid `d`), this raises a
   `MatchError` rather than returning `{:error, reason}` like every other
   function in this module. Same issue in `get_cluster_assignment/2` at
   clustering.ex:101 (`:ok = Index.add(quantizer, centroids)` — bare match).
   Severity: **medium**.

8. **lib/faiss_ex/index.ex:259 — `from_file/2` hardcodes
   `description: "from_file"`** as a placeholder rather than reading the
   actual index factory description back from FAISS (if the C API exposes
   it) or leaving it `nil`. Misleading if a caller later inspects
   `index.description` expecting the factory string. Severity: **low**.

9. **No `@spec` annotations anywhere** in `index.ex`, `clustering.ex`, or
   `nif.ex` (the latter is `@moduledoc false` so more excusable, but public
   modules `FaissEx.Index` and `FaissEx.Clustering` have zero `@spec`s despite
   fairly complex tagged-tuple return types (`{:ok, %{distances: ..., labels:
   ...}}`, etc.) that would benefit greatly from typespecs, both for
   documentation and for the OTP 27+/Elixir 1.20 compiler type checker and
   Dialyzer. Severity: **medium** (repo-wide gap).

10. **lib/faiss_ex/shared.ex:4-12 — `Shared.unwrap!/1` is defined but never
    used anywhere in the library** (grep shows no call sites in `index.ex`,
    `clustering.ex`, or tests). Either it's meant as a public convenience for
    library consumers (in which case it belongs on `FaissEx.Index`/
    `FaissEx.Clustering` as bang variants, e.g. `Index.add!/2`,
    `Index.search!/3`) or it's dead code. `@moduledoc false` on `Shared` means
    consumers likely won't discover it. Severity: **low**.

11. **lib/faiss_ex/clustering.ex:34 — `trained?: boolean()` typespec field
    exists on `Clustering` but `Index` has no equivalent field**, relying
    instead on a NIF round-trip (`Index.trained?/1`) that hits the C layer
    every call. Inconsistent design — `Clustering.trained?` is a cached local
    boolean while `Index.trained?/1` always makes a NIF call. Not wrong, but
    worth deciding on one convention. Severity: **low**.

## Suggestions

12. **Missing API coverage vs the C API** (per CLAUDE.md and nif.ex): no
    `remove_ids`, `range_search`, search-with-params (e.g. `nprobe` for IVF),
    or `merge` wrapped in `Index`. If these exist in `libfaiss_c` and are
    planned, this is expected pre-1.0 scope; otherwise worth README callout.
    `nif_get_num_gpus/0` is exposed only via `FaissEx.NIF` directly (tests
    call it as `FaissEx.NIF.nif_get_num_gpus()`, e.g. index_test.exs:323) —
    there's no `FaissEx.Index.num_gpus/0` or similar convenience wrapper on
    the public API surface, forcing users to reach into the internal `NIF`
    module (marked `@moduledoc false`). Severity: low, but a documented gap.

13. **README concurrency section (README.md:352-357) is good** but doesn't
    mention `Clustering` — is concurrent `train`/`get_cluster_assignment` on
    the same `Clustering` struct safe? Given finding #1 (quantizer rebuilt
    per call), concurrent `get_cluster_assignment/2` calls are currently
    *accidentally* safe (each call creates its own temp index) — this
    behavior would need re-verification once #1 is fixed to reuse
    `clustering.index`, since a shared `Index` ref would then be subject to
    the same "concurrent writes unsafe" rule. Flag as a note for whoever
    fixes #1.

14. **Test coverage gaps**:
    - No test for `Index.new/3` with an invalid `:metric` (would currently
      raise `KeyError` uncaught — see #6).
    - No test for `add_with_ids/3` on a non-`IDMap` index (does the NIF
      return a clean `{:error, _}` or crash?).
    - No test for `reconstruct/2` on an IVF or HNSW index (only Flat is
      tested at index_test.exs:124-145) — `reconstruct` semantics differ
      across index types in FAISS.
    - No test for `Index.clone/1`, `to_file/from_file`, or
      `compute_residuals/3` failure paths (invalid ref, bad path, mismatched
      dims).
    - `Clustering` tests (clustering_test.exs) have no test asserting
      `clustering.index` is populated/reused after `train/2`, which would
      have caught the #1 dead-field issue.
    - GPU tests are appropriately tagged `:cuda` and excluded by default —
      no issue there.
    - No test exercises `Index.new/3` with `device: {:cuda, _}` opt path
      (index.ex:84-87) even under the `:cuda` tag.

15. **index.ex:332-333 — `to_binary/1` accepts `is_list(str)` (charlist) via
    `List.to_string/1`** but this isn't documented in `@doc` for `new/3` or
    `to_file/2`/`from_file/1`; minor, low-value flexibility that adds a
    private helper and an untested branch (no test passes a charlist path).
    Consider dropping the charlist branch and requiring binaries, per
    "explicit over implicit."

16. **lib/faiss_ex/index.ex — `new/3`, `add/2`, etc. return `:ok`/`{:ok, _}`
    consistently** which is good; error variants correctly return
    `{:error, reason}` pass-through from the NIF. This part of the API is
    consistent and idiomatic — no action needed, noted only because dimension
    #2 of the review asked about it explicitly.

17. **`@doc` coverage is generally good** (every public function in
    `index.ex`/`clustering.ex` has a `@doc`), but none include `@spec`
    (see #9) and none use doctests beyond `new/3` in index.ex:63-68 — could
    add doctests to `add/2`/`search/3` for extra compile-time verification
    given examples are already present in docstrings.

18. **mix.exs ex_doc config** was not read in this pass (not in the requested
    file list) — flagging as unverified; worth checking `groups_for_modules`
    given `FaissEx.NIF` is `@moduledoc false` (correctly hidden) but
    `FaissEx.Shared` is also `@moduledoc false` — confirm ex_doc doesn't leak
    internals into generated docs.

19. **lib/faiss_ex/index.ex:196-207, 214-228 — `reconstruct/2` and
    `compute_residuals/3` do not validate that `keys` values are within
    `[0, ntotal)` before calling the NIF**; relies entirely on the C/NIF layer
    to reject out-of-range keys. Not necessarily wrong (let the NIF own
    validation) but worth confirming the NIF returns a clean `{:error, _}`
    rather than crashing/segfaulting on bad keys (C review territory, flagged
    here for cross-reference).

## Notes on Struct Design (dimension 5)

- `Clustering.index` (clustering.ex:33) is set once in `train/2` and never
  read — see finding #1, this is the most actionable structural fix in the
  review.
- `Index` has no `trained?` field (always a NIF round trip via
  `trained?/1`), while `Clustering` caches `trained?` as a struct field set
  once at train time and never re-verified — asymmetric design, see #11.
- `Index.description` is a real value from `new/3` but a placeholder string
  `"from_file"` from `from_file/1` (index.ex:259) — see #8.
