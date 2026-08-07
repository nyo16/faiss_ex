# Technical Debt Report — faiss_ex

**Date**: 2026-07-12 | **Scope**: lib/ (4 modules, 668 lines), c_src/faiss_ex_nif.c (1001 lines), test/
**Credo**: not installed (no `credo` dep) — analysis done manually.

Prioritized by value-to-effort. The codebase is small and clean; this is polish, not rescue.

## 1. Redundant `case` pass-throughs in Index (6 occurrences) — Effort: Low

- **Location**: `lib/faiss_ex/index.ex:149-152` (add), `:168-171` (add_with_ids), `:205-208` (train), `:230-233` (reset), `:281-284` (to_file), `:333-336` (trained?)
- **Pattern**: `case NIF.nif_x(...) do :ok -> :ok; {:error, _} = err -> err end` — an identity transform. The NIF already returns exactly `:ok | {:error, msg}`.
- **Extraction**: none needed — delete the case and call the NIF directly, as `ntotal/1` and `dim/1` already do (`index.ex:320-322`, `:326-328`). Right now the module mixes both styles; converging on direct delegation removes ~25 lines.
- **Note**: `trained?/1` is the same shape for the `{:ok, val}` case.

## 2. Duplicated decode-and-chunk pipeline (5 occurrences) — Effort: Low-Medium

- **Location**: `lib/faiss_ex/index.ex:189` and `:190` (search, distances + labels), `:249` (reconstruct), `:269` (compute_residuals); `lib/faiss_ex/clustering.ex:95` (get_centroids)
- **Pattern**: `bin |> Shared.binary_to_floats() |> Enum.chunk_every(width)` (int64 variant on line 190). Passes the three-strikes rule at 5 sites.
- **Extraction**: `Shared.decode_f32_rows(bin, width)` / `Shared.decode_i64_rows(bin, width)`. Implementing them as a single-pass recursive binary walk also fixes the perf audit's "two passes, double list allocation" finding for free — one extraction, two debts retired.

## 3. Dead code: `Shared.floats_to_binary/1` — Effort: Trivial

- **Location**: `lib/faiss_ex/shared.ex:5-7`
- **Pattern**: defined, never called anywhere in `lib/` or `test/` (`encode_vectors!/2` is the real encoder). Also the audit's architecture finding.
- **Extraction**: delete it.

## 4. Metric enum duplicated across the NIF boundary — Effort: Low

- **Location**: `lib/faiss_ex/index.ex:104-115` ↔ `c_src/faiss_ex_nif.c:193`
- **Pattern**: `:l2 → 1`, `:inner_product → 0` hardcoded in Elixir, cast blind to `FaissMetricType` in C. Silent wrong-metric search if FAISS renumbers on a version bump.
- **Extraction**: module attribute (`@metrics %{l2: 1, inner_product: 0}`) with a comment naming the C enum, plus a `_Static_assert(METRIC_L2 == 1 && METRIC_INNER_PRODUCT == 0, ...)` in the C file so a FAISS bump that renumbers fails at compile time instead of at search time. (Also an audit high finding.)

## 5. Duplicated tmp-file naming in tests (3 occurrences) — Effort: Low

- **Location**: `test/faiss_ex/index_test.exs:172`, `:296`, `:359`
- **Pattern**: `Path.join(System.tmp_dir!(), "faiss_ex_*_#{:rand.uniform(100_000)}.index")` + `try/after File.rm` — copy-pasted, and collision-prone under `async: true`.
- **Extraction**: ExUnit's built-in `@tag :tmp_dir` (`test "...", %{tmp_dir: dir}`) — deletes the naming, the randomness, and the manual cleanup in one move.

## Explicitly not debt (leave alone)

- **C lock/resource boilerplate** (17 lock acquisitions, 23 NIF functions): idiomatic NIF code; error handling is already centralized in `make_faiss_error`/`make_error_msg` (`c_src/faiss_ex_nif.c:121`). Macro-extracting the lock dance would hide control flow in the one place where visible control flow is a safety feature. Revisit only if the file grows well past ~1500 lines.
- **Single C file**: at 1001 lines with clear section structure, splitting adds build complexity for no cohesion gain.
- **`to_binary/1` charlist helper** (`index.ex:387-388`): two callers, three lines — fine as is.

## Tooling recommendation

Add `{:credo, "~> 1.7", only: [:dev, :test], runtime: false}` and a CI step. Marginal for 5 files, but it's the standing guard against consistency drift, and this skill's automated step currently can't run.

## Suggested order

1 → 3 → 5 (mechanical, ~30 min total), then 2 (pairs with the perf fix), then 4 (touches C, wants the version-bump story thought through). All affected paths have test coverage, including the new untrained/lifecycle/concurrency tests from the audit PR.
