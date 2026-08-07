# Testing Review: HEAD~4..HEAD (test/)

## Summary
New tests are well-structured (good `describe` grouping, correct tag placement, correct exclusion logic in `test_helper.exs`), but several NEW behaviors introduced in this diff are untested, and the concurrency test is a crash/liveness smoke test rather than a real race-detection test.

## Iron Law Violations
None outright, but `concurrency_test.exs` uses `async: false` without modifying global/application state — it only touches a locally-created `Index`. Per Iron Law 1 (ASYNC BY DEFAULT) this should be `async: true` unless there's a resource-contention reason (e.g. avoiding CPU competition with other dirty-scheduler-heavy tests) — if that's the reason, say so in a comment.

## Issues Found

### Critical
- [ ] `test/faiss_ex/concurrency_test.exs:20-25` — Each of the 8 workers is pinned to a single operation for all 300 iterations (`rem(i,4)`: 2 workers reset-only, 2 add-only, 2 search-only, 2 ntotal-only). This exercises "some concurrent mix" but never a worker that interleaves add→search→reset itself, and the mix ratio is fixed/deterministic across runs — it does not stress the lock under varied interleavings. Consider randomizing the op per-iteration per-worker (`Enum.random([:add, :search, :reset, :ntotal])`) to better fuzz interleavings across runs.
- [ ] No test exercises the write-lock (`enif_rwlock_rwlock`) held during `train` concurrently with reads/writes (e.g. `Index.train` racing `Index.search`/`Index.add` on IVF). This is the most contention-heavy write path (long-running under dirty scheduler) and is unexercised by the new rwlock coverage.
- [ ] `lib/faiss_ex/shared.ex:65-67` (`encode_row` non-number branch) and `lib/faiss_ex/shared.ex:47-49` (`encode_batch` non-list-row branch) are new/changed single-pass validation paths with zero test coverage. Add: `Index.add(index, [[1.0, "x", 3.0]])` → `ArgumentError` "row 0 contains a non-number", and `Index.add(index, [[1.0,2.0,3.0], "bad"])` → "row 1 is not a list of numbers".
- [ ] `Shared.encode_vectors!([], _dim)` (empty list) path is untested — `Index.add(index, [])` behavior (should be a no-op `:ok` with `n=0`) is unverified.
- [ ] `Index.add_with_ids/3` mismatched `ids` vs `data` length (e.g. 2 vectors, 1 id) is untested. Validation is NIF-side only per the diff description; no test confirms the error is surfaced correctly rather than reading OOB.
- [ ] `Clustering.train/2` error propagation is untested — e.g. wrong-dimension `data` (should raise via `encode_vectors!`) or NIF-level training failure (e.g. `k` > number of training points). Given the diff explicitly adds quantizer reuse in `train`, a failure mid-`with` (quantizer created but `nif_train_clustering` errors) leaves an orphaned quantizer index — worth a test asserting `clustering.index` stays `nil` and `trained?` stays `false` on error.

### Warnings
- [ ] `c_src/faiss_ex_nif.c:472-480` (contiguous vs non-contiguous `reconstruct_n` fast path) — `test/faiss_ex/index_test.exs:135` only calls `Index.reconstruct(index, [0, 1])` (contiguous ascending). The non-contiguous fallback loop (`faiss_Index_reconstruct` per-key) is never exercised. Add a case like `Index.reconstruct(index, [1, 0])` or `[0, 2]` (with a gap) to hit `contiguous = 0`.
- [ ] `test/faiss_ex/concurrency_test.exs` — no assertion bounds `ntotal` mid-run or checks for internal consistency beyond "doesn't crash + still works after". This is acceptable for a smoke test but the module/test name ("does not crash the VM") undersells that it's purely a liveness/crash check, not a correctness-under-race check; consider renaming or adding a comment clarifying scope.
- [ ] `test/faiss_ex/index_test.exs:399-401` and `:403-411` — NUL-byte rejection tests only cover `Index.new/3` factory string and file paths; the diff description implies NUL-byte validation was added broadly in the NIF, but there's no test for NUL bytes elsewhere the NIF accepts a string.
- [ ] `test/faiss_ex/clustering_test.exs` — no test for `k` values validation edge case where `k > n` (fewer training points than clusters), which FAISS errors on; would be a natural test alongside the new metric/dim validation additions.

### Suggestions
- [ ] `test/faiss_ex/concurrency_test.exs:9-10` — `@workers 8` / `@iterations 300` (2400 total ops) is a reasonable smoke-test size; consider tagging with `:slow` (already done) and optionally exposing iteration count via env var for CI vs local dev tuning — not required.
- [ ] `test/faiss_ex/index_test.exs:378-392` — good use of exact-message `assert_raise/3`; consider also asserting the error for a completely empty row list `Index.add(index, [[]])` to pin the boundary at `count == 0 != dim`.
- [ ] Consider a property-based test (StreamData) for `encode_vectors!/2`: for any `dim` and any list of `dim` numbers, encode+decode roundtrips to the same floats (within f32 precision) — good fit since it's a pure encode/decode roundtrip.

## Pre-existing (not introduced by this diff)
None flagged as in-scope; all findings above concern new/changed code in this diff.
