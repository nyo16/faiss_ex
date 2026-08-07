# Scratchpad — faiss-upgrade

Decisions, dead-ends, and open questions during planning/execution.

## Decisions

- **Pin v1.14.3** (latest tag, 2026-06-13). C reviewer independently verified all
  used C API signatures identical on `main`; web research confirmed no breaking
  changes across 1.11–1.14.3.
- **rwlock over mutex**: preserves concurrent searches (FAISS search is itself
  OpenMP-parallel); mutex would serialize the common read-heavy path.
- **Raise ArgumentError for developer errors** (bad metric, ragged batch),
  keep `{:error, binary}` for runtime FAISS errors. Matches existing
  `validate_length!` behavior.
- **v0.2.0** not 0.1.1: locking changes observable behavior (mutations now
  serialize) and API additions (num_gpus/0).

## Open questions / verify during execution

- Metal backend claim (web research says default-on for Apple Silicon in
  v1.14.2+) — NOT independently verified; check cmake output in task 1.3.
  Treat with suspicion; the C reviewer did not corroborate.
- `-lstdc++` drop (2.9): test link on both macOS and Linux CI before removing.

## Execution notes (post-run)

- Metal claim CONFIRMED (FAISS_ENABLE_METAL=ON default on Apple Silicon in
  v1.14.3) but harmless: adds OBJCXX language only; our faiss/faiss_c targets
  link no Metal framework. No cmake pin added.
- Beyond plan: dim/ntotal/is_trained getter NIFs moved to dirty CPU — they
  block on the rwlock and a writer can hold it for seconds (H1 rationale).
- Bench: batch-add scenarios are GC-dominated (NIF add of 10k vectors ≈
  0.4ms; ~10ms BEAM major-GC pauses on the 20MB corpus make add-1000 ≈
  add-10000). Probe script confirmed; README note added. Not a regression.
- `-lc++` on Darwin verified locally; Linux `-lstdc++` path unchanged, still
  untested on Linux CI (flag identical to pre-change behavior there).
- New test tag `:no_cuda` (excluded when USE_CUDA=true) for non-CUDA-build
  error-path assertions.

## Dead-ends

- (none)
