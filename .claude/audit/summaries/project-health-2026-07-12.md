# Project Health Report — faiss_ex

**Date**: 2026-07-12
**Commit**: 3d0a7e1 (master) — v0.2.0, FAISS 1.14.3
**Mode**: Full audit (5 parallel specialist auditors + quick checks)

## Executive Summary

**Overall Health: 83/100 — Grade B (Good)**

faiss_ex is a healthy, well-engineered NIF library. The native layer — historically the riskiest part of any NIF project — audited clean: zero-copy binary handling, consistent overflow guards (`safe_mul`), strict size validation before every FAISS pointer handoff, correct lock release on all error paths, and a lock/dirty-scheduler contract that matches its documentation exactly. Dependencies are minimal, correctly scoped, and current (FAISS v1.14.3 is the latest upstream release).

The one weak category is **test health (62/100)**: the concurrency suite — the only test evidence for the RW-lock safety contract — is tagged `:slow` and never runs in a default `mix test`, and the resource destructor/GC path has no test at all. The top cross-cutting risk is the **mutable FAISS tag pin** (flagged independently by two auditors): the build clones `--branch v1.14.3` with no commit-SHA or checksum verification.

No critical-override conditions triggered: compile is clean under `--warnings-as-errors`, 48/48 default tests pass, no hardcoded secrets, no exploitable vulnerability found.

## Category Scores

| Category | Score | Grade | Weight | Notes |
|---|---|---|---|---|
| Dependencies | 92/100 | A | 15% | FAISS pin current + in sync; only patch-level lag |
| Architecture | 88/100 | B | 20% | Clean layering; lock contract verified against C source |
| Security | 85/100 | B | 25% | Memory-safe NIF; deductions for source-fetch integrity |
| Performance | 84/100 | B | 25% | Native layer near-optimal; cost sits in list-based API |
| Tests | 62/100 | D | 15% | Concurrency + destructor contracts untested by default |

**Weighted overall**: 88×0.20 + 84×0.25 + 85×0.25 + 62×0.15 + 92×0.15 = **83 (B)**

## Critical / High Issues

1. **Concurrency suite excluded from default runs** — `test/concurrency_test.exs:9` is `@moduletag :slow`; `mix test` never exercises the RW-lock contract documented at `lib/faiss_ex/index.ex:39-53`. Split out a fast concurrency smoke test (small index, a few dozen concurrent add/search tasks) that always runs, and ensure CI includes `--include slow`.
2. **FAISS pinned to a mutable tag, unverified** — `Makefile:11` / `mix.exs:72` pin `v1.14.3` (a movable tag) and clone with `--depth 1 --branch` and no integrity check. A re-pointed tag silently compiles different C++ into the BEAM on the next cache-miss build. Pin the commit SHA and assert `git rev-parse HEAD` post-clone. *(Found independently by security + dependency auditors.)*
3. **Resource destructor/GC path untested** — module docs promise destructor-based cleanup safety; no test drops an index/clustering resource under load or exercises GC-triggered destruction.
4. **List-only result decoding dominates at scale** — `lib/faiss_ex/index.ex:184-196` (and reconstruct/residuals/centroids) unconditionally decode NIF binaries into lists of lists (~4× memory blowup; `search(batch_10k, k: 100)` ≈ 32 MB heap from 8 MB of binaries). An opt-in `output: :binary` needs no C changes.
5. **Metric-type enum duplicated across the NIF boundary** — `lib/faiss_ex/index.ex:104-115` hardcodes `:l2 → 1` / `:inner_product → 0`, cast blind to `FaissMetricType` in `c_src/faiss_ex_nif.c:193`. Nothing enforces sync across FAISS bumps; drift would silently search the wrong metric.
6. **Unquoted build-env interpolation in Makefile recipes** — `Makefile:110-124,140`: `FAISS_GIT_REV`/`FAISS_PREFIX` expand unquoted in shell recipes; a poisoned CI env var executes at compile time. Quote and validate (`^[A-Za-z0-9._/-]+$`).

## Medium / Low Issues (by category)

- **Tests**: untrained-index add/search error path has no always-run unit test; `Clustering.get_centroids/1` before `train/2` untested; `reconstruct/2` on non-direct-map index types, `compute_residuals/3` batch case, `add_with_ids/3` on plain Flat, and invalid factory strings all untested; tmp-file naming uses `:rand.uniform(100_000)` (collision risk under `async: true`) — prefer `@tag :tmp_dir` or `System.unique_integer/1`.
- **Performance**: result decode makes two passes (`binary_to_floats` → `chunk_every`); result binaries allocated inside the read-lock critical section in `nif_search_index`/`nif_reconstruct_batch`/`nif_compute_residuals`; `encode_batch` pays one spare full-payload copy (`shared.ex:39-45`); default `FAISS_OPT_LEVEL=generic` leaves x86 AVX2 throughput unused.
- **Architecture**: dead code `Shared.floats_to_binary/1` (never called); `nif_new_index`/`nif_new_clustering` on normal schedulers while `faiss_index_factory` can allocate nontrivially for composite factory strings (documented as intentional — re-confirm for large factories).
- **Dependencies**: elixir_make 0.9.0 → 0.10.0, ex_doc/benchee patch-level lag; no OTP minimum declared in mix.exs (worth documenting for a NIF package); `mix deps.audit` unavailable (mix_audit not a dep) — not worth adding for 3 dev/build-only deps; run `mix hex.audit` in CI instead.

## Action Plan

**Immediate (this week)**
1. Add an always-run concurrency smoke test; keep the heavy suite under `:slow` but run `--include slow` in CI.
2. Pin FAISS to a commit SHA in both `Makefile` and `mix.exs` (keep them in sync per project convention) and verify post-clone.
3. Add unit tests: untrained-index ops, `get_centroids/1` before train, destructor/GC exercise.

**Short-term (this month)**
4. Quote/validate env-var expansions in Makefile recipes.
5. Replace hardcoded metric integers with a shared constant + compile-time assertion against the C enum.
6. Add opt-in `output: :binary` to search/reconstruct/residuals/centroids; single-pass decode for the list path.
7. Bump elixir_make/ex_doc/benchee (patch/minor); delete `Shared.floats_to_binary/1`.

**Long-term (next quarter)**
8. Move result-binary allocation out of read-lock critical sections in the C layer.
9. Document/auto-detect `FAISS_OPT_LEVEL` for AVX2 builds on x86.
10. Declare an OTP minimum in mix.exs; add `mix hex.audit` to CI.

## Method

Quick checks (compile `--warnings-as-errors`, `mix hex.audit`, `mix xref graph --format stats`, `mix test`) all passed before agent fan-out. Five parallel specialist auditors wrote to `.claude/audit/reports/` (arch-review, perf-audit, security-audit, test-audit, deps-audit); findings deduplicated in `.claude/audit/summaries/consolidated.md`. Scores per `scoring-methodology.md` weighting (security/perf 25%, arch 20%, tests/deps 15%). Scores are project-internal — track trends here, don't compare across projects.
