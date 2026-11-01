# Review triage — cycle 1 (2026-11-01)

Reports: `c-nif.md`, `elixir.md`, `security.md` (second attempt; the first panel hit a 429 rate limit).

| Source | ID | Severity | Decision | Fix |
|---|---|---|---|---|
| C | B | BLOCKER | fixed | `train` n = 0 never reaches FAISS (PCA/RQ → BLAS `cblas_ssyrk` LDC = 0 → Accelerate `exit()`); answers from `is_trained`; regression test on PCA8,Flat / PCAR8,SQ8 |
| C | W | WARNING | fixed | Makefile requires libfaiss to positively link the NIF's OpenMP runtime (`$(LIBOMP_PREFIX)/lib/libomp.dylib` / `[libgomp.so`), else `FAISS_EX_NO_OMP` |
| C | S | SUGGESTION | fixed | `g_omp_default` under `#ifndef FAISS_EX_NO_OMP` |
| (C, pre-limit) | — | BLOCKER | fixed in #15 | IVF `reconstruct_n` unwritten rows → sentinel tag |
| Elixir | B1 | BLOCKER | fixed (docs) | `omp_threads` must be in config.exs/prod.exs (NIF loads before runtime.exs in releases); runtime values via `set_omp_threads/1` in `start/2`. No Application module added (keeps "no processes") |
| Elixir | W1 | WARNING | fixed | `set_omp_threads/1` and config capped at 0..4096 |
| Elixir | W2 | WARNING | fixed | reconstruct/residual docs describe FAISS's own errors |
| Elixir | W3 | WARNING | fixed | starvation test pins `omp_threads: 2` and asserts waiters are still blocked |
| Elixir | W4 | WARNING | fixed | race test runs on both `Flat` and `IDMap,Flat` |
| Elixir | S* | SUGGESTION | skipped | error-shape nits, M1/M4 wording, misaligned-test strength — not defects |
| Security | W1 | WARNING | fixed | `from_file` docs: io_flags 6 narrows, does not make untrusted files safe |
| Security | W2 | WARNING | fixed | NIF rejects index files with d ≤ 0 (repro: IDMap,Flat with outer d = 0); regression test |
| Security | W3 | WARNING | fixed | `io_flags` restricted to 0/2/4/6 (8 crashed on on-disk IVF) |
| Security | W4 | WARNING | **blocked** | `hex` environment / reviewers / tag ruleset need repo-admin; token gets HTTP 403. Comment in publish.yml made truthful |
| Security | S1 | SUGGESTION | fixed | `FAISS_GIT_SHA` must be 40 lowercase hex chars |
| Security | S4 | SUGGESTION | fixed | verify-faiss comment: commit only, not working tree |
| Security | S5 | SUGGESTION | fixed | CUDA container pinned by digest; Dependabot for mix |
| Security | S6 | SUGGESTION | fixed | publish requires the tagged commit to be on master |
| Security | S7 | SUGGESTION | fixed | NIF rejects metrics other than L2/IP |
| Security | S8 | SUGGESTION | fixed | libomp must exist (`$(wildcard ...)`), not just `brew --prefix` output |
| Security | S2, S3 | SUGGESTION | skipped | paths with spaces (elixir_make warns); make expands env values by design |
| Security | S9 | SUGGESTION | already fixed | CLAUDE.md single-source note was updated in Phase 1 |
