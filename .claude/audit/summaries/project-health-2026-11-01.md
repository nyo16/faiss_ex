# Project Health Report — faiss_ex

**Date**: 2026-11-01
**Commit**: a397362 (master) — v0.3.2, FAISS v1.15.1 (latest upstream; SHA-verified)
**Mode**: Full audit (quick-mode gate → 5 parallel specialist auditors → consolidation)
**Previous**: 2026-07-12 @ 3d0a7e1 — 83/100 (B)

## Executive Summary

**Overall Health: 70/100 — Grade C (Needs Attention)**

The drop from 83 is not a code regression: across the prior findings 15 are resolved, 3 partly resolved, 16 still open, 0 regressed. Test health jumped 62 → 89. This round went deeper than the last (concurrency was measured, call paths were traced into the FAISS 1.15.1 source, and build configuration was audited), and it found three runtime-throughput problems that predate the last audit. Perf scores 48 because of them. The C layer still fully matches the CLAUDE.md lock/unwind/scheduler spec (20/20 NIFs), and no memory-safety bug was found.

Trend caveat: the rubric adaptations for a non-Phoenix NIF library differ from the 2026-07-12 run, so read the category deltas as a change in what was audited, not as the code getting worse.

**Critical override: not triggered.** Compiles clean under `--warnings-as-errors`, 85/85 tests pass, `mix hex.audit` is clean, there are no hardcoded secrets, and no vulnerability is reachable without the caller feeding untrusted input to documented APIs.

## Category Scores

| Category | Score | Grade | Weight | Prior | Notes |
|---|---|---|---|---|---|
| Architecture | 71 | C | 20% | 88 | Layering/lock spec exact; Clustering state split; build config not single-sourced |
| Performance | 48 | F | 25% | 84 | Dirty-CPU pool starvation, OpenMP oversubscription, AVX2 opt level is a no-op |
| Security | 74 | C | 25% | 85 | Memory-safe NIF; GPU device abort, untrusted index files/factory strings, CI supply chain |
| Tests | 89 | B | 15% | 62 | Concurrency/destructor gaps closed; one vacuous test; concurrent paths beyond add/search untested |
| Dependencies | 80 | B | 15% | 92 | Hex clean, FAISS current; macos-14 runner retires tomorrow |

**Weighted**: 71×0.20 + 48×0.25 + 74×0.25 + 89×0.15 + 80×0.15 = **70.05 → 70 (C)**

Orchestrator adjustments:
- **Arch M1 dropped as a false positive.** It claimed the GPU index's resources stay alive only through the Elixir struct field. In fact `faiss/gpu/GpuIndex.h:289` holds a `std::shared_ptr<GpuResources>`, and the `StandardGpuResources` wrapper (`StandardGpuResources.h:265`) owns only a `shared_ptr` to the impl. The security audit independently agrees.
- **Arch M2 takes M1's slot.** M2 is the Clustering state split; it is the same criterion (layer/state ownership) with the same −5, so the arch score stays at 71. `consolidated.md` shows 76 because it does not apply this substitution.
- **Deps "macos-14 retired 2026-11-02" confirmed** via the [GitHub changelog](https://github.blog/changelog/2026-10-01-github-actions-macos-14-runner-image-retirement/).

## Critical / High Issues

1. **CI macOS arm64 job dies tomorrow.** `.github/workflows/ci.yml:135` uses `macos-14`, which is retired on 2026-11-02. Switch to `macos-15`. It is the only Apple Silicon build check. *(deps)*
2. **Lock waiters starve the dirty-CPU scheduler pool, VM-wide.** Every locking NIF is `DIRTY_JOB_CPU_BOUND` (`c_src/faiss_ex_nif.c:1029-1049`), and `enif_rwlock_*lock` blocks the scheduler thread. Measured: one 3.9 s IVF train plus 10 queued `ntotal` calls made a search on an *unrelated* index take 3644 ms instead of 1.7 ms. Fix: move the getters to dirty IO; for search/reconstruct/residuals, `tryrlock` and fall back to a dirty-IO variant when the lock is contended. *(perf; correlates with test H6)*
3. **OpenMP oversubscription.** No `omp_set_num_threads` is called anywhere in the NIF, so every dirty scheduler spawns a full OpenMP team even for a 1-query search. Measured: 10 procs × 200 single-query searches took 1845 ms by default vs 622 ms with `OMP_NUM_THREADS=1`. Large batches (1000 queries) do benefit from OpenMP (102 vs 164 ms). Fix: a per-call `:threads` option plus documentation of the tradeoff. *(perf)*
4. **`FAISS_OPT_LEVEL=avx2|avx512` has no effect.** `Makefile:183,188` build only the `faiss`/`faiss_c` targets; FAISS 1.15.1 puts its SIMD code in separate `faiss_avx2`/`faiss_c_avx2` targets, which are never built or shipped (verified at `Makefile:181-207`). x86 users who set it get generic code. Fix: default to `dd` (dynamic dispatch) in both `Makefile` and `mix.exs`. *(perf)*
5. **Vacuous test.** `test/faiss_ex/clustering_test.exs:98-99` ("stays untrained") hits the `index: nil` Elixir clause and never reaches the NIF, so it passes regardless of C behavior. Assert that `get_centroids/1` returns `"no centroids available"` instead. *(tests; correlates with arch M2)*
6. **Concurrency coverage stops at add/search.** The two-lock `nif_train_clustering` path, and reconstruct/residuals/clone/`to_file`/`add_with_ids` racing a writer (e.g. reconstruct vs reset), are never exercised concurrently (`concurrency_smoke_test.exs:21-28`, `concurrency_test.exs:23-28,56-59`). *(tests)*

## Medium Issues (deduplicated)

- **Build artifacts not keyed on `USE_CUDA`/`FAISS_OPT_LEVEL`** (`Makefile:28-29,176-178,210`; CI cache key likewise). After a CPU build, `USE_CUDA=true mix compile` silently keeps the CPU NIF and `num_gpus/0` returns 0. Put both options in the build dir name. *(arch, deps, perf, tests)*
- **Clustering state split** (`lib/faiss_ex/clustering.ex:73-81`). `train/2` mutates the shared C resource but returns a new struct. Older structs then see the new centroids via `get_centroids/1` but the old quantizer via `get_cluster_assignment/2`, and the untrained original reads trained centroids while `trained?: false`. Choose one ownership model. *(arch)*
- **`nif_new_index` on a normal scheduler with caller-supplied factory strings.** Measured at 4–176 ms (`LSHr`, d=2048), and `"IVF100M,Flat"` allocates GBs. The `Index.new/3` docs invite user-supplied descriptions. Move it to dirty CPU, drop the "routinely user input" wording, and update the CLAUDE.md scheduler list. *(perf, security, arch)*
- **Negative GPU device id → FAISS assert → `abort()`** of the VM (`c_src/faiss_ex_nif.c:746-759`; GPU builds only). Add a `device < 0` check in the C NIF and a guard in Elixir. *(security)*
- **`from_file/2` on an untrusted index**: with `io_flags: 0`, an on-disk IVF index makes FAISS mmap read-write whatever path it names. Document this, or default to `IO_FLAG_ONDISK_READ_ONLY`/reject `ilod`. *(security)*
- **CI actions pinned to movable tags**, including the publish job that holds `HEX_API_KEY`; `ci.yml` has no `permissions:` block. Pin to SHAs. *(security, deps)*
- **Binary API gaps**: Clustering and `compute_residuals/3` are list-only (8× memory on decode) despite the moduledoc saying all hot paths take binaries. *(perf, arch)*
- **One `StandardGpuResources` per `cpu_to_gpu`**, reserving up to 1.5 GiB of GPU memory per index (`c_src/faiss_ex_nif.c:751-752`). *(perf)*
- **Declared minimum Elixir (`~> 1.18`) / OTP 27 never built in CI**; CI runs only Elixir 1.19 / OTP 28. *(tests, deps)*
- **Empty-train asymmetry**: `Index.train(idx, [])` returns `:ok` and leaves the index untrained, while `Clustering.train(c, [])` errors; neither is tested. Unbounded search `k` is untested. *(tests, arch, security)*
- **README toolchain floor is wrong**: it says CMake 3.17+, but FAISS needs 3.24 + C++20, so the "exact CI recipe" fails on stock Ubuntu 22.04 apt cmake (3.22). *(deps)*

## Low (selected)

- FAISS rev/SHA/build defaults are duplicated between `Makefile` and `mix.exs` with no drift check; the SHA is verified only on a fresh clone, never on a restored cache.
- Makefile recipes still interpolate env vars unquoted (prior finding, still open).
- Naming drift: `d` vs `dim`, inconsistent `get_` prefixes, and the same count mismatch raises in `compute_residuals` but returns `{:error,_}` in `add_with_ids`.
- `mix test --cover` fails its default 90% threshold only because of the `FaissEx.NIF` stubs (12.5%); add `test_coverage: [ignore_modules: [FaissEx.NIF]]`. Excluding the stubs, coverage is ~96%; C is not measured.
- The `ubuntu-22.04` runners are deprecated (removal 2027-04-17); patch-level Hex lag (elixir_make 0.9.0 → 0.10.1, ex_doc, benchee); credo is present but never run; `mix hex.audit` is not in CI.

## Action Plan

**Immediate (today/this week)**
1. `ci.yml:135` `macos-14` → `macos-15`.
2. Fix the vacuous `clustering_test.exs:98-99` assertion.
3. Add a `device < 0` guard in `nif_index_cpu_to_gpu` and in Elixir.
4. Change the default `FAISS_OPT_LEVEL` to `dd` and key the build dir on `FAISS_OPT_LEVEL` + `USE_CUDA` (this fixes both the AVX2 no-op and the stale CUDA build).

**Short-term (this month)**
5. Scheduler rework: getters to dirty IO, and a `tryrlock` + dirty-IO fallback for read NIFs; move `nif_new_index` to dirty CPU and update the CLAUDE.md scheduler list.
6. Add a `:threads` option (per-call `omp_set_num_threads`) and document the batch-vs-concurrency tradeoff.
7. Resolve the Clustering ownership model (fresh resource per train, or trained/quantizer state held in C).
8. Extend the concurrency tests: two-lock clustering train, and reconstruct/residuals/clone/`to_file` racing writers.
9. Pin actions to SHAs, add `permissions:`, add `mix hex.audit` to CI, and add a min-version CI matrix row.
10. Document the `from_file/2` trust boundary and the factory-string DoS; correct the README CMake/C++ floor.

**Long-term (next quarter)**
11. Binary API for Clustering and `compute_residuals/3`.
12. Shared GPU resources across indexes (opt-in pool).
13. Single-source FAISS build config (mix.exs → Makefile env only), and re-verify the SHA on cache restore.
14. Benchmark concurrency and non-contiguous reconstruct scenarios.

## Method

Quick gate (`mix compile --warnings-as-errors`, `mix hex.audit`, `mix xref graph --format stats` → 0 cycles, `mix test` → 85/85, `mix test --cover`) passed before fan-out. Five auditors (phoenix-patterns-analyst, general task ×2, security-analyzer, testing-reviewer) wrote to `.claude/audit/reports/`; context-supervisor merged them into `.claude/audit/summaries/consolidated.md`. Perf findings 2–3 were measured with throwaway `mix run` probes (since deleted); finding 4 was verified against `Makefile:181-207`. Track trends within this project only.
