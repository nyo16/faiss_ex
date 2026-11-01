# Consolidated Summary — faiss_ex audit (2026-11-01, a397362, v0.3.2 / FAISS v1.15.1)

**Strategy**: Compress (audit priorities: scores, critical findings, correlations kept in full)
**Input**: 5 files (arch-review, perf-audit, security-audit, test-audit, deps-audit), ~19k tokens. `techdebt.md` ignored as stale.
**Output**: ~4.5k tokens (~76% reduction)
**Orchestrator corrections applied**: arch M1 dropped as a false positive (score re-credited); deps macos-14 retirement confirmed and escalated to Immediate.

## Health Scores

| Category | Score | Prior | Rationale |
|---|---|---|---|
| Architecture | **76** (report: 71, +5 re-credited) | 88 | The C layer matches the CLAUDE.md lock/scheduler spec exactly. Points are lost for naming/convention drift (−9), non-uniform list/binary inputs (−5), and build config that is duplicated and not keyed on build settings (−10). The M1 −5 "layer boundaries" deduction is removed: the false positive is explained below. |
| Performance | **48** | 84 | Not a code regression. Three issues were measured for the first time and were already present: dirty-CPU pool starvation (N1), OpenMP oversubscription (N2), and an `avx2`/`avx512` opt level that does nothing (N3). Lock and scheduler criteria are at or near 0. |
| Security | **74** | 85 | No memory-safety bugs. The losses are new DoS paths into FAISS (negative GPU device, factory strings), an mmap read/write primitive via untrusted index files, and Actions pinned by tag in the publish job. |
| Testing | **89** (B) | 62 | 79.66% coverage (~96% when the NIF stubs are excluded); no sleeps; 0.4 s run. Points lost: only 18/33 NIF error branches reachable from the public API are tested (−7), plus 2 timing-dependent patterns (−4). |
| Dependencies | **80** | 92 | `hex.audit` is clean and FAISS v1.15.1 is the latest release with its SHA verified. Points lost: runner images retired or deprecated, a build cache keyed only by rev, and a CMake floor that is documented wrongly. |

**Arch M1 false positive** (GPU resources kept alive only by the Elixir struct, claimed use-after-free at `c_src/faiss_ex_nif.c:62-71,766-776`, `lib/faiss_ex/index.ex:85,415-425`): `faiss/gpu/GpuIndex.h:289` holds `std::shared_ptr<GpuResources> resources_`, and the `StandardGpuResources.h:265` wrapper owns only a shared_ptr to the impl. Freeing the wrapper resource therefore cannot free resources the index still uses. The security report agrees (`StandardGpuResources.h:204,265`; `GpuIndex.cu:63`).

## Critical / High Findings

No Critical findings.

**H1 — IMMEDIATE: the `macos-14` CI runner is retired 2026-11-02 (tomorrow)** · Deps (CI)
- `.github/workflows/ci.yml:135` (`{ runner: macos-14, label: macOS arm64 }`), comment at `ci.yml:34`. Confirmed by the github.blog changelog of 2026-10-01.
- Impact: this is the only Apple Silicon coverage of the darwin-only Makefile path (`Makefile:106-116` libomp lookup, `install_name_tool`). From tomorrow the job will fail or sit queued on every push and PR.
- Fix: switch to `macos-15` (or `macos-26`/`macos-latest`) and update the comment. The cache key already includes `runner.arch`.

**H2 — Threads waiting on one index's lock fill the dirty-CPU pool and stall every other index** · Perf N1 (+P5), Test H2 (see Correlations)
- `c_src/faiss_ex_nif.c:1029-1036,1040-1041,1048-1049`: every lock-taking NIF is `DIRTY_JOB_CPU_BOUND` and blocks in `enif_rwlock_*` (`:416,560,717,730`).
- Measured: a 1-query search on a *different* index took **3643.8 ms** (baseline 1.7 ms) while 10 `ntotal` callers waited behind a `train` write lock.
- The moduledoc claim at `index.ex:78-79` and the CLAUDE.md spec ("getters → CPU-bound") both encode the cause.
- Fix:
  1. Move the `ntotal`/`is_trained` getters to `DIRTY_JOB_IO_BOUND`.
  2. For heavy reads, `enif_rwlock_tryrlock` on the CPU entry and fall back to `enif_schedule_nif` of an IO-bound variant. Never hold a rwlock across a reschedule.
  3. Update CLAUDE.md and the moduledoc.

**H3 — OpenMP teams multiply across dirty schedulers (~3× throughput loss under concurrency)** · Perf N2
- `c_src/faiss_ex_nif.c` never calls `omp_set_num_threads`, and the C API has no equivalent. FAISS opens a team even for a single query (`faiss/utils/distances.cpp:301,342,652`), so up to ~100 threads compete for 10 cores.
- Measured: 10×200 concurrent single-query searches took 1845 ms with defaults vs 622 ms with `OMP_NUM_THREADS=1`.
- Fix:
  1. Add a per-call or app-env `:threads` option, and call `omp_set_num_threads(n)` on the dirty thread before the `faiss_*` call. Link the NIF against libomp/libgomp.
  2. Document the trade-off in the README "Concurrency" section, including `KMP_BLOCKTIME=0`.
  3. Recommend `libopenblas-openmp-dev` on Linux.

**H4 — `FAISS_OPT_LEVEL=avx2|avx512` has no effect: the generic `faiss`/`faiss_c` targets are built and shipped** · Perf N3 (supersedes P9)
- `Makefile:183,188,191,200`; README `:130-145` documents the override.
- In v1.15.1 the AVX code lives in the separate targets `faiss_avx2`/`faiss_avx512` (`faiss/CMakeLists.txt:454-495`; `c_api/CMakeLists.txt:44-83`).
- Fix: default to `FAISS_OPT_LEVEL=dd` (runtime dispatch) in both `Makefile:24` and `mix.exs:80`, or build `faiss_$(LEVEL)` + `faiss_c_$(LEVEL)`. Add the level to the build dir path (see M1) and update the README.

**H5 — Tautological "stays untrained" assertion hides C-side clustering state** · Test H1 (see Correlations: arch M2)
- `test/faiss_ex/clustering_test.exs:98-99` calls `get_cluster_assignment` on the original struct, whose `index` is `nil`. It hits the short-circuit at `lib/faiss_ex/clustering.ex:103-105` and never reaches the NIF.
- Fix: `assert {:error, "no centroids available"} = Clustering.get_centroids(clustering)`. Tighten `:96` to `msg =~ "clusters"`.

**H6 — No test runs concurrent writers against the two-lock clustering path or most read-locked ops** · Test H2
- Concurrency tests only call add/search/ntotal/reset/train: `test/faiss_ex/concurrency_smoke_test.exs:21-28` and `concurrency_test.exs:23-28,56-59`.
- Never raced:
  - `nif_train_clustering`, which takes 2 locks (`c_src:907-908`)
  - centroids `:948`, reconstruct `:560`, residuals `:628`, clone `:261`, `to_file` `:655`, `add_with_ids` `:348`
  - reconstruct/residuals vs `reset`, which is exactly the use-after-free case the read lock prevents.
- Fix:
  1. Add `:slow` race tests: reset/add vs reconstruct, residuals, clone, `to_file`.
  2. Add a race of `Clustering.train` vs `get_centroids` and a quantizer search.
  3. Add a two-task smoke version.

## Medium Findings Reported by 2+ Agents

**M1 — FAISS build dir, cmake config and NIF artifact are not keyed on `USE_CUDA`/`FAISS_OPT_LEVEL`; the CI cache key is not either** (found by: ArchAudit M3, DepsAudit N1, PerfAudit N3, TestAudit M5)
- Build keyed only by rev: `Makefile:28-29`. cmake runs only when `build/Makefile` is missing (`Makefile:176`). `$(NIF_SO)` ignores `NIF_CFLAGS` (`Makefile:210`; flags at `:69,89-95`). CI key is `RUNNER_OS-ARCH-faiss-REV` only (`ci.yml:46,163`).
- Impact: flipping `USE_CUDA=true` keeps the CPU NIF, so `num_gpus/0` returns 0. A fresh `_build` with a warm cache links GPU symbols against a CPU-only `libfaiss_c`, which fails at load time (macOS `-undefined dynamic_lookup`, `Makefile:57-60`) [INFERENCE]. Makefile cmake-flag edits are never built in CI.
- Fix:
  1. Name the build dir after the config: `FAISS_BUILD := $(FAISS_SRC)/build-$(FAISS_OPT_LEVEL)-{gpu|cpu}`.
  2. Add a `.build_config`/`.nif_flags` stamp that is rewritten only when its content changes, and make it a prerequisite of `$(NIF_SO)` and the lib copies.
  3. Add a hash of the build inputs to the CI key.
  4. Drop the `rm -rf` workaround from README `:143-144`.

**M2 — `nif_new_index` runs `faiss_index_factory` on a normal scheduler with caller-controlled factory strings** (found by: PerfAudit N4, SecAudit N2, ArchAudit prior #2 — arch counts this as accepted spec)
- Code: `c_src/faiss_ex_nif.c:1028` (flag 0), factory call at `:237`. The doc at `lib/faiss_ex/index.ex:133-134` invites "user input".
- Measured: 4–176 ms (`LSHr` d=2048 → 175.7 ms).
- Security impact: `IVF<n>M`/`IMI2x<nbit>` can allocate gigabytes or hit int overflow/UB (`faiss/index_factory.cpp:274,291-300`), and a recursive `std::regex` is a stack risk [INFERENCE].
- Fix:
  1. Flag `nif_new_index` as `DIRTY_JOB_CPU_BOUND`.
  2. Cap `byte_size(description)` (e.g. 256).
  3. Change the doc to say descriptions must be trusted or allow-listed.
  4. Update the CLAUDE.md dirty-scheduler list. The spec itself must change, or arch will keep scoring this as compliant.

**M3 — The Binary API stops at `Index`: Clustering and `compute_residuals` are list-only** (found by: PerfAudit N5, ArchAudit L2)
- List-only call sites: `lib/faiss_ex/clustering.ex:73,88-91,108-110`; `lib/faiss_ex/index.ex:336-349` (residuals) and `:235` (`search/3`, which raises `FunctionClauseError` on a binary).
- The moduledoc claim at `index.ex:41` is false. The binary validators are private to `Index` (`index.ex:453-481`).
- Measured: 8× decode amplification (1.2 MB of binary → 4.83 MB of terms).
- Fix:
  1. Move `encode_vector_data!/2` and `encode_id_data!/1` into `Shared`.
  2. Accept binaries in `Clustering.train/2`, `get_cluster_assignment/2` and `compute_residuals/3`.
  3. Add `get_centroids_binary/1`.
  4. Fix the moduledoc.

**M4 — GitHub Actions pinned to mutable tags (one major behind) in the job that holds `HEX_API_KEY`; no `hex.audit` gate** (found by: SecAudit N4 [Medium], DepsAudit N7 [Low])
- `.github/workflows/publish.yml:19,21,30` (`checkout@v6`, `setup-beam@v1`, `cache@v5`), with the key at `:79`. Same pattern in `ci.yml:23,25,51,58,76,78,90,92,100,106,139,141,166`.
- Latest versions: checkout v7.0.1, cache v6.1.0, setup-beam v1.24.1.
- Fix:
  1. Pin every action by SHA with a `# vX.Y.Z` comment, and add Dependabot for `github-actions`.
  2. Bump to checkout@v7 / cache@v6.
  3. Run `mix hex.audit` in the setup job and before `mix hex.publish`.

**M5 — The declared Elixir/OTP floor is never built in CI** (found by: TestAudit M6 [Medium], DepsAudit N5 [Low])
- `mix.exs:16` declares `elixir: "~> 1.18"` with no OTP constraint, while README `:19-20` says OTP 27+. CI and publish pin only 1.19/28.0 (`ci.yml:12-13`, `publish.yml:9-10`).
- Fix: add a `{1.18, 27}` floor row and a `{1.20, 29}` latest row to the Linux job, set `OTP_VERSION: "28"`, and align the README with mix.exs.

**M6 — `train` with n=0 behaves differently for Index and Clustering, and neither path is tested** (found by: TestAudit M1 [Medium], ArchAudit L6 [Low])
- `nif_train_index` returns `:ok` for n=0 and leaves the index untrained (`c_src/faiss_ex_nif.c:447-451`). `nif_train_clustering` returns "n must be positive" (`:894`), after `clustering.ex:78` has already allocated a quantizer.
- Fix: pick one contract for both NIFs, then pin it with tests (`Index.train(ivf, [])`, `Clustering.train(c, [])`).

**M7 — Search `k` is an unbounded, caller-driven allocation, and no test covers its error branches** (found by: TestAudit M2 [Medium], SecAudit P3 [Low, accepted])
- `c_src/faiss_ex_nif.c:372,376,398-412`; `k` is passed straight through at `lib/faiss_ex/index.ex:238,261`.
- Fix: add tests for `k=0`, `k=1.5` and `k=2**62` on both `search` and `search_binary`. Cap `k` if it can come from untrusted input.

### Lower-severity overlaps (merged, Low in all reports)
- **FAISS rev/SHA/defaults stated 2–3× across Makefile and mix.exs, with no drift guard** (found by: ArchAudit L1, DepsAudit N8, SecAudit P1-residual)
  - Locations: `Makefile:13-24` (rev at `:14` and again at `:19`); `mix.exs:9-10,72-85`. The CI key parses only `Makefile:14` (`ci.yml:44,162`), while the build uses the mix.exs value.
  - Bumping `:14` without `:19` silently disables SHA verification (`Makefile:22`).
  - Fix: add `FAISS_DEFAULT_REV` and make the Makefile the single source; `make_env/0` forwards only variables that are set. Add a CI check that the files agree and a `$(warning)` when the SHA is empty.
- **`nif_get_num_gpus` on a normal scheduler; first call initializes CUDA** (found by: ArchAudit L7, PerfAudit P8): `c_src/faiss_ex_nif.c:811-815,1045`. Fix: `DIRTY_JOB_IO_BOUND`.
- **`ClusteringResource` caches neither `k` nor `d`; validation and centroid allocation happen under the lock** (found by: ArchAudit L6, PerfAudit N7)
  - `c_src/faiss_ex_nif.c:73-76,907-918,948-970`.
  - Fix: cache them at `nif_new_clustering`, validate and allocate before the locks, check that the quantizer dim matches, and remove the `goto unlock` validation branches.

## Cross-Category Correlations

1. **Perf ↔ Test ↔ Spec: concurrency failures are invisible to every gate.**
   - Perf H2 (dirty-CPU starvation, `c_src:1029-1049`) and H3 (OpenMP oversubscription) only show up when many callers run at once.
   - Test H6 never races the affected read paths. The bench is single-process (`bench/faiss_ex_bench.exs:41-112`, perf N8). CI sets no `timeout-minutes` (test L9), so a lock deadlock would hang for 6 h.
   - Arch scored the scheduler placement ✅ because CLAUDE.md lists getters as CPU-bound: the spec encodes the perf defect. Fixing H2 needs a spec change, a concurrency bench scenario, and a race test together.
2. **Arch ↔ Test: Clustering's struct/resource split hides behind a tautological test.**
   - Arch M2 [Medium, arch only]: `train/2` mutates the shared C `ref` but returns a new struct with its own quantizer (`lib/faiss_ex/clustering.ex:73-81,88-91,104-111`). So `get_centroids` and `get_cluster_assignment` can disagree, and `c.trained?` stays false while centroids exist.
   - The test at `clustering_test.exs:98-99` (H5) passes *because of* that split: it reads the struct's `nil` index, not C state.
   - The two-lock train path is never raced (H6). Fix M2 (pick the value model or the resource model) together with H5: the corrected assertion will exercise the C state M2 is about.
3. **Arch/Deps/Perf ↔ Test: build-config defects ship green.**
   - M1 (unkeyed build/cache) and H4 (no-op AVX level) are invisible to CI. The CI cache key ignores cmake flags (`ci.yml:46,163`), and CUDA is never compiled in any workflow (test M4, `c_src:741-808` is dead code as far as CI knows).
   - A broken GPU build, or an x86 build without AVX2, passes every check.
4. **Security ↔ Perf ↔ Test ↔ Arch: GPU code paths are unverifiable.**
   - Security N1: a negative device id gets past FAISS's upper-bound-only checks and `abort()`s the VM (`c_src/faiss_ex_nif.c:746-759`; `lib/faiss_ex/index.ex:178,413-416`).
   - Perf N6: one `StandardGpuResources` per index reserves up to 1.5 GiB each (`c_src:751-752`).
   - Arch M1 was a false positive.
   - None of this can be confirmed or regression-tested, because test M4 found no CUDA job (`ci.yml`, `test_helper.exs:5-8`). Add at least a compile-only CUDA container job plus `:no_cuda` tests for `index.ex:178,413`.
5. **Security ↔ Perf ↔ Test: factory strings are trusted input, but nothing tests or limits them.** M2 (DoS and measured latency) correlates with test M3: the factory-failure branch `c_src:241` and other `make_faiss_error` paths are untested, so a regression in exception conversion would abort the VM undetected.
6. **Security ↔ Test: untrusted index files.**
   - Security N3 [Medium, sec only]: with `io_flags=0`, an on-disk IVF file can make the VM mmap any path read+write (`lib/faiss_ex/index.ex:362-372`, `c_src:684`).
   - Test prior #13 / L7: `:io_flags` and corrupt-file loading are untested (`index_test.exs:328,332`).
   - Fix both together: document the trust boundary, recommend `IO_FLAG_READ_ONLY`, and add round-trip and corrupt-file tests.
7. **Perf/Arch ↔ Security: widening the Binary API widens the alignment UB.** M3 extends binary pass-through. Security N5 [Low] notes that caller sub-binaries reach `float*`/`idx_t*` unaligned (`c_src:186-187,306,418,…`; `index.ex:455-466`). Add the alignment check-or-copy inside the shared `encode_vector_data!` when it moves to `Shared`.
8. **Deps ↔ Security: publish-job supply chain.** Tag-pinned actions (M4), cache-restored FAISS trees that are never re-verified (deps N6, prefix `restore-keys` at `publish.yml:36-37`), no `permissions:` block in ci.yml, unvalidated tag interpolation into `sed`, and no protected environment (security N6, `publish.yml:41-42`) all converge on the job that holds `HEX_API_KEY`.
9. **Deps ↔ Test: toolchain matrix.**
   - Moving off the deprecated `ubuntu-22.04` runners (deps N4, `ci.yml:18,73,87,137`) to 24.04 also fixes the CMake ≥3.24 floor (deps N3, README `:22` wrongly says 3.17+).
   - Pair that move with the M5 floor/latest matrix rows.

## Prior-Findings Trend

| Category | Prior total | Resolved | Partially resolved | Still open | Regressed | Notes |
|---|---|---|---|---|---|---|
| Architecture | 3 | 2 | 0 | 1 | 0 | Open: `nif_new_index` on a normal scheduler (accepted as spec by arch; escalated by perf and security → M2) |
| Performance | 9 | 4 (P1,P2,P3,P7) | 0 | 4 (P5,P6,P8,P9) | 0 | P4 is WONTFIX (measured). P5 re-assessed as worse (→ H2); P9's override is broken (→ H4) |
| Security | 4 | 2 (P1,P4) | 0 | 2 (P2, P3 accepted) | 0 | P1 residual: empty SHA or custom rev builds unverified |
| Testing | 15 | 6 (#1–6) | 2 (#10,#12) | 7 (#7,8,9,11,13,14,15) | 0 | #7/#8/#9 fold into test M3 |
| Dependencies | 4 | 1 | 1 (OTP floor documented, not enforced) | 2 (Hex minor lag, no `hex.audit` in CI) | 0 | |
| **Total** | 35 | 15 | 3 | 16 | **0** | Every score drop comes from wider scope or newly measured issues, not code regressions |

## Coverage

| File | Represented | Key items |
|---|---|---|
| arch-review.md | Yes | M1 (dropped, re-credited), M2 (corr. 2), M3→M1, L1/L6/L7 merged, L2→M3, prior #2→M2 |
| perf-audit.md | Yes | H2, H3, H4, M1, M2, M3, N6 (corr. 4), N7/P8 merged, N8 (corr. 1) |
| security-audit.md | Yes | M2, M4, M7, N1 (corr. 4), N3 (corr. 6), N5 (corr. 7), N6 (corr. 8), M1-FP confirmation |
| test-audit.md | Yes | H5, H6, M1 (M5), M5 (M6), M6 (M5), M2 (M7), M3/M4/L7/L9 (correlations) |
| deps-audit.md | Yes | H1, M1, M4, M5, rev-sync merge, N3/N4/N6 (correlations) |

## Coverage Gaps
None. Single-report Mediums are left out of the findings list by caller rule; they appear only through the correlations above: arch M2; perf N6; security N1, N3; test M3, M4; deps N3, N4.
