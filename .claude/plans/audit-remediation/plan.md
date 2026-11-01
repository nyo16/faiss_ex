# Plan: Audit Remediation (2026-11-01 health audit → v0.4.0)

**Status:** READY — D1–D4 confirmed (all recommended options) and all 4 deferrals approved by the user on 2026-11-01
**Depth:** standard (planned from audit; no research agents — the findings are the research, Iron Law 7)
**Created:** 2026-11-01
**Input:** `.claude/audit/summaries/project-health-2026-11-01.md`, `.claude/audit/summaries/consolidated.md`, `.claude/audit/reports/*.md`
**Baseline:** a397362 (v0.3.2, FAISS v1.15.1), 85/85 tests, health 70/100 (C)

## Goal

Close every finding from the 2026-11-01 audit, or have the user explicitly defer it:
- unblock CI (macOS 14 runner retires 2026-11-02);
- remove the VM-wide dirty-scheduler starvation and the OpenMP oversubscription;
- make `FAISS_OPT_LEVEL` real and key the build on its configuration;
- fix the Clustering state split;
- close the NIF error-branch and concurrency test gaps;
- harden the build and publish supply chain.

Ship it as v0.4.0, which contains behavior changes (see Phase 7).

## Decisions (confirmed by user 2026-11-01: recommended column for D1–D4)

| ID | Question | Recommended | Alternatives |
|---|---|---|---|
| D1 | How lock waiters stop starving the dirty-CPU pool (perf N1) | **Trylock on the CPU entry. On contention, release any pre-allocated result and `enif_schedule_nif` the same op to a `DIRTY_JOB_IO_BOUND` variant that blocks.** Waiting moves to the IO pool (size it with `+SDio`). Blocking-lock fairness is kept. ~1 helper + 1 mode flag per NIF. | (b) Trylock → `:busy` → Elixir retries with backoff: no thread ever blocks, but up to ~16 ms of extra latency, and writers can starve under constant reads. A writer-pending flag fixes that but leaks if the writer process is killed mid-backoff. (c) Document only (`+SDcpu`, shard hot-written indexes). |
| D2 | OpenMP thread control (perf N2) | **One global knob: `config :faiss_ex, omp_threads: n` (read at NIF load) plus `FaissEx.set_omp_threads/1` at runtime.** It is stored in a C atomic and applied with `omp_set_num_threads` before every `faiss_*` call (the ICV is per thread). Default `0` keeps the FAISS default, so there is no regression for large batches. | (b) A per-call `:threads` option on search/add/train. (c) Both. |
| D3 | Clustering ownership model (arch M2) | **Value model: `train/2` allocates a fresh clustering resource and a fresh quantizer.** The input struct is never mutated, so old structs stay self-consistent. | (b) Resource model: the C clustering resource owns the quantizer (`enif_keep_resource`), and `trained?`/`index` are derived from C state. This keeps mutation semantics like `Index`. |
| D4 | `from_file/2` on untrusted files (security N3) | **Keep the default `io_flags: 0` and document the trust boundary.** Document the flag values and recommend `io_flags: 6` (`IO_FLAG_READ_ONLY ||| IO_FLAG_ONDISK_SAME_DIR`) for untrusted files. | (b) Default to `6`. Safe by default, but a breaking change for writable on-disk IVF users. |

Tasks below implement the recommended column. Tasks that depend on a decision are marked `{D#}`. Under D3 the two-lock race test in 6.2 is N/A.

## Deferred (approved by user 2026-11-01; not dropped, follow-up plans)

| Finding | Why defer |
|---|---|
| perf N6: one `StandardGpuResources` per `cpu_to_gpu` (up to 1.5 GiB each) | Needs CUDA hardware to validate a shared-resources API. Same deferral as the faiss-upgrade plan. |
| test M4 (part): run the `:cuda` test suite in CI | Needs a GPU runner. A compile-only CUDA syntax job (2.9) and `:no_cuda` tests (5.11) are in scope. |
| perf P6: `add`/`train` hold the write lock for the whole batch (no `chunk_size:`) | New API with an atomicity tradeoff (a chunked add is not all-or-nothing). Better as its own feature plan. |
| arch L4: `d` vs `dim`, mixed `get_` prefixes | A breaking rename of the public API. Do it in a 1.0 API pass, not a remediation release. |

---

## Phase 0 — CI hotfix (land today) [ci]

- [x] 0.1 **[deps N2 / H1]** `.github/workflows/ci.yml:135` `macos-14` → `macos-15`. Update the comments at `ci.yml:33-36` and `:121-122` that name macos-14. — PR #8, all 6 checks green (macOS arm64 41 s on macos-15).

**Verify:** push a branch; the `Tests (macOS arm64)` job is green on `macos-15`.

## Phase 1 — Build system (`Makefile`, `mix.exs`) [build]

- [x] 1.1 **[arch L1, deps N8-pin]** Single-source the FAISS defaults in the Makefile: — done; `make_env/0` deleted outright (make inherits env; see scratchpad)
  - Add `FAISS_DEFAULT_REV := v1.15.1`, then `FAISS_GIT_REV ?= $(FAISS_DEFAULT_REV)` and `ifeq ($(FAISS_GIT_REV),$(FAISS_DEFAULT_REV))` for the SHA (this replaces the literals at `Makefile:14,19`).
  - In `mix.exs`, delete `@faiss_git_rev`/`@faiss_git_sha` (`:9-10`). `make_env/0` (`:72-91`) forwards only the env vars that are actually set: `USE_CUDA`, `FAISS_OPT_LEVEL`, `FAISS_GIT_REPO`, `FAISS_GIT_REV`, `FAISS_GIT_SHA`, `FAISS_PREFIX`, `LIBOMP_PREFIX`.
  - Update the CLAUDE.md "keep in sync in BOTH" note: the Makefile is now the single source.
- [x] 1.2 **[arch M3, deps N1, perf N3, test M5]** Key the FAISS build on its configuration: — `build-dd-cpu`, `build-generic-cpu` verified
  - `FAISS_BUILD := $(FAISS_SRC)/build-$(FAISS_OPT_LEVEL)-$(if $(filter true,$(USE_CUDA)),gpu,cpu)`.
  - `cmake -B $(notdir $(FAISS_BUILD))`. The source tree stays shared per rev.
- [x] 1.3 **[arch M3, deps N1]** Add a config stamp: — printf+cmp+mv (Make 3.81 has no `$(file)`); dd→generic→dd re-copies libs and re-links
  - `$(PRIV_DIR)/.build_config` holds `$(FAISS_BUILD) $(NIF_CFLAGS) $(NIF_LDFLAGS)`. A shell compare rewrites it only when the content changes.
  - Make it a prerequisite of `$(NIF_SO)` and of both `$(LIB_DIR)` copy rules. Without it, switching back to an older cached build dir would not re-copy its libs, because their mtime is older.
  - Applies to the source path (`Makefile:191-213`) and the system path (`:131-151`).
- [x] 1.4 **[perf N3 / H4, P9]** Default `FAISS_OPT_LEVEL` to `dd` (`Makefile:24`, and drop the default in `mix.exs:80` per 1.1). FAISS builds runtime SIMD dispatch into the main `faiss` target for x86_64 and aarch64 (`faiss/CMakeLists.txt:541-620`). — `$(error)` branch: `faiss_c_avx2` links `faiss_avx2` by name; only dd/generic accepted
  - Keep the `generic|avx2|avx512` overrides working: for `avx2`/`avx512*`, build `faiss_$(LEVEL)` + `faiss_c_$(LEVEL)` and install them as `libfaiss`/`libfaiss_c` (copy plus `install_name_tool -id`). Otherwise say in the README that `dd` is the only supported optimized level and `$(error)` on the others. **Pick the former** unless `faiss_c_avx2` links against `libfaiss_avx2` under a different install name. Check with `otool -L`/`readelf -d`, and record the finding in the scratchpad.
- [x] 1.5 **[deps N6, sec P1-residual]** — recursive var `verify_faiss_sha` (not `define`) so it composes in one recipe line; `deadbeef` fails closed before any copy
  - Factor the SHA check (`Makefile:163-172`) into a `define verify_faiss_sha`. Run it in the clone recipe and in a `verify-faiss` phony target that `$(NIF_SO)` takes as an **order-only** prerequisite, so it re-verifies on every `make` (milliseconds) without forcing a rebuild.
  - Emit `$(warning ...)` when `FAISS_GIT_SHA` is empty.
- [x] 1.6 **[sec P2]** Validate `FAISS_GIT_REV` against `^[A-Za-z0-9._/-]+$` and `FAISS_GIT_REPO` against `^https://`, with `$(error)` on mismatch. Quote every path and URL expansion in recipes: `"$(FAISS_SRC)"`, `"$(FAISS_GIT_REPO)"`, `"$(SYSTEM_LIB_DIR)"`, `"$(LIB_DIR)"`, `"$(PRIV_DIR)"` (`Makefile:134,142,150,161-173,178,183,188,193-213`). — pure-make `strip_chars` allow-list (no `$(shell)` on untrusted input); repo also restricted to `[A-Za-z0-9._/:~%+@-]`
- [x] 1.7 **[deps N8-CUDA]** When `USE_CUDA=true`, parse `nvcc --version` and `$(error)` below 12 with the README remediation. — in the configure recipe; verified with a fake nvcc 11.8
- [x] 1.8 **[deps N3]** Before configure (Step 2), parse `cmake --version` and `$(error)` below 3.24 with a remediation line (Kitware apt repo or `pip install cmake`). — verified with a fake cmake 3.22
- [x] 1.9 **[test L8]** `mix.exs` `project/0`: `test_coverage: [ignore_modules: [FaissEx.NIF], summary: [threshold: 90]]`. — 96.81% total
- [x] 1.10 **[deps prior: Hex lag]** Bump elixir_make 0.9.0 → 0.10.x (read its CHANGELOG for `make_env`/`make_targets` changes first), plus ex_doc and benchee (patch). Update the `mix.exs:65-67` requirements only if needed. — elixir_make 0.10.1 (only change: Elixir 1.14+), ex_doc 0.40.4, benchee 1.5.1; requirement `~> 0.10`
- [x] 1.11 **[deps N3, N8-libomp, test/arch L9 partial]** README build section:
  - CMake 3.24+ and a C++20 compiler (GCC 10+ / Clang 12+ / AppleClang 14+) at `README:21-22`.
  - Ubuntu recipe → 24.04 (`README:46-51`).
  - libomp: auto-detected via `brew --prefix libomp`, override with `LIBOMP_PREFIX` (`README:39-40`).
  - `FAISS_OPT_LEVEL` section (`README:130-145`): `dd` is the default; delete the `rm -rf` workaround (1.2 makes it unnecessary).
  - CUDA ≥ 12 is enforced (1.7).

**Verify:**
- `mix compile` from a clean `_build`: a fresh `dd` FAISS build (~10 min) lands in `~/.cache/faiss_ex/faiss-v1.15.1/build-dd-cpu/`.
- `mix test --include slow` is green.
- `FAISS_OPT_LEVEL=generic mix compile` reuses or creates `build-generic-cpu` and **re-links the NIF**: the stamp changed and `priv/lib` is re-copied. Check with `otool -L` and the mtime.
- Switching back to `dd` restores the `dd` libs.
- Corrupting the SHA (`FAISS_GIT_SHA=deadbeef mix compile`) fails closed even with a warm cache.
- `mix test --cover` exits 0.

## Phase 2 — CI and supply chain (`.github/`) [ci]

- [x] 2.1 **[deps N4]** `ubuntu-22.04` → `ubuntu-24.04` (`ci.yml:18,73,87`; `publish.yml:15`) and `ubuntu-22.04-arm` → `ubuntu-24.04-arm` (`ci.yml:137`).
- [x] 2.2 **[test M5, deps N1]** Cache keys: — `OS-ARCH-ImageOS-faiss-REV-MAKEFILEHASH`
  - Add `${ImageOS}` (the runner image) and `hashFiles('Makefile')` to the FAISS key (`ci.yml:44-46,162-163`). The OS-major alone would let a 22.04-built FAISS restore onto 24.04.
  - Derive the rev from `FAISS_DEFAULT_REV` (1.1) instead of `sed` on `FAISS_GIT_REV ?=`.
- [x] 2.3 **[test L10]** Give the FAISS cache step `id: faiss` (`ci.yml:50`). Run `deps.get`/`compile` when either cache missed (`ci.yml:65-69`). — moot after 2.7: deps/_build cache dropped, every test job always runs deps.get/compile (NIF only; FAISS cached)
- [x] 2.4 **[sec N4, deps N7]** — checkout v7.0.1, cache v6.1.0 (v6.0.0 notes: ESM migration only), setup-beam v1.24.1; Dependabot with 7-day cooldown (zizmor)
  - Bump `actions/checkout@v6` → v7 and `actions/cache@v5` → v6 (read the release notes for `cache/restore` input changes).
  - Pin every action by commit SHA with a `# vX.Y.Z` comment in both workflows.
  - Add `.github/dependabot.yml` (`package-ecosystem: github-actions`, weekly).
- [x] 2.5 **[sec N6, test L9]** Top-level `permissions: contents: read` in `ci.yml`. Add `timeout-minutes`: 60 for `setup`, 30 for `test`/`test-native`/`format`, 90 for `publish`. — lint 15, test/test-native 60 (cold FAISS build in-job), cuda-syntax 30, publish 90
- [x] 2.6 **[deps prior: hex.audit]** Run `mix hex.audit` in the CI `setup` job after `deps.get`, and in `publish.yml` before `mix hex.publish`. — in the Lint job (setup removed)
- [x] 2.7 **[test M6, deps N5]** — self-contained matrixed test job; setup job removed
  - Add a Linux `{elixir, otp}` matrix to the `test` job: `{1.18, 27}` (floor) and `{1.20, 29}` (latest), keeping `{1.19, 28}`. The `setup`/`test` cache handoff must be matrixed too, or the test job made self-contained like `test-native`.
  - `OTP_VERSION: "28.0"` → `"28"` (`ci.yml:13`, `publish.yml:10`).
  - README `:19-20`: "Elixir 1.18+ on OTP 27+ (the CI-tested floor)".
- [x] 2.8 **[sec N6, deps N6]** `publish.yml`: — restore-only exact key; user action still pending (flagged in PR #10)
  - Validate `TAG_VERSION` against `^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$` before the `sed` (`:39-43`).
  - Add `environment: hex` to the job.
  - Replace the prefix `restore-keys: Linux-X64-faiss-` (`:36-37`) with the exact CI key format from 2.2.
  - **User action (GitHub settings, not code):** create the `hex` environment with required reviewers, move the `HEX_API_KEY` secret into it, and add a tag-protection rule for `v*`.
- [x] 2.9 **[test M4 (part)]** New `cuda-syntax` job: — reuses `make verify-faiss`; adds `-Werror`; caught a real `faiss_get_num_gpus` bug (fixed)
  - Container `nvidia/cuda:12.6.3-devel-ubuntu24.04`; `apt-get install erlang-dev git`.
  - Shallow-clone FAISS at the pinned rev and verify the SHA (headers only, no build).
  - Run `cc -std=c11 -fsyntax-only -Wall -Wextra -DFAISS_GPU_ENABLED -I<erts> -I<faiss> -I/usr/local/cuda/include c_src/faiss_ex_nif.c`.
  - Catches bit-rot in the `#ifdef FAISS_GPU_ENABLED` block (`c_src:741-808`), which Phases 3–4 edit.
- [x] 2.10 **[test L8]** The Linux `test` job runs `mix test --include slow --cover`.
- [x] 2.11 **[deps: credo unused]** Run `mix credo --strict` locally and fix the findings, then add it to the `format` job. If the findings are pure style noise against this codebase's conventions, remove credo from `mix.exs:68` instead. Record which in the scratchpad. — kept; 1 finding fixed; runs in Lint

**Verify:** the PR's CI is green on every matrix row. `actionlint` and `zizmor` (if available) run clean on `.github/workflows/`. The `cuda-syntax` job fails when a deliberate typo is introduced inside the GPU block (try it locally in the container or on a throwaway commit, then revert). — ✅ PR #10: 9/9 checks green; actionlint + zizmor clean; typo test exit 1 in the container

## Phase 3 — C NIF correctness and safety (`c_src/faiss_ex_nif.c`) [c]

Follow the CLAUDE.md lock-unwinding spec: validation and allocation go above the lock, there is one `unlock:` label, and terms are built after release.

- [x] 3.1 **[sec N1]** `nif_index_cpu_to_gpu`: `if (device < 0) return make_error_msg(env, "device must be non-negative");` after the arg parse (`:746-749`). The Elixir guard is 5.4. — GPU block syntax-checked in the CUDA container
- [x] 3.2 **[perf N4, sec N2, arch prior]** `nif_new_index` → `ERL_NIF_DIRTY_JOB_CPU_BOUND` (`:1028`). Reject descriptions over 256 bytes with "description too long", next to the NUL check (`:231`). — `MAX_DESCRIPTION_BYTES`
- [x] 3.3 **[perf P8, arch L7]** `nif_get_num_gpus` → `ERL_NIF_DIRTY_JOB_IO_BOUND` (`:1045`).
- [x] 3.4 **[arch L6, perf N7]** Cache `d` and `k` in `ClusteringResource` (`:73-76`) at `nif_new_clustering` (`:871`). — centroids status is a `const char *`, plus a "centroids size mismatch" guard before the memcpy
  - `nif_train_clustering`: size check and quantizer-dim check (`idx_res->dim == clust_res->d`, "quantizer dim mismatch") go **above** both locks. The critical section shrinks to the `faiss_Clustering_train` call (`:907-925`).
  - `nif_get_clustering_centroids`: `enif_alloc_binary(k*d*4)` goes above the lock (`:962`). The "no centroids" check stays inside as a status int, and the binary is released after unlock on that path.
  - Update the CLAUDE.md lock-unwinding paragraph. Its example of "validation must read locked state" (`faiss_Clustering_d`) no longer applies.
- [x] 3.5 **[sec N5]** Alignment: — plus `free_owned`; empty inputs get a static dummy pointer (covers 3.7's non-NULL requirement for every NIF)
  - Add helper `static int aligned_input(const ErlNifBinary *b, size_t align, const void **out, void **owned)`. It returns the binary pointer when aligned, otherwise an `enif_alloc` copy that the caller frees after unlock.
  - Apply it to every `float*`/`idx_t*` input: `:186-187,306,351-352,418,464,520-522,615-616,625,921`.
- [x] 3.6 **[arch L8]** Replace the copy-pasted per-NIF size validation (`:289-302,329-345,375-397,447-460,504-512,593-611`) with one helper: `static const char *check_rows(ErlNifSInt64 n, int dim, size_t elem, const ErlNifBinary *b)`, which returns NULL or a message. The existing messages stay byte-identical, because tests assert them. — signature takes the mismatch message as a 5th arg (three distinct messages exist); `rows_bytes` sizes results
- [x] 3.7 **[test M1, arch L6 n=0]** `nif_train_index`: remove the n==0 → `:ok` short-circuit (`:447-451`) and let FAISS decide. A Flat train with n=0 is a no-op `:ok`. An IVF train with n=0 raises "Number of training points (0) should be at least as large as number of clusters" → `{:error, _}`. **Verify the FAISS behavior with a probe before relying on it.** Pass a non-NULL dummy pointer when `bin.size == 0`. — probe: Flat/HNSW32/IDMap,Flat `:ok`; IVF4,Flat and PQ2 `{:error, "...Number of training points (0) should be at least as large as number of clusters (4|256)"}`
- [x] 3.8 [test] Tests for 3.1–3.7 go in Phase 5 (5.11) with the Elixir guards. — done in 5.11 (description cap, train n=0, quantizer dim via clustering, alignment, device guards)

**Verify:** `mix compile --warnings-as-errors`; `mix test --include slow`. Audit the diff against the 20-row NIF → lock → scheduler table in `.claude/audit/reports/arch-review.md` and update that table in CLAUDE.md terms.

## Phase 4 — Scheduler starvation and OpenMP (`c_src`, `Makefile`, `lib/faiss_ex.ex`) [c] {D1, D2}

- [x] 4.1 **[perf N1 / H2]** {D1} Lock helper and reschedule: — `LockMode` + `LOCKED_NIF(name)` macro (impl + `_io` twin) + `RESCHEDULE_BLOCKING`; every reschedule path releases result binaries, alignment copies, and (train_clustering) the first lock
  - Add `static int try_lock_or_reschedule(...)`, or a `mode` parameter (`MODE_CPU_TRY` / `MODE_IO_BLOCK`) on each locked NIF's implementation.
  - The CPU entry does parse → validate → allocate → `enif_rwlock_tryrlock`/`tryrwlock`. On `EBUSY` it releases the pre-allocated result binary and returns `enif_schedule_nif(env, "<name>_io", ERL_NIF_DIRTY_JOB_IO_BOUND, <impl>_io, argc, argv)`. The IO variant re-runs the same code with blocking acquisition.
  - **No lock is ever held across a reschedule.**
  - Apply to clone, add, add_with_ids, search, train, reset, reconstruct, residuals, train_clustering, and centroids.
  - `nif_train_clustering` takes two locks: trylock the index, then trylock the clustering. On failure of the second, release the first and reschedule. The order stays quantizer → clustering.
- [x] 4.2 **[perf N1, P5]** `nif_get_index_ntotal`/`nif_get_index_is_trained` → `ERL_NIF_DIRTY_JOB_IO_BOUND` (`:1040-1041`). They do no CPU work.
- [x] 4.3 **[perf N2 / H3]** {D2} OpenMP control: — iomp5 / no-libomp → `-DFAISS_EX_NO_OMP` (knob ignored); `load_info` validated in `FaissEx.NIF` (raises on bad config)
  - Linux: link the NIF with `-fopenmp`. macOS: `-Xclang -fopenmp -I$(LIBOMP_PREFIX)/include -L$(LIBOMP_PREFIX)/lib -lomp`, plus `LIBOMP_PREFIX` detection for the `FAISS_PREFIX` path.
  - `#include <omp.h>`; `static _Atomic int g_omp_threads`; capture `omp_get_max_threads()` at load as the default.
  - `apply_omp_threads()` runs right before each `faiss_*` compute call (search, add, add_with_ids, train, reconstruct, residuals, train_clustering). It calls `omp_set_num_threads(g > 0 ? g : default)`, because the ICV is per thread and dirty threads are reused.
  - New NIF `nif_set_omp_threads/1` (normal scheduler; an atomic store). `load_info` comes from `Application.get_env(:faiss_ex, :omp_threads, 0)` in `FaissEx.NIF`'s `@on_load`.
- [x] 4.4 **[perf N2]** Public API: `FaissEx.set_omp_threads(non_neg_integer) :: :ok`, with a doc explaining the tradeoff (measured: 1-thread is ~3× faster for many concurrent single queries and ~1.6× slower for one 1000-query batch). — re-measured: ~2.8× / ~1.5× (Flat 100k×128)
- [x] 4.5 **[perf N8]** `bench/faiss_ex_bench.exs`: — `System.schedulers_dirty_cpu/0` does not exist; uses `:erlang.system_info(:dirty_cpu_schedulers_online)`. Parallel scenario uses a 100k index (10k shows no OpenMP effect)
  - A parallel scenario (`parallel: System.schedulers_online()`) of single-query search, with and without `set_omp_threads(1)`.
  - A starvation scenario: train on index A with `System.schedulers_dirty_cpu()` waiters, then search on index B.
  - Shuffled-key `reconstruct`, `reconstruct_binary`, and k=100 list vs `search_binary`.
  - One IVF and one HNSW index.
  - Move `Index.new` into `before_each` (`:45,50,56,60`).
- [x] 4.6 **[perf N1, N2; correlation 1]** Docs: — README "Concurrency tuning" subsection under Thread Safety
  - Index moduledoc "Concurrency" (`index.ex:65-79`): the waiting model and `+SDio` sizing.
  - README "Concurrency": OMP threads, `KMP_BLOCKTIME=0`, `libopenblas-openmp-dev` on Linux.
  - CLAUDE.md dirty-scheduler list and locking convention.
  - Append a 2026-11 update to `.claude/solutions/otp-issues/nif-rwlock-getters-must-be-dirty-faissex-20260711.md`: a dirty NIF that is merely *blocked* still consumes a dirty-CPU thread.
- [x] 4.7 [test] **[perf N1 regression, test H2]** `:slow` test in `concurrency_test.exs`: — IVF1024 on 80k×64 (~1.7 s)
  - Start a ≥1 s IVF train on A in a Task, then spawn `System.schedulers_dirty_cpu()` Tasks calling `Index.search(A, …)`.
  - Assert that `Index.search(B, …)` returns **while the train Task is still alive** (`Process.alive?`). The assertion is about ordering, not wall-clock thresholds.
  - Before 4.1 it fails, because B queues behind the blocked dirty-CPU threads.
- [x] 4.8 [test] `set_omp_threads/1`: an invalid argument raises `FunctionClauseError`. Search results are identical with 1 thread and with the default. Its effect on thread count is not unit-tested (bench only). — `test/faiss_ex/omp_threads_test.exs`

**Verify:** ✅
- The 4.7 test fails on the pre-4.1 commit and passes after. — fails on the Phase 3 NIF with "search on an idle index waited for training on a different index"; passes after
- `otool -L priv/libfaiss_ex.so` (macOS) or `ldd` (Linux) shows exactly **one** OpenMP runtime, the same one `libfaiss` links. — macOS: `/opt/homebrew/opt/libomp/lib/libomp.dylib` only, same as libfaiss. Linux: covered by CI test runs (gcc → libgomp both sides)
- The bench starvation scenario drops from ~3.6 s to the single-search baseline order of magnitude. — 0.036 ms contended vs 0.025 ms uncontended
- 10×200 concurrent single-query searches with `set_omp_threads(1)` reproduce the ~3× win. — 1529/1635 ms → 586/557 ms (~2.8×); batch 78/88 → 123/126 ms
- `mix test --include slow` is green. — 90 passed

## Phase 5 — Elixir API (`lib/`) [elixir] {D3, D4}

- [x] 5.1 **[arch M2]** {D3} `Clustering.train/2` (`clustering.ex:73-81`): — empty data checked via the encoded row count, so binaries are covered too
  - Validate `n > 0` before any allocation ("n must be positive", matching the NIF).
  - Then `NIF.nif_new_clustering(d, k)` → a fresh `ref`, plus a fresh quantizer → `nif_train_clustering`.
  - Return `%{clustering | ref: new_ref, index: quantizer, trained?: true}`. The input struct is never mutated.
  - Update the moduledoc and `train/2` doc to describe the value semantics.
- [x] 5.2 **[arch L2, perf N5 / M3]** Move `encode_vector_data!/2` and `encode_id_data!/1` from `index.ex:453-481` into `Shared` (the test-only-helper rule in CLAUDE.md is unaffected). — `compute_residuals/3` decodes `compute_residuals_binary/3`; `get_centroids/1` decodes `get_centroids_binary/1`
  - Accept binary input in `Index.search/3` (`:235`), `compute_residuals/3` (`:337-338`), `Clustering.train/2`, and `Clustering.get_cluster_assignment/2`.
  - Add `Clustering.get_centroids_binary/1` and `Index.compute_residuals_binary/3`.
  - Fix the moduledoc Binary API list (`index.ex:41-58`).
- [x] 5.3 **[arch L3]** `add_with_ids/3` raises `ArgumentError` ("got N ids for M vectors, expected one id per vector") on a count mismatch, matching `compute_residuals/3`. Delete the caveat at `index.ex:136-139` and update the doc at `:209-210`. The NIF check stays as a backstop.
- [x] 5.4 **[arch L5, sec N1]** `Keyword.validate!` in `Index.new/3` (`[:metric, :device]`) and `from_file/2` (`[:io_flags]`).
  - A bad `:device` raises `ArgumentError` with an explicit message instead of `CaseClauseError` (`:176-179`).
  - Guard `device_id` with `is_integer(device_id) and device_id >= 0` in `cpu_to_gpu/2` (`:413-416`) and in the `{:cuda, id}` device option.
- [x] 5.5 **[sec N3]** {D4} `from_file/2` doc: a **Trust boundary** section explaining the on-disk IVF arbitrary-path read/write mmap, and an `:io_flags` table (`2` read-only, `4` on-disk same dir, `8` skip IVF data), recommending `6` for untrusted files. Add the same warning to README `:243-244`.
- [x] 5.6 **[sec N2, M2]** `Index.new/3` doc (`index.ex:130-134`): factory descriptions are **trusted input**. Allow-list them if they are user-derived; note the 256-byte cap (3.2) and that FAISS can allocate gigabytes for `IVF<n>M`.
- [x] 5.7 **[arch L9]** Fix the stale "lists only" docs: `lib/faiss_ex.ex:6-7` and `index.ex:6`.
- [x] 5.8 **[test M1]** Document the `train/2` n=0 contract from 3.7 (FAISS decides: no-op for Flat, error when training is required).
- [x] 5.9 **[sec P3, test M2]** `search/3` doc: `k` drives an `n*k` allocation, so cap it if it is caller-derived. Behavior is unchanged (`safe_mul` already guards the overflow).
- [x] 5.10 `mix format`; `@spec` updates for every changed signature.
- [x] 5.11 [test] **[test H1, M1, M2, M3, M4-part, L5, L7, prior #7–#15, inventory ✗ #1,4,6,8,9,10,11,12,14,19,20,23,27,28,31; sec N1, N5]** The new and changed tests are listed in the "Test additions" appendix below. — all rows covered; **found and fixed a VM abort** (FAISS exceptions inside OpenMP regions: residual key ≥ ntotal, IDMap reconstruct, HNSW contiguous past ntotal) — see scratchpad

**Verify:** `mix compile --warnings-as-errors`, `mix format --check-formatted`, `mix credo --strict` (if kept per 2.11), `mix test --include slow`, `mix docs` with no warnings. — ✅ all clean; PR #13

## Phase 6 — Test hygiene (`test/`) [test]

- [x] 6.1 **[test H1 / H5]** `clustering_test.exs:98-99` → `assert {:error, "no centroids available"} = Clustering.get_centroids(clustering)`. Line `:96` → `assert msg =~ "clusters"`. Under D3 this now checks that the original resource is untouched. — landed in PR #13 (same file as 5.11)
- [x] 6.2 **[test H2 / H6]** `:slow` races on one Flat index: workers run `reset`/`add`/`add_with_ids` (IDMap) against `reconstruct([0])`, `compute_residuals`, `clone`, and `to_file` (`@tag :tmp_dir`). Accept `{:error, msg}` with `is_binary(msg)`, because a key can fall past `ntotal` after a reset. Assert no crash and that the index still works afterwards. — on `IDMap,Flat` (exercises the former abort path); PR #14
  - Add a two-task, barrier-started version to `concurrency_smoke_test.exs`.
  - Clustering two-lock race: **N/A under D3**. Train never shares resources, so the path is unreachable concurrently through the public API (note this in the test file). Under D3(b), add a race of `train` vs `get_centroids` vs a quantizer search instead.
- [x] 6.3 **[test L1]** `index_test.exs:294-302`: deterministic HNSW data (`for i <- 0..99, do: List.duplicate(i * 1.0, 8)`).
- [x] 6.4 **[test L2]** `concurrency_test.exs`: `@moduletag timeout: 180_000`.
- [x] 6.5 **[test L3]** Generate per-worker op lists in the test process (seeded by `--seed`), then pass them into the Tasks (`concurrency_test.exs:23`).
- [x] 6.6 **[test L4]** Start barrier in `concurrency_smoke_test.exs:18-33` (`receive :go`).
- [x] 6.7 **[test L5]** Delete `index_test.exs:365-374` (it calls a private NIF; `:683-689` already covers it).
- [x] 6.8 **[test L6]**
  - Delete `clustering_test.exs:30-46`.
  - Strengthen `:13-27` to assert that the sorted centroids are ≈ `[[0,0,0,1],[1,0,0,0]]` (delta).
  - `index_test.exs:64-80` → `assert [[0], [2]] = labels`.

**Verify:** `mix test --include slow` passes 5× with different seeds (`for s in 1 2 3 4 5; do mix test --include slow --seed $s; done`); `mix test --cover` ≥ 90%. — ✅ 131 passed × seeds 1–5 (also `--warnings-as-errors`); coverage 100%

## Phase 7 — Release v0.4.0 [docs]

- [x] 7.1 CHANGELOG `0.4.0`: — also lists the VM-abort fix, num_gpus fix, INT64_MAX key, new_index dirty, CI floor/CMake/CUDA checks
  - **Changed (behavior):**
    - `add_with_ids` raises on a count mismatch.
    - Options are validated (typos raise).
    - `Index.train(idx, [])` errors for indexes that need training.
    - `Clustering.train/2` no longer mutates its input.
    - `FAISS_OPT_LEVEL` defaults to `dd` (a FAISS rebuild on upgrade).
    - Locked NIFs reschedule to dirty IO under contention.
  - **Added:**
    - Binary API for Clustering and residuals.
    - `set_omp_threads/1`.
    - Description length cap.
  - **Fixed:**
    - Starvation.
    - `FAISS_OPT_LEVEL` override no-op.
    - Stale CUDA build.
    - Negative device abort.
    - Alignment UB.
  - **Security:**
    - `from_file` trust docs.
    - CI hardening.
- [x] 7.2 `@version "0.4.0"` (`mix.exs:4`); README version references.
- [x] 7.3 CLAUDE.md final pass covering 1.1, 3.4, and 4.1–4.3 (locking, dirty-scheduler list, OpenMP, build keying, single-source pin). — plus OpenMP-region exception rule, verify-faiss, cuda-syntax, stacked-PR CI notes
- [x] 7.4 Re-run the `/phx:audit --quick` gate; optionally a full `/phx:audit` to record the trend. — quick gate clean (compile, hex.audit, xref stats, 127 tests); `mix deps.audit` unavailable (mix_audit not a dep); full audit not run

**Verify:** `mix hex.build` succeeds; `mix docs` is clean; CI green on all rows. — hex.build ✅ (0.4.0), docs ✅; CI: PR #15

---

## Appendix — Test additions (5.11)

Each row is one test, asserting the exact message or a substring. All are consumer-visible error and contract paths.

| Area | Test | Finding |
|---|---|---|
| new | `Index.new(4, "NotAnIndex")` → `{:error, msg}` with a FAISS message | test M3, inv #4, prior #7 |
| new | `Index.new(4, ~c"Flat")` → `{:error, "invalid arguments"}` | inv #1 |
| new | 257-byte description → `{:error, "description too long"}` | sec N2 (3.2) |
| new | `Index.new(4, "Flat", metrc: :l2)` raises `ArgumentError` | arch L5 |
| new | `device: :gpu` raises `ArgumentError`; `device: {:cuda, -1}` raises `FunctionClauseError`/`ArgumentError` | arch L5, sec N1 |
| add_with_ids | on plain `"Flat"` → `{:error, msg}` | test M3, inv #8, prior #8 |
| add_with_ids | count mismatch (incl. `[]` vs `[1]`) raises `ArgumentError` | arch L3, inv #6 |
| search | `k = 0` → "k must be positive" on `search` and `search_binary` | test M2, inv #10 |
| search | `k = 1.5` → "invalid arguments"; `k = 2**62` → `{:error, _}` | inv #9, #11, #12 |
| search | wrong dim on `search`/`search_binary` raises; `search(idx, [], k)` → empty rows | prior #11, #12 |
| search | binary input to `search/3` returns lists | 5.2 |
| train | IVF4 with 2 vectors → `{:error, msg}`; IVF `train([])` → error; Flat `train([])` → `:ok`; wrong dim raises | test M1, M3, inv #14, prior #11–12 |
| reconstruct | IVF without a direct map → `{:error, msg}`; `reconstruct(idx, [])` → `{:ok, []}` | test M3, prior #9, #12 |
| residuals | key ≥ ntotal → `{:error, _}`; successful n=3 batch equals the manual `xs - reconstruct`; wrong dim raises; binary variant matches the list | inv #19, prior #10–11, 5.2 |
| file I/O | `to_file(idx, ~c"/x")` → "invalid arguments"; `from_file(p, io_flags: :x)` → "invalid arguments" (or `ArgumentError` after 5.4); corrupt file → `{:error, msg}`; `io_flags: 2` round-trip; message substrings at `:328,332` | inv #20, #23, test L7, prior #13 |
| clustering | `Clustering.new(4.0, 2)` → "invalid arguments"; `train(c, [])` → "n must be positive" with no quantizer allocated; old struct untouched after train (D3); two trains from one struct are independent; binary train and `get_centroids_binary` | inv #28, #31, arch M2 |
| gpu (`:no_cuda`) | `Index.new(4, "Flat", device: {:cuda, 0})` and `cpu_to_gpu(index)` → "GPU support not compiled"; a forged `%Index{device: {:cuda, 0}}` passed to `gpu_to_cpu` → error; `FaissEx.num_gpus() == 0` | test M4, inv #27, prior #15 |
| alignment | `binary_part(<<0, payload::binary>>, 1, size)` given to `add`, `search_binary`, `reconstruct_binary` gives the same results as the aligned binary | sec N5 |

## Finding → task coverage

| Source | Findings → tasks |
|---|---|
| deps | N1→1.2/1.3/2.2 · N2→0.1 · N3→1.8/1.11 · N4→2.1 · N5→2.7 · N6→1.5/2.8 · N7→2.4/2.6 · N8→1.1/1.7/1.11 · Hex lag→1.10 · credo→2.11 |
| security | P1-res→1.5 · P2→1.6 · P3→5.9/tests · N1→3.1/5.4 · N2→3.2/5.6 · N3→5.5 {D4} · N4→2.4 · N5→3.5/tests · N6→2.5/2.8 |
| perf | P5→4.2 · P6→**deferred** · P8→3.3 · P9/N3→1.4 · N1→4.1/4.2/4.7 {D1} · N2→4.3/4.4 {D2} · N4→3.2 · N5→5.2 · N6→**deferred** · N7→3.4 · N8→4.5 · (P4 WONTFIX by measurement, no task) |
| arch | M1 false positive, no task · M2→5.1 {D3}/6.1 · M3→1.2/1.3 · L1→1.1 · L2→5.2 · L3→5.3 · L4→**deferred** · L5→5.4 · L6→3.4/3.7 · L7→3.3 · L8→3.6 · L9→5.7 |
| tests | H1→6.1 · H2→6.2/4.7 · M1→3.7/5.8/tests · M2→tests · M3→tests · M4→2.9/tests + GPU-runner suite **deferred** · M5→2.2 · M6→2.7 · L1→6.3 · L2→6.4 · L3→6.5 · L4→6.6 · L5→6.7 · L6→6.8 · L7→tests · L8→1.9/2.10 · L9→2.5 · L10→2.3 · prior #7–#15 + inventory ✗ rows→appendix |

## Risks

1. **`dd` on macOS arm64 is untested by this project.** Mitigation: Phase 1 verify does a clean build and runs the full suite on this machine, and CI covers Linux x86/arm and macOS Intel/arm. If `dd` fails on any target, fall back to `generic` on that arch in the Makefile and record why.
2. **Two OpenMP runtimes in one process** (the NIF linking a different libomp than libfaiss) cause hangs or crashes. Mitigation: 4.3 links against the exact `LIBOMP_PREFIX` FAISS was configured with, and the Phase 4 verify checks `otool -L`/`ldd`. A system `FAISS_PREFIX` built with Intel `iomp5` is unsupported for `set_omp_threads`; detect it, make the knob a no-op, and document that.
3. **D1 moves waiting to the dirty-IO pool**, which also serves file I/O (`prim_file`). Under heavy write contention, file I/O can queue. Mitigation: document `+SDio`. The regression test only proves the CPU pool is protected. Option (b) was rejected for its writer-starvation and leak hazards (see the scratchpad).
4. **Behavior changes in a 0.x minor** (add_with_ids raises, option validation, train n=0). Mitigation: list them in CHANGELOG "Changed". The Elixir guards raise with explicit messages.
5. **Rescheduling must not leak the pre-allocated result binary.** Every reschedule path calls `enif_release_binary` first. Review it per NIF; ASan is not practical in-BEAM (see the scratchpad).
6. **The CI cache-key change triggers ~10 min FAISS rebuilds** on every Makefile edit. Accepted, because correctness beats cache hits (test M5).

## Execution notes

- Order: Phase 0 (today) → 1 → 2 → 3 → 4 → 5 → 6 → 7. Phases 3 and 5 are independent of each other but both precede 4 (4.1 touches every locked NIF after 3.4/3.6 reshape them). 6 can run alongside 5.
- One PR per phase. Phase 0 is a standalone hotfix commit to master.
- macOS cannot compile the GPU block locally. The 2.9 `cuda-syntax` job is the only check for 3.1/4.1 edits inside `#ifdef FAISS_GPU_ENABLED`, so land 2.9 before Phase 3.
