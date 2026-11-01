# Scratchpad — audit-remediation

## Verified facts (2026-11-01, during planning)

- **macOS 14 runner retirement 2026-11-02**: [GitHub changelog 2026-10-01](https://github.blog/changelog/2026-10-01-github-actions-macos-14-runner-image-retirement/).
- **`FAISS_OPT_LEVEL=dd`** builds runtime SIMD dispatch into the main `faiss` target (`faiss/CMakeLists.txt:541-548`).
  - x86_64 per-file AVX2/AVX512 flags: `:550-581`. aarch64 branch: `:603`.
  - No separate install for dd (`:813-814`).
  - `avx2`/`avx512*` use the separate `faiss_avx2`/`faiss_c_avx2` targets, which are `EXCLUDE_FROM_ALL` unless selected (`faiss/CMakeLists.txt:454-495`, `c_api/CMakeLists.txt:44-83`).
  - Our Makefile builds only `faiss`/`faiss_c` (`Makefile:183,188`), so `avx2` is a silent no-op. Confirmed.
- **FAISS IO flags** (`faiss/index_io.h`): `IO_FLAG_READ_ONLY=2`, `IO_FLAG_ONDISK_SAME_DIR=4`, `IO_FLAG_SKIP_IVF_DATA=8`, `IO_FLAG_MMAP = 8 | 0x646f0000`.
  - `READ_ONLY` alone still lets a hostile file name an arbitrary path to mmap read-only (an info leak).
  - `SAME_DIR` strips the directory component, so we recommend `6`.
- **GPU C API headers** include `cuda_runtime_api.h`/`cublas_v2.h` (`c_api/gpu/StandardGpuResources_c.h:13`, `GpuResources_c.h:13-14`, `DeviceUtils_c.h:13-14`). A GPU syntax check therefore needs CUDA headers, hence the container job (2.9).
- **The FAISS C API exposes no OpenMP thread control** (grep of `c_api/` for omp: none relevant). The NIF must `#include <omp.h>` and link the runtime itself.
- **GPU resources lifetime is sound.** `GpuIndex.h:289` holds `std::shared_ptr<GpuResources>`, and `StandardGpuResources.h:265` wraps a `shared_ptr` to the impl. Arch M1 was a false positive: no task.

## Decision rationale

- **D1 (IO reschedule) over (b) trylock + Elixir backoff:**
  - (b) never blocks a thread, but plain trylock polling lets a stream of overlapping readers starve a writer indefinitely.
  - Fixing that needs a writer-pending counter. If the writer process is killed during its backoff `receive`, nothing decrements the counter and readers are locked out forever.
  - Making it leak-proof needs a token resource whose destructor decrements it. That is too much new invariant surface for a remediation release.
  - (a) keeps pthread rwlock fairness and is about 1 helper. Its cost (waiting on the IO pool) is documented and tunable with `+SDio`.
- **D2 global knob over per-call:** the measured tradeoff is workload-level (many small concurrent queries vs a few large batches), not call-level. A per-call option can be added later without breaking the global one.
- **D3 value model:**
  - It matches the struct-returning API that callers already use (`{:ok, trained} = train(c, data)`), costs one extra `faiss_Clustering_new` per train (trivial), and removes the shared-mutable state entirely.
  - Side effect: the two-lock `nif_train_clustering` path can no longer be raced through the public API. The C locks stay as defense in depth.
- **D4 document over changing the default:** a changed default silently breaks writable on-disk IVF users. Documentation plus a recommended flag value is the conservative choice; flip it if the user prefers safe-by-default.
- **3.7 (train n=0) "let FAISS decide"** over unifying on error or on `:ok`:
  - Removing the NIF short-circuit makes Flat (no training needed) a no-op, and makes IVF fail with FAISS's own message.
  - That is the honest contract. It must be verified with a probe before relying on it: FAISS `Clustering::train_encoded` throws when `n < k`.

## Dead ends / rejected

- **`dlsym(RTLD_DEFAULT, "omp_set_num_threads")`** to bind whatever runtime libfaiss loaded, with no link flags. Rejected: ERTS `dlopen`s NIFs `RTLD_LOCAL`, so libfaiss's dependencies are not in the global scope on Linux, and the lookup is unreliable. Explicit linking against the same `LIBOMP_PREFIX`, plus an `otool`/`ldd` check, is boring and verifiable.
- **Alignment checks in Elixir:** impossible, because Elixir cannot observe binary pointer alignment. The check lives in C (3.5).
- **ASan/UBSan fuzzing (security recommendation):** a NIF loaded into BEAM needs `DYLD_INSERT_LIBRARIES`/`LD_PRELOAD` of the sanitizer runtime into `beam.smp`. Painful on macOS. Not a finding, so not a task. Consider a Linux-only CI experiment later.
- **Wall-clock regression test for starvation:** flaky. 4.7 asserts ordering instead (B's search returns while A's train Task is alive).

## Open items to record during /phx:work

- 1.4: whether `faiss_c_avx2` links `libfaiss_avx2` under an install name we can rewrite, which decides between supporting the avx2/avx512 overrides and `$(error)`.
- 2.11: credo kept or removed, and why.
- 3.7: the FAISS probe result for IVF/Flat/HNSW `train` with n=0.

## Work log

- **1.1 deviation: `make_env/0` deleted outright** instead of "forward only the env vars that are set". `System.cmd` (elixir_make) inherits the parent environment, so forwarding set vars is a no-op; the Makefile `?=` defaults are now the only defaults.
- **1.4 resolved → `$(error)` branch.** `c_api/CMakeLists.txt:44-45`: `faiss_c_avx2` links `faiss_avx2` PRIVATE, so `libfaiss_c_avx2` records `libfaiss_avx2` as its dependency. Renaming the pair to `libfaiss`/`libfaiss_c` would need a dependency rewrite (`install_name_tool -change` on macOS, `patchelf` on Linux, not standard). `dd` already dispatches AVX2/AVX-512/SVE at runtime (`faiss/CMakeLists.txt:544-610`), so only `dd`/`generic` are accepted.
- **1.7/1.8 placement:** the nvcc and cmake version checks run inside the configure recipe (Step 2), not as parse-time `$(error)`. They fail the same way but cost nothing on warm builds.
- **Local GNU Make is 3.81** (macOS): Makefile avoids `$(file)`; stamp uses `printf` + `cmp` + `mv`.
- **Stacked PRs had no CI:** `pull_request: branches: [master]` filters on the *base* branch, so #9 (base = #8's branch) got no checks. Phase 2 drops the filter. #9 gets CI once #8 merges and GitHub retargets it to master.
- **2.7: `setup` job removed**, test job self-contained (the plan allowed either). hex.audit moved into a `Lint` job (format + hex.audit + credo) instead of `setup`. Timeouts: lint 15, test/test-native 60 (not 30: a cold FAISS dd build runs inside the job now), cuda-syntax 30, publish 90.
- **2.11: credo kept.** One finding (alias order in clustering.ex), fixed. credo needs only `loadpaths`, so it never triggers a FAISS build in CI.
- **2.9 found a real bug:** `nif_get_num_gpus` called `faiss_get_num_gpus()` without including `c_api/gpu/DeviceUtils_c.h` and without its `int*` out-param (`int faiss_get_num_gpus(int* p_output)`), so it returned the status code (0) as the GPU count. GCC 14 rejects the implicit declaration. Fixed in the Phase 2 commit. Typo test: the job fails (exit 1) on a deliberate typo in the GPU block.
- zizmor flagged `artipacked` (×5) and `dependabot-cooldown`; fixed with `persist-credentials: false` and `cooldown.default-days: 7`.
- **3.7 probe (resolved):** train n=0 → Flat/HNSW32/IDMap,Flat `:ok`; IVF4,Flat and PQ2 `{:error, "...Number of training points (0) should be at least as large as number of clusters"}`.
- **4.x measurements:** OpenMP effect needs a large index — 10k×128 Flat shows none (single query not split); 100k×128: 10×200 concurrent single-query 1529/1635 ms → 586/557 ms with 1 thread; 1000-query batch 78/88 → 123/126 ms. Starvation bench after fix: 0.036 ms contended vs 0.025 ms. `System.schedulers_dirty_cpu/0` doesn't exist → `:erlang.system_info(:dirty_cpu_schedulers_online)`.
- **NEW CRITICAL (found writing 5.11 tests, not in the audit): FAISS exceptions inside OpenMP regions abort the VM.** `Index::compute_residual_n` (`omp parallel for`) and base `Index::reconstruct_n` (`omp parallel for if (ni > 1000)` — aborts even when serialized by the if-clause) call `reconstruct`, which throws for key ≥ ntotal or index types without reconstruct; the exception cannot reach the C API's catch → `std::terminate`. Repros: `compute_residuals(flat, [v], [5])`; any reconstruct on `IDMap,Flat`; HNSW contiguous reconstruct past ntotal (exit 134). Fix: residuals loop `faiss_Index_compute_residual`; reconstruct uses `reconstruct_n` only for Flat/IVF (`faiss_IndexFlat_cast`/`faiss_IndexIVF_cast`; their overrides range-check before any parallel work) after an under-lock ntotal check, else per key. INT64_MAX keys rejected (IndexFlatCodes::reconstruct = reconstruct_n(key,1) → key+1 overflow). Dead end: chunking reconstruct_n to ≤1000 keys — serialized regions still terminate.
- **5.11 deviations:** "IVF without a direct map" errors only on non-contiguous keys (contiguous takes IVF's reconstruct_n override, which needs no direct map) — test asserts both. 6.1/6.3/6.7/6.8 committed with Phase 5 (same files); PR #14 holds the concurrency files only.
