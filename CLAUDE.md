# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Elixir NIF bindings for FAISS (Facebook AI Similarity Search) via the C API (`libfaiss_c`). No external dependencies beyond `elixir_make`. Supports vector similarity search (L2, inner product), index factory, k-means clustering, and optional CUDA GPU support (CUDA 12.x required). FAISS is pinned at v1.15.1 via `FAISS_DEFAULT_REV`, with `FAISS_DEFAULT_SHA` holding the commit the tag must resolve to (verified post-clone and on every `make`; tags are mutable refs). The `Makefile` is the single source of every build default — `mix.exs` has no `make_env`; make inherits the caller's environment, and CI derives its cache keys from `FAISS_DEFAULT_REV`.

## Build & Test Commands

```bash
mix compile          # First build clones + compiles FAISS from source (~5-15 min), cached at ~/.cache/faiss_ex/
mix test             # Run all tests (excludes :cuda and :slow by default)
mix test test/faiss_ex/index_test.exs       # Run a single test file
mix test test/faiss_ex/index_test.exs:48    # Run a single test by line number
USE_CUDA=true mix test --include cuda       # Run with GPU tests
mix format           # Format code
```

## Build Pitfalls

- FAISS C API headers are at `c_api/` (NOT `faiss/c_api/`) in the source tree
- The Makefile uses `NIF_CFLAGS`/`NIF_LDFLAGS` (not `CFLAGS`/`LDFLAGS`) to avoid leaking env vars into FAISS's cmake build. The cmake/make calls are wrapped in `env -u CFLAGS -u LDFLAGS -u CXXFLAGS`.
- macOS Apple Silicon requires OpenMP hints for cmake (`-DOpenMP_ROOT=/opt/homebrew/opt/libomp`)
- macOS rpath: `install_name_tool -id @rpath/libfaiss.dylib` (no `lib/` prefix — avoids double lib/ in path)
- `elixir_make` sets `MIX_APP_PATH` automatically; do not reintroduce a `make_env/0` (build defaults live only in the Makefile)
- FAISS is cloned to `~/.cache/faiss_ex/faiss-<rev>/` and built in-tree in one directory per configuration: `build-<FAISS_OPT_LEVEL>-<cpu|gpu>/`
- `$(PRIV_DIR)/.build_config` is a config stamp (build dir + NIF flags), rewritten only when it changes; the NIF and both `priv/lib` copies depend on it so switching configurations re-copies and re-links
- `FAISS_OPT_LEVEL` accepts only `dd` (default, runtime SIMD dispatch) and `generic`; FAISS's `avx2`/`avx512`/`sve` levels build differently named libs (`libfaiss_c_avx2` links `libfaiss_avx2`) and are rejected with `$(error)`
- The Makefile must stay GNU Make 3.81-compatible (macOS ships 3.81): no `$(file ...)`, no `.ONESHELL`
- `FAISS_PREFIX` env var: when set, uses system-installed FAISS instead of building from source
- The pinned commit is re-checked on every `make` by the phony `verify-faiss` target (order-only prerequisite of the configure step and the NIF), so a tampered warm cache fails closed. `FAISS_GIT_REV`/`FAISS_GIT_REPO` are allow-listed with pure-make `strip_chars` (never pass them through `$(shell)` unvalidated); recipe paths are quoted
- CMake ≥ 3.24 and (with `USE_CUDA=true`) nvcc ≥ 12 are checked inside the configure recipe
- The macOS dev box has no CUDA: code inside `#ifdef FAISS_GPU_ENABLED` is compiled only by CI's `cuda-syntax` job (CUDA 12.6 container, `-fsyntax-only -Werror`). Reproduce locally with `docker run` on that image (see `.github/workflows/ci.yml`). Real GPU runs: DGX Spark at `10.0.20.26` (GB10, aarch64, CUDA 13.0) — build with `USE_CUDA=true CUDAARCHS=121`, OpenBLAS from `~/opt/openblas` via `CMAKE_PREFIX_PATH`
- `USE_CUDA=true` adds `-I$(CUDA_HOME)/include` (default `/usr/local/cuda`) to the NIF compile — FAISS's GPU C API headers include `cuda_runtime_api.h`; without it no CUDA build compiled (fixed in v0.4.1)
- Linux NIF link uses `-Wl,--no-as-needed` before `-lfaiss_c -lfaiss`: `libfaiss_c`'s RUNPATH points into the FAISS build dir in `~/.cache`, so without a direct `libfaiss.so` dependency (dropped by Ubuntu's default `--as-needed`) a release copied without the cache cannot load the NIF. CI's "NIF loads without the FAISS build cache" step guards this on every platform
- `pull_request` CI has no base-branch filter so stacked PRs get checks; job names are not required checks (ruleset disabled), so renaming jobs is safe

## Architecture

```
Elixir (FaissEx.Index / FaissEx.Clustering)
  → NIF stubs (FaissEx.NIF)
    → C (c_src/faiss_ex_nif.c)
      → libfaiss_c → libfaiss (C++)
```

**Key modules:**
- `lib/faiss_ex/index.ex` — Index struct and all index operations (new, add, search, train, clone, reset, reconstruct, residuals, file I/O, GPU transfer). Accepts flat lists (single vector), lists of lists (batch), or raw f32-native binaries (Binary API: add/add_with_ids/train/search/compute_residuals take binary input; `search_binary/3`, `reconstruct_binary/2`, `compute_residuals_binary/3` return raw binaries). Options are validated with `Keyword.validate!`; caller mistakes the library can detect (bad option/metric/device, id/key count mismatch, wrong dims) raise `ArgumentError`, FAISS-side failures return `{:error, msg}`.
- `lib/faiss_ex/clustering.ex` — K-means clustering (new, train, get_centroids, get_centroids_binary, get_cluster_assignment). **Value semantics:** `train/2` allocates a fresh clustering resource *and* a fresh Flat quantizer per call and never mutates its input struct — do not reuse the input `ref`. FAISS populates the quantizer with the centroids; `get_cluster_assignment/2` searches that same quantizer (do not rebuild it).
- `lib/faiss_ex/nif.ex` — Raw NIF function stubs. All functions prefixed `nif_`. Returns `{:ok, result}`, `:ok`, or `{:error, binary_message}`. `@on_load` passes the validated `:omp_threads` app env as `load_info`.
- `lib/faiss_ex/shared.ex` — `encode_vectors!/2` (single-pass list→f32 binary with per-row dim validation, raises ArgumentError; a flat-accumulator rewrite measured 2.4× slower — keep the per-row iodata design), `encode_ids!/1` (single-pass id list→{n, s64 binary}), `encode_vector_data!/2` / `encode_id_data!/1` (list-or-binary front doors used by Index and Clustering; binaries pass through after a byte-size check), single-pass row decoders (`binary_to_float_rows/2`, `binary_to_int64_rows/2`) plus the flat decode helpers they build on (`binary_to_floats`, `binary_to_int64s`). Encode-side comprehension helpers (`floats_to_binary`, `int64s_to_binary`) were test-only and now live as private helpers in the test files — do not re-add them here.
- `c_src/faiss_ex_nif.c` — Single C file with all NIF implementations. Uses NIF resource types with destructors for index/clustering/GPU resource lifecycle.

**No FAISS call may throw inside an OpenMP region (C layer):** an exception cannot cross an OpenMP parallel region (even one serialized by an `if` clause) to the C API's try/catch — the runtime calls `std::terminate` and the whole VM aborts. `Index::compute_residual_n` and the base `Index::reconstruct_n` call `reconstruct` inside such regions, so the NIF never calls `faiss_Index_compute_residual_n` (it loops `faiss_Index_compute_residual`) and uses `faiss_Index_reconstruct_n` only for Flat/IVF (`has_safe_reconstruct_n`, whose overrides range-check up front) after an under-lock `ntotal` check; everything else reconstructs per key. Regression tests: `index_test.exs` "reconstruction errors never abort the VM". Before calling any other batched FAISS entry point, check its implementation for `#pragma omp` around throwing code.

**Batched reconstruct may skip rows (C layer):** `IndexIVF::reconstruct_n` writes only rows whose *id* lies in the key range, so with sparse `add_with_ids` ids some rows stay unwritten. The NIF tags every row with `UNWRITTEN_ROW_TAG` (a signaling-NaN bit pattern) before the batched call and returns `{:error, "key out of range"}` if any row is still tagged — never return an `enif_alloc_binary` buffer FAISS may not have fully written.

**NIF resource types:** `FaissIndex`, `FaissClustering`, `FaissGpuResources` (when `FAISS_GPU_ENABLED`). BEAM GC handles cleanup via destructors.

**Locking convention (C layer):** every index/clustering resource carries an `ErlNifRWLock`. Mutations (add/add_with_ids/train/reset/train_clustering) take the write lock; reads (search/reconstruct/residuals/write_index/clone/GPU transfer/centroids/ntotal/is_trained getters) take the read lock. Immutable shapes are cached at creation and read lock-free: `IndexResource.dim` (`Index::d`) and `ClusteringResource.d`/`.k`; all size checks and result sizing use the cache. `nif_train_clustering` locks two resources — fixed order: quantizer index first, then clustering — and checks `idx_res->dim == clust_res->d` ("quantizer dim mismatch") before locking. Any NIF that can block on a lock must run on a dirty scheduler.

**Lock unwinding (C layer):** validate before acquiring the lock wherever the inputs allow. Size checks (`check_rows`, which keeps each NIF's existing mismatch message byte-identical), input alignment (`aligned_input`), and the result `enif_alloc_binary` all belong *above* the lock — the critical section then holds nothing but the `faiss_*` call and its one unlock; `free_owned` releases alignment copies after the unlock. The one place validation must read locked state is the centroids pointer in `nif_get_clustering_centroids`: its critical section records only a `const char *` status and the copy, unlocks once, and builds every `ERL_NIF_TERM` after the release — a lock must never cover term construction or error formatting. If a future NIF needs several locked checks, give it a single `unlock:` label that every branch reaches by `goto`, releasing locks in the mirror of the acquisition order. Never hand-write an unlock per error branch: a missed one wedges the resource forever and neither the build nor the test suite can catch it.

**Input alignment (C layer):** Elixir sub-binaries (`binary_part/3`, match tails) can start at any byte offset; every `float*`/`idx_t*` input goes through `aligned_input`, which passes aligned data through and copies misaligned data. Empty inputs get a valid dummy pointer, so FAISS never sees NULL. Exception: `train` with n = 0 never reaches FAISS — PCA/RQ trainers pass it to BLAS (`cblas_ssyrk` LDC = 0) and macOS Accelerate's error handler exits the process; the NIF answers from `is_trained` instead.

**Contention → dirty IO (C layer):** a dirty NIF blocked on a lock still occupies its scheduler thread, so waiting must never happen on the dirty-CPU pool. Every locked compute/mutation NIF is declared with `LOCKED_NIF(name)` over `name##_impl(env, argc, argv, LockMode)`: the registered dirty-CPU entry validates, allocates, then `acquire_read`/`acquire_write` with `LOCK_TRY`; on contention it releases everything it allocated (and the first lock, in `train_clustering`) and returns `RESCHEDULE_BLOCKING(name, ...)` → `enif_schedule_nif` of the `name##_io` entry (dirty IO, `LOCK_BLOCK`), which re-runs the same code and blocks. No lock is ever held across a reschedule. A new locked NIF must follow this pattern; `concurrency_test.exs` has the ordering regression test.

**Dirty schedulers:** new_index plus the `LOCKED_NIF` set — add, add_with_ids, search, train, reset, reconstruct, residuals, clone, centroids, train_clustering — → `DIRTY_JOB_CPU_BOUND` (their `_io` twins are reached only via `enif_schedule_nif`, never registered); write/read index, GPU transfer, the ntotal/is_trained getters (no CPU work, block in place), and `nif_get_num_gpus` (CUDA runtime init) → `DIRTY_JOB_IO_BOUND`. Only `nif_new_clustering`, `nif_get_index_dim` (immutable cached field, no lock), and `nif_set_omp_threads` (atomic store) stay on normal schedulers.

**OpenMP (C layer + Makefile):** `FaissEx.set_omp_threads/1` / `config :faiss_ex, omp_threads: n` (passed as NIF `load_info`, validated in `FaissEx.NIF`) store a count in `g_omp_threads` (C11 atomic). `apply_omp_threads()` must run before every FAISS compute call (search, add, add_with_ids, train, reconstruct, residuals, train_clustering) because `omp_set_num_threads` sets a per-thread ICV and dirty threads are reused; 0 means the `omp_get_max_threads()` captured at load. The NIF links the same OpenMP runtime as libfaiss (macOS: `-Xclang -fopenmp` + `LIBOMP_PREFIX` libomp; Linux: `-fopenmp` → libgomp) — verify with `otool -L`/`ldd` that exactly one runtime appears. A system FAISS linked to `libiomp5` (or macOS without libomp) builds with `-DFAISS_EX_NO_OMP` and the setting is ignored.

**Data flow:** Elixir encodes lists to f32/s64 binaries via `Shared` → passed to NIF → NIF returns raw binaries → Elixir decodes back to lists. The Binary API skips both conversions: binary input passes through after a byte-size divisibility check; `*_binary` functions return the NIF binaries as-is.

## Test Tags

- `@tag :cuda` — GPU tests, excluded unless `USE_CUDA=true`
- `@tag :no_cuda` — asserts non-CUDA-build error paths, excluded when `USE_CUDA=true`
- `@tag :slow` — always excluded by default (`mix test --include slow` to run)
- `@tag :cuda` goes before individual `test`, NOT before `describe`; use `@describetag` inside a `describe` block
