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

## Architecture

```
Elixir (FaissEx.Index / FaissEx.Clustering)
  → NIF stubs (FaissEx.NIF)
    → C (c_src/faiss_ex_nif.c)
      → libfaiss_c → libfaiss (C++)
```

**Key modules:**
- `lib/faiss_ex/index.ex` — Index struct and all index operations (new, add, search, train, clone, reset, reconstruct, file I/O, GPU transfer). Accepts flat lists (single vector), lists of lists (batch), or raw f32-native binaries (Binary API: add/train/add_with_ids take binary input; `search_binary/3` and `reconstruct_binary/2` return raw binaries).
- `lib/faiss_ex/clustering.ex` — K-means clustering (new, train, get_centroids, get_cluster_assignment). Training creates a Flat quantizer index that FAISS populates with the centroids; `get_cluster_assignment/2` searches that same quantizer (do not rebuild it).
- `lib/faiss_ex/nif.ex` — Raw NIF function stubs. All functions prefixed `nif_`. Returns `{:ok, result}`, `:ok`, or `{:error, binary_message}`.
- `lib/faiss_ex/shared.ex` — `encode_vectors!/2` (single-pass list→f32 binary with per-row dim validation, raises ArgumentError; a flat-accumulator rewrite measured 2.4× slower — keep the per-row iodata design), `encode_ids!/1` (single-pass id list→{n, s64 binary}), single-pass row decoders (`binary_to_float_rows/2`, `binary_to_int64_rows/2`) plus the flat decode helpers they build on (`binary_to_floats`, `binary_to_int64s`). Encode-side comprehension helpers (`floats_to_binary`, `int64s_to_binary`) were test-only and now live as private helpers in the test files — do not re-add them here.
- `c_src/faiss_ex_nif.c` — Single C file with all NIF implementations. Uses NIF resource types with destructors for index/clustering/GPU resource lifecycle.

**NIF resource types:** `FaissIndex`, `FaissClustering`, `FaissGpuResources` (when `FAISS_GPU_ENABLED`). BEAM GC handles cleanup via destructors.

**Locking convention (C layer):** every index/clustering resource carries an `ErlNifRWLock`. Mutations (add/add_with_ids/train/reset/train_clustering) take the write lock; reads (search/reconstruct/residuals/write_index/clone/GPU transfer/centroids/ntotal/is_trained getters) take the read lock. Immutable shapes are cached at creation and read lock-free: `IndexResource.dim` (`Index::d`) and `ClusteringResource.d`/`.k`; all size checks and result sizing use the cache. `nif_train_clustering` locks two resources — fixed order: quantizer index first, then clustering — and checks `idx_res->dim == clust_res->d` ("quantizer dim mismatch") before locking. Any NIF that can block on a lock must run on a dirty scheduler.

**Lock unwinding (C layer):** validate before acquiring the lock wherever the inputs allow. Size checks (`check_rows`, which keeps each NIF's existing mismatch message byte-identical), input alignment (`aligned_input`), and the result `enif_alloc_binary` all belong *above* the lock — the critical section then holds nothing but the `faiss_*` call and its one unlock; `free_owned` releases alignment copies after the unlock. The one place validation must read locked state is the centroids pointer in `nif_get_clustering_centroids`: its critical section records only a `const char *` status and the copy, unlocks once, and builds every `ERL_NIF_TERM` after the release — a lock must never cover term construction or error formatting. If a future NIF needs several locked checks, give it a single `unlock:` label that every branch reaches by `goto`, releasing locks in the mirror of the acquisition order. Never hand-write an unlock per error branch: a missed one wedges the resource forever and neither the build nor the test suite can catch it.

**Input alignment (C layer):** Elixir sub-binaries (`binary_part/3`, match tails) can start at any byte offset; every `float*`/`idx_t*` input goes through `aligned_input`, which passes aligned data through and copies misaligned data. Empty inputs get a valid dummy pointer, so FAISS never sees NULL (e.g. `train` with n = 0, which FAISS decides: no-op for Flat/HNSW, error for IVF/PQ).

**Dirty schedulers:** new_index, add, add_with_ids, search, train, reset, reconstruct, residuals, clone, centroids, and the ntotal/is_trained getters → `DIRTY_JOB_CPU_BOUND`; write/read index, GPU transfer, `nif_get_num_gpus` (CUDA runtime init) → `DIRTY_JOB_IO_BOUND`. Only `nif_new_clustering` and `nif_get_index_dim` (immutable cached field, no lock) stay on normal schedulers.

**Data flow:** Elixir encodes lists to f32/s64 binaries via `Shared` → passed to NIF → NIF returns raw binaries → Elixir decodes back to lists. The Binary API skips both conversions: binary input passes through after a byte-size divisibility check; `*_binary` functions return the NIF binaries as-is.

## Test Tags

- `@tag :cuda` — GPU tests, excluded unless `USE_CUDA=true`
- `@tag :no_cuda` — asserts non-CUDA-build error paths, excluded when `USE_CUDA=true`
- `@tag :slow` — always excluded by default (`mix test --include slow` to run)
- `@tag :cuda` goes before individual `test`, NOT before `describe`; use `@describetag` inside a `describe` block
