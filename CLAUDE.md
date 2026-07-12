# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Elixir NIF bindings for FAISS (Facebook AI Similarity Search) via the C API (`libfaiss_c`). No external dependencies beyond `elixir_make`. Supports vector similarity search (L2, inner product), index factory, k-means clustering, and optional CUDA GPU support (CUDA 12.x required). FAISS is pinned at v1.14.3 via `FAISS_GIT_REV` — defined in BOTH `Makefile` and `mix.exs` (`make_env/0`); keep them in sync.

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
- `Mix.Project.app_path()` cannot be called in `make_env/0` — `elixir_make` sets `MIX_APP_PATH` automatically
- FAISS is cloned to `~/.cache/faiss_ex/faiss-<rev>/` and built in-tree (`build/` subdir)
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
- `lib/faiss_ex/shared.ex` — `encode_vectors!/2` (single-pass list→f32 binary with per-row dim validation, raises ArgumentError; a flat-accumulator rewrite measured 2.4× slower — keep the per-row iodata design), `encode_ids!/1` (single-pass id list→{n, s64 binary}), single-pass row decoders (`binary_to_float_rows/2`, `binary_to_int64_rows/2`) plus flat decode helpers (`binary_to_floats`, `int64s_to_binary`, `binary_to_int64s`).
- `c_src/faiss_ex_nif.c` — Single C file with all NIF implementations. Uses NIF resource types with destructors for index/clustering/GPU resource lifecycle.

**NIF resource types:** `FaissIndex`, `FaissClustering`, `FaissGpuResources` (when `FAISS_GPU_ENABLED`). BEAM GC handles cleanup via destructors.

**Locking convention (C layer):** every index/clustering resource carries an `ErlNifRWLock`. Mutations (add/add_with_ids/train/reset/train_clustering) take the write lock; reads (search/reconstruct/residuals/write_index/clone/GPU transfer/centroids/ntotal/is_trained getters) take the read lock. The dim getter is lock-free: `IndexResource` caches the immutable `Index::d` at wrap time (`res->dim`), and all size checks use the cache. `nif_train_clustering` locks two resources — fixed order: quantizer index first, then clustering. Any NIF that can block on a lock must run on a dirty scheduler.

**Dirty schedulers:** add, add_with_ids, search, train, reset, reconstruct, residuals, clone, centroids, and the ntotal/is_trained getters → `DIRTY_JOB_CPU_BOUND`; write/read index, GPU transfer → `DIRTY_JOB_IO_BOUND`. `nif_new_index`, `nif_new_clustering`, `nif_get_num_gpus`, and `nif_get_index_dim` (immutable cached field, no lock) stay on normal schedulers.

**Data flow:** Elixir encodes lists to f32/s64 binaries via `Shared` → passed to NIF → NIF returns raw binaries → Elixir decodes back to lists. The Binary API skips both conversions: binary input passes through after a byte-size divisibility check; `*_binary` functions return the NIF binaries as-is.

## Test Tags

- `@tag :cuda` — GPU tests, excluded unless `USE_CUDA=true`
- `@tag :no_cuda` — asserts non-CUDA-build error paths, excluded when `USE_CUDA=true`
- `@tag :slow` — always excluded by default (`mix test --include slow` to run)
- `@tag :cuda` goes before individual `test`, NOT before `describe`; use `@describetag` inside a `describe` block
