# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Elixir NIF bindings for FAISS (Facebook AI Similarity Search) via the C API (`libfaiss_c`). No external dependencies beyond `elixir_make`. Supports vector similarity search (L2, inner product), index factory, k-means clustering, and optional CUDA GPU support.

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
- `lib/faiss_ex/index.ex` — Index struct and all index operations (new, add, search, train, clone, reset, reconstruct, file I/O, GPU transfer). Accepts flat lists (single vector) or lists of lists (batch).
- `lib/faiss_ex/clustering.ex` — K-means clustering (new, train, get_centroids, get_cluster_assignment). Creates a temporary Flat index as quantizer during training.
- `lib/faiss_ex/nif.ex` — Raw NIF function stubs. All functions prefixed `nif_`. Returns `{:ok, result}`, `:ok`, or `{:error, binary_message}`.
- `lib/faiss_ex/shared.ex` — Binary conversion helpers (`floats_to_binary`, `binary_to_floats`, `int64s_to_binary`, `binary_to_int64s`) and `unwrap!/1`.
- `c_src/faiss_ex_nif.c` — Single C file with all NIF implementations. Uses NIF resource types with destructors for index/clustering/GPU resource lifecycle.

**NIF resource types:** `FaissIndex`, `FaissClustering`, `FaissGpuResources` (when `FAISS_GPU_ENABLED`). BEAM GC handles cleanup via destructors.

**Dirty schedulers:** add, add_with_ids, search, train → `DIRTY_JOB_CPU_BOUND`; write/read index, GPU transfer → `DIRTY_JOB_IO_BOUND`.

**Data flow:** Elixir converts lists to f32/s64 binaries via `Shared` helpers → passed to NIF → NIF returns raw binaries → Elixir decodes back to lists.

## Test Tags

- `@tag :cuda` — GPU tests, excluded unless `USE_CUDA=true`
- `@tag :slow` — always excluded by default
- `@tag :cuda` goes before individual `test`, NOT before `describe`; use `@describetag` inside a `describe` block
