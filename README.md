# FaissEx

Elixir NIF bindings for [FAISS](https://github.com/facebookresearch/faiss) — Facebook's library for efficient similarity search and clustering of dense vectors.

Binds to FAISS via its official C API (`libfaiss_c`). No external dependencies beyond `elixir_make`.

## Features

- Vector similarity search (L2, inner product)
- Index factory support (`"Flat"`, `"IVF4,Flat"`, `"HNSW32"`, etc.)
- Add, search, train, clone, reset, reconstruct
- File I/O (save/load indexes)
- K-means clustering
- GPU support (CUDA)

## Prerequisites

- Erlang/OTP 27+ with NIF support
- Elixir 1.18+
- A C/C++ compiler (clang or gcc)
- **Either** a system FAISS installation **or** CMake 3.17+ to build from source (the default)

There are no precompiled binaries: the first `mix compile` builds FAISS and the
NIF from source as shared libraries, cached at `~/.cache/faiss_ex/` (~5–15 min,
once per FAISS version). Install the platform packages below first.

### macOS setup

```bash
# Compiler (if you don't have Xcode / command line tools yet)
xcode-select --install

# Build tools + OpenMP
brew install cmake libomp
```

- BLAS comes from the built-in Accelerate framework — nothing to install
- The build looks for libomp at `$HOMEBREW_PREFIX/opt/libomp` (defaults to
  `/opt/homebrew` on Apple Silicon); set `HOMEBREW_PREFIX` if yours differs

### Linux setup (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev
```

- `libopenblas-dev` provides the BLAS/LAPACK libraries FAISS links against
- OpenMP (`libgomp`) ships with gcc — nothing extra to install
- This is the exact recipe this project's CI uses (ubuntu-22.04)

### Linux setup (Fedora/RHEL)

```bash
sudo dnf install -y gcc gcc-c++ make cmake openblas-devel
```

### GPU (optional)

- CUDA toolkit **12.x** — FAISS 1.12.0+ dropped CUDA 11 (see the
  [GPU Support](#gpu-support) section)
- Build with `USE_CUDA=true mix compile`

## Installation

Add `faiss_ex` to your dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:faiss_ex, "~> 0.1.0"}
  ]
end
```

Then fetch and compile:

```bash
mix deps.get
mix compile
```

### Option A: Build FAISS from source (default)

By default, the first `mix compile` clones FAISS from GitHub and builds it from source. This takes ~5-15 minutes but requires no pre-installed FAISS. The result is cached at `~/.cache/faiss_ex/` — subsequent builds are fast.

```bash
# Just works — no extra setup needed (besides cmake and a C++ compiler)
mix compile
```

### Option B: Use a system-installed FAISS

If you already have FAISS installed (via a package manager or custom build), point `FAISS_PREFIX` at it to skip building from source entirely:

```bash
# macOS (Homebrew)
brew install faiss
FAISS_PREFIX=$(brew --prefix faiss) mix compile

# Ubuntu/Debian
sudo apt install libfaiss-dev
FAISS_PREFIX=/usr mix compile

# Conda
conda install -c pytorch faiss-cpu
FAISS_PREFIX=$CONDA_PREFIX mix compile

# Custom install location
FAISS_PREFIX=/opt/faiss mix compile
```

`FAISS_PREFIX` should point to a directory containing `include/` (with `c_api/` headers) and `lib/` (with `libfaiss` and `libfaiss_c` shared libraries).

> **Note:** Most system packages ship `libfaiss` but not `libfaiss_c` (the C API wrapper). If you get linker errors about `libfaiss_c`, you may need to build from source — just omit `FAISS_PREFIX` and let FaissEx handle it.

### Build configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FAISS_PREFIX` | *(unset)* | Path to system FAISS install. When set, skips building from source |
| `FAISS_OPT_LEVEL` | `generic` | SIMD optimization level (see below). Only used when building from source |
| `USE_CUDA` | `false` | Set to `true` to enable GPU support |
| `FAISS_GIT_REPO` | `https://github.com/facebookresearch/faiss.git` | FAISS git repository (ignored when `FAISS_PREFIX` is set) |
| `FAISS_GIT_REV` | `v1.14.3` | FAISS version tag or commit (ignored when `FAISS_PREFIX` is set) |

### SIMD optimization

FAISS uses SIMD instructions for fast distance computation. By default it builds with `generic` (portable) code. Set `FAISS_OPT_LEVEL` to enable optimized codepaths for your CPU:

| Value | Platform | Instructions |
|-------|----------|-------------|
| `generic` | Any | Portable, no special instructions |
| `avx2` | x86-64 | AVX2 + FMA (most Intel/AMD since ~2015) |
| `avx512` | x86-64 | AVX-512 (Intel Xeon, AMD Zen 4+) |
| `sve` | aarch64 | SVE (ARM Neoverse V1+) |

```bash
# Build with AVX2 optimizations
FAISS_OPT_LEVEL=avx2 mix compile

# Force a fresh FAISS build after changing opt level
rm -rf ~/.cache/faiss_ex && FAISS_OPT_LEVEL=avx2 mix compile
```

> **Note:** Apple Silicon (M1-M4) uses NEON which is always enabled — `FAISS_OPT_LEVEL` has no effect on ARM macOS.

### Runtime dependencies (releases & Docker)

The compiled artifact in `priv/` contains the NIF plus **shared** `libfaiss` /
`libfaiss_c` libraries, which link against system libraries at runtime. If you
build in one environment and run in another (multi-stage Docker builds, `mix
release` to a different host), the **runtime** image/host also needs:

- **Linux**: `libopenblas0` and `libgomp1` (plus a matching libc)

  ```dockerfile
  # in the runtime stage of your Dockerfile
  RUN apt-get update && apt-get install -y libopenblas0 libgomp1 && rm -rf /var/lib/apt/lists/*
  ```

- **macOS**: `brew install libomp` (Accelerate is part of the OS)

Build and runtime must use the same OS/architecture — the cached FAISS build is
not portable across platforms.

## Usage

All functions accept **plain lists** (single vector) or **lists of lists** (batch):

```elixir
# Single vector
FaissEx.Index.add(index, [1.0, 2.0, 3.0])
# Batch of vectors
FaissEx.Index.add(index, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
```

### Creating an index and searching

```elixir
# Create a flat L2 index with 128 dimensions
{:ok, index} = FaissEx.Index.new(128, "Flat")

# Add vectors
:ok = FaissEx.Index.add(index, [[0.1, 0.2, ...], [0.3, 0.4, ...]])

# Search for 5 nearest neighbors
{:ok, %{distances: distances, labels: labels}} = FaissEx.Index.search(index, query, 5)
# distances: [[float]] — n rows of k distances
# labels: [[integer]] — n rows of k vector indices
```

### Index types

FAISS uses [index factory strings](https://github.com/facebookresearch/faiss/wiki/The-index-factory) to create different index types:

```elixir
# Flat (exact search, no training needed)
{:ok, index} = FaissEx.Index.new(128, "Flat")

# IVF with flat quantizer (needs training)
{:ok, index} = FaissEx.Index.new(128, "IVF256,Flat")

# HNSW graph-based index
{:ok, index} = FaissEx.Index.new(128, "HNSW32")

# Inner product metric instead of L2
{:ok, index} = FaissEx.Index.new(128, "Flat", metric: :inner_product)
```

### Training (IVF, PQ indexes)

Some index types require training before adding vectors:

```elixir
{:ok, index} = FaissEx.Index.new(128, "IVF256,Flat")

# Train on representative data (list of lists)
training_data = for _ <- 1..10_000, do: for(_ <- 1..128, do: :rand.uniform())
:ok = FaissEx.Index.train(index, training_data)

# Now add and search
:ok = FaissEx.Index.add(index, training_data)
{:ok, results} = FaissEx.Index.search(index, query, 10)
```

### Adding vectors with IDs

```elixir
{:ok, index} = FaissEx.Index.new(128, "IDMap,Flat")

vectors = for _ <- 1..100, do: for(_ <- 1..128, do: :rand.uniform())
ids = Enum.to_list(1000..1099)
:ok = FaissEx.Index.add_with_ids(index, vectors, ids)
```

### Saving and loading

```elixir
:ok = FaissEx.Index.to_file(index, "/tmp/my_index.faiss")
{:ok, loaded} = FaissEx.Index.from_file("/tmp/my_index.faiss")
```

### Cloning and resetting

```elixir
{:ok, copy} = FaissEx.Index.clone(index)
:ok = FaissEx.Index.reset(index)  # clear all vectors
```

### Index properties

```elixir
{:ok, dim} = FaissEx.Index.dim(index)
{:ok, count} = FaissEx.Index.ntotal(index)
{:ok, trained?} = FaissEx.Index.trained?(index)
```

### Reconstructing vectors

```elixir
{:ok, vectors} = FaissEx.Index.reconstruct(index, [0, 5, 10])
# vectors: [[float]] — 3 rows of dim floats
```

### Computing residuals

Compute the difference between vectors and their quantized reconstructions:

```elixir
{:ok, index} = FaissEx.Index.new(4, "Flat")
vectors = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
:ok = FaissEx.Index.add(index, vectors)

{:ok, residuals} = FaissEx.Index.compute_residuals(index, vectors, [0, 1])
# For a flat index, residuals are zero (exact reconstruction)
```

### Batch search

Search with multiple query vectors at once for better throughput:

```elixir
{:ok, index} = FaissEx.Index.new(128, "Flat")
:ok = FaissEx.Index.add(index, corpus_vectors)

# Search 100 queries at once
{:ok, %{distances: distances, labels: labels}} = FaissEx.Index.search(index, batch_of_queries, 10)
# distances: [[float]] — 100 rows of 10 distances
# labels: [[integer]] — 100 rows of 10 vector IDs
```

## Example: Semantic Search with External Embeddings

FaissEx works with embeddings from any source — OpenAI, Cohere, Bumblebee, etc.
Just pass them as lists of floats:

```elixir
# Embeddings from any source (OpenAI, Cohere, Voyage, etc.)
embeddings = [
  [0.023, -0.041, 0.067, ...],  # 1536-dim for text-embedding-3-small
  [0.011, -0.032, 0.089, ...],
  # ...
]

dim = length(hd(embeddings))

# For cosine similarity, normalize vectors to unit length first
normalize = fn vec ->
  norm = :math.sqrt(Enum.reduce(vec, 0.0, fn x, acc -> acc + x * x end))
  Enum.map(vec, &(&1 / norm))
end

normalized = Enum.map(embeddings, normalize)

{:ok, index} = FaissEx.Index.new(dim, "Flat", metric: :inner_product)
:ok = FaissEx.Index.add(index, normalized)

# Query with a new embedding
query = normalize.(query_floats)
{:ok, %{distances: scores, labels: indices}} = FaissEx.Index.search(index, query, 10)
# scores: [[float]] — similarity scores (higher = more similar)
# indices: [[integer]] — vector IDs
```

## Choosing an Index

| Index | Training | Memory | Speed | Recall | Best for |
|-------|----------|--------|-------|--------|----------|
| `"Flat"` | No | High | Slow (brute force) | 100% | < 100K vectors, exact results |
| `"IVFx,Flat"` | Yes | High | Fast | ~95% | 100K-10M vectors |
| `"HNSWx"` | No | High | Very fast | ~99% | Read-heavy workloads |
| `"IVFx,PQy"` | Yes | Low | Fast | ~90% | 1M+ vectors, memory constrained |
| `"IVFx,SQfp16"` | Yes | Medium | Fast | ~97% | 1M+ vectors, good recall/memory balance |
| `"Flat"` + `:inner_product` | No | High | Slow | 100% | Cosine similarity (normalize first) |

`x` = number of IVF clusters (typical: `4*sqrt(n)`), `y` = PQ subquantizers (typical: `dim/4`).

### Practical guidelines

- **Under 100K vectors**: Use `"Flat"`. It's brute force but fast enough and gives exact results.
- **100K to 1M vectors**: Use `"IVF{nlist},Flat"` where `nlist = 4*sqrt(n)`. Train on at least `40*nlist` vectors.
- **1M+ vectors**: Use `"IVF{nlist},PQ{m}"` or `"HNSW32"`. PQ compresses vectors to save memory.
- **Cosine similarity**: Normalize your vectors to unit length first, then use `:inner_product` metric.

```elixir
# Cosine similarity search — normalize vectors to unit length, then use inner product
{:ok, index} = FaissEx.Index.new(128, "Flat", metric: :inner_product)
:ok = FaissEx.Index.add(index, normalized_vectors)
```

### IDMap wrapper

FAISS indexes assign sequential IDs by default (0, 1, 2...). Wrap with `"IDMap,"` to use your own IDs:

```elixir
# Custom IDs with any index type
{:ok, index} = FaissEx.Index.new(128, "IDMap,Flat")
{:ok, index} = FaissEx.Index.new(128, "IDMap,HNSW32")

:ok = FaissEx.Index.add_with_ids(index, vectors, [42, 99, 1337])

{:ok, %{labels: labels}} = FaissEx.Index.search(index, query, 5)
# labels now contain your custom IDs (42, 99, 1337...)
```

## K-means Clustering

### Basic clustering

```elixir
# Cluster 128-dimensional vectors into 10 groups
{:ok, clustering} = FaissEx.Clustering.new(128, 10)

data = for _ <- 1..5000, do: for(_ <- 1..128, do: :rand.uniform())
{:ok, trained} = FaissEx.Clustering.train(clustering, data)

{:ok, centroids} = FaissEx.Clustering.get_centroids(trained)
# centroids: [[float]] — 10 rows of 128 floats, center of each cluster
```

### Assigning vectors to clusters

```elixir
# Which cluster does each vector belong to?
{:ok, %{labels: labels, distances: distances}} =
  FaissEx.Clustering.get_cluster_assignment(trained, data)
# labels: [[integer]] — 5000 rows of 1 cluster ID each
# distances: [[float]] — 5000 rows of 1 distance each
```

### Clustering for preprocessing

Use clustering to build an IVF quantizer or to reduce a dataset:

```elixir
# Cluster embeddings, then keep only vectors near centroids
{:ok, clustering} = FaissEx.Clustering.new(dim, num_clusters)
{:ok, trained} = FaissEx.Clustering.train(clustering, embeddings)
{:ok, %{labels: labels, distances: dists}} =
  FaissEx.Clustering.get_cluster_assignment(trained, embeddings)

# Filter to vectors within distance threshold
threshold = 10.0
nearby = Enum.zip(embeddings, List.flatten(dists))
  |> Enum.filter(fn {_vec, dist} -> dist < threshold end)
  |> Enum.map(&elem(&1, 0))
```

## Thread Safety

Indexes can be shared freely across processes — the NIF layer guards every index (and clustering object) with a read-write lock:

- **Concurrent reads** (search, reconstruct, save, clone, property getters) run in parallel with each other
- **Mutations** (add, train, reset) take the write lock and serialize against all other operations on the same index — racing an `add` against a `search` is safe and cannot crash the VM
- Heavy operations (add, search, train) run on BEAM dirty CPU schedulers so they don't block normal Erlang processes, even while waiting on the lock
- File I/O (to_file, from_file) runs on dirty IO schedulers

For write-heavy concurrent workloads, note that mutations are serialized per index — shard across multiple indexes if you need parallel ingestion.

## Architecture

FaissEx uses Erlang NIF (Native Implemented Functions) to call FAISS through its C API:

```
Elixir (FaissEx.Index) → NIF stubs (FaissEx.NIF) → C (faiss_ex_nif.c) → libfaiss_c → libfaiss (C++)
```

- **Single C file**: All NIF code lives in `c_src/faiss_ex_nif.c`
- **NIF resources**: FAISS index pointers are wrapped in NIF resource types with destructors, so BEAM GC handles cleanup
- **No processes**: FaissEx uses plain functions and data — no GenServers or supervision trees
- **No dependencies**: Vectors flow in/out as plain lists; binary conversion happens internally

## Troubleshooting

### `OpenMP not found` during build

On macOS, install libomp:

```bash
brew install libomp
```

On Linux, OpenMP ships with gcc; make sure `build-essential` (or `gcc-c++`) is installed.

### `Could NOT find BLAS` / `LAPACK not found` during build (Linux)

FAISS needs a BLAS implementation:

```bash
# Debian/Ubuntu
sudo apt-get install -y libopenblas-dev

# Fedora/RHEL
sudo dnf install -y openblas-devel
```

macOS never needs this — the Accelerate framework provides BLAS.

### `Library not loaded: libfaiss_c.dylib`

The shared libraries weren't installed correctly. Clean and rebuild:

```bash
mix clean
mix compile
```

### Build takes too long

The first build compiles FAISS from source. This is cached at `~/.cache/faiss_ex/`. To force a clean FAISS rebuild:

```bash
rm -rf ~/.cache/faiss_ex/
mix compile
```

### `cmake` not found

```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt install cmake
```

## GPU Support

### Building with CUDA

```bash
USE_CUDA=true mix compile
```

This requires the CUDA toolkit to be installed. The build adds `-DFAISS_ENABLE_GPU=ON` to cmake and compiles GPU-specific NIF functions.

> **Note:** FAISS 1.12.0+ (including the pinned v1.14.3) requires **CUDA 12.x** — CUDA 11 support was dropped upstream. If you are stuck on CUDA 11, build with `FAISS_GIT_REV=v1.11.0`.

### Moving indexes to GPU

```elixir
# Create index on CPU first
{:ok, index} = FaissEx.Index.new(128, "Flat")
:ok = FaissEx.Index.add(index, vectors)

# Move to GPU (device 0)
{:ok, gpu_index} = FaissEx.Index.cpu_to_gpu(index, 0)

# Search on GPU (faster for large datasets)
{:ok, results} = FaissEx.Index.search(gpu_index, query, 10)

# Move back to CPU (e.g. for saving to file)
{:ok, cpu_index} = FaissEx.Index.gpu_to_cpu(gpu_index)
:ok = FaissEx.Index.to_file(cpu_index, "/tmp/index.faiss")
```

### GPU at index creation time

```elixir
{:ok, gpu_index} = FaissEx.Index.new(128, "Flat", device: {:cuda, 0})
```

### Checking GPU availability

```elixir
num_gpus = FaissEx.num_gpus()
# Returns 0 if not compiled with CUDA or no GPUs available
```

### GPU lifecycle notes

- GPU resources (`FaissStandardGpuResources`) must outlive GPU indexes. FaissEx handles this automatically — the `%FaissEx.Index{}` struct holds both the GPU resources ref and the index ref, so BEAM GC keeps both alive.
- You can create indexes on CPU, add vectors, then move to GPU for fast search.
- File I/O requires CPU indexes — move back with `gpu_to_cpu/1` before saving.
- Concurrent reads (search) on the same GPU index are safe. Concurrent writes (add) are not.

### Running GPU tests

```bash
USE_CUDA=true mix test --include cuda
```

## Benchmarks

Measured on Apple M4 Max (16 cores, 64 GB RAM), Elixir 1.20.0, OTP 29, FAISS v1.14.3.
Index: 10,000 vectors, 128 dimensions, Flat L2.

| Operation | ips | avg | median | 99th % |
|-----------|-----|-----|--------|--------|
| reconstruct 1 vector | 141.05 K | 7.09 μs | 6.96 μs | 13.79 μs |
| add 1 vector | 83.64 K | 11.96 μs | 10.50 μs | 31.17 μs |
| compute_residuals 1 vector | 17.48 K | 57.21 μs | 54.50 μs | 117.78 μs |
| reconstruct 10 vectors | 16.68 K | 59.96 μs | 57.75 μs | 118.00 μs |
| search k=10, 1 query | 8.48 K | 117.94 μs | 114.13 μs | 263.71 μs |
| compute_residuals 10 vectors | 7.36 K | 135.85 μs | 128.17 μs | 247.56 μs |
| kmeans assign 100 vectors | 3.87 K | 258.16 μs | 249.75 μs | 494.97 μs |
| search k=10, 10 queries | 2.97 K | 337.12 μs | 335.83 μs | 381.58 μs |
| reconstruct 100 vectors | 1.10 K | 911.66 μs | 847.73 μs | 1786.82 μs |
| compute_residuals 100 vectors | 1.09 K | 914.24 μs | 881.77 μs | 1223.25 μs |
| search k=10, 100 queries | 0.45 K | 2.20 ms | 2.18 ms | 2.47 ms |
| kmeans k=50, 1000 vectors | 0.26 K | 3.87 ms | 3.63 ms | 4.70 ms |
| kmeans k=10, 1000 vectors | 0.22 K | 4.50 ms | 4.09 ms | 5.83 ms |
| add 1000 vectors | 0.090 K | 11.08 ms | 11.62 ms | 21.81 ms |
| add 10000 vectors | 0.069 K | 14.45 ms | 13.39 ms | 36.66 ms |

> **Note:** the batch `add` scenarios are dominated by list→binary encoding
> and BEAM garbage collection of the benchmark corpus, not by FAISS — the
> NIF-side add of 10,000 vectors is ~0.4 ms. This is also why `add 1000`
> and `add 10000` land so close together.

Run benchmarks yourself:

```bash
mix run bench/faiss_ex_bench.exs
```

## License

Apache License 2.0.
