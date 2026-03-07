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

- Erlang/OTP with NIF support
- CMake 3.17+
- C/C++ compiler (gcc or clang)
- On macOS Apple Silicon: `brew install libomp`
- For GPU support: CUDA toolkit

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

The first build clones and compiles FAISS from source (~5-15 min). Subsequent builds use the cached version at `~/.cache/faiss_ex/`.

### Build configuration

Set these environment variables before `mix compile`:

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_CUDA` | `false` | Set to `true` to enable GPU support |
| `FAISS_GIT_REPO` | `https://github.com/facebookresearch/faiss.git` | FAISS git repository |
| `FAISS_GIT_REV` | `v1.10.0` | FAISS version tag or commit |

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

- **Concurrent reads** (search) on the same index are safe
- **Concurrent writes** (add) on the same index are **not safe** — serialize writes or use separate indexes
- Heavy operations (add, search, train) run on BEAM dirty CPU schedulers so they don't block normal Erlang processes
- File I/O (to_file, from_file) runs on dirty IO schedulers

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
{:ok, num_gpus} = FaissEx.NIF.nif_get_num_gpus()
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

Measured on Apple M4 Max (16 cores, 64 GB RAM), Elixir 1.20.0-rc.1, OTP 28.3, FAISS v1.10.0.
Index: 10,000 vectors, 128 dimensions, Flat L2.

| Operation | ips | avg | median | 99th % |
|-----------|-----|-----|--------|--------|
| reconstruct 1 vector | 215.33 K | 4.64 μs | 4.75 μs | 11 μs |
| add 1 vector | 133.28 K | 7.50 μs | 6.92 μs | 14.54 μs |
| reconstruct 10 vectors | 23.57 K | 42.42 μs | 42.21 μs | 79.14 μs |
| compute_residuals 1 vector | 17.00 K | 58.83 μs | 55.92 μs | 124.25 μs |
| search k=10, 1 query | 12.05 K | 82.95 μs | 80.92 μs | 104.25 μs |
| compute_residuals 10 vectors | 7.72 K | 129.46 μs | 124.46 μs | 217.88 μs |
| search k=10, 10 queries | 5.13 K | 194.90 μs | 190.42 μs | 266.53 μs |
| reconstruct 100 vectors | 1.89 K | 528.36 μs | 499.75 μs | 930.06 μs |
| compute_residuals 100 vectors | 1.11 K | 903.15 μs | 866.35 μs | 1325.00 μs |
| search k=10, 100 queries | 0.58 K | 1.72 ms | 1.72 ms | 1.96 ms |
| add 1000 vectors | 0.38 K | 2.65 ms | 2.58 ms | 3.67 ms |
| kmeans k=10, 1000 vectors | 0.108 K | 9.22 ms | 9.00 ms | 13.87 ms |
| kmeans k=50, 1000 vectors | 0.098 K | 10.18 ms | 9.97 ms | 14.32 ms |
| add 10000 vectors | 0.018 K | 56.52 ms | 55.05 ms | 71.46 ms |

Run benchmarks yourself:

```bash
mix run bench/faiss_ex_bench.exs
```

## License

Apache License 2.0.
