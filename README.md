# FaissEx

Elixir NIF bindings for [FAISS](https://github.com/facebookresearch/faiss) — Facebook's library for efficient similarity search and clustering of dense vectors.

Binds to FAISS via its official C API (`libfaiss_c`) and integrates with [Nx](https://github.com/elixir-nx/nx) tensors.

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

### Creating an index and searching

```elixir
# Create a flat L2 index with 128 dimensions
{:ok, index} = FaissEx.Index.new(128, "Flat")

# Add vectors (Nx tensors, f32)
vectors = Nx.random_uniform({1000, 128}, type: {:f, 32})
:ok = FaissEx.Index.add(index, vectors)

# Search for 5 nearest neighbors
query = Nx.random_uniform({1, 128}, type: {:f, 32})
{:ok, %{distances: distances, labels: labels}} = FaissEx.Index.search(index, query, 5)
# distances: {1, 5} f32 tensor
# labels: {1, 5} s64 tensor of vector indices
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

# Train on representative data
training_data = Nx.random_uniform({10_000, 128}, type: {:f, 32})
:ok = FaissEx.Index.train(index, training_data)

# Now add and search
:ok = FaissEx.Index.add(index, training_data)
{:ok, results} = FaissEx.Index.search(index, query, 10)
```

### Adding vectors with IDs

```elixir
{:ok, index} = FaissEx.Index.new(128, "IDMap,Flat")

vectors = Nx.random_uniform({100, 128}, type: {:f, 32})
ids = Nx.tensor(Enum.to_list(1000..1099), type: {:s, 64})
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
keys = Nx.tensor([0, 5, 10], type: {:s, 64})
{:ok, vectors} = FaissEx.Index.reconstruct(index, keys)
# vectors: {3, dim} f32 tensor
```

### K-means clustering

```elixir
{:ok, clustering} = FaissEx.Clustering.new(128, 10)  # 128-dim, 10 clusters

data = Nx.random_uniform({5000, 128}, type: {:f, 32})
{:ok, trained} = FaissEx.Clustering.train(clustering, data)

{:ok, centroids} = FaissEx.Clustering.get_centroids(trained)
# centroids: {10, 128} f32 tensor

# Assign vectors to nearest cluster
{:ok, %{labels: labels}} = FaissEx.Clustering.get_cluster_assignment(trained, data)
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

## License

Apache License 2.0. See [LICENSE](LICENSE).
