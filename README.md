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

All functions that accept vectors work with **Nx tensors**, **plain lists**, or **lists of lists**:

```elixir
# All three are equivalent:
FaissEx.Index.add(index, Nx.tensor([[1.0, 2.0, 3.0]], type: {:f, 32}))
FaissEx.Index.add(index, [1.0, 2.0, 3.0])
FaissEx.Index.add(index, [[1.0, 2.0, 3.0]])
```

### Creating an index and searching

```elixir
# Create a flat L2 index with 128 dimensions
{:ok, index} = FaissEx.Index.new(128, "Flat")

# Add vectors — Nx tensors or plain lists
:ok = FaissEx.Index.add(index, some_nx_tensor)
:ok = FaissEx.Index.add(index, [[0.1, 0.2, ...], [0.3, 0.4, ...]])

# Search for 5 nearest neighbors
{:ok, %{distances: distances, labels: labels}} = FaissEx.Index.search(index, query, 5)
# distances: {n, 5} f32 tensor
# labels: {n, 5} s64 tensor of vector indices
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

### Computing residuals

Compute the difference between vectors and their quantized reconstructions:

```elixir
{:ok, index} = FaissEx.Index.new(4, "Flat")
vectors = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], type: {:f, 32})
:ok = FaissEx.Index.add(index, vectors)

keys = Nx.tensor([0, 1], type: {:s, 64})
{:ok, residuals} = FaissEx.Index.compute_residuals(index, vectors, keys)
# For a flat index, residuals are zero (exact reconstruction)
```

### Batch search

Search with multiple query vectors at once for better throughput:

```elixir
{:ok, index} = FaissEx.Index.new(128, "Flat")
:ok = FaissEx.Index.add(index, corpus_vectors)

# Search 100 queries at once
queries = Nx.tensor(batch_of_queries, type: {:f, 32})  # {100, 128}
{:ok, %{distances: distances, labels: labels}} = FaissEx.Index.search(index, queries, 10)
# distances: {100, 10} - top 10 distances per query
# labels: {100, 10} - top 10 vector IDs per query
```

## Example: Semantic Search with Bumblebee Embeddings

A complete example using [Bumblebee](https://github.com/elixir-nx/bumblebee) to generate sentence embeddings and FaissEx to search them.

### Setup

Add Bumblebee and an EXLA/Torchx backend to your deps:

```elixir
def deps do
  [
    {:faiss_ex, "~> 0.1.0"},
    {:bumblebee, "~> 0.6"},
    {:exla, "~> 0.9"}
  ]
end
```

### Building an embedding index

```elixir
# Load a sentence-transformers model
{:ok, model_info} = Bumblebee.load_model({:hf, "sentence-transformers/all-MiniLM-L6-v2"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "sentence-transformers/all-MiniLM-L6-v2"})

serving = Bumblebee.Text.text_embedding(model_info, tokenizer,
  compile: [batch_size: 32, sequence_length: 128],
  defn_options: [compiler: EXLA]
)

# Your document corpus
documents = [
  "The cat sat on the mat",
  "A dog played in the park",
  "Elixir is a functional programming language",
  "Phoenix is a web framework for Elixir",
  "FAISS enables fast similarity search",
  "Machine learning models generate embeddings",
  "Vectors can represent semantic meaning",
  "The quick brown fox jumps over the lazy dog"
]

# Generate embeddings for all documents
embeddings =
  documents
  |> Enum.map(fn doc ->
    %{embedding: embedding} = Nx.Serving.run(serving, doc)
    embedding
  end)
  |> Nx.stack()

# embeddings: {8, 384} f32 tensor (MiniLM outputs 384-dim vectors)
{n, dim} = Nx.shape(embeddings)

# Normalize for cosine similarity
norms = Nx.LinAlg.norm(embeddings, axes: [1], keep_axes: true)
normalized = Nx.divide(embeddings, norms)

# Build the FAISS index
{:ok, index} = FaissEx.Index.new(dim, "Flat", metric: :inner_product)
:ok = FaissEx.Index.add(index, normalized)
```

### Querying

```elixir
# Embed the query
query_text = "functional programming with Elixir"
%{embedding: query_embedding} = Nx.Serving.run(serving, query_text)

# Normalize
query_norm = Nx.LinAlg.norm(query_embedding)
query_normalized = Nx.divide(query_embedding, query_norm)

# Search for top 3 most similar documents
{:ok, %{distances: scores, labels: indices}} =
  FaissEx.Index.search(index, query_normalized, 3)

# Display results
indices
|> Nx.to_flat_list()
|> Enum.zip(Nx.to_flat_list(scores))
|> Enum.each(fn {idx, score} ->
  IO.puts("#{Float.round(score, 4)} - #{Enum.at(documents, idx)}")
end)

# Output (similarity scores, higher = more similar):
# 0.8234 - Elixir is a functional programming language
# 0.7891 - Phoenix is a web framework for Elixir
# 0.4012 - Machine learning models generate embeddings
```

### Scaling up with IVF

For larger corpora (100K+ documents), use an IVF index:

```elixir
n_documents = length(documents)
nlist = max(round(4 * :math.sqrt(n_documents)), 1)

{:ok, index} = FaissEx.Index.new(dim, "IVF#{nlist},Flat", metric: :inner_product)

# Train on your embeddings (or a representative sample)
:ok = FaissEx.Index.train(index, normalized)
:ok = FaissEx.Index.add(index, normalized)

# Save for later use
:ok = FaissEx.Index.to_file(index, "embeddings.index")
```

### Persistent search service

A simple pattern for serving search in a Phoenix app:

```elixir
defmodule MyApp.SemanticSearch do
  @index_path "priv/embeddings.index"

  def load_index do
    {:ok, index} = FaissEx.Index.from_file(@index_path)
    index
  end

  def search(index, serving, query_text, k \\ 5) do
    %{embedding: embedding} = Nx.Serving.run(serving, query_text)
    norm = Nx.LinAlg.norm(embedding)
    normalized = Nx.divide(embedding, norm)

    {:ok, %{distances: scores, labels: indices}} =
      FaissEx.Index.search(index, normalized, k)

    Enum.zip(
      Nx.to_flat_list(indices),
      Nx.to_flat_list(scores)
    )
  end
end
```

### Using OpenAI / other external embeddings

If your embeddings come from an external API as lists of floats:

```elixir
# Embeddings from any source (OpenAI, Cohere, Voyage, etc.)
raw_embeddings = [
  [0.023, -0.041, 0.067, ...],  # 1536-dim for text-embedding-3-small
  [0.011, -0.032, 0.089, ...],
  # ...
]

dim = length(hd(raw_embeddings))
embeddings = Nx.tensor(raw_embeddings, type: {:f, 32})

# Normalize and index
norms = Nx.LinAlg.norm(embeddings, axes: [1], keep_axes: true)
normalized = Nx.divide(embeddings, norms)

{:ok, index} = FaissEx.Index.new(dim, "Flat", metric: :inner_product)
:ok = FaissEx.Index.add(index, normalized)

# Query with a new embedding from the same API
query_embedding = Nx.tensor([query_floats], type: {:f, 32})
query_normalized = Nx.divide(query_embedding, Nx.LinAlg.norm(query_embedding))
{:ok, results} = FaissEx.Index.search(index, query_normalized, 10)
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
# Cosine similarity search
normalized = Nx.divide(vectors, Nx.LinAlg.norm(vectors, axes: [1], keep_axes: true))
{:ok, index} = FaissEx.Index.new(128, "Flat", metric: :inner_product)
:ok = FaissEx.Index.add(index, normalized)
```

### IDMap wrapper

FAISS indexes assign sequential IDs by default (0, 1, 2...). Wrap with `"IDMap,"` to use your own IDs:

```elixir
# Custom IDs with any index type
{:ok, index} = FaissEx.Index.new(128, "IDMap,Flat")
{:ok, index} = FaissEx.Index.new(128, "IDMap,HNSW32")

ids = Nx.tensor([42, 99, 1337], type: {:s, 64})
:ok = FaissEx.Index.add_with_ids(index, vectors, ids)

{:ok, %{labels: labels}} = FaissEx.Index.search(index, query, 5)
# labels now contain your custom IDs (42, 99, 1337...)
```

## K-means Clustering

### Basic clustering

```elixir
# Cluster 128-dimensional vectors into 10 groups
{:ok, clustering} = FaissEx.Clustering.new(128, 10)

data = Nx.iota({5000, 128}, type: {:f, 32})
{:ok, trained} = FaissEx.Clustering.train(clustering, data)

{:ok, centroids} = FaissEx.Clustering.get_centroids(trained)
# centroids: {10, 128} f32 tensor — center of each cluster
```

### Assigning vectors to clusters

```elixir
# Which cluster does each vector belong to?
{:ok, %{labels: labels, distances: distances}} =
  FaissEx.Clustering.get_cluster_assignment(trained, data)
# labels: {5000, 1} s64 tensor of cluster IDs
# distances: {5000, 1} f32 tensor of distances to nearest centroid
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
mask = Nx.less(Nx.reshape(dists, {n}), threshold)
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
- **Nx integration**: Vectors flow in/out as Nx tensors; binary data is passed directly to FAISS with zero copy where possible

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

## License

Apache License 2.0. See [LICENSE](LICENSE).
