defmodule FaissEx do
  @moduledoc """
  Elixir NIF bindings for [FAISS](https://github.com/facebookresearch/faiss) —
  Facebook AI Similarity Search.

  FaissEx provides fast vector similarity search and clustering through
  FAISS's C API. All data flows as plain Elixir lists — no external
  dependencies beyond `elixir_make`.

  ## Quick start

      # Create a flat L2 index
      {:ok, index} = FaissEx.Index.new(128, "Flat")

      # Add vectors (list of lists)
      vectors = for _ <- 1..1000, do: for(_ <- 1..128, do: :rand.uniform())
      :ok = FaissEx.Index.add(index, vectors)

      # Search for 5 nearest neighbors
      {:ok, %{distances: distances, labels: labels}} =
        FaissEx.Index.search(index, hd(vectors), 5)

  ## Modules

    * `FaissEx.Index` — create, populate, search, and manage FAISS indexes
    * `FaissEx.Clustering` — k-means clustering and cluster assignment

  ## Index types

  FAISS uses [index factory strings](https://github.com/facebookresearch/faiss/wiki/The-index-factory):

    * `"Flat"` — exact brute-force search (no training needed)
    * `"IVF256,Flat"` — inverted file index (requires training)
    * `"HNSW32"` — graph-based approximate search
    * `"IVF256,PQ32"` — product quantization (compressed, requires training)
    * `"IDMap,Flat"` — flat index with custom vector IDs
  """
end
