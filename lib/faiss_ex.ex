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

  @doc """
  Returns the number of available CUDA GPUs.

  Returns `0` when GPU support is not compiled in (the default build).
  """
  @spec num_gpus() :: non_neg_integer()
  def num_gpus do
    {:ok, n} = FaissEx.NIF.nif_get_num_gpus()
    n
  end

  @doc """
  Sets the number of OpenMP threads FAISS uses inside each call
  (search, add, train, reconstruct, residuals, clustering).

  `0` restores the runtime default (`OMP_NUM_THREADS`, or the core count).
  The setting is global, takes effect on the next FAISS call, and can also
  be set at boot with `config :faiss_ex, omp_threads: n`.

  ## Choosing a value

  Every FAISS call already runs on a BEAM dirty scheduler, and by default
  each one also starts a team of one OpenMP thread per core. With many
  concurrent callers the two thread pools oversubscribe the CPU:

    * **Many concurrent small requests** (single-query searches from many
      processes) — use `1`. Concurrency comes from the dirty schedulers;
      measured ~2.8× higher throughput than the default (Flat, 100k × 128).
    * **Few large batches** (one 1000-query search, bulk `add`, `train`) —
      keep the default `0`. Measured ~1.5× slower with `1` thread.

  See the README's "Concurrency tuning" section for the measurements and
  Linux-specific advice (`libopenblas-openmp-dev`, `KMP_BLOCKTIME`).

  A NIF built without a shareable OpenMP runtime (a system FAISS linked
  against Intel's `libiomp5`) accepts and ignores the setting.
  """
  @spec set_omp_threads(non_neg_integer()) :: :ok
  def set_omp_threads(threads) when is_integer(threads) and threads >= 0 do
    :ok = FaissEx.NIF.nif_set_omp_threads(threads)
  end
end
