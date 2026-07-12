defmodule FaissEx.Clustering do
  @moduledoc """
  K-means clustering using FAISS.

  Partitions vectors into `k` clusters and returns cluster centroids.
  Also supports assigning new vectors to their nearest cluster.

  ## Example

      {:ok, clustering} = FaissEx.Clustering.new(128, 10)

      data = for _ <- 1..5000, do: for(_ <- 1..128, do: :rand.uniform())
      {:ok, trained} = FaissEx.Clustering.train(clustering, data)

      {:ok, centroids} = FaissEx.Clustering.get_centroids(trained)
      # centroids: 10 lists of 128 floats each

      {:ok, %{labels: labels}} =
        FaissEx.Clustering.get_cluster_assignment(trained, [hd(data)])
      # labels: [[cluster_id]]
  """

  alias FaissEx.NIF
  alias FaissEx.Shared
  alias FaissEx.Index

  defstruct [:ref, :k, :d, :index, :trained?]

  @typedoc """
  Handle to a FAISS clustering object.

    * `:ref` — NIF resource reference (freed by the BEAM GC)
    * `:k` — number of clusters
    * `:d` — vector dimension
    * `:index` — quantizer index; after `train/2` it holds the `k` centroids
      and is reused for `get_cluster_assignment/2`
    * `:trained?` — whether `train/2` has run
  """
  @type t :: %__MODULE__{
          ref: reference(),
          k: pos_integer(),
          d: pos_integer(),
          index: Index.t() | nil,
          trained?: boolean()
        }

  @typedoc "Error reason reported by FAISS or the NIF layer."
  @type error :: {:error, String.t()}

  @doc """
  Creates a new clustering object.

  ## Parameters

    * `d` - vector dimension
    * `k` - number of clusters
  """
  @spec new(pos_integer(), pos_integer()) :: {:ok, t()} | error()
  def new(d, k) do
    case NIF.nif_new_clustering(d, k) do
      {:ok, ref} ->
        {:ok, %__MODULE__{ref: ref, k: k, d: d, index: nil, trained?: false}}

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Trains the clustering on the given data.

  Creates a flat index internally for the quantizer; after training it holds
  the `k` centroids. `data` must be a list of lists of floats with inner
  dimension `d`.
  """
  @spec train(t(), [[number()]]) :: {:ok, t()} | error()
  def train(%__MODULE__{ref: ref, d: d} = clustering, data) when is_list(data) do
    {n, data_bin} = Shared.encode_vectors!(data, d)

    # On training failure the discarded quantizer is not leaked — its NIF
    # resource is freed by the BEAM GC via the resource destructor.
    with {:ok, quantizer} <- Index.new(d, "Flat"),
         :ok <- NIF.nif_train_clustering(ref, n, data_bin, quantizer.ref) do
      {:ok, %__MODULE__{clustering | index: quantizer, trained?: true}}
    end
  end

  @doc """
  Returns the cluster centroids as a list of lists `[[float]]` with shape `{k, d}`.
  """
  @spec get_centroids(t()) :: {:ok, [[float()]]} | error()
  def get_centroids(%__MODULE__{ref: ref, d: d}) do
    case NIF.nif_get_clustering_centroids(ref) do
      {:ok, {_k, _d, centroids_bin}} ->
        {:ok, centroids_bin |> Shared.binary_to_floats() |> Enum.chunk_every(d)}

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Assigns vectors to their nearest cluster.

  Searches the quantizer index populated during `train/2` — the trained
  centroids are already indexed, so no per-call rebuild is needed.

  Returns `{:ok, %{labels: [[integer]], distances: [[float]]}}`.
  """
  @spec get_cluster_assignment(t(), Index.vectors()) ::
          {:ok, %{distances: [[float()]], labels: [[integer()]]}} | error()
  def get_cluster_assignment(%__MODULE__{index: nil}, _data) do
    {:error, "clustering not trained"}
  end

  def get_cluster_assignment(%__MODULE__{index: %Index{} = quantizer}, data)
      when is_list(data) do
    Index.search(quantizer, data, 1)
  end
end
