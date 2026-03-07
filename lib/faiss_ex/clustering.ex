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

  @type t :: %__MODULE__{
          ref: reference(),
          k: pos_integer(),
          d: pos_integer(),
          index: Index.t() | nil,
          trained?: boolean()
        }

  @doc """
  Creates a new clustering object.

  ## Parameters

    * `d` - vector dimension
    * `k` - number of clusters
  """
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

  Creates a flat index internally for the quantizer.
  `data` must be a list of lists of floats with inner dimension `d`.
  """
  def train(%__MODULE__{ref: ref, d: d} = clustering, data) when is_list(data) do
    {n, flat} = Shared.to_flat_floats(data)
    data_bin = Shared.floats_to_binary(flat)

    {:ok, quantizer} = Index.new(d, "Flat")

    case NIF.nif_train_clustering(ref, n, data_bin, quantizer.ref) do
      :ok ->
        {:ok, %__MODULE__{clustering | index: quantizer, trained?: true}}

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Returns the cluster centroids as a list of lists `[[float]]` with shape `{k, d}`.
  """
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

  Returns `{:ok, %{labels: [[integer]], distances: [[float]]}}`.
  """
  def get_cluster_assignment(%__MODULE__{index: nil}, _data) do
    {:error, "clustering not trained"}
  end

  def get_cluster_assignment(%__MODULE__{d: d} = clustering, data) when is_list(data) do
    with {:ok, centroids} <- get_centroids(clustering),
         {:ok, quantizer} <- Index.new(d, "Flat") do
      :ok = Index.add(quantizer, centroids)
      Index.search(quantizer, data, 1)
    end
  end
end
