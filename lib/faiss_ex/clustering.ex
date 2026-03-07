defmodule FaissEx.Clustering do
  @moduledoc """
  FAISS clustering (k-means) operations.
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

  Creates a flat index internally for the quantizer if no index is provided.
  The `tensor` must be `{:f, 32}` with shape `{n, d}`.
  """
  def train(%__MODULE__{ref: ref, d: d} = clustering, tensor) do
    tensor = tensor |> Nx.as_type({:f, 32}) |> ensure_2d(d)
    Shared.validate_type!(tensor, {:f, 32})
    {n, ^d} = Nx.shape(tensor)
    data = Nx.to_binary(tensor)

    {:ok, quantizer} = Index.new(d, "Flat")

    case NIF.nif_train_clustering(ref, n, data, quantizer.ref) do
      :ok ->
        {:ok, %__MODULE__{clustering | index: quantizer, trained?: true}}

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Returns the cluster centroids as an `{:f, 32}` tensor of shape `{k, d}`.
  """
  def get_centroids(%__MODULE__{ref: ref}) do
    case NIF.nif_get_clustering_centroids(ref) do
      {:ok, {k, d, centroids_bin}} ->
        {:ok, centroids_bin |> Nx.from_binary({:f, 32}) |> Nx.reshape({k, d})}

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Assigns vectors to their nearest cluster.

  Returns `%{labels: tensor, distances: tensor}` where labels are cluster IDs.
  """
  def get_cluster_assignment(%__MODULE__{index: nil}, _tensor) do
    {:error, "clustering not trained"}
  end

  def get_cluster_assignment(%__MODULE__{d: d} = clustering, tensor) do
    tensor = tensor |> Nx.as_type({:f, 32}) |> ensure_2d(d)
    Shared.validate_type!(tensor, {:f, 32})

    with {:ok, centroids} <- get_centroids(clustering),
         {:ok, quantizer} <- Index.new(d, "Flat") do
      :ok = Index.add(quantizer, centroids)
      Index.search(quantizer, tensor, 1)
    end
  end

  defp ensure_2d(tensor, d) do
    case Nx.shape(tensor) do
      {^d} ->
        Nx.reshape(tensor, {1, d})

      {_, ^d} ->
        tensor

      shape ->
        raise ArgumentError, "expected tensor of shape {#{d}} or {n, #{d}}, got #{inspect(shape)}"
    end
  end
end
