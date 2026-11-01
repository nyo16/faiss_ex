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

  ## Value semantics

  A clustering struct is a value: `train/2` never changes the struct you
  pass in. Each call builds a fresh FAISS clustering object and a fresh
  quantizer and returns them in a new struct, so an untrained struct can be
  trained several times (different data, concurrently from different
  processes) and every returned struct stays self-consistent — its
  centroids and its quantizer always come from the same training run.

  ## Binary input and output

  `train/2` and `get_cluster_assignment/2` also accept a row-major
  `f32-native` binary (`d` floats per vector), and `get_centroids_binary/1`
  returns the centroids as one such binary — see the *Binary API* section of
  `FaissEx.Index`.
  """

  alias FaissEx.Index
  alias FaissEx.NIF
  alias FaissEx.Shared

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
    with {:ok, ref} <- NIF.nif_new_clustering(d, k) do
      {:ok, %__MODULE__{ref: ref, k: k, d: d, index: nil, trained?: false}}
    end
  end

  @doc """
  Trains k-means on `data` and returns a new, trained clustering.

  `data` is a list of vectors with `d` elements each, or a row-major
  `f32-native` binary. The input struct is not modified (see *Value
  semantics* in the moduledoc): the result carries its own clustering
  object and a Flat quantizer index holding the `k` centroids.

  Returns `{:error, "n must be positive"}` for empty data without
  allocating anything, and `{:error, reason}` from FAISS when training
  fails (for example, fewer points than clusters).
  """
  @spec train(t(), Index.vector_data()) :: {:ok, t()} | error()
  def train(%__MODULE__{d: d, k: k} = clustering, data) when is_list(data) or is_binary(data) do
    case Shared.encode_vector_data!(data, d) do
      {0, _} ->
        {:error, "n must be positive"}

      {n, data_bin} ->
        # On failure the fresh clustering object and quantizer are dropped
        # and freed by the BEAM GC via their resource destructors.
        with {:ok, ref} <- NIF.nif_new_clustering(d, k),
             {:ok, quantizer} <- Index.new(d, "Flat"),
             :ok <- NIF.nif_train_clustering(ref, n, data_bin, quantizer.ref) do
          {:ok, %__MODULE__{clustering | ref: ref, index: quantizer, trained?: true}}
        end
    end
  end

  @doc """
  Returns the cluster centroids as a list of lists `[[float]]` with shape `{k, d}`.

  Returns `{:error, "no centroids available"}` for a clustering that has
  not been trained.
  """
  @spec get_centroids(t()) :: {:ok, [[float()]]} | error()
  def get_centroids(%__MODULE__{d: d} = clustering) do
    with {:ok, centroids_bin} <- get_centroids_binary(clustering) do
      {:ok, Shared.binary_to_float_rows(centroids_bin, d)}
    end
  end

  @doc """
  Returns the cluster centroids as one raw binary — the zero-decode
  counterpart of `get_centroids/1`: `k * d` `f32-native` floats, row-major.
  """
  @spec get_centroids_binary(t()) :: {:ok, binary()} | error()
  def get_centroids_binary(%__MODULE__{ref: ref}) do
    with {:ok, {_k, _d, centroids_bin}} <- NIF.nif_get_clustering_centroids(ref) do
      {:ok, centroids_bin}
    end
  end

  @doc """
  Assigns vectors to their nearest cluster.

  `data` accepts the same list or binary forms as `FaissEx.Index.search/3`.
  Searches the quantizer index populated during `train/2` — the trained
  centroids are already indexed, so no per-call rebuild is needed.

  Returns `{:ok, %{labels: [[integer]], distances: [[float]]}}`.
  """
  @spec get_cluster_assignment(t(), Index.vector_data()) ::
          {:ok, %{distances: [[float()]], labels: [[integer()]]}} | error()
  def get_cluster_assignment(%__MODULE__{index: nil}, _data) do
    {:error, "clustering not trained"}
  end

  def get_cluster_assignment(%__MODULE__{index: %Index{} = quantizer}, data)
      when is_list(data) or is_binary(data) do
    Index.search(quantizer, data, 1)
  end
end
