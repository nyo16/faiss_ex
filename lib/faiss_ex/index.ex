defmodule FaissEx.Index do
  @moduledoc """
  Create, populate, search, and manage FAISS indexes.

  An index holds a collection of vectors and supports nearest-neighbor search.
  Vectors are passed as flat lists (single vector) or lists of lists (batch).

  ## Creating and searching

      {:ok, index} = FaissEx.Index.new(4, "Flat")
      :ok = FaissEx.Index.add(index, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

      {:ok, %{distances: distances, labels: labels}} =
        FaissEx.Index.search(index, [1.0, 0.0, 0.0, 0.0], 2)
      # labels => [[0, 1]], distances => [[0.0, 2.0]]

  ## Index types

  Use [FAISS index factory strings](https://github.com/facebookresearch/faiss/wiki/The-index-factory)
  to create different index types:

    * `"Flat"` — exact brute-force search
    * `"IVF256,Flat"` — inverted file index (call `train/2` first)
    * `"HNSW32"` — graph-based approximate search
    * `"IDMap,Flat"` — flat index with custom vector IDs

  ## File I/O

      :ok = FaissEx.Index.to_file(index, "/tmp/my.index")
      {:ok, loaded} = FaissEx.Index.from_file("/tmp/my.index")

  ## GPU support

  Requires building with `USE_CUDA=true`:

      {:ok, gpu_index} = FaissEx.Index.cpu_to_gpu(index, 0)
      {:ok, cpu_index} = FaissEx.Index.gpu_to_cpu(gpu_index)
  """

  alias FaissEx.NIF
  alias FaissEx.Shared

  defstruct [:dim, :ref, :device, :description, :gpu_resources_ref]

  @typedoc """
  Handle to a FAISS index.

    * `:dim` — vector dimension the index was created with
    * `:ref` — NIF resource reference (freed by the BEAM GC)
    * `:device` — `:host` or `{:cuda, device_id}`
    * `:description` — factory string the index was built from, or `nil`
      when loaded from a file
    * `:gpu_resources_ref` — NIF resource backing a GPU index, `nil` on host
  """
  @type t :: %__MODULE__{
          dim: pos_integer(),
          ref: reference(),
          device: :host | {:cuda, non_neg_integer()},
          description: String.t() | nil,
          gpu_resources_ref: reference() | nil
        }

  @typedoc "A single vector as a flat list, or a batch as a list of vectors."
  @type vectors :: [number()] | [[number()]]

  @typedoc "Error reason reported by FAISS or the NIF layer."
  @type error :: {:error, String.t()}

  @doc """
  Creates a new FAISS index using the index factory.

  ## Options

    * `:metric` - distance metric, `:l2` (default) or `:inner_product`
    * `:device` - `:host` (default) or `{:cuda, device_id}`

  ## Examples

      iex> {:ok, index} = FaissEx.Index.new(128, "Flat")
      iex> index.dim
      128
  """
  @spec new(pos_integer(), String.t(), keyword()) :: {:ok, t()} | error()
  def new(dim, description, opts \\ []) do
    metric = Keyword.get(opts, :metric, :l2)
    device = Keyword.get(opts, :device, :host)

    metric_int =
      case metric do
        :l2 ->
          1

        :inner_product ->
          0

        other ->
          raise ArgumentError,
                "invalid metric #{inspect(other)}, expected :l2 or :inner_product"
      end

    desc_bin = to_binary(description)

    with {:ok, ref} <- NIF.nif_new_index(dim, desc_bin, metric_int) do
      index = %__MODULE__{
        dim: dim,
        ref: ref,
        device: :host,
        description: description
      }

      case device do
        :host -> {:ok, index}
        {:cuda, device_id} -> cpu_to_gpu(index, device_id)
      end
    end
  end

  @doc """
  Adds vectors to the index.

  Accepts a flat list of floats (single vector) or a list of lists (batch).
  Raises `ArgumentError` if any vector does not have `dim` elements.

  ## Examples

      :ok = FaissEx.Index.add(index, [1.0, 2.0, 3.0])
      :ok = FaissEx.Index.add(index, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  """
  @spec add(t(), vectors()) :: :ok | error()
  def add(%__MODULE__{ref: ref, dim: dim}, data) when is_list(data) do
    {n, bin} = Shared.encode_vectors!(data, dim)

    case NIF.nif_add_to_index(ref, n, bin) do
      :ok -> :ok
      {:error, _} = err -> err
    end
  end

  @doc """
  Adds vectors with explicit IDs.

  ## Examples

      :ok = FaissEx.Index.add_with_ids(index, [[1.0, 2.0]], [100])
  """
  @spec add_with_ids(t(), vectors(), [integer()]) :: :ok | error()
  def add_with_ids(%__MODULE__{ref: ref, dim: dim}, data, ids)
      when is_list(data) and is_list(ids) do
    {n, data_bin} = Shared.encode_vectors!(data, dim)
    ids_bin = Shared.int64s_to_binary(ids)

    case NIF.nif_add_with_ids_to_index(ref, n, data_bin, ids_bin) do
      :ok -> :ok
      {:error, _} = err -> err
    end
  end

  @doc """
  Searches the index for the `k` nearest neighbors.

  Returns `{:ok, %{distances: [[float]], labels: [[integer]]}}`.

  When `k` exceeds the number of indexed vectors, FAISS pads the missing
  positions with label `-1` (and an implementation-defined distance).
  """
  @spec search(t(), vectors(), pos_integer()) ::
          {:ok, %{distances: [[float()]], labels: [[integer()]]}} | error()
  def search(%__MODULE__{ref: ref, dim: dim}, data, k) when is_list(data) do
    {n, data_bin} = Shared.encode_vectors!(data, dim)

    case NIF.nif_search_index(ref, n, data_bin, k) do
      {:ok, {distances_bin, labels_bin}} ->
        distances = distances_bin |> Shared.binary_to_floats() |> Enum.chunk_every(k)
        labels = labels_bin |> Shared.binary_to_int64s() |> Enum.chunk_every(k)
        {:ok, %{distances: distances, labels: labels}}

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Trains the index (required for IVF, PQ, etc.).
  """
  @spec train(t(), vectors()) :: :ok | error()
  def train(%__MODULE__{ref: ref, dim: dim}, data) when is_list(data) do
    {n, bin} = Shared.encode_vectors!(data, dim)

    case NIF.nif_train_index(ref, n, bin) do
      :ok -> :ok
      {:error, _} = err -> err
    end
  end

  @doc """
  Deep-copies the index.
  """
  @spec clone(t()) :: {:ok, t()} | error()
  def clone(%__MODULE__{} = index) do
    case NIF.nif_clone_index(index.ref) do
      {:ok, new_ref} ->
        {:ok, %__MODULE__{index | ref: new_ref, device: :host, gpu_resources_ref: nil}}

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Removes all vectors from the index.
  """
  @spec reset(t()) :: :ok | error()
  def reset(%__MODULE__{ref: ref}) do
    case NIF.nif_reset_index(ref) do
      :ok -> :ok
      {:error, _} = err -> err
    end
  end

  @doc """
  Reconstructs vectors at the given keys.

  `keys` is a list of integer indices.
  Returns `{:ok, [[float]]}` with shape n * dim.
  """
  @spec reconstruct(t(), [integer()]) :: {:ok, [[float()]]} | error()
  def reconstruct(%__MODULE__{ref: ref, dim: dim}, keys) when is_list(keys) do
    n = length(keys)
    keys_bin = Shared.int64s_to_binary(keys)

    case NIF.nif_reconstruct_batch(ref, n, keys_bin) do
      {:ok, result_bin} ->
        {:ok, result_bin |> Shared.binary_to_floats() |> Enum.chunk_every(dim)}

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Computes residuals: `xs - reconstruct(keys)`.

  Returns `{:ok, [[float]]}`.
  """
  @spec compute_residuals(t(), vectors(), [integer()]) :: {:ok, [[float()]]} | error()
  def compute_residuals(%__MODULE__{ref: ref, dim: dim}, xs, keys)
      when is_list(xs) and is_list(keys) do
    {n, data_bin} = Shared.encode_vectors!(xs, dim)
    keys_bin = Shared.int64s_to_binary(keys)

    case NIF.nif_compute_residuals(ref, n, data_bin, keys_bin) do
      {:ok, result_bin} ->
        {:ok, result_bin |> Shared.binary_to_floats() |> Enum.chunk_every(dim)}

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Writes the index to a file.
  """
  @spec to_file(t(), String.t()) :: :ok | error()
  def to_file(%__MODULE__{ref: ref}, path) do
    case NIF.nif_write_index(ref, to_binary(path)) do
      :ok -> :ok
      {:error, _} = err -> err
    end
  end

  @doc """
  Reads an index from a file.

  The returned struct has `description: nil` — the factory string used to
  build the index is not stored in the file format.

  ## Options

    * `:io_flags` - FAISS IO flags (default `0`)
  """
  @spec from_file(String.t(), keyword()) :: {:ok, t()} | error()
  def from_file(path, opts \\ []) do
    io_flags = Keyword.get(opts, :io_flags, 0)

    case NIF.nif_read_index(to_binary(path), io_flags) do
      {:ok, ref} ->
        {:ok, dim} = NIF.nif_get_index_dim(ref)

        {:ok,
         %__MODULE__{
           dim: dim,
           ref: ref,
           device: :host,
           description: nil
         }}

      {:error, _} = err ->
        err
    end
  end

  @doc "Returns the total number of indexed vectors."
  @spec ntotal(t()) :: {:ok, non_neg_integer()} | error()
  def ntotal(%__MODULE__{ref: ref}) do
    NIF.nif_get_index_ntotal(ref)
  end

  @doc "Returns the dimension of the index."
  @spec dim(t()) :: {:ok, pos_integer()} | error()
  def dim(%__MODULE__{ref: ref}) do
    NIF.nif_get_index_dim(ref)
  end

  @doc "Returns whether the index is trained."
  @spec trained?(t()) :: {:ok, boolean()} | error()
  def trained?(%__MODULE__{ref: ref}) do
    case NIF.nif_get_index_is_trained(ref) do
      {:ok, val} -> {:ok, val}
      {:error, _} = err -> err
    end
  end

  @doc """
  Moves a CPU index to GPU.

  Returns `{:error, "index already on GPU"}` if the index is already there.
  """
  @spec cpu_to_gpu(t(), non_neg_integer()) :: {:ok, t()} | error()
  def cpu_to_gpu(index, device_id \\ 0)

  def cpu_to_gpu(%__MODULE__{device: :host} = index, device_id) do
    case NIF.nif_index_cpu_to_gpu(index.ref, device_id) do
      {:ok, {gpu_res_ref, gpu_idx_ref}} ->
        {:ok,
         %__MODULE__{
           index
           | ref: gpu_idx_ref,
             device: {:cuda, device_id},
             gpu_resources_ref: gpu_res_ref
         }}

      {:error, _} = err ->
        err
    end
  end

  def cpu_to_gpu(%__MODULE__{device: {:cuda, _}}, _device_id) do
    {:error, "index already on GPU"}
  end

  @doc """
  Moves a GPU index back to CPU.
  """
  @spec gpu_to_cpu(t()) :: {:ok, t()} | error()
  def gpu_to_cpu(%__MODULE__{device: {:cuda, _}} = index) do
    case NIF.nif_index_gpu_to_cpu(index.ref) do
      {:ok, cpu_ref} ->
        {:ok,
         %__MODULE__{
           index
           | ref: cpu_ref,
             device: :host,
             gpu_resources_ref: nil
         }}

      {:error, _} = err ->
        err
    end
  end

  defp to_binary(str) when is_binary(str), do: str
  defp to_binary(str) when is_list(str), do: List.to_string(str)
end
