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

  ## Binary API

  Every hot-path function also works directly with raw binaries, skipping
  the list encoding/decoding entirely — the only per-call cost is a byte
  size check. Use it when vectors already arrive as binaries (an HTTP
  response body, a database blob, a file read):

      # 32-bit native-endian floats, row-major: vector 0, then vector 1, ...
      embeddings = HTTPClient.fetch_embeddings!(texts)
      :ok = FaissEx.Index.add(index, embeddings)

  The layout contract is `f32-native`, row-major, `dim` floats per vector;
  the number of vectors is inferred from `byte_size(binary)`. IDs and keys
  are 64-bit native-endian signed integers (`s64-native`).

    * Binary input — `add/2`, `train/2`, and `add_with_ids/3` accept a
      binary anywhere they accept lists (mixing forms is fine).
    * Binary output — `search_binary/3` and `reconstruct_binary/2` return
      raw result binaries instead of decoded lists. The list-returning
      functions never change shape based on input type.

  Decode result binaries with a comprehension when needed:

      for <<f::float-32-native <- distances_bin>>, do: f
      for <<i::signed-native-64 <- labels_bin>>, do: i

  ## Concurrency

  An index can be shared freely across processes. The NIF layer guards
  every index with a read-write lock:

    * Mutations — `add/2`, `add_with_ids/3`, `train/2`, `reset/1` — take
      the write lock and serialize against all other operations on the
      same index.
    * Read-only operations — `search/3`, `reconstruct/2`,
      `compute_residuals/3`, `to_file/2`, `clone/1` and the property
      getters — take the read lock, so concurrent searches still run in
      parallel.

  Heavy operations run on dirty schedulers and never block the BEAM's
  normal schedulers, even while waiting on the lock.
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

  @typedoc """
  Vectors as lists, or a row-major `f32-native` binary (`dim` floats per
  vector; the vector count is inferred from the byte size).
  """
  @type vector_data :: vectors() | binary()

  @typedoc "Vector IDs/keys as a list of integers, or an `s64-native` binary."
  @type id_data :: [integer()] | binary()

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

  Accepts a flat list of floats (single vector), a list of lists (batch),
  or a row-major `f32-native` binary (see the *Binary API* section in the
  moduledoc). Raises `ArgumentError` if any vector does not have `dim`
  elements, or if a binary's size is not a multiple of `dim * 4` bytes.

  ## Examples

      :ok = FaissEx.Index.add(index, [1.0, 2.0, 3.0])
      :ok = FaissEx.Index.add(index, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      :ok = FaissEx.Index.add(index, <<1.0::float-32-native, 2.0::float-32-native,
                                       3.0::float-32-native>>)
  """
  @spec add(t(), vector_data()) :: :ok | error()
  def add(%__MODULE__{ref: ref, dim: dim}, data) when is_list(data) or is_binary(data) do
    {n, bin} = encode_vector_data!(data, dim)

    case NIF.nif_add_to_index(ref, n, bin) do
      :ok -> :ok
      {:error, _} = err -> err
    end
  end

  @doc """
  Adds vectors with explicit IDs.

  Vectors and IDs each accept lists or raw binaries (`f32-native` vectors,
  `s64-native` IDs), in any combination. Returns
  `{:error, "binary size mismatch"}` when the ID count does not match the
  vector count.

  ## Examples

      :ok = FaissEx.Index.add_with_ids(index, [[1.0, 2.0]], [100])
      :ok = FaissEx.Index.add_with_ids(index, vectors_bin, <<100::signed-native-64>>)
  """
  @spec add_with_ids(t(), vector_data(), id_data()) :: :ok | error()
  def add_with_ids(%__MODULE__{ref: ref, dim: dim}, data, ids)
      when (is_list(data) or is_binary(data)) and (is_list(ids) or is_binary(ids)) do
    {n, data_bin} = encode_vector_data!(data, dim)
    {_n_ids, ids_bin} = encode_id_data!(ids)

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
        distances = Shared.binary_to_float_rows(distances_bin, k)
        labels = Shared.binary_to_int64_rows(labels_bin, k)
        {:ok, %{distances: distances, labels: labels}}

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Searches the index and returns raw result binaries — the zero-decode
  counterpart of `search/3`.

  `queries` is a row-major `f32-native` binary; the query count `n` is
  inferred from its byte size. Returns
  `{:ok, %{n: n, distances: distances, labels: labels}}` where `distances`
  is `n * k` `f32-native` floats and `labels` is `n * k` `s64-native`
  integers, both row-major (`k` results per query). See the *Binary API*
  section in the moduledoc for decoding examples.
  """
  @spec search_binary(t(), binary(), pos_integer()) ::
          {:ok, %{n: non_neg_integer(), distances: binary(), labels: binary()}} | error()
  def search_binary(%__MODULE__{ref: ref, dim: dim}, queries, k) when is_binary(queries) do
    {n, queries_bin} = encode_vector_data!(queries, dim)

    case NIF.nif_search_index(ref, n, queries_bin, k) do
      {:ok, {distances_bin, labels_bin}} ->
        {:ok, %{n: n, distances: distances_bin, labels: labels_bin}}

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Trains the index (required for IVF, PQ, etc.).

  Accepts the same list or binary vector forms as `add/2`.
  """
  @spec train(t(), vector_data()) :: :ok | error()
  def train(%__MODULE__{ref: ref, dim: dim}, data) when is_list(data) or is_binary(data) do
    {n, bin} = encode_vector_data!(data, dim)

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
    {n, keys_bin} = Shared.encode_ids!(keys)

    case NIF.nif_reconstruct_batch(ref, n, keys_bin) do
      {:ok, result_bin} ->
        {:ok, Shared.binary_to_float_rows(result_bin, dim)}

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Reconstructs vectors at the given keys as one raw binary — the
  zero-decode counterpart of `reconstruct/2`.

  `keys` is a list of integers or an `s64-native` binary. Returns
  `{:ok, binary}` where the binary is `n * dim` `f32-native` floats,
  row-major — useful for bulk export without decode-then-chunk overhead.
  """
  @spec reconstruct_binary(t(), id_data()) :: {:ok, binary()} | error()
  def reconstruct_binary(%__MODULE__{ref: ref}, keys)
      when is_list(keys) or is_binary(keys) do
    {n, keys_bin} = encode_id_data!(keys)

    case NIF.nif_reconstruct_batch(ref, n, keys_bin) do
      {:ok, result_bin} -> {:ok, result_bin}
      {:error, _} = err -> err
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
        {:ok, Shared.binary_to_float_rows(result_bin, dim)}

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

  defp encode_vector_data!(data, dim) when is_list(data), do: Shared.encode_vectors!(data, dim)

  defp encode_vector_data!(data, dim) when is_binary(data) do
    row_size = dim * 4

    case rem(byte_size(data), row_size) do
      0 ->
        {div(byte_size(data), row_size), data}

      _ ->
        raise ArgumentError,
              "binary has #{byte_size(data)} bytes, expected a multiple of " <>
                "#{row_size} (dim #{dim} * 4 bytes per f32)"
    end
  end

  defp encode_id_data!(ids) when is_list(ids), do: Shared.encode_ids!(ids)

  defp encode_id_data!(ids) when is_binary(ids) do
    case rem(byte_size(ids), 8) do
      0 ->
        {div(byte_size(ids), 8), ids}

      _ ->
        raise ArgumentError,
              "ids binary has #{byte_size(ids)} bytes, expected a multiple of " <>
                "8 (one s64 per id)"
    end
  end
end
