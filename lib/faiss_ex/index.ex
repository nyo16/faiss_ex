defmodule FaissEx.Index do
  @moduledoc """
  Create, populate, search, and manage FAISS indexes.

  An index holds a collection of vectors and supports nearest-neighbor search.
  Vectors are passed as flat lists (single vector), lists of lists (batch),
  or row-major `f32-native` binaries (see *Binary API* below).

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

    * Binary input — `add/2`, `add_with_ids/3`, `train/2`, `search/3`,
      `compute_residuals/3`, `reconstruct_binary/2` (keys), and
      `FaissEx.Clustering.train/2` / `FaissEx.Clustering.get_cluster_assignment/2`
      accept a binary anywhere they accept lists (mixing forms is fine).
    * Binary output — `search_binary/3`, `reconstruct_binary/2`,
      `compute_residuals_binary/3`, and `FaissEx.Clustering.get_centroids_binary/1`
      return raw result binaries instead of decoded lists. The
      list-returning functions never change shape based on input type.

  A binary may be a sub-binary at any byte offset (`binary_part/3`, a
  pattern-match tail); the NIF copies it to an aligned buffer if needed.

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

  Waiting never occupies the dirty-CPU pool. A call that finds its index's
  lock taken (say, a `search/3` while `train/2` holds the write lock)
  moves to a dirty-IO scheduler and waits there, so dirty-CPU threads stay
  free for work on other indexes. Many callers waiting on one busy index
  therefore share the dirty-IO pool (10 threads by default) with file I/O;
  raise it with `+SDio N` in `vm.args` (or `ELIXIR_ERL_OPTIONS="+SDio N"`)
  if you expect more concurrent waiters than that.

  FAISS itself parallelizes each call with OpenMP; see
  `FaissEx.set_omp_threads/1` for tuning many concurrent small calls.
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
    * `:device` - `:host` (default) or `{:cuda, device_id}` with a
      non-negative integer `device_id`

  Unknown options raise `ArgumentError`.

  ## Errors

  Caller mistakes that this library can detect — an unknown option,
  `:metric`, or `:device` — raise `ArgumentError`. A `dim` or `description`
  that FAISS rejects returns `{:error, reason}`, because those routinely
  come from configuration.

  ## Factory descriptions are trusted input

  `description` is parsed by FAISS's index factory and decides how much
  memory the index allocates — an `"IVF1048576,Flat"` or `"IVF4M_HNSW32"`
  quantizer can reserve gigabytes before any vector is added. Treat it like
  code, not data: if it is derived from user input, map the input onto an
  allow-list of descriptions you have sized. Descriptions longer than 256
  bytes return `{:error, "description too long"}`.

  ## Examples

      iex> {:ok, index} = FaissEx.Index.new(128, "Flat")
      iex> index.dim
      128
  """
  @spec new(pos_integer(), String.t(), keyword()) :: {:ok, t()} | error()
  def new(dim, description, opts \\ []) do
    opts = Keyword.validate!(opts, metric: :l2, device: :host)

    # Values mirror FaissMetricType in FAISS's c_api/Index_c.h; the C side
    # static-asserts them, so a FAISS bump that renumbers the enum fails to
    # compile instead of silently searching the wrong metric.
    metric_int =
      case opts[:metric] do
        :l2 ->
          1

        :inner_product ->
          0

        other ->
          raise ArgumentError,
                "invalid metric #{inspect(other)}, expected :l2 or :inner_product"
      end

    device = validate_device!(opts[:device])

    with {:ok, ref} <- NIF.nif_new_index(dim, description, metric_int) do
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

  defp validate_device!(:host), do: :host

  defp validate_device!({:cuda, id} = device) when is_integer(id) and id >= 0, do: device

  defp validate_device!(other) do
    raise ArgumentError,
          "invalid device #{inspect(other)}, expected :host or {:cuda, device_id} " <>
            "with a non-negative integer device_id"
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
    {n, bin} = Shared.encode_vector_data!(data, dim)
    NIF.nif_add_to_index(ref, n, bin)
  end

  @doc """
  Adds vectors with explicit IDs.

  Vectors and IDs each accept lists or raw binaries (`f32-native` vectors,
  `s64-native` IDs), in any combination. Raises `ArgumentError` when the
  ID count does not match the vector count. Indexes that do not support
  explicit IDs (a plain `"Flat"`; wrap it as `"IDMap,Flat"`) return
  `{:error, reason}`.

  ## Examples

      :ok = FaissEx.Index.add_with_ids(index, [[1.0, 2.0]], [100])
      :ok = FaissEx.Index.add_with_ids(index, vectors_bin, <<100::signed-native-64>>)
  """
  @spec add_with_ids(t(), vector_data(), id_data()) :: :ok | error()
  def add_with_ids(%__MODULE__{ref: ref, dim: dim}, data, ids)
      when (is_list(data) or is_binary(data)) and (is_list(ids) or is_binary(ids)) do
    {n, data_bin} = Shared.encode_vector_data!(data, dim)
    {n_ids, ids_bin} = Shared.encode_id_data!(ids)

    if n_ids != n do
      raise ArgumentError, "got #{n_ids} ids for #{n} vectors, expected one id per vector"
    end

    NIF.nif_add_with_ids_to_index(ref, n, data_bin, ids_bin)
  end

  @doc """
  Searches the index for the `k` nearest neighbors.

  Queries accept the same list or binary forms as `add/2`. Returns
  `{:ok, %{distances: [[float]], labels: [[integer]]}}` — one row of `k`
  results per query — whatever the input form.

  When `k` exceeds the number of indexed vectors, FAISS pads the missing
  positions with label `-1` (and an implementation-defined distance).

  `k` sizes the result: the NIF allocates `n * k * 12` bytes (an `f32`
  distance and an `s64` label per result) before searching. Cap `k` when it
  comes from a caller. `k` must be a positive integer.
  """
  @spec search(t(), vector_data(), pos_integer()) ::
          {:ok, %{distances: [[float()]], labels: [[integer()]]}} | error()
  def search(%__MODULE__{ref: ref, dim: dim}, data, k) when is_list(data) or is_binary(data) do
    {n, data_bin} = Shared.encode_vector_data!(data, dim)

    with {:ok, {distances_bin, labels_bin}} <- NIF.nif_search_index(ref, n, data_bin, k) do
      distances = Shared.binary_to_float_rows(distances_bin, k)
      labels = Shared.binary_to_int64_rows(labels_bin, k)
      {:ok, %{distances: distances, labels: labels}}
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
  section in the moduledoc for decoding examples, and `search/3` for the
  `k` sizing note.
  """
  @spec search_binary(t(), binary(), pos_integer()) ::
          {:ok, %{n: non_neg_integer(), distances: binary(), labels: binary()}} | error()
  def search_binary(%__MODULE__{ref: ref, dim: dim}, queries, k) when is_binary(queries) do
    {n, queries_bin} = Shared.encode_vector_data!(queries, dim)

    with {:ok, {distances_bin, labels_bin}} <- NIF.nif_search_index(ref, n, queries_bin, k) do
      {:ok, %{n: n, distances: distances_bin, labels: labels_bin}}
    end
  end

  @doc """
  Trains the index (required for IVF, PQ, etc.).

  Accepts the same list or binary vector forms as `add/2`.

  An empty training set goes to FAISS, which decides: indexes that need no
  training (`"Flat"`, `"HNSW32"`) return `:ok`, while indexes that do
  (`"IVF4,Flat"`, `"PQ8"`) return `{:error, reason}` — k-means needs at
  least as many points as clusters.
  """
  @spec train(t(), vector_data()) :: :ok | error()
  def train(%__MODULE__{ref: ref, dim: dim}, data) when is_list(data) or is_binary(data) do
    {n, bin} = Shared.encode_vector_data!(data, dim)
    NIF.nif_train_index(ref, n, bin)
  end

  @doc """
  Deep-copies the index.
  """
  @spec clone(t()) :: {:ok, t()} | error()
  def clone(%__MODULE__{} = index) do
    with {:ok, new_ref} <- NIF.nif_clone_index(index.ref) do
      {:ok, %__MODULE__{index | ref: new_ref, device: :host, gpu_resources_ref: nil}}
    end
  end

  @doc """
  Removes all vectors from the index.
  """
  @spec reset(t()) :: :ok | error()
  def reset(%__MODULE__{ref: ref}) do
    NIF.nif_reset_index(ref)
  end

  @doc """
  Reconstructs vectors at the given keys.

  `keys` is a list of integer indices. Returns `{:ok, [[float]]}` with shape
  n * dim, or `{:error, "key out of range"}` for a negative key or a
  contiguous range that cannot address any index.
  """
  @spec reconstruct(t(), [integer()]) :: {:ok, [[float()]]} | error()
  def reconstruct(%__MODULE__{ref: ref, dim: dim}, keys) when is_list(keys) do
    {n, keys_bin} = Shared.encode_ids!(keys)

    with {:ok, result_bin} <- NIF.nif_reconstruct_batch(ref, n, keys_bin) do
      {:ok, Shared.binary_to_float_rows(result_bin, dim)}
    end
  end

  @doc """
  Reconstructs vectors at the given keys as one raw binary — the
  zero-decode counterpart of `reconstruct/2`.

  `keys` is a list of integers or an `s64-native` binary. Returns
  `{:ok, binary}` where the binary is `n * dim` `f32-native` floats,
  row-major — useful for bulk export without decode-then-chunk overhead.
  Returns `{:error, "key out of range"}` for a negative key.
  """
  @spec reconstruct_binary(t(), id_data()) :: {:ok, binary()} | error()
  def reconstruct_binary(%__MODULE__{ref: ref}, keys)
      when is_list(keys) or is_binary(keys) do
    {n, keys_bin} = Shared.encode_id_data!(keys)
    NIF.nif_reconstruct_batch(ref, n, keys_bin)
  end

  @doc """
  Computes residuals: `xs - reconstruct(keys)`.

  `xs` accepts the same list or binary forms as `add/2`, `keys` a list of
  integers or an `s64-native` binary. Returns `{:ok, [[float]]}`, or
  `{:error, "key out of range"}` for a negative key.

  Raises `ArgumentError` if a vector does not have `dim` elements, if a key
  is not an integer, or if the key count does not match the vector count.
  """
  @spec compute_residuals(t(), vector_data(), id_data()) :: {:ok, [[float()]]} | error()
  def compute_residuals(%__MODULE__{dim: dim} = index, xs, keys)
      when (is_list(xs) or is_binary(xs)) and (is_list(keys) or is_binary(keys)) do
    with {:ok, result_bin} <- compute_residuals_binary(index, xs, keys) do
      {:ok, Shared.binary_to_float_rows(result_bin, dim)}
    end
  end

  @doc """
  Computes residuals and returns them as one raw binary — the zero-decode
  counterpart of `compute_residuals/3`.

  Takes the same arguments and raises on the same mistakes. Returns
  `{:ok, binary}` with `n * dim` `f32-native` floats, row-major.
  """
  @spec compute_residuals_binary(t(), vector_data(), id_data()) :: {:ok, binary()} | error()
  def compute_residuals_binary(%__MODULE__{ref: ref, dim: dim}, xs, keys)
      when (is_list(xs) or is_binary(xs)) and (is_list(keys) or is_binary(keys)) do
    {n, data_bin} = Shared.encode_vector_data!(xs, dim)
    {n_keys, keys_bin} = Shared.encode_id_data!(keys)

    if n_keys != n do
      raise ArgumentError,
            "got #{n_keys} keys for #{n} vectors, expected one key per vector"
    end

    NIF.nif_compute_residuals(ref, n, data_bin, keys_bin)
  end

  @doc """
  Writes the index to a file.
  """
  @spec to_file(t(), String.t()) :: :ok | error()
  def to_file(%__MODULE__{ref: ref}, path) do
    NIF.nif_write_index(ref, path)
  end

  @doc """
  Reads an index from a file.

  The returned struct has `description: nil` — the factory string used to
  build the index is not stored in the file format.

  ## Options

    * `:io_flags` - FAISS IO flags, a bitwise OR of the values below
      (default `0`). Unknown options raise `ArgumentError`.

  | Flag | Value | Effect |
  |------|-------|--------|
  | `IO_FLAG_READ_ONLY` | `2` | Open on-disk inverted lists read-only |
  | `IO_FLAG_ONDISK_SAME_DIR` | `4` | Resolve on-disk inverted-list files in the index file's directory, ignoring the stored path |
  | `IO_FLAG_SKIP_IVF_DATA` | `8` | Load an IVF index's structure without its inverted lists |

  ## Trust boundary

  Only load files you trust with the default flags. An index file can
  describe an on-disk IVF index (`OnDiskInvertedLists`) whose inverted
  lists live in a *separate* file named by a path stored inside the index
  file. Loading it memory-maps that path — read-write unless
  `IO_FLAG_READ_ONLY` is set — so a crafted file can make this process
  map, read, and modify an arbitrary file it has access to.

  For files from an untrusted source, pass `io_flags: 6`
  (`IO_FLAG_READ_ONLY ||| IO_FLAG_ONDISK_SAME_DIR`): the stored path is
  replaced by the index file's own directory and the mapping is read-only.
  `IO_FLAG_READ_ONLY` alone is not enough — it still maps whatever path
  the file names, read-only.
  """
  @spec from_file(String.t(), keyword()) :: {:ok, t()} | error()
  def from_file(path, opts \\ []) do
    io_flags = opts |> Keyword.validate!(io_flags: 0) |> Keyword.fetch!(:io_flags)

    with {:ok, ref} <- NIF.nif_read_index(path, io_flags) do
      # Hard match on purpose: we just opened this index, so a failure here
      # is an invariant violation, not a caller error.
      {:ok, dim} = NIF.nif_get_index_dim(ref)

      {:ok,
       %__MODULE__{
         dim: dim,
         ref: ref,
         device: :host,
         description: nil
       }}
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
    NIF.nif_get_index_is_trained(ref)
  end

  @doc """
  Moves a CPU index to GPU.

  `device_id` must be a non-negative integer. Returns
  `{:error, "index already on GPU"}` if the index is already there.
  """
  @spec cpu_to_gpu(t(), non_neg_integer()) :: {:ok, t()} | error()
  def cpu_to_gpu(index, device_id \\ 0)

  def cpu_to_gpu(%__MODULE__{device: :host} = index, device_id)
      when is_integer(device_id) and device_id >= 0 do
    with {:ok, {gpu_res_ref, gpu_idx_ref}} <- NIF.nif_index_cpu_to_gpu(index.ref, device_id) do
      {:ok,
       %__MODULE__{
         index
         | ref: gpu_idx_ref,
           device: {:cuda, device_id},
           gpu_resources_ref: gpu_res_ref
       }}
    end
  end

  def cpu_to_gpu(%__MODULE__{device: {:cuda, _}}, device_id)
      when is_integer(device_id) and device_id >= 0 do
    {:error, "index already on GPU"}
  end

  @doc """
  Moves a GPU index back to CPU.

  Returns `{:error, "index already on host"}` if the index is already there.
  """
  @spec gpu_to_cpu(t()) :: {:ok, t()} | error()
  def gpu_to_cpu(%__MODULE__{device: {:cuda, _}} = index) do
    with {:ok, cpu_ref} <- NIF.nif_index_gpu_to_cpu(index.ref) do
      {:ok,
       %__MODULE__{
         index
         | ref: cpu_ref,
           device: :host,
           gpu_resources_ref: nil
       }}
    end
  end

  def gpu_to_cpu(%__MODULE__{device: :host}) do
    {:error, "index already on host"}
  end
end
