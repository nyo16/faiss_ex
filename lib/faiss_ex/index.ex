defmodule FaissEx.Index do
  @moduledoc """
  FAISS index operations for vector similarity search.
  """

  alias FaissEx.NIF
  alias FaissEx.Shared

  defstruct [:dim, :ref, :device, :description, :gpu_resources_ref]

  @type t :: %__MODULE__{
          dim: pos_integer(),
          ref: reference(),
          device: :host | {:cuda, non_neg_integer()},
          description: String.t(),
          gpu_resources_ref: reference() | nil
        }

  @metric_map %{l2: 1, inner_product: 0}

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
  def new(dim, description, opts \\ []) do
    metric = Keyword.get(opts, :metric, :l2)
    device = Keyword.get(opts, :device, :host)

    metric_int = Map.fetch!(@metric_map, metric)
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

  Accepts an Nx tensor of type `{:f, 32}` with shape `{n, dim}` or `{dim}`.
  """
  def add(%__MODULE__{ref: ref, dim: dim}, tensor) do
    tensor = tensor |> Nx.as_type({:f, 32}) |> ensure_2d(dim)
    Shared.validate_type!(tensor, {:f, 32})
    {n, ^dim} = Nx.shape(tensor)
    data = Nx.to_binary(tensor)

    case NIF.nif_add_to_index(ref, n, data) do
      :ok -> :ok
      {:error, _} = err -> err
    end
  end

  @doc """
  Adds vectors with explicit IDs.

  `ids` must be an `{:s, 64}` tensor of shape `{n}`.
  """
  def add_with_ids(%__MODULE__{ref: ref, dim: dim} = _index, tensor, ids) do
    tensor = tensor |> Nx.as_type({:f, 32}) |> ensure_2d(dim)
    Shared.validate_type!(tensor, {:f, 32})
    ids = Nx.as_type(ids, {:s, 64})
    Shared.validate_type!(ids, {:s, 64})

    {n, ^dim} = Nx.shape(tensor)
    data = Nx.to_binary(tensor)
    ids_bin = Nx.to_binary(ids)

    case NIF.nif_add_with_ids_to_index(ref, n, data, ids_bin) do
      :ok -> :ok
      {:error, _} = err -> err
    end
  end

  @doc """
  Searches the index for the `k` nearest neighbors.

  Returns `%{distances: tensor, labels: tensor}` where:
    * `distances` has shape `{n, k}` and type `{:f, 32}`
    * `labels` has shape `{n, k}` and type `{:s, 64}`
  """
  def search(%__MODULE__{ref: ref, dim: dim}, tensor, k) do
    tensor = tensor |> Nx.as_type({:f, 32}) |> ensure_2d(dim)
    Shared.validate_type!(tensor, {:f, 32})
    {n, ^dim} = Nx.shape(tensor)
    data = Nx.to_binary(tensor)

    case NIF.nif_search_index(ref, n, data, k) do
      {:ok, {distances_bin, labels_bin}} ->
        distances = distances_bin |> Nx.from_binary({:f, 32}) |> Nx.reshape({n, k})
        labels = labels_bin |> Nx.from_binary({:s, 64}) |> Nx.reshape({n, k})
        {:ok, %{distances: distances, labels: labels}}

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Trains the index (required for IVF, PQ, etc.).
  """
  def train(%__MODULE__{ref: ref, dim: dim} = _index, tensor) do
    tensor = tensor |> Nx.as_type({:f, 32}) |> ensure_2d(dim)
    Shared.validate_type!(tensor, {:f, 32})
    {n, ^dim} = Nx.shape(tensor)
    data = Nx.to_binary(tensor)

    case NIF.nif_train_index(ref, n, data) do
      :ok -> :ok
      {:error, _} = err -> err
    end
  end

  @doc """
  Deep-copies the index.
  """
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
  def reset(%__MODULE__{ref: ref}) do
    case NIF.nif_reset_index(ref) do
      :ok -> :ok
      {:error, _} = err -> err
    end
  end

  @doc """
  Reconstructs vectors at the given keys.

  `keys` is an `{:s, 64}` tensor of indices.
  Returns an `{:f, 32}` tensor of shape `{n, dim}`.
  """
  def reconstruct(%__MODULE__{ref: ref, dim: dim}, keys) do
    keys = Nx.as_type(keys, {:s, 64})
    Shared.validate_type!(keys, {:s, 64})
    keys_flat = Nx.reshape(keys, {Nx.size(keys)})
    n = Nx.size(keys_flat)
    keys_bin = Nx.to_binary(keys_flat)

    case NIF.nif_reconstruct_batch(ref, n, keys_bin) do
      {:ok, result_bin} ->
        {:ok, result_bin |> Nx.from_binary({:f, 32}) |> Nx.reshape({n, dim})}

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Computes residuals: `xs - reconstruct(keys)`.
  """
  def compute_residuals(%__MODULE__{ref: ref, dim: dim}, xs, keys) do
    xs = xs |> Nx.as_type({:f, 32}) |> ensure_2d(dim)
    Shared.validate_type!(xs, {:f, 32})
    keys = Nx.as_type(keys, {:s, 64})
    Shared.validate_type!(keys, {:s, 64})

    {n, ^dim} = Nx.shape(xs)
    data = Nx.to_binary(xs)
    keys_bin = Nx.to_binary(Nx.reshape(keys, {n}))

    case NIF.nif_compute_residuals(ref, n, data, keys_bin) do
      {:ok, result_bin} ->
        {:ok, result_bin |> Nx.from_binary({:f, 32}) |> Nx.reshape({n, dim})}

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Writes the index to a file.
  """
  def to_file(%__MODULE__{ref: ref}, path) do
    case NIF.nif_write_index(ref, to_binary(path)) do
      :ok -> :ok
      {:error, _} = err -> err
    end
  end

  @doc """
  Reads an index from a file.

  ## Options

    * `:io_flags` - FAISS IO flags (default `0`)
  """
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
           description: "from_file"
         }}

      {:error, _} = err ->
        err
    end
  end

  @doc "Returns the total number of indexed vectors."
  def ntotal(%__MODULE__{ref: ref}) do
    NIF.nif_get_index_ntotal(ref)
  end

  @doc "Returns the dimension of the index."
  def dim(%__MODULE__{ref: ref}) do
    NIF.nif_get_index_dim(ref)
  end

  @doc "Returns whether the index is trained."
  def trained?(%__MODULE__{ref: ref}) do
    case NIF.nif_get_index_is_trained(ref) do
      {:ok, val} -> {:ok, val}
      {:error, _} = err -> err
    end
  end

  @doc """
  Moves a CPU index to GPU.
  """
  def cpu_to_gpu(%__MODULE__{device: :host} = index, device_id \\ 0) do
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

  @doc """
  Moves a GPU index back to CPU.
  """
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

  # Ensures a 1D tensor {dim} becomes {1, dim}
  defp ensure_2d(tensor, dim) do
    case Nx.shape(tensor) do
      {^dim} ->
        Nx.reshape(tensor, {1, dim})

      {_, ^dim} ->
        tensor

      shape ->
        raise ArgumentError,
              "expected tensor of shape {#{dim}} or {n, #{dim}}, got #{inspect(shape)}"
    end
  end

  defp to_binary(str) when is_binary(str), do: str
  defp to_binary(str) when is_list(str), do: List.to_string(str)
end
