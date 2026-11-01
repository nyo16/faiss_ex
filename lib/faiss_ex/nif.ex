defmodule FaissEx.NIF do
  @moduledoc false
  @on_load :load_nif

  defp load_nif do
    case :code.priv_dir(:faiss_ex) do
      {:error, :bad_name} ->
        {:error,
         {:load_failed, ~c"could not find priv dir for :faiss_ex (is the application loaded?)"}}

      priv_dir ->
        path = :filename.join(priv_dir, ~c"libfaiss_ex")
        :erlang.load_nif(path, omp_threads_config())
    end
  end

  # Initial OpenMP thread count for FAISS calls; FaissEx.set_omp_threads/1
  # changes it at runtime. Validated here because a bad value would otherwise
  # surface as an opaque NIF load failure.
  defp omp_threads_config do
    case Application.get_env(:faiss_ex, :omp_threads, 0) do
      n when is_integer(n) and n >= 0 and n <= 0x7FFFFFFF ->
        n

      other ->
        raise ArgumentError,
              "config :faiss_ex, omp_threads: expected a non-negative integer, got: #{inspect(other)}"
    end
  end

  # Index
  def nif_new_index(_dim, _description, _metric), do: :erlang.nif_error(:undef)
  def nif_clone_index(_ref), do: :erlang.nif_error(:undef)
  def nif_add_to_index(_ref, _n, _data), do: :erlang.nif_error(:undef)
  def nif_add_with_ids_to_index(_ref, _n, _data, _ids), do: :erlang.nif_error(:undef)
  def nif_search_index(_ref, _n, _data, _k), do: :erlang.nif_error(:undef)
  def nif_train_index(_ref, _n, _data), do: :erlang.nif_error(:undef)
  def nif_reset_index(_ref), do: :erlang.nif_error(:undef)
  def nif_reconstruct_batch(_ref, _n, _keys), do: :erlang.nif_error(:undef)
  def nif_compute_residuals(_ref, _n, _data, _keys), do: :erlang.nif_error(:undef)
  def nif_write_index(_ref, _path), do: :erlang.nif_error(:undef)
  def nif_read_index(_path, _io_flags), do: :erlang.nif_error(:undef)
  def nif_get_index_dim(_ref), do: :erlang.nif_error(:undef)
  def nif_get_index_ntotal(_ref), do: :erlang.nif_error(:undef)
  def nif_get_index_is_trained(_ref), do: :erlang.nif_error(:undef)

  # GPU
  def nif_index_cpu_to_gpu(_ref, _device), do: :erlang.nif_error(:undef)
  def nif_index_gpu_to_cpu(_ref), do: :erlang.nif_error(:undef)
  def nif_get_num_gpus, do: :erlang.nif_error(:undef)

  # Clustering
  def nif_new_clustering(_d, _k), do: :erlang.nif_error(:undef)
  def nif_train_clustering(_clust_ref, _n, _data, _idx_ref), do: :erlang.nif_error(:undef)
  def nif_get_clustering_centroids(_clust_ref), do: :erlang.nif_error(:undef)

  # Runtime
  def nif_set_omp_threads(_n), do: :erlang.nif_error(:undef)
end
