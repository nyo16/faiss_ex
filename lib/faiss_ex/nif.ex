defmodule FaissEx.NIF do
  @moduledoc false
  @on_load :load_nif

  defp load_nif do
    path = :filename.join(:code.priv_dir(:faiss_ex), ~c"libfaiss_ex")
    :erlang.load_nif(path, 0)
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
end
