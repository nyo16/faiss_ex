defmodule FaissEx.IndexTest do
  use ExUnit.Case, async: true

  alias FaissEx.Index

  describe "new/3" do
    test "creates a flat L2 index" do
      assert {:ok, %Index{dim: 128, device: :host}} = Index.new(128, "Flat")
    end

    test "creates a flat inner product index" do
      assert {:ok, %Index{dim: 64}} = Index.new(64, "Flat", metric: :inner_product)
    end

    test "creates an IVF index" do
      assert {:ok, %Index{dim: 32}} = Index.new(32, "IVF8,Flat")
    end
  end

  describe "add/2 and ntotal/1" do
    test "adds a single vector" do
      {:ok, index} = Index.new(4, "Flat")
      assert :ok = Index.add(index, [1.0, 2.0, 3.0, 4.0])
      assert {:ok, 1} = Index.ntotal(index)
    end

    test "adds a batch of vectors" do
      {:ok, index} = Index.new(4, "Flat")

      vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
      ]

      assert :ok = Index.add(index, vectors)
      assert {:ok, 3} = Index.ntotal(index)
    end
  end

  describe "search/3" do
    test "finds exact nearest neighbors in flat L2 index" do
      {:ok, index} = Index.new(4, "Flat")

      vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ]

      :ok = Index.add(index, vectors)

      {:ok, %{distances: distances, labels: labels}} =
        Index.search(index, [1.0, 0.0, 0.0, 0.0], 2)

      assert [[label_0, _label_1]] = labels
      assert [[dist_0, _dist_1]] = distances

      assert label_0 == 0
      assert_in_delta dist_0, 0.0, 1.0e-6
    end

    test "searches with multiple queries" do
      {:ok, index} = Index.new(3, "Flat")

      :ok =
        Index.add(index, [
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]
        ])

      {:ok, %{labels: labels}} =
        Index.search(index, [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], 1)

      assert [[0], [2]] = labels
    end
  end

  describe "train/2 with IVF index" do
    test "trains and searches IVF index" do
      {:ok, index} = Index.new(8, "IVF4,Flat")

      training_data = for _ <- 1..200, do: for(_ <- 1..8, do: :rand.uniform())

      assert :ok = Index.train(index, training_data)
      assert {:ok, true} = Index.trained?(index)

      :ok = Index.add(index, training_data)
      assert {:ok, 200} = Index.ntotal(index)

      query = [for(_ <- 1..8, do: :rand.uniform())]
      assert {:ok, %{distances: _, labels: _}} = Index.search(index, query, 5)
    end
  end

  describe "clone/1" do
    test "creates an independent copy" do
      {:ok, index} = Index.new(4, "Flat")
      :ok = Index.add(index, [1.0, 2.0, 3.0, 4.0])

      {:ok, cloned} = Index.clone(index)
      assert {:ok, 1} = Index.ntotal(cloned)

      :ok = Index.add(index, [5.0, 6.0, 7.0, 8.0])
      assert {:ok, 2} = Index.ntotal(index)
      assert {:ok, 1} = Index.ntotal(cloned)
    end
  end

  describe "reset/1" do
    test "clears all vectors" do
      {:ok, index} = Index.new(4, "Flat")
      :ok = Index.add(index, [1.0, 2.0, 3.0, 4.0])
      assert {:ok, 1} = Index.ntotal(index)

      :ok = Index.reset(index)
      assert {:ok, 0} = Index.ntotal(index)
    end
  end

  describe "reconstruct/2" do
    test "reconstructs vectors from flat index" do
      {:ok, index} = Index.new(4, "Flat")

      vectors = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0]
      ]

      :ok = Index.add(index, vectors)

      {:ok, reconstructed} = Index.reconstruct(index, [0, 1])

      assert length(reconstructed) == 2
      assert length(hd(reconstructed)) == 4

      Enum.zip(List.flatten(vectors), List.flatten(reconstructed))
      |> Enum.each(fn {expected, actual} ->
        assert_in_delta expected, actual, 1.0e-6
      end)
    end

    test "reconstructs vectors with non-contiguous keys (per-key fallback)" do
      {:ok, index} = Index.new(4, "Flat")

      vectors = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
      ]

      :ok = Index.add(index, vectors)

      # gapped and descending keys must not take the reconstruct_n fast path
      {:ok, [row2, row0]} = Index.reconstruct(index, [2, 0])

      Enum.zip(Enum.at(vectors, 2) ++ Enum.at(vectors, 0), row2 ++ row0)
      |> Enum.each(fn {expected, actual} ->
        assert_in_delta expected, actual, 1.0e-6
      end)
    end
  end

  describe "to_file/2 and from_file/1" do
    @tag :tmp_dir
    test "round-trips an index through a file", %{tmp_dir: tmp_dir} do
      {:ok, index} = Index.new(4, "Flat")
      :ok = Index.add(index, [[1.0, 2.0, 3.0, 4.0]])

      path = Path.join(tmp_dir, "roundtrip.index")

      assert :ok = Index.to_file(index, path)
      assert {:ok, loaded} = Index.from_file(path)
      assert {:ok, 1} = Index.ntotal(loaded)
      assert loaded.dim == 4
    end
  end

  describe "dim/1 and trained?/1" do
    test "returns index properties" do
      {:ok, index} = Index.new(64, "Flat")
      assert {:ok, 64} = Index.dim(index)
      assert {:ok, true} = Index.trained?(index)
    end
  end

  describe "add_with_ids/3" do
    test "adds vectors with custom IDs" do
      {:ok, index} = Index.new(4, "IDMap,Flat")

      vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
      ]

      assert :ok = Index.add_with_ids(index, vectors, [100, 200])
      assert {:ok, 2} = Index.ntotal(index)
    end
  end

  describe "GPU operations" do
    @tag :cuda
    test "moves index to GPU and back" do
      {:ok, index} = Index.new(4, "Flat")
      :ok = Index.add(index, [1.0, 2.0, 3.0, 4.0])

      {:ok, gpu_index} = Index.cpu_to_gpu(index, 0)
      assert gpu_index.device == {:cuda, 0}
      assert gpu_index.gpu_resources_ref != nil

      {:ok, cpu_index} = Index.gpu_to_cpu(gpu_index)
      assert cpu_index.device == :host
      assert {:ok, 1} = Index.ntotal(cpu_index)
    end
  end

  describe "compute_residuals/3" do
    test "computes residuals with lists" do
      {:ok, index} = Index.new(3, "Flat")
      :ok = Index.add(index, [[1.0, 2.0, 3.0]])

      {:ok, residuals} = Index.compute_residuals(index, [[1.0, 2.0, 3.0]], [0])

      residuals
      |> List.flatten()
      |> Enum.each(fn r -> assert_in_delta r, 0.0, 1.0e-6 end)
    end

    test "residual against a different key is the vector difference" do
      # Key 0 is byte-order invariant and yields an all-zero residual, so the
      # identity case above passes even if keys are byte-swapped or ignored.
      # This one only passes if the key actually selects vector 1.
      {:ok, index} = Index.new(3, "Flat")
      :ok = Index.add(index, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

      assert {:ok, [[a, b, c]]} = Index.compute_residuals(index, [[1.0, 2.0, 3.0]], [1])

      Enum.each([a, b, c], &assert_in_delta(&1, -3.0, 1.0e-6))
    end

    test "a negative key is rejected instead of reading out of bounds" do
      {:ok, index} = Index.new(3, "Flat")
      :ok = Index.add(index, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

      assert {:error, "key out of range"} =
               Index.compute_residuals(index, [[0.0, 0.0, 0.0]], [-1])
    end

    test "raises when the key count does not match the vector count" do
      {:ok, index} = Index.new(3, "Flat")
      :ok = Index.add(index, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

      assert_raise ArgumentError,
                   "got 1 keys for 2 vectors, expected one key per vector",
                   fn ->
                     Index.compute_residuals(index, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [0])
                   end
    end

    test "raises when a key is not an integer" do
      {:ok, index} = Index.new(3, "Flat")
      :ok = Index.add(index, [[1.0, 2.0, 3.0]])

      assert_raise ArgumentError, "id 0 is not an integer: :nope", fn ->
        Index.compute_residuals(index, [[1.0, 2.0, 3.0]], [:nope])
      end
    end
  end

  describe "inner product metric" do
    test "returns higher scores for more similar vectors" do
      {:ok, index} = Index.new(3, "Flat", metric: :inner_product)

      :ok =
        Index.add(index, [
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.9, 0.1, 0.0]
        ])

      {:ok, %{labels: [[label | _]]}} = Index.search(index, [1.0, 0.0, 0.0], 1)
      assert label == 0
    end
  end

  describe "HNSW index" do
    test "creates and searches HNSW index" do
      {:ok, index} = Index.new(8, "HNSW32")
      assert {:ok, true} = Index.trained?(index)

      vectors = for i <- 0..99, do: List.duplicate(i * 1.0, 8)
      :ok = Index.add(index, vectors)

      assert {:ok, 100} = Index.ntotal(index)

      {:ok, %{distances: distances, labels: labels}} = Index.search(index, hd(vectors), 5)
      assert length(hd(distances)) == 5
      assert length(hd(labels)) == 5
      assert hd(hd(labels)) == 0
    end
  end

  describe "GPU error paths (non-CUDA build)" do
    @tag :no_cuda
    test "cpu_to_gpu without GPU support returns error" do
      {:ok, index} = Index.new(4, "Flat")
      assert {:error, "GPU support not compiled"} = Index.cpu_to_gpu(index, 0)
    end

    test "cpu_to_gpu on an index already on GPU returns error" do
      {:ok, %Index{} = index} = Index.new(4, "Flat")
      gpu_index = %Index{index | device: {:cuda, 0}}
      assert {:error, "index already on GPU"} = Index.cpu_to_gpu(gpu_index, 0)
    end

    test "gpu_to_cpu on an index already on the host returns error" do
      {:ok, index} = Index.new(4, "Flat")
      assert {:error, "index already on host"} = Index.gpu_to_cpu(index)
    end
  end

  describe "file I/O error paths" do
    test "to_file into a nonexistent directory returns error" do
      {:ok, index} = Index.new(4, "Flat")
      assert {:error, msg} = Index.to_file(index, "/nonexistent_dir_faiss_ex/x.index")
      assert msg =~ "could not open /nonexistent_dir_faiss_ex/x.index for writing"
    end

    test "from_file on a nonexistent path returns error" do
      assert {:error, msg} = Index.from_file("/nonexistent_dir_faiss_ex/missing.index")
      assert msg =~ "could not open /nonexistent_dir_faiss_ex/missing.index for reading"
    end

    @tag :tmp_dir
    test "from_file on a file that is not an index returns error", %{tmp_dir: tmp_dir} do
      path = Path.join(tmp_dir, "corrupt.index")
      File.write!(path, "not a faiss index at all")

      assert {:error, msg} = Index.from_file(path)
      assert msg =~ "not recognized"
    end

    @tag :tmp_dir
    test "from_file with io_flags: 2 (read-only) round-trips", %{tmp_dir: tmp_dir} do
      {:ok, index} = Index.new(4, "Flat")
      :ok = Index.add(index, [[1.0, 2.0, 3.0, 4.0]])
      path = Path.join(tmp_dir, "ro.index")
      :ok = Index.to_file(index, path)

      assert {:ok, loaded} = Index.from_file(path, io_flags: 2)
      assert {:ok, %{labels: [[0]]}} = Index.search(loaded, [1.0, 2.0, 3.0, 4.0], 1)
    end

    test "non-binary path and non-integer io_flags are rejected" do
      {:ok, index} = Index.new(4, "Flat")
      assert {:error, "invalid arguments"} = Index.to_file(index, ~c"/x")
      assert {:error, "invalid arguments"} = Index.from_file("/x.index", io_flags: :x)
    end

    test "unknown from_file options raise" do
      assert_raise ArgumentError, ~r/unknown keys \[:io_flag\]/, fn ->
        Index.from_file("/x.index", io_flag: 2)
      end
    end

    @tag :tmp_dir
    test "from_file sets description to nil", %{tmp_dir: tmp_dir} do
      {:ok, index} = Index.new(4, "Flat")
      :ok = Index.add(index, [[1.0, 2.0, 3.0, 4.0]])
      path = Path.join(tmp_dir, "desc.index")

      :ok = Index.to_file(index, path)
      {:ok, loaded} = Index.from_file(path)
      assert loaded.description == nil
    end
  end

  describe "edge cases" do
    test "search with k greater than ntotal pads labels with -1" do
      {:ok, index} = Index.new(3, "Flat")
      :ok = Index.add(index, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

      {:ok, %{labels: [labels_row]}} = Index.search(index, [1.0, 0.0, 0.0], 5)

      assert length(labels_row) == 5
      assert Enum.take(labels_row, 2) == [0, 1]
      assert Enum.drop(labels_row, 2) == [-1, -1, -1]
    end

    test "search returns -1 labels when index is empty" do
      {:ok, index} = Index.new(4, "Flat")
      {:ok, %{labels: [labels_row]}} = Index.search(index, [0.0, 0.0, 0.0, 0.0], 3)
      assert labels_row == [-1, -1, -1]
    end

    test "adding after reset works" do
      {:ok, index} = Index.new(3, "Flat")
      :ok = Index.add(index, [1.0, 0.0, 0.0])
      assert {:ok, 1} = Index.ntotal(index)

      :ok = Index.reset(index)
      assert {:ok, 0} = Index.ntotal(index)

      :ok = Index.add(index, [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
      assert {:ok, 2} = Index.ntotal(index)
    end

    @tag :tmp_dir
    test "file round-trip preserves search results", %{tmp_dir: tmp_dir} do
      {:ok, index} = Index.new(3, "Flat")

      :ok =
        Index.add(index, [
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]
        ])

      path = Path.join(tmp_dir, "search-roundtrip.index")

      :ok = Index.to_file(index, path)
      {:ok, loaded} = Index.from_file(path)

      {:ok, %{labels: orig_labels}} = Index.search(index, [1.0, 0.0, 0.0], 3)
      {:ok, %{labels: loaded_labels}} = Index.search(loaded, [1.0, 0.0, 0.0], 3)

      assert orig_labels == loaded_labels
    end

    test "clone preserves search results independently" do
      {:ok, index} = Index.new(3, "Flat")
      :ok = Index.add(index, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

      {:ok, cloned} = Index.clone(index)

      :ok = Index.add(cloned, [0.0, 0.0, 1.0])
      assert {:ok, 2} = Index.ntotal(index)
      assert {:ok, 3} = Index.ntotal(cloned)

      {:ok, %{labels: [[orig_label]]}} = Index.search(index, [1.0, 0.0, 0.0], 1)
      {:ok, %{labels: [[clone_label]]}} = Index.search(cloned, [1.0, 0.0, 0.0], 1)
      assert orig_label == 0
      assert clone_label == 0
    end
  end

  describe "untrained index error paths" do
    test "add to an untrained IVF index returns error" do
      {:ok, index} = Index.new(8, "IVF4,Flat")
      assert {:ok, false} = Index.trained?(index)

      assert {:error, msg} = Index.add(index, List.duplicate(0.5, 8))
      assert msg =~ "train"
      assert {:ok, 0} = Index.ntotal(index)
    end

    test "search on an untrained IVF index returns error" do
      {:ok, index} = Index.new(8, "IVF4,Flat")

      assert {:error, msg} = Index.search(index, List.duplicate(0.5, 8), 1)
      assert msg =~ "train"
    end
  end

  describe "resource lifecycle" do
    test "index created in another process survives its owner's death" do
      parent = self()

      {pid, mref} =
        spawn_monitor(fn ->
          {:ok, index} = Index.new(4, "Flat")
          :ok = Index.add(index, [1.0, 2.0, 3.0, 4.0])
          send(parent, {:index, index})
        end)

      assert_receive {:index, index}
      assert_receive {:DOWN, ^mref, :process, ^pid, :normal}

      # The creating process is gone; the resource must still be alive and
      # usable from here (refcounted, not owner-bound)
      assert {:ok, 1} = Index.ntotal(index)
      assert {:ok, %{labels: [[0]]}} = Index.search(index, [1.0, 2.0, 3.0, 4.0], 1)
    end

    test "dropping index and clustering resources under churn does not crash the VM" do
      vectors =
        List.duplicate([1.0, 0.0, 0.0, 0.0], 50) ++
          List.duplicate([0.0, 0.0, 0.0, 1.0], 50)

      # Each task's resources become unreachable when it exits, so their NIF
      # destructors run as the dead heaps are collected
      for _ <- 1..25 do
        task =
          Task.async(fn ->
            {:ok, index} = Index.new(4, "Flat")
            :ok = Index.add(index, vectors)
            {:ok, _} = Index.search(index, hd(vectors), 3)

            {:ok, clustering} = FaissEx.Clustering.new(4, 2)
            {:ok, _trained} = FaissEx.Clustering.train(clustering, vectors)
            :ok
          end)

        assert Task.await(task) == :ok
      end

      # The NIF must still be fully functional after the churn
      {:ok, index} = Index.new(4, "Flat")
      :ok = Index.add(index, [1.0, 0.0, 0.0, 0.0])
      assert {:ok, 1} = Index.ntotal(index)
    end
  end

  describe "input validation" do
    test "raises on invalid metric" do
      assert_raise ArgumentError,
                   "invalid metric :cosine, expected :l2 or :inner_product",
                   fn -> Index.new(4, "Flat", metric: :cosine) end
    end

    test "raises on ragged batch with the offending row index" do
      {:ok, index} = Index.new(3, "Flat")

      assert_raise ArgumentError, "row 1 has 2 elements, expected 3", fn ->
        Index.add(index, [[1.0, 2.0, 3.0], [1.0, 2.0]])
      end
    end

    test "raises on short single vector" do
      {:ok, index} = Index.new(3, "Flat")

      assert_raise ArgumentError, "row 0 has 2 elements, expected 3", fn ->
        Index.add(index, [1.0, 2.0])
      end
    end

    test "raises on non-number element with row index" do
      {:ok, index} = Index.new(3, "Flat")

      assert_raise ArgumentError, ~s(row 0 contains a non-number: "x"), fn ->
        Index.add(index, [[1.0, "x", 3.0]])
      end
    end

    test "raises on non-list row in a batch" do
      {:ok, index} = Index.new(3, "Flat")

      assert_raise ArgumentError, ~s(row 1 is not a list of numbers: "bad"), fn ->
        Index.add(index, [[1.0, 2.0, 3.0], "bad"])
      end
    end

    test "adding an empty batch is a no-op" do
      {:ok, index} = Index.new(4, "Flat")
      assert :ok = Index.add(index, [])
      assert {:ok, 0} = Index.ntotal(index)
    end

    test "add_with_ids raises when the id count does not match the vector count" do
      {:ok, index} = Index.new(4, "IDMap,Flat")

      assert_raise ArgumentError, "got 1 ids for 2 vectors, expected one id per vector", fn ->
        Index.add_with_ids(index, [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], [100])
      end

      assert_raise ArgumentError, "got 1 ids for 0 vectors, expected one id per vector", fn ->
        Index.add_with_ids(index, [], [1])
      end

      assert {:ok, 0} = Index.ntotal(index)
    end

    test "rejects non-positive dimension" do
      assert {:error, "dim must be positive"} = Index.new(0, "Flat")
      assert {:error, "dim must be positive"} = Index.new(-4, "Flat")
    end

    test "rejects NUL byte in factory description" do
      assert {:error, "string contains NUL byte"} = Index.new(4, <<"Flat", 0, "X">>)
    end

    test "rejects NUL byte in file paths" do
      {:ok, index} = Index.new(4, "Flat")

      assert {:error, "string contains NUL byte"} =
               Index.to_file(index, <<"/tmp/bad", 0, "path">>)

      assert {:error, "string contains NUL byte"} =
               Index.from_file(<<"/tmp/bad", 0, "path">>)
    end
  end

  describe "binary API" do
    alias FaissEx.Shared

    test "add with a binary matches add with lists" do
      vectors = for i <- 0..9, do: for(j <- 1..4, do: i * 10.0 + j)

      {:ok, list_index} = Index.new(4, "Flat")
      {:ok, bin_index} = Index.new(4, "Flat")

      :ok = Index.add(list_index, vectors)
      :ok = Index.add(bin_index, encode(vectors))

      assert {:ok, 10} = Index.ntotal(bin_index)

      query = hd(vectors)
      {:ok, list_results} = Index.search(list_index, query, 3)
      {:ok, bin_results} = Index.search(bin_index, query, 3)
      assert list_results == bin_results
    end

    test "train with a binary matches train with lists" do
      data = for _ <- 1..200, do: for(_ <- 1..8, do: :rand.uniform())

      {:ok, index} = Index.new(8, "IVF4,Flat")
      assert :ok = Index.train(index, encode(data))
      assert {:ok, true} = Index.trained?(index)

      :ok = Index.add(index, encode(data))
      assert {:ok, 200} = Index.ntotal(index)
    end

    test "search_binary returns raw binaries matching search/3" do
      vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

      {:ok, index} = Index.new(3, "Flat")
      :ok = Index.add(index, vectors)

      queries = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
      {:ok, list_results} = Index.search(index, queries, 2)

      {:ok, %{n: 2, distances: dist_bin, labels: labels_bin}} =
        Index.search_binary(index, encode(queries), 2)

      assert Shared.binary_to_float_rows(dist_bin, 2) == list_results.distances
      assert Shared.binary_to_int64_rows(labels_bin, 2) == list_results.labels
    end

    test "reconstruct_binary matches reconstruct/2 for list and binary keys" do
      vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

      {:ok, index} = Index.new(3, "Flat")
      :ok = Index.add(index, vectors)

      {:ok, from_lists} = Index.reconstruct(index, [0, 1])

      {:ok, bin_from_list_keys} = Index.reconstruct_binary(index, [0, 1])
      {:ok, bin_from_bin_keys} = Index.reconstruct_binary(index, encode_ids([0, 1]))

      assert bin_from_list_keys == bin_from_bin_keys
      assert Shared.binary_to_float_rows(bin_from_list_keys, 3) == from_lists
    end

    test "add_with_ids accepts mixed list/binary forms" do
      vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
      ids = [100, 200]
      vectors_bin = encode(vectors)
      ids_bin = encode_ids(ids)

      combos = [
        {vectors, ids_bin},
        {vectors_bin, ids},
        {vectors_bin, ids_bin}
      ]

      for {data, id_arg} <- combos do
        {:ok, index} = Index.new(4, "IDMap,Flat")
        assert :ok = Index.add_with_ids(index, data, id_arg)
        assert {:ok, 2} = Index.ntotal(index)
        assert {:ok, %{labels: [[100]]}} = Index.search(index, hd(vectors), 1)
      end
    end

    test "raises on a binary not divisible by dim * 4 with actual sizes" do
      {:ok, index} = Index.new(4, "Flat")

      assert_raise ArgumentError,
                   "binary has 13 bytes, expected a multiple of 16 (dim 4 * 4 bytes per f32)",
                   fn -> Index.add(index, <<0::size(13 * 8)>>) end
    end

    test "raises on an ids binary not divisible by 8" do
      {:ok, index} = Index.new(4, "IDMap,Flat")
      vectors_bin = encode([[1.0, 0.0, 0.0, 0.0]])

      assert_raise ArgumentError,
                   "ids binary has 3 bytes, expected a multiple of 8 (one s64 per id)",
                   fn -> Index.add_with_ids(index, vectors_bin, <<1, 2, 3>>) end
    end

    test "adding an empty binary is a no-op" do
      {:ok, index} = Index.new(4, "Flat")
      assert :ok = Index.add(index, <<>>)
      assert {:ok, 0} = Index.ntotal(index)
    end

    test "add_with_ids raises on mismatched binary sizes" do
      {:ok, index} = Index.new(4, "IDMap,Flat")
      vectors_bin = encode([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

      assert_raise ArgumentError, "got 1 ids for 2 vectors, expected one id per vector", fn ->
        Index.add_with_ids(index, vectors_bin, <<100::signed-native-64>>)
      end
    end

    test "search_binary on an empty query binary returns empty results" do
      {:ok, index} = Index.new(4, "Flat")
      :ok = Index.add(index, [[1.0, 0.0, 0.0, 0.0]])

      assert {:ok, %{n: 0, distances: <<>>, labels: <<>>}} =
               Index.search_binary(index, <<>>, 2)
    end
  end

  describe "num_gpus" do
    test "FaissEx.num_gpus/0 returns a non-negative integer" do
      count = FaissEx.num_gpus()
      assert is_integer(count) and count >= 0
    end
  end

  describe "reconstruction key range" do
    # FAISS bounds-checks only the upper end: IndexFlat::reconstruct tests
    # `key < ntotal`, which every negative key passes, and reconstruct_n's
    # `i0 + ni <= ntotal` signed-overflows near INT64_MAX and passes too.
    # Both read out of bounds and return heap contents as vector data, so the
    # NIF rejects them before calling FAISS.
    @max_int64 9_223_372_036_854_775_807

    setup do
      {:ok, index} = Index.new(3, "Flat")
      :ok = Index.add(index, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      %{index: index}
    end

    test "rejects a negative key on the batched path", %{index: index} do
      assert {:error, "key out of range"} = Index.reconstruct(index, [-1])
    end

    test "rejects a negative key on the per-key path", %{index: index} do
      # Non-contiguous, so this goes through faiss_Index_reconstruct one key
      # at a time — the path with no bounds check at all.
      assert {:error, "key out of range"} = Index.reconstruct(index, [-1, -5])
    end

    test "rejects a key range whose end overflows int64", %{index: index} do
      assert {:error, "key out of range"} = Index.reconstruct(index, [@max_int64])

      assert {:error, "key out of range"} =
               Index.reconstruct(index, [@max_int64 - 1, @max_int64])
    end

    test "rejects a negative key through reconstruct_binary/2", %{index: index} do
      assert {:error, "key out of range"} = Index.reconstruct_binary(index, [-1])
    end

    test "still reconstructs valid keys, contiguous and not", %{index: index} do
      assert {:ok, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]} = Index.reconstruct(index, [0, 1])
      assert {:ok, [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]} = Index.reconstruct(index, [1, 0])
    end

    test "still lets FAISS reject a key past ntotal", %{index: index} do
      assert {:error, message} = Index.reconstruct(index, [5])
      assert message =~ "ntotal"
    end
  end

  describe "new/3 errors and options" do
    test "a description FAISS cannot parse returns its message" do
      assert {:error, msg} = Index.new(4, "NotAnIndex")
      assert msg =~ "could not parse index string NotAnIndex"
    end

    test "a non-binary description is rejected" do
      assert {:error, "invalid arguments"} = Index.new(4, ~c"Flat")
    end

    test "descriptions are capped at 256 bytes" do
      assert {:error, "description too long"} = Index.new(4, String.duplicate("F", 257))

      # 256 bytes passes the cap and reaches the factory parser
      assert {:error, msg} = Index.new(4, String.duplicate("F", 256))
      assert msg =~ "could not parse index string"
    end

    test "unknown options raise" do
      assert_raise ArgumentError, ~r/unknown keys \[:metrc\]/, fn ->
        Index.new(4, "Flat", metrc: :l2)
      end
    end

    test "invalid devices raise before any index is created" do
      for device <- [:gpu, {:cuda, -1}, {:cuda, 0.0}, :cuda] do
        assert_raise ArgumentError, ~r/invalid device/, fn ->
          Index.new(4, "Flat", device: device)
        end
      end
    end

    test "cpu_to_gpu rejects a negative device id" do
      {:ok, index} = Index.new(4, "Flat")
      assert_raise FunctionClauseError, fn -> Index.cpu_to_gpu(index, -1) end
    end
  end

  describe "add_with_ids/3 on an index without id support" do
    test "returns FAISS's error" do
      {:ok, index} = Index.new(4, "Flat")
      assert {:error, msg} = Index.add_with_ids(index, [[1.0, 0.0, 0.0, 0.0]], [7])
      assert msg =~ "add_with_ids not implemented"
      assert {:ok, 0} = Index.ntotal(index)
    end
  end

  describe "search/3 contracts" do
    setup do
      {:ok, index} = Index.new(4, "Flat")
      :ok = Index.add(index, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
      %{index: index}
    end

    test "k must be a positive integer", %{index: index} do
      query = [1.0, 0.0, 0.0, 0.0]
      assert {:error, "k must be positive"} = Index.search(index, query, 0)
      assert {:error, "k must be positive"} = Index.search_binary(index, encode([query]), 0)
      assert {:error, "invalid arguments"} = Index.search(index, query, 1.5)
    end

    test "a k whose result size overflows is rejected before allocating", %{index: index} do
      assert {:error, "size overflow"} =
               Index.search(index, [1.0, 0.0, 0.0, 0.0], Integer.pow(2, 62))
    end

    test "queries of the wrong dimension raise", %{index: index} do
      assert_raise ArgumentError, "row 0 has 3 elements, expected 4", fn ->
        Index.search(index, [1.0, 0.0, 0.0], 1)
      end

      assert_raise ArgumentError, ~r/binary has 12 bytes, expected a multiple of 16/, fn ->
        Index.search_binary(index, encode([[1.0, 0.0, 0.0]]), 1)
      end
    end

    test "no queries return no rows", %{index: index} do
      assert {:ok, %{distances: [], labels: []}} = Index.search(index, [], 3)
    end

    test "binary queries return decoded lists", %{index: index} do
      queries = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]

      assert {:ok, %{labels: [[0, 1], [1, 0]]} = from_binary} =
               Index.search(index, encode(queries), 2)

      assert {:ok, from_binary} == Index.search(index, queries, 2)
    end
  end

  describe "train/2 contracts" do
    test "IVF with fewer points than clusters returns FAISS's error" do
      {:ok, index} = Index.new(4, "IVF4,Flat")
      assert {:error, msg} = Index.train(index, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
      assert msg =~ "Number of training points (2)"
      assert {:ok, false} = Index.trained?(index)
    end

    test "an empty training set errors when training is required" do
      {:ok, index} = Index.new(4, "IVF4,Flat")
      assert {:error, msg} = Index.train(index, [])
      assert msg =~ "Number of training points (0)"
      assert {:ok, false} = Index.trained?(index)
    end

    test "an empty training set is a no-op when no training is needed" do
      {:ok, index} = Index.new(4, "Flat")
      assert :ok = Index.train(index, [])
      assert :ok = Index.train(index, <<>>)
    end

    test "training data of the wrong dimension raises" do
      {:ok, index} = Index.new(4, "IVF4,Flat")

      assert_raise ArgumentError, "row 0 has 3 elements, expected 4", fn ->
        Index.train(index, [[1.0, 0.0, 0.0]])
      end
    end
  end

  # Every case below aborted the whole VM before FAISS calls were kept out of
  # OpenMP parallel regions (an exception inside one cannot reach the C API's
  # catch): they must come back as error tuples.
  describe "reconstruction errors never abort the VM" do
    test "an index type without reconstruct" do
      {:ok, index} = Index.new(4, "IDMap,Flat")
      :ok = Index.add_with_ids(index, [[1.0, 0.0, 0.0, 0.0]], [7])

      assert {:error, msg} = Index.reconstruct(index, [0])
      assert msg =~ "reconstruct not implemented"
      assert {:error, _} = Index.compute_residuals(index, [[1.0, 0.0, 0.0, 0.0]], [0])
    end

    test "an id map with a hole in a contiguous key run" do
      {:ok, index} = Index.new(4, "IDMap2,Flat")
      :ok = Index.add_with_ids(index, [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]], [10, 12])

      assert {:error, _} = Index.reconstruct(index, [10, 11, 12])

      assert Index.reconstruct(index, [12, 10]) ==
               {:ok, [[2.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]}
    end

    test "more than 1000 contiguous keys past ntotal on a non-Flat index" do
      {:ok, index} = Index.new(4, "HNSW32")
      :ok = Index.add(index, for(i <- 0..1999, do: [i * 1.0, 0.0, 0.0, 0.0]))

      assert {:error, _} = Index.reconstruct_binary(index, Enum.to_list(1500..2600))
    end

    test "a residual key past ntotal" do
      {:ok, index} = Index.new(4, "Flat")
      :ok = Index.add(index, [[1.0, 0.0, 0.0, 0.0]])

      assert {:error, msg} = Index.compute_residuals(index, [[0.0, 0.0, 0.0, 0.0]], [5])
      assert msg =~ "ntotal"
    end

    test "an IVF index without a direct map, non-contiguous keys" do
      {:ok, index} = Index.new(4, "IVF2,Flat")
      data = for i <- 0..99, do: [i * 1.0, 0.0, 0.0, 1.0]
      :ok = Index.train(index, data)
      :ok = Index.add(index, data)

      assert {:error, msg} = Index.reconstruct(index, [5, 2])
      assert is_binary(msg)

      # Contiguous keys still take FAISS's batched path, which needs no
      # direct map
      assert Index.reconstruct(index, [2, 3]) ==
               {:ok, [[2.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0]]}
    end

    test "no keys reconstruct no rows" do
      {:ok, index} = Index.new(4, "Flat")
      assert {:ok, []} = Index.reconstruct(index, [])
    end
  end

  describe "compute_residuals contracts" do
    setup do
      {:ok, index} = Index.new(2, "Flat")
      :ok = Index.add(index, [[1.0, 1.0], [2.0, 4.0], [-3.0, 0.5]])
      %{index: index}
    end

    test "a batch equals xs minus the reconstructed keys", %{index: index} do
      xs = [[0.0, 0.0], [10.0, 10.0], [1.0, -1.0]]
      keys = [2, 0, 1]

      {:ok, reconstructed} = Index.reconstruct(index, keys)

      expected =
        Enum.zip_with(xs, reconstructed, fn x, r -> Enum.zip_with(x, r, &(&1 - &2)) end)

      assert {:ok, ^expected} = Index.compute_residuals(index, xs, keys)
    end

    test "binary inputs and the binary variant agree with lists", %{index: index} do
      xs = [[0.0, 0.0], [10.0, 10.0], [1.0, -1.0]]
      keys = [2, 0, 1]
      {:ok, from_lists} = Index.compute_residuals(index, xs, keys)

      assert {:ok, ^from_lists} = Index.compute_residuals(index, encode(xs), encode_ids(keys))
      assert {:ok, bin} = Index.compute_residuals_binary(index, encode(xs), keys)
      assert bin == encode(from_lists)
    end

    test "vectors of the wrong dimension raise", %{index: index} do
      assert_raise ArgumentError, "row 0 has 3 elements, expected 2", fn ->
        Index.compute_residuals(index, [[1.0, 2.0, 3.0]], [0])
      end
    end
  end

  describe "GPU calls on a non-CUDA build" do
    @describetag :no_cuda

    test "creating on a CUDA device reports the missing support" do
      assert {:error, "GPU support not compiled"} = Index.new(4, "Flat", device: {:cuda, 0})
    end

    test "cpu_to_gpu with the default device reports the missing support" do
      {:ok, index} = Index.new(4, "Flat")
      assert {:error, "GPU support not compiled"} = Index.cpu_to_gpu(index)
    end

    test "gpu_to_cpu on a struct claiming a CUDA device reports the missing support" do
      {:ok, %Index{} = index} = Index.new(4, "Flat")
      forged = %Index{index | device: {:cuda, 0}}
      assert {:error, "GPU support not compiled"} = Index.gpu_to_cpu(forged)
    end

    test "num_gpus is zero" do
      assert FaissEx.num_gpus() == 0
    end
  end

  describe "misaligned binary input" do
    # binary_part/3 at offset 1 yields a sub-binary whose data pointer is
    # misaligned for f32/s64; the NIF must copy it rather than hand FAISS a
    # misaligned pointer, and results must match the aligned input.
    setup do
      vectors = for i <- 0..7, do: [i * 1.0, i * 2.0, 0.0, 1.0]
      %{vectors: vectors, payload: encode(vectors)}
    end

    test "add, search_binary, and reconstruct_binary match aligned input", ctx do
      {:ok, aligned} = Index.new(4, "Flat")
      {:ok, misaligned} = Index.new(4, "Flat")

      :ok = Index.add(aligned, ctx.payload)
      :ok = Index.add(misaligned, misalign(ctx.payload))

      queries = encode(Enum.take(ctx.vectors, 3))
      {:ok, expected} = Index.search_binary(aligned, queries, 2)
      assert {:ok, ^expected} = Index.search_binary(misaligned, misalign(queries), 2)

      keys = encode_ids([5, 2, 7])
      {:ok, expected_rows} = Index.reconstruct_binary(aligned, keys)
      assert {:ok, ^expected_rows} = Index.reconstruct_binary(misaligned, misalign(keys))

      assert expected_rows ==
               encode([Enum.at(ctx.vectors, 5), Enum.at(ctx.vectors, 2), Enum.at(ctx.vectors, 7)])
    end
  end

  defp misalign(bin), do: binary_part(<<0, bin::binary>>, 1, byte_size(bin))

  # Reference encoders, module-level so the scoping matches reality: `defp`
  # inside a `describe` compiles into the module regardless. Kept local rather
  # than in `test/support/` — the project has no `elixirc_paths` override and
  # these are one-line comprehensions.
  defp encode(rows) do
    for f <- List.flatten(rows), into: <<>>, do: <<f::float-32-native>>
  end

  defp encode_ids(ids) do
    for i <- ids, into: <<>>, do: <<i::signed-native-64>>
  end
end
