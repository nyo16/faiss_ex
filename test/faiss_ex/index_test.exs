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

      assert [[0]] = Enum.take(labels, 1)
      assert [_, [2]] = labels
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
  end

  describe "to_file/2 and from_file/1" do
    test "round-trips an index through a file" do
      {:ok, index} = Index.new(4, "Flat")
      :ok = Index.add(index, [[1.0, 2.0, 3.0, 4.0]])

      path = Path.join(System.tmp_dir!(), "faiss_ex_test_#{:rand.uniform(100_000)}.index")

      try do
        assert :ok = Index.to_file(index, path)
        assert {:ok, loaded} = Index.from_file(path)
        assert {:ok, 1} = Index.ntotal(loaded)
        assert loaded.dim == 4
      after
        File.rm(path)
      end
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

      vectors = for _ <- 1..100, do: for(_ <- 1..8, do: :rand.uniform())
      :ok = Index.add(index, vectors)

      assert {:ok, 100} = Index.ntotal(index)

      {:ok, %{distances: distances, labels: labels}} = Index.search(index, hd(vectors), 5)
      assert length(hd(distances)) == 5
      assert length(hd(labels)) == 5
      assert hd(hd(labels)) == 0
    end
  end

  describe "edge cases" do
    test "search returns -1 labels when index is empty" do
      {:ok, index} = Index.new(4, "Flat")
      {:ok, %{labels: [labels_row]}} = Index.search(index, [0.0, 0.0, 0.0, 0.0], 3)
      assert labels_row == [-1, -1, -1]
    end

    test "search with n=0 returns empty results via NIF" do
      {:ok, index} = Index.new(4, "Flat")
      :ok = Index.add(index, [[1.0, 0.0, 0.0, 0.0]])

      {:ok, {distances_bin, labels_bin}} =
        FaissEx.NIF.nif_search_index(index.ref, 0, <<>>, 2)

      assert byte_size(distances_bin) == 0
      assert byte_size(labels_bin) == 0
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

    test "file round-trip preserves search results" do
      {:ok, index} = Index.new(3, "Flat")

      :ok =
        Index.add(index, [
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]
        ])

      path = Path.join(System.tmp_dir!(), "faiss_ex_roundtrip_#{:rand.uniform(100_000)}.index")

      try do
        :ok = Index.to_file(index, path)
        {:ok, loaded} = Index.from_file(path)

        {:ok, %{labels: orig_labels}} = Index.search(index, [1.0, 0.0, 0.0], 3)
        {:ok, %{labels: loaded_labels}} = Index.search(loaded, [1.0, 0.0, 0.0], 3)

        assert orig_labels == loaded_labels
      after
        File.rm(path)
      end
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

  describe "get_num_gpus" do
    test "returns a non-negative integer" do
      {:ok, count} = FaissEx.NIF.nif_get_num_gpus()
      assert is_integer(count) and count >= 0
    end
  end
end
