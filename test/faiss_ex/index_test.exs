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
      vector = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: {:f, 32})

      assert :ok = Index.add(index, vector)
      assert {:ok, 1} = Index.ntotal(index)
    end

    test "adds a batch of vectors" do
      {:ok, index} = Index.new(4, "Flat")

      vectors =
        Nx.tensor(
          [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
          ],
          type: {:f, 32}
        )

      assert :ok = Index.add(index, vectors)
      assert {:ok, 3} = Index.ntotal(index)
    end
  end

  describe "search/3" do
    test "finds exact nearest neighbors in flat L2 index" do
      {:ok, index} = Index.new(4, "Flat")

      vectors =
        Nx.tensor(
          [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
          ],
          type: {:f, 32}
        )

      :ok = Index.add(index, vectors)

      query = Nx.tensor([1.0, 0.0, 0.0, 0.0], type: {:f, 32})
      {:ok, %{distances: distances, labels: labels}} = Index.search(index, query, 2)

      assert Nx.shape(distances) == {1, 2}
      assert Nx.shape(labels) == {1, 2}

      # First result should be vector 0 with distance 0
      assert Nx.to_number(labels[0][0]) == 0
      assert_in_delta Nx.to_number(distances[0][0]), 0.0, 1.0e-6
    end

    test "searches with multiple queries" do
      {:ok, index} = Index.new(3, "Flat")

      vectors =
        Nx.tensor(
          [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
          ],
          type: {:f, 32}
        )

      :ok = Index.add(index, vectors)

      queries =
        Nx.tensor(
          [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
          ],
          type: {:f, 32}
        )

      {:ok, %{labels: labels}} = Index.search(index, queries, 1)
      assert Nx.to_number(labels[0][0]) == 0
      assert Nx.to_number(labels[1][0]) == 2
    end
  end

  describe "train/2 with IVF index" do
    test "trains and searches IVF index" do
      {:ok, index} = Index.new(8, "IVF4,Flat")

      # Need enough training data (at least k * 39 for IVF)
      key = Nx.Random.key(42)
      {training_data, key} = Nx.Random.uniform(key, shape: {200, 8}, type: {:f, 32})

      assert :ok = Index.train(index, training_data)
      assert {:ok, true} = Index.trained?(index)

      :ok = Index.add(index, training_data)
      assert {:ok, 200} = Index.ntotal(index)

      {query, _key} = Nx.Random.uniform(key, shape: {1, 8}, type: {:f, 32})
      assert {:ok, %{distances: _, labels: _}} = Index.search(index, query, 5)
    end
  end

  describe "clone/1" do
    test "creates an independent copy" do
      {:ok, index} = Index.new(4, "Flat")
      vector = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: {:f, 32})
      :ok = Index.add(index, vector)

      {:ok, cloned} = Index.clone(index)
      assert {:ok, 1} = Index.ntotal(cloned)

      # Adding to original doesn't affect clone
      :ok = Index.add(index, Nx.tensor([5.0, 6.0, 7.0, 8.0], type: {:f, 32}))
      assert {:ok, 2} = Index.ntotal(index)
      assert {:ok, 1} = Index.ntotal(cloned)
    end
  end

  describe "reset/1" do
    test "clears all vectors" do
      {:ok, index} = Index.new(4, "Flat")
      :ok = Index.add(index, Nx.tensor([1.0, 2.0, 3.0, 4.0], type: {:f, 32}))
      assert {:ok, 1} = Index.ntotal(index)

      :ok = Index.reset(index)
      assert {:ok, 0} = Index.ntotal(index)
    end
  end

  describe "reconstruct/2" do
    test "reconstructs vectors from flat index" do
      {:ok, index} = Index.new(4, "Flat")

      vectors =
        Nx.tensor(
          [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0]
          ],
          type: {:f, 32}
        )

      :ok = Index.add(index, vectors)

      keys = Nx.tensor([0, 1], type: {:s, 64})
      {:ok, reconstructed} = Index.reconstruct(index, keys)

      assert Nx.shape(reconstructed) == {2, 4}
      assert Nx.to_number(Nx.subtract(vectors, reconstructed) |> Nx.sum()) |> abs() < 1.0e-6
    end
  end

  describe "to_file/2 and from_file/1" do
    test "round-trips an index through a file" do
      {:ok, index} = Index.new(4, "Flat")
      :ok = Index.add(index, Nx.tensor([[1.0, 2.0, 3.0, 4.0]], type: {:f, 32}))

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

      vectors =
        Nx.tensor(
          [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
          ],
          type: {:f, 32}
        )

      ids = Nx.tensor([100, 200], type: {:s, 64})
      assert :ok = Index.add_with_ids(index, vectors, ids)
      assert {:ok, 2} = Index.ntotal(index)
    end
  end

  describe "GPU operations" do
    @tag :cuda
    test "moves index to GPU and back" do
      {:ok, index} = Index.new(4, "Flat")
      :ok = Index.add(index, Nx.tensor([1.0, 2.0, 3.0, 4.0], type: {:f, 32}))

      {:ok, gpu_index} = Index.cpu_to_gpu(index, 0)
      assert gpu_index.device == {:cuda, 0}
      assert gpu_index.gpu_resources_ref != nil

      {:ok, cpu_index} = Index.gpu_to_cpu(gpu_index)
      assert cpu_index.device == :host
      assert {:ok, 1} = Index.ntotal(cpu_index)
    end
  end

  describe "get_num_gpus" do
    test "returns a non-negative integer" do
      {:ok, count} = FaissEx.NIF.nif_get_num_gpus()
      assert is_integer(count) and count >= 0
    end
  end
end
