defmodule FaissEx.ClusteringTest do
  use ExUnit.Case, async: true

  alias FaissEx.Clustering

  @two_blobs List.duplicate([1.0, 0.0, 0.0, 0.0], 50) ++
               List.duplicate([0.0, 0.0, 0.0, 1.0], 50)

  describe "new/2" do
    test "creates a clustering object" do
      assert {:ok, %Clustering{d: 8, k: 4, trained?: false}} = Clustering.new(8, 4)
    end
  end

  describe "train/2 and get_centroids/1" do
    test "finds the centroid of each blob" do
      {:ok, clustering} = Clustering.new(4, 2)

      {:ok, trained} = Clustering.train(clustering, @two_blobs)
      assert trained.trained?

      {:ok, centroids} = Clustering.get_centroids(trained)
      assert_centroids(Enum.sort(centroids), [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    end

    test "accepts binary data and returns the same centroids as a binary" do
      {:ok, clustering} = Clustering.new(4, 2)
      {:ok, trained} = Clustering.train(clustering, encode(@two_blobs))

      {:ok, centroids} = Clustering.get_centroids(trained)
      {:ok, centroids_bin} = Clustering.get_centroids_binary(trained)

      assert byte_size(centroids_bin) == 2 * 4 * 4
      assert encode(centroids) == centroids_bin
      assert_centroids(Enum.sort(centroids), [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    end
  end

  describe "value semantics" do
    test "train leaves the input struct and its FAISS object untouched" do
      {:ok, clustering} = Clustering.new(4, 2)
      {:ok, trained} = Clustering.train(clustering, @two_blobs)

      assert trained.ref != clustering.ref
      assert %Clustering{trained?: false, index: nil} = clustering
      assert {:error, "no centroids available"} = Clustering.get_centroids(clustering)

      assert {:error, "clustering not trained"} =
               Clustering.get_cluster_assignment(clustering, [[1.0, 0.0, 0.0, 0.0]])
    end

    test "two trains from one struct are independent" do
      {:ok, clustering} = Clustering.new(2, 1)

      {:ok, near_origin} = Clustering.train(clustering, List.duplicate([0.0, 0.0], 10))
      {:ok, far_away} = Clustering.train(clustering, List.duplicate([5.0, 5.0], 10))

      # The second train must not overwrite the first result's centroids or
      # quantizer
      assert {:ok, [[a, b]]} = Clustering.get_centroids(near_origin)
      assert_in_delta a, 0.0, 1.0e-5
      assert_in_delta b, 0.0, 1.0e-5

      assert {:ok, [[c, d]]} = Clustering.get_centroids(far_away)
      assert_in_delta c, 5.0, 1.0e-5
      assert_in_delta d, 5.0, 1.0e-5

      assert {:ok, %{distances: [[dist]]}} =
               Clustering.get_cluster_assignment(near_origin, [[0.0, 0.0]])

      assert_in_delta dist, 0.0, 1.0e-5
    end
  end

  describe "get_cluster_assignment/2" do
    test "train populates the quantizer with k centroids and reuses it" do
      {:ok, clustering} = Clustering.new(4, 2)
      {:ok, trained} = Clustering.train(clustering, @two_blobs)

      # FAISS Clustering::train adds the k centroids to the quantizer index
      assert {:ok, 2} = FaissEx.Index.ntotal(trained.index)

      # The assigned label must match the argmin over centroid L2 distances
      {:ok, centroids} = Clustering.get_centroids(trained)
      query = [1.0, 0.0, 0.0, 0.0]

      expected_label =
        centroids
        |> Enum.map(fn c ->
          Enum.zip(c, query) |> Enum.map(fn {a, b} -> (a - b) * (a - b) end) |> Enum.sum()
        end)
        |> Enum.with_index()
        |> Enum.min()
        |> elem(1)

      {:ok, %{labels: [[label]]}} = Clustering.get_cluster_assignment(trained, [query])
      assert label == expected_label
    end

    test "accepts binary queries" do
      {:ok, clustering} = Clustering.new(4, 2)
      {:ok, trained} = Clustering.train(clustering, @two_blobs)
      queries = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

      assert Clustering.get_cluster_assignment(trained, encode(queries)) ==
               Clustering.get_cluster_assignment(trained, queries)
    end
  end

  describe "input validation" do
    test "rejects non-positive dimension" do
      assert {:error, "d must be positive"} = Clustering.new(0, 4)
      assert {:error, "d must be positive"} = Clustering.new(-1, 4)
    end

    test "rejects non-positive cluster count" do
      assert {:error, "k must be positive"} = Clustering.new(8, 0)
      assert {:error, "k must be positive"} = Clustering.new(8, -2)
    end

    test "rejects a non-integer dimension" do
      assert {:error, "invalid arguments"} = Clustering.new(4.0, 2)
    end

    test "rejects empty training data before allocating anything" do
      {:ok, clustering} = Clustering.new(4, 2)
      assert {:error, "n must be positive"} = Clustering.train(clustering, [])
      assert {:error, "n must be positive"} = Clustering.train(clustering, <<>>)
    end

    test "raises on training data of the wrong dimension" do
      {:ok, clustering} = Clustering.new(4, 2)

      assert_raise ArgumentError, "row 0 has 3 elements, expected 4", fn ->
        Clustering.train(clustering, [[1.0, 0.0, 0.0]])
      end
    end
  end

  describe "edge cases" do
    test "train with fewer points than clusters returns error and stays untrained" do
      {:ok, clustering} = Clustering.new(4, 10)
      data = for i <- 1..5, do: [i / 1, 0.0, 0.0, 0.0]

      assert {:error, msg} = Clustering.train(clustering, data)
      assert msg =~ "clusters"

      # The failed train allocated its own FAISS object; the input's is untouched
      assert {:error, "no centroids available"} = Clustering.get_centroids(clustering)
    end

    test "untrained clustering returns error for assignment" do
      {:ok, clustering} = Clustering.new(4, 2)

      assert {:error, "clustering not trained"} =
               Clustering.get_cluster_assignment(clustering, [[1.0, 0.0, 0.0, 0.0]])
    end

    test "get_centroids before train returns error" do
      {:ok, clustering} = Clustering.new(4, 2)
      assert {:error, "no centroids available"} = Clustering.get_centroids(clustering)
      assert {:error, "no centroids available"} = Clustering.get_centroids_binary(clustering)
    end
  end

  defp assert_centroids(actual, expected) do
    for {a_row, e_row} <- Enum.zip(actual, expected), {a, e} <- Enum.zip(a_row, e_row) do
      assert_in_delta a, e, 1.0e-5
    end
  end

  defp encode(rows) do
    for f <- List.flatten(rows), into: <<>>, do: <<f::float-32-native>>
  end
end
