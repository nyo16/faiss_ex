defmodule FaissEx.ClusteringTest do
  use ExUnit.Case, async: true

  alias FaissEx.Clustering

  describe "new/2" do
    test "creates a clustering object" do
      assert {:ok, %Clustering{d: 8, k: 4, trained?: false}} = Clustering.new(8, 4)
    end
  end

  describe "train/2 and get_centroids/1" do
    test "trains clustering and returns centroids" do
      {:ok, clustering} = Clustering.new(4, 2)

      data =
        List.duplicate([1.0, 0.0, 0.0, 0.0], 50) ++
          List.duplicate([0.0, 0.0, 0.0, 1.0], 50)

      {:ok, trained} = Clustering.train(clustering, data)
      assert trained.trained?

      {:ok, centroids} = Clustering.get_centroids(trained)
      assert length(centroids) == 2
      assert length(hd(centroids)) == 4
    end
  end

  describe "get_cluster_assignment/2" do
    test "assigns vectors to nearest clusters" do
      {:ok, clustering} = Clustering.new(4, 2)

      data =
        List.duplicate([1.0, 0.0, 0.0, 0.0], 50) ++
          List.duplicate([0.0, 0.0, 0.0, 1.0], 50)

      {:ok, trained} = Clustering.train(clustering, data)

      {:ok, %{labels: labels, distances: distances}} =
        Clustering.get_cluster_assignment(trained, [[1.0, 0.0, 0.0, 0.0]])

      assert length(labels) == 1
      assert length(hd(labels)) == 1
      assert length(distances) == 1
      assert length(hd(distances)) == 1
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
  end

  describe "edge cases" do
    test "untrained clustering returns error for assignment" do
      {:ok, clustering} = Clustering.new(4, 2)

      assert {:error, "clustering not trained"} =
               Clustering.get_cluster_assignment(clustering, [[1.0, 0.0, 0.0, 0.0]])
    end
  end
end
