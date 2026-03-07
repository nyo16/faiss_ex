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

      # Two clear clusters
      data =
        Nx.concatenate([
          Nx.broadcast(Nx.tensor([1.0, 0.0, 0.0, 0.0], type: {:f, 32}), {50, 4}),
          Nx.broadcast(Nx.tensor([0.0, 0.0, 0.0, 1.0], type: {:f, 32}), {50, 4})
        ])

      {:ok, trained} = Clustering.train(clustering, data)
      assert trained.trained?

      {:ok, centroids} = Clustering.get_centroids(trained)
      assert Nx.shape(centroids) == {2, 4}
    end
  end

  describe "get_cluster_assignment/2" do
    test "assigns vectors to nearest clusters" do
      {:ok, clustering} = Clustering.new(4, 2)

      data =
        Nx.concatenate([
          Nx.broadcast(Nx.tensor([1.0, 0.0, 0.0, 0.0], type: {:f, 32}), {50, 4}),
          Nx.broadcast(Nx.tensor([0.0, 0.0, 0.0, 1.0], type: {:f, 32}), {50, 4})
        ])

      {:ok, trained} = Clustering.train(clustering, data)

      query = Nx.tensor([[1.0, 0.0, 0.0, 0.0]], type: {:f, 32})

      {:ok, %{labels: labels, distances: distances}} =
        Clustering.get_cluster_assignment(trained, query)

      assert Nx.shape(labels) == {1, 1}
      assert Nx.shape(distances) == {1, 1}
    end
  end

  describe "raw list inputs" do
    test "train with list of lists" do
      {:ok, clustering} = Clustering.new(3, 2)

      data =
        List.duplicate([1.0, 0.0, 0.0], 50) ++
          List.duplicate([0.0, 0.0, 1.0], 50)

      {:ok, trained} = Clustering.train(clustering, data)
      assert trained.trained?

      {:ok, centroids} = Clustering.get_centroids(trained)
      assert Nx.shape(centroids) == {2, 3}
    end

    test "get_cluster_assignment with list query" do
      {:ok, clustering} = Clustering.new(3, 2)

      data =
        List.duplicate([1.0, 0.0, 0.0], 50) ++
          List.duplicate([0.0, 0.0, 1.0], 50)

      {:ok, trained} = Clustering.train(clustering, data)

      {:ok, %{labels: labels}} =
        Clustering.get_cluster_assignment(trained, [[1.0, 0.0, 0.0]])

      assert Nx.shape(labels) == {1, 1}
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
