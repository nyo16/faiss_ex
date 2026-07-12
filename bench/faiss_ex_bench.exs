alias FaissEx.Index
alias FaissEx.Clustering
alias FaissEx.Shared

dim = 128
num_vectors = 10_000
k = 10

# Pre-generate data as lists of lists
:rand.seed(:exsss, {42, 42, 42})
vectors = for _ <- 1..num_vectors, do: for(_ <- 1..dim, do: :rand.uniform())
query_1 = [for(_ <- 1..dim, do: :rand.uniform())]
query_10 = for _ <- 1..10, do: for(_ <- 1..dim, do: :rand.uniform())
query_100 = for _ <- 1..100, do: for(_ <- 1..dim, do: :rand.uniform())

# Pre-encoded binaries for the binary fast-path scenarios
{^num_vectors, vectors_bin} = Shared.encode_vectors!(vectors, dim)
{100, query_100_bin} = Shared.encode_vectors!(query_100, dim)

# Build a populated Flat index for search/reconstruct/residual benchmarks
{:ok, flat_index} = Index.new(dim, "Flat")
:ok = Index.add(flat_index, vectors)

keys_1 = [0]
keys_10 = Enum.to_list(0..9)
keys_100 = Enum.to_list(0..99)

# Pre-slice data for residual benchmarks
data_1 = Enum.take(vectors, 1)
data_10 = Enum.take(vectors, 10)
data_100 = Enum.take(vectors, 100)

# Clustering data (smaller for reasonable bench times)
cluster_data = for _ <- 1..1000, do: for(_ <- 1..dim, do: :rand.uniform())

# Pre-trained clustering for assignment benchmarks
{:ok, assign_clustering} = Clustering.new(dim, 10)
{:ok, assign_trained} = Clustering.train(assign_clustering, cluster_data)
assign_data_100 = Enum.take(cluster_data, 100)

Benchee.run(
  %{
    # --- Add ---
    "add 1 vector" => fn ->
      {:ok, idx} = Index.new(dim, "Flat")
      :ok = Index.add(idx, hd(query_1))
    end,
    "add 1000 vectors" => {
      fn vectors_1k ->
        {:ok, idx} = Index.new(dim, "Flat")
        :ok = Index.add(idx, vectors_1k)
      end,
      before_scenario: fn _ -> Enum.take(vectors, 1000) end
    },
    "add 10000 vectors" => fn ->
      {:ok, idx} = Index.new(dim, "Flat")
      :ok = Index.add(idx, vectors)
    end,
    "add 10000 vectors (binary)" => fn ->
      {:ok, idx} = Index.new(dim, "Flat")
      :ok = Index.add(idx, vectors_bin)
    end,

    # --- Search ---
    "search k=#{k}, 1 query" => fn ->
      {:ok, _} = Index.search(flat_index, query_1, k)
    end,
    "search k=#{k}, 10 queries" => fn ->
      {:ok, _} = Index.search(flat_index, query_10, k)
    end,
    "search k=#{k}, 100 queries" => fn ->
      {:ok, _} = Index.search(flat_index, query_100, k)
    end,
    "search k=#{k}, 100 queries (binary)" => fn ->
      {:ok, _} = Index.search_binary(flat_index, query_100_bin, k)
    end,

    # --- Reconstruct ---
    "reconstruct 1 vector" => fn ->
      {:ok, _} = Index.reconstruct(flat_index, keys_1)
    end,
    "reconstruct 10 vectors" => fn ->
      {:ok, _} = Index.reconstruct(flat_index, keys_10)
    end,
    "reconstruct 100 vectors" => fn ->
      {:ok, _} = Index.reconstruct(flat_index, keys_100)
    end,

    # --- Compute Residuals ---
    "compute_residuals 1 vector" => fn ->
      {:ok, _} = Index.compute_residuals(flat_index, data_1, keys_1)
    end,
    "compute_residuals 10 vectors" => fn ->
      {:ok, _} = Index.compute_residuals(flat_index, data_10, keys_10)
    end,
    "compute_residuals 100 vectors" => fn ->
      {:ok, _} = Index.compute_residuals(flat_index, data_100, keys_100)
    end,

    # --- Clustering ---
    "kmeans k=10, 1000 vectors" => fn ->
      {:ok, c} = Clustering.new(dim, 10)
      {:ok, _} = Clustering.train(c, cluster_data)
    end,
    "kmeans k=50, 1000 vectors" => fn ->
      {:ok, c} = Clustering.new(dim, 50)
      {:ok, _} = Clustering.train(c, cluster_data)
    end,
    "kmeans assign 100 vectors" => fn ->
      {:ok, _} = Clustering.get_cluster_assignment(assign_trained, assign_data_100)
    end
  },
  warmup: 2,
  time: 5,
  memory_time: 2,
  print: [configuration: true, benchmarking: true]
)
