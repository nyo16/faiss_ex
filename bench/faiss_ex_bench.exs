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

# Approximate indexes: same corpus, IVF (trained on it) and HNSW
{:ok, ivf_index} = Index.new(dim, "IVF100,Flat")
:ok = Index.train(ivf_index, vectors_bin)
:ok = Index.add(ivf_index, vectors_bin)
{:ok, hnsw_index} = Index.new(dim, "HNSW32")
:ok = Index.add(hnsw_index, vectors_bin)

keys_1 = [0]
keys_10 = Enum.to_list(0..9)
keys_100 = Enum.to_list(0..99)
# Non-contiguous keys take the per-key reconstruct path instead of reconstruct_n
keys_100_shuffled = Enum.shuffle(0..(num_vectors - 1)) |> Enum.take(100)
{100, keys_100_shuffled_bin} = Shared.encode_ids!(keys_100_shuffled)

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

# Index creation happens in before_each so add scenarios time only the add
new_flat = fn _ ->
  {:ok, idx} = Index.new(dim, "Flat")
  idx
end

Benchee.run(
  %{
    # --- Add ---
    "add 1 vector" => {fn idx -> :ok = Index.add(idx, hd(query_1)) end, before_each: new_flat},
    "add 1000 vectors" => {
      fn idx -> :ok = Index.add(idx, Enum.take(vectors, 1000)) end,
      before_each: new_flat
    },
    "add 10000 vectors" => {fn idx -> :ok = Index.add(idx, vectors) end, before_each: new_flat},
    "add 10000 vectors (binary)" => {
      fn idx -> :ok = Index.add(idx, vectors_bin) end,
      before_each: new_flat
    },

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
    "search k=100, 100 queries" => fn ->
      {:ok, _} = Index.search(flat_index, query_100, 100)
    end,
    "search k=100, 100 queries (binary)" => fn ->
      {:ok, _} = Index.search_binary(flat_index, query_100_bin, 100)
    end,
    "search IVF100 k=#{k}, 100 queries (binary)" => fn ->
      {:ok, _} = Index.search_binary(ivf_index, query_100_bin, k)
    end,
    "search HNSW32 k=#{k}, 100 queries (binary)" => fn ->
      {:ok, _} = Index.search_binary(hnsw_index, query_100_bin, k)
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
    "reconstruct 100 shuffled keys" => fn ->
      {:ok, _} = Index.reconstruct(flat_index, keys_100_shuffled)
    end,
    "reconstruct 100 shuffled keys (binary)" => fn ->
      {:ok, _} = Index.reconstruct_binary(flat_index, keys_100_shuffled_bin)
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

# --- Concurrency: many callers, one query each ---------------------------
#
# `parallel:` runs each scenario from System.schedulers_online() processes at
# once. FAISS opens an OpenMP team per call even for one query once the
# database is large enough to split, so with the default thread count the
# dirty schedulers and the OpenMP teams oversubscribe the CPU; one thread
# per call does not. 10k vectors is too small to split — use 100k.
big_n = 100_000
big_bin = for _ <- 1..(big_n * dim), into: <<>>, do: <<:rand.uniform()::float-32-native>>
{:ok, big_index} = Index.new(dim, "Flat")
:ok = Index.add(big_index, big_bin)
big_query = binary_part(big_bin, 0, dim * 4)

Benchee.run(
  %{
    "concurrent search 100k k=#{k}, 1 query (omp default)" => {
      fn -> {:ok, _} = Index.search_binary(big_index, big_query, k) end,
      before_scenario: fn input ->
        :ok = FaissEx.set_omp_threads(0)
        input
      end
    },
    "concurrent search 100k k=#{k}, 1 query (omp 1 thread)" => {
      fn -> {:ok, _} = Index.search_binary(big_index, big_query, k) end,
      before_scenario: fn input ->
        :ok = FaissEx.set_omp_threads(1)
        input
      end,
      after_scenario: fn _ -> :ok = FaissEx.set_omp_threads(0) end
    }
  },
  parallel: System.schedulers_online(),
  warmup: 1,
  time: 5,
  print: [configuration: true, benchmarking: true]
)

# --- Starvation: waiters on a busy index vs a search on another ----------
#
# One process trains IVF1024 on index A (seconds of write-lock hold) while
# one waiter per dirty-CPU scheduler searches A. Then time a single search
# on an unrelated Flat index B. Waiters reschedule their lock wait onto the
# dirty-IO pool, so B should cost about as much as an uncontended search.
starve_dim = 64
{:ok, busy} = Index.new(starve_dim, "IVF1024,Flat")
{:ok, other} = Index.new(starve_dim, "Flat")
starve_query = List.duplicate(0.0, starve_dim)
:ok = Index.add(other, [starve_query])

starve_training =
  for _ <- 1..(80_000 * starve_dim), into: <<>>, do: <<:rand.uniform()::float-32-native>>

{baseline_us, {:ok, _}} = :timer.tc(fn -> Index.search(other, starve_query, 1) end)

trainer = Task.async(fn -> Index.train(busy, starve_training) end)
Process.sleep(200)

waiters =
  for _ <- 1..:erlang.system_info(:dirty_cpu_schedulers_online) do
    Task.async(fn -> Index.search(busy, starve_query, 1) end)
  end

Process.sleep(200)
{contended_us, {:ok, _}} = :timer.tc(fn -> Index.search(other, starve_query, 1) end)
trainer_alive? = Process.alive?(trainer.pid)
:ok = Task.await(trainer, :infinity)
Task.await_many(waiters, :infinity)

IO.puts("""

Starvation scenario (search on idle index B while A trains with \
#{length(waiters)} waiters on A):
  uncontended search on B: #{Float.round(baseline_us / 1000, 3)} ms
  contended search on B:   #{Float.round(contended_us / 1000, 3)} ms \
(A still training when B answered: #{trainer_alive?})
""")
