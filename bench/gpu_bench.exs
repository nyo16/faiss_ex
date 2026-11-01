alias FaissEx.Index

dim = 128
n = 500_000
k = 10

median_ms = fn f, runs ->
  f.()
  times = for _ <- 1..runs, do: elem(:timer.tc(f), 0)
  Enum.at(Enum.sort(times), div(runs, 2)) / 1000
end

once_ms = fn f -> elem(:timer.tc(f), 0) / 1000 end

IO.puts("generating #{n} x #{dim} f32 ...")
:rand.seed(:exsss, {7, 7, 7})
rows = for _ <- 1..n, do: (for _ <- 1..dim, into: <<>>, do: <<:rand.uniform()::float-32-native>>)
corpus = IO.iodata_to_binary(rows)
queries = fn q -> binary_part(corpus, 0, q * dim * 4) end

row = fn label, cpu, gpu ->
  IO.puts(
    String.pad_trailing(label, 44) <>
      String.pad_leading(:erlang.float_to_binary(cpu, decimals: 2), 12) <>
      String.pad_leading(:erlang.float_to_binary(gpu, decimals: 2), 12) <>
      String.pad_leading(:erlang.float_to_binary(cpu / gpu, decimals: 1) <> "x", 10)
  )
end

IO.puts("\n" <> String.pad_trailing("scenario (ms)", 44) <> String.pad_leading("CPU", 12) <> String.pad_leading("GPU", 12) <> String.pad_leading("speedup", 10))

# ---- Flat ----
{:ok, cpu} = Index.new(dim, "Flat")
cpu_add = once_ms.(fn -> :ok = Index.add(cpu, corpus) end)
{:ok, gpu} = Index.new(dim, "Flat", device: {:cuda, 0})
gpu_add = once_ms.(fn -> :ok = Index.add(gpu, corpus) end)
row.("Flat add 500k", cpu_add, gpu_add)

for q <- [1, 100, 1_000, 10_000] do
  qs = queries.(q)
  runs = if q >= 1_000, do: 3, else: 11
  c = median_ms.(fn -> {:ok, _} = Index.search_binary(cpu, qs, k) end, runs)
  g = median_ms.(fn -> {:ok, _} = Index.search_binary(gpu, qs, k) end, runs)
  row.("Flat search #{q} queries, k=#{k}", c, g)
end

{:ok, cpu_from} = Index.new(dim, "Flat")
:ok = Index.add(cpu_from, binary_part(corpus, 0, 100_000 * dim * 4))
xfer = once_ms.(fn -> {:ok, _} = Index.cpu_to_gpu(cpu_from, 0) end)
IO.puts("cpu_to_gpu, Flat 100k x 128: #{Float.round(xfer, 1)} ms")

# ---- IVF ----
train = binary_part(corpus, 0, 100_000 * dim * 4)
{:ok, ivf_cpu} = Index.new(dim, "IVF1024,Flat")
{:ok, ivf_gpu} = Index.new(dim, "IVF1024,Flat", device: {:cuda, 0})
row.(
  "IVF1024 train 100k",
  once_ms.(fn -> :ok = Index.train(ivf_cpu, train) end),
  once_ms.(fn -> :ok = Index.train(ivf_gpu, train) end)
)

row.(
  "IVF1024 add 500k",
  once_ms.(fn -> :ok = Index.add(ivf_cpu, corpus) end),
  once_ms.(fn -> :ok = Index.add(ivf_gpu, corpus) end)
)

for q <- [1, 1_000, 10_000] do
  qs = queries.(q)
  runs = if q >= 1_000, do: 5, else: 21
  row.(
    "IVF1024 search #{q} queries, k=#{k}",
    median_ms.(fn -> {:ok, _} = Index.search_binary(ivf_cpu, qs, k) end, runs),
    median_ms.(fn -> {:ok, _} = Index.search_binary(ivf_gpu, qs, k) end, runs)
  )
end

# ---- Concurrency: 20 processes x 100 single-query searches, Flat 500k ----
q1 = queries.(1)

conc = fn idx ->
  once_ms.(fn ->
    1..20
    |> Enum.map(fn _ -> Task.async(fn -> for _ <- 1..100, do: {:ok, _} = Index.search_binary(idx, q1, k) end) end)
    |> Task.await_many(600_000)
  end)
end

:ok = FaissEx.set_omp_threads(0)
c_default = conc.(cpu)
:ok = FaissEx.set_omp_threads(1)
c_one = conc.(cpu)
g_one = conc.(gpu)
:ok = FaissEx.set_omp_threads(0)

IO.puts("\n20 procs x 100 single-query searches, Flat 500k (ms):")
IO.puts("  CPU omp default: #{Float.round(c_default, 0)}")
IO.puts("  CPU omp 1:       #{Float.round(c_one, 0)}")
IO.puts("  GPU:             #{Float.round(g_one, 0)}")
