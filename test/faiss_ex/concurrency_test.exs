defmodule FaissEx.ConcurrencyTest do
  # async: false is intentional — these tests saturate the dirty CPU
  # schedulers; running them alongside other suites would just add noise
  # to an already timing-sensitive stress test.
  use ExUnit.Case, async: false

  alias FaissEx.Index

  @moduletag :slow
  @moduletag timeout: 180_000

  @dim 16
  @workers 8
  @iterations 300

  test "randomized add/search/reset interleavings on a shared index do not crash the VM" do
    {:ok, index} = Index.new(@dim, "Flat")
    vector = List.duplicate(1.0, @dim)

    # Op lists are drawn here, in the test process: ExUnit seeds :rand from
    # --seed only in the test process, so Task processes would get fresh,
    # unseeded state and make failures unreproducible.
    op_lists =
      for _ <- 1..@workers do
        for _ <- 1..@iterations, do: Enum.random([:add, :search, :reset, :ntotal])
      end

    tasks =
      for ops <- op_lists do
        Task.async(fn ->
          Enum.each(ops, fn
            :add -> :ok = Index.add(index, vector)
            :search -> {:ok, _} = Index.search(index, vector, 5)
            :reset -> :ok = Index.reset(index)
            :ntotal -> {:ok, _} = Index.ntotal(index)
          end)

          :done
        end)
      end

    assert Task.await_many(tasks, 120_000) == List.duplicate(:done, @workers)

    # Index must still be functional after the hammering
    :ok = Index.reset(index)
    :ok = Index.add(index, vector)
    assert {:ok, %{labels: [[0]]}} = Index.search(index, vector, 1)
  end

  test "train racing search and add on a shared IVF index does not crash the VM" do
    {:ok, index} = Index.new(8, "IVF4,Flat")
    training = for _ <- 1..512, do: for(_ <- 1..8, do: :rand.uniform())
    query = List.duplicate(0.5, 8)

    # Workers race real kmeans training (long write-lock hold) against
    # searches and adds. Until a train wins, add/search legitimately return
    # {:error, "...not trained..."} — crash-safety is what's under test.
    tasks =
      for i <- 1..6 do
        Task.async(fn ->
          Enum.each(1..25, fn _ ->
            result =
              case rem(i, 3) do
                0 -> Index.train(index, training)
                1 -> Index.search(index, query, 3)
                2 -> Index.add(index, [query])
              end

            case result do
              :ok -> :ok
              {:ok, _} -> :ok
              {:error, msg} when is_binary(msg) -> :ok
            end
          end)

          :done
        end)
      end

    assert Task.await_many(tasks, 120_000) == List.duplicate(:done, 6)

    # Index must still be functional afterwards
    :ok = Index.train(index, training)
    :ok = Index.add(index, training)
    assert {:ok, %{labels: [labels]}} = Index.search(index, hd(training), 3)
    assert length(labels) == 3
  end

  # Writers (reset/add/add_with_ids) race readers (reconstruct, residuals,
  # clone, to_file) on ONE shared index. After a reset, key 0 may be past
  # ntotal, and IDMap rejects plain add / positional reconstruct, so
  # {:error, msg} is a legitimate outcome — crash-safety is what's under test.
  #
  # The clustering two-lock race (quantizer index + clustering in
  # nif_train_clustering) is N/A here: under value semantics
  # Clustering.train/2 allocates a fresh clustering and quantizer on every
  # call and never shares them between calls, so those two locks cannot be
  # contended through the public API. The C locks remain as defense in depth.
  @tag :tmp_dir
  test "writers racing reconstruct/residuals/clone/to_file on a shared index do not crash the VM",
       %{tmp_dir: tmp_dir} do
    dim = 8
    {:ok, index} = Index.new(dim, "IDMap,Flat")
    random_vector = fn -> for _ <- 1..dim, do: :rand.uniform() end

    # Drawn in the test process so --seed reproduces the interleaving inputs
    writer_ops =
      for _ <- 1..div(@workers, 2) do
        for _ <- 1..@iterations do
          case Enum.random([:reset, :add, :add_with_ids]) do
            :reset -> :reset
            op -> {op, random_vector.()}
          end
        end
      end

    reader_ops =
      for _ <- 1..div(@workers, 2) do
        for _ <- 1..@iterations do
          case Enum.random([:reconstruct, :residuals, :clone, :to_file]) do
            :residuals -> {:residuals, random_vector.()}
            op -> op
          end
        end
      end

    check = fn
      :ok -> :ok
      {:ok, _} -> :ok
      {:error, msg} when is_binary(msg) -> :ok
    end

    writers =
      for ops <- writer_ops do
        Task.async(fn ->
          Enum.each(ops, fn
            :reset ->
              check.(Index.reset(index))

            {:add, vec} ->
              check.(Index.add(index, [vec]))

            {:add_with_ids, vec} ->
              check.(Index.add_with_ids(index, [vec], [System.unique_integer([:positive])]))
          end)

          :done
        end)
      end

    readers =
      for {ops, w} <- Enum.with_index(reader_ops) do
        Task.async(fn ->
          ops
          |> Enum.with_index()
          |> Enum.each(fn
            {:reconstruct, _} ->
              check.(Index.reconstruct(index, [0]))

            {{:residuals, vec}, _} ->
              check.(Index.compute_residuals(index, [vec], [0]))

            {:clone, _} ->
              check.(Index.clone(index))

            {:to_file, i} ->
              check.(Index.to_file(index, Path.join(tmp_dir, "race-#{w}-#{i}.index")))
          end)

          :done
        end)
      end

    tasks = writers ++ readers
    assert Task.await_many(tasks, 120_000) == List.duplicate(:done, length(tasks))

    # Index must still be functional afterwards
    vec = List.duplicate(0.5, dim)
    :ok = Index.reset(index)
    :ok = Index.add_with_ids(index, [vec], [42])
    assert {:ok, %{labels: [[42]]}} = Index.search(index, vec, 1)
  end

  # Regression for dirty-scheduler starvation. A NIF blocked on an rwlock
  # still occupies its dirty scheduler thread, so before contended NIFs
  # rescheduled their wait onto the dirty-IO pool, one waiter per dirty-CPU
  # thread on a busy index A queued *every* other dirty-CPU NIF — including a
  # search on an unrelated index B — behind A's training. The assertion is
  # about ordering: B answers while A is still training.
  test "waiters on a busy index leave the dirty-CPU pool free for other indexes" do
    dim = 64
    {:ok, busy} = Index.new(dim, "IVF1024,Flat")
    {:ok, other} = Index.new(dim, "Flat")
    query = List.duplicate(0.0, dim)
    :ok = Index.add(other, [query])

    # ~1.7 s of k-means on an M-series CPU, longer on CI runners
    training = for _ <- 1..(80_000 * dim), into: <<>>, do: <<:rand.uniform()::float-32-native>>

    trainer = Task.async(fn -> Index.train(busy, training) end)
    # Setup only: let the trainer take the write lock before waiters arrive
    Process.sleep(200)

    waiters =
      for _ <- 1..:erlang.system_info(:dirty_cpu_schedulers_online) do
        Task.async(fn -> Index.search(busy, query, 1) end)
      end

    Process.sleep(200)

    assert {:ok, %{labels: [[0]]}} = Index.search(other, query, 1)

    assert Process.alive?(trainer.pid),
           "search on an idle index waited for training on a different index"

    assert :ok = Task.await(trainer, 60_000)
    assert Enum.all?(Task.await_many(waiters, 60_000), &match?({:ok, _}, &1))
  end
end
