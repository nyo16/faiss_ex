defmodule FaissEx.ConcurrencyTest do
  # async: false is intentional — these tests saturate the dirty CPU
  # schedulers; running them alongside other suites would just add noise
  # to an already timing-sensitive stress test.
  use ExUnit.Case, async: false

  alias FaissEx.Index

  @moduletag :slow

  @dim 16
  @workers 8
  @iterations 300

  test "randomized add/search/reset interleavings on a shared index do not crash the VM" do
    {:ok, index} = Index.new(@dim, "Flat")
    vector = List.duplicate(1.0, @dim)

    tasks =
      for _ <- 1..@workers do
        Task.async(fn ->
          Enum.each(1..@iterations, fn _ ->
            case Enum.random([:add, :search, :reset, :ntotal]) do
              :add -> :ok = Index.add(index, vector)
              :search -> {:ok, _} = Index.search(index, vector, 5)
              :reset -> :ok = Index.reset(index)
              :ntotal -> {:ok, _} = Index.ntotal(index)
            end
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
