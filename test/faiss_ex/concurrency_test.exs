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
end
