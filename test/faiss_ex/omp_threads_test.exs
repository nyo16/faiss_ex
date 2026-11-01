defmodule FaissEx.OmpThreadsTest do
  # async: false — the OpenMP thread count is global to the VM
  use ExUnit.Case, async: false

  alias FaissEx.Index

  setup do
    on_exit(fn -> FaissEx.set_omp_threads(0) end)
  end

  test "rejects negative and non-integer thread counts" do
    for bad <- [-1, 2.0, :all] do
      assert_raise FunctionClauseError, fn -> FaissEx.set_omp_threads(bad) end
    end
  end

  test "search results do not depend on the thread count" do
    {:ok, index} = Index.new(16, "Flat")
    :rand.seed(:exsss, {1, 2, 3})
    corpus = for _ <- 1..2_000, do: for(_ <- 1..16, do: :rand.uniform())
    queries = Enum.take(corpus, 200)
    :ok = Index.add(index, corpus)

    :ok = FaissEx.set_omp_threads(0)
    {:ok, default_threads} = Index.search(index, queries, 10)

    :ok = FaissEx.set_omp_threads(1)
    {:ok, one_thread} = Index.search(index, queries, 10)

    assert one_thread == default_threads
  end
end
