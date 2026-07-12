defmodule FaissEx.ConcurrencySmokeTest do
  # Fast, always-run slice of the RW-lock contract. The randomized stress
  # suite in concurrency_test.exs stays behind :slow; this one is small and
  # deterministic so a default `mix test` exercises concurrent readers and
  # writers on a shared index at least once.
  use ExUnit.Case, async: true

  alias FaissEx.Index

  @dim 8
  @workers 4
  @iterations 50

  test "concurrent add/search/ntotal on a shared index stays consistent" do
    {:ok, index} = Index.new(@dim, "Flat")
    vector = List.duplicate(1.0, @dim)

    tasks =
      for _ <- 1..@workers do
        Task.async(fn ->
          Enum.each(1..@iterations, fn _ ->
            :ok = Index.add(index, vector)

            {:ok, %{labels: [[label | _]]}} = Index.search(index, vector, 1)
            assert label >= 0

            {:ok, n} = Index.ntotal(index)
            assert n > 0
          end)

          :done
        end)
      end

    assert Task.await_many(tasks, 30_000) == List.duplicate(:done, @workers)

    # No resets in the mix, so the final count is exact
    expected = @workers * @iterations
    assert {:ok, ^expected} = Index.ntotal(index)
  end
end
