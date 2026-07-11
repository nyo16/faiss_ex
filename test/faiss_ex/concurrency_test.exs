defmodule FaissEx.ConcurrencyTest do
  use ExUnit.Case, async: false

  alias FaissEx.Index

  @moduletag :slow

  @dim 16
  @workers 8
  @iterations 300

  test "concurrent add/search/reset on a shared index does not crash the VM" do
    {:ok, index} = Index.new(@dim, "Flat")
    vector = List.duplicate(1.0, @dim)

    tasks =
      for i <- 1..@workers do
        Task.async(fn ->
          Enum.each(1..@iterations, fn _ ->
            case rem(i, 4) do
              0 -> :ok = Index.reset(index)
              1 -> :ok = Index.add(index, vector)
              2 -> {:ok, _} = Index.search(index, vector, 5)
              3 -> {:ok, _} = Index.ntotal(index)
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
end
