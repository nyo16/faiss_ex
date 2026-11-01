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
          await_go()

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

    release(tasks)
    assert Task.await_many(tasks, 30_000) == List.duplicate(:done, @workers)

    # No resets in the mix, so the final count is exact
    expected = @workers * @iterations
    assert {:ok, ^expected} = Index.ntotal(index)
  end

  # Two-task slice of the shared-index writer/reader race in
  # concurrency_test.exs: one writer resets and re-adds, one reader
  # reconstructs, clones and computes residuals. Fixed op sequences keep it
  # deterministic in shape; {:error, msg} is legitimate when key 0 is past
  # ntotal right after a reset (or IDMap rejects positional reconstruct).
  test "writer racing reconstruct/clone/residuals on a shared IDMap index" do
    {:ok, index} = Index.new(@dim, "IDMap,Flat")
    vector = List.duplicate(1.0, @dim)

    check = fn
      :ok -> :ok
      {:ok, _} -> :ok
      {:error, msg} when is_binary(msg) -> :ok
    end

    writer =
      Task.async(fn ->
        await_go()

        Enum.each(1..@iterations, fn i ->
          if rem(i, 5) == 0, do: :ok = Index.reset(index)
          :ok = Index.add_with_ids(index, [vector], [i])
        end)

        :done
      end)

    reader =
      Task.async(fn ->
        await_go()

        Enum.each(1..@iterations, fn _ ->
          check.(Index.reconstruct(index, [0]))
          {:ok, %Index{}} = Index.clone(index)
          check.(Index.compute_residuals(index, [vector], [0]))
        end)

        :done
      end)

    tasks = [writer, reader]
    release(tasks)
    assert Task.await_many(tasks, 30_000) == [:done, :done]

    :ok = Index.reset(index)
    :ok = Index.add_with_ids(index, [vector], [42])
    assert {:ok, %{labels: [[42]]}} = Index.search(index, vector, 1)
  end

  # Start barrier: tasks block here until release/1, so they all begin
  # hammering the index together instead of staggered by spawn order.
  defp await_go do
    receive do
      :go -> :ok
    end
  end

  defp release(tasks), do: Enum.each(tasks, &send(&1.pid, :go))
end
