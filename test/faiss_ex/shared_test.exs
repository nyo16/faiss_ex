defmodule FaissEx.SharedTest do
  use ExUnit.Case, async: true

  alias FaissEx.Shared

  describe "binary_to_float_rows/2" do
    test "decodes an empty binary to an empty list" do
      assert Shared.binary_to_float_rows(<<>>, 4) == []
    end

    test "decodes a single row" do
      bin = Shared.floats_to_binary([1.0, 2.0, 3.0])
      assert Shared.binary_to_float_rows(bin, 3) == [[1.0, 2.0, 3.0]]
    end

    test "decodes many rows" do
      rows = for i <- 0..9, do: for(j <- 1..4, do: i * 10.0 + j)
      {10, bin} = Shared.encode_vectors!(rows, 4)
      assert Shared.binary_to_float_rows(bin, 4) == rows
    end

    test "round-trips f32-representable values exactly" do
      rows = [[0.5, -0.25, 1024.0], [1.0e-3, -1.0e3, 0.0]]

      decoded =
        rows
        |> List.flatten()
        |> Shared.floats_to_binary()
        |> Shared.binary_to_float_rows(3)

      Enum.zip(List.flatten(rows), List.flatten(decoded))
      |> Enum.each(fn {expected, actual} ->
        assert_in_delta expected, actual, 1.0e-6
      end)
    end

    test "matches binary_to_floats plus chunking" do
      floats = Enum.map(1..12, &(&1 * 1.0))
      bin = Shared.floats_to_binary(floats)

      assert Shared.binary_to_float_rows(bin, 4) ==
               bin |> Shared.binary_to_floats() |> Enum.chunk_every(4)
    end
  end

  describe "binary_to_int64_rows/2" do
    test "decodes an empty binary to an empty list" do
      assert Shared.binary_to_int64_rows(<<>>, 5) == []
    end

    test "round-trips rows of int64s including negatives" do
      rows = [[0, -1, 9_223_372_036_854_775_807], [-9_223_372_036_854_775_808, 42, 7]]

      bin = rows |> List.flatten() |> Shared.int64s_to_binary()
      assert Shared.binary_to_int64_rows(bin, 3) == rows
    end
  end

  describe "encode_ids!/1" do
    test "encodes an empty list" do
      assert Shared.encode_ids!([]) == {0, <<>>}
    end

    test "returns the count and the same binary as int64s_to_binary" do
      ids = [100, -5, 0, 9_999_999_999]
      assert {4, bin} = Shared.encode_ids!(ids)
      assert bin == Shared.int64s_to_binary(ids)
      assert Shared.binary_to_int64s(bin) == ids
    end

    test "raises on a non-integer with its position" do
      assert_raise ArgumentError, "id 2 is not an integer: 3.5", fn ->
        Shared.encode_ids!([1, 2, 3.5])
      end
    end
  end
end
