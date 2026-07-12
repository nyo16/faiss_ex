defmodule FaissEx.Shared do
  @moduledoc false

  @doc false
  def floats_to_binary(list) when is_list(list) do
    for f <- list, into: <<>>, do: <<f::float-32-native>>
  end

  @doc false
  def binary_to_floats(bin) when is_binary(bin) do
    for <<f::float-32-native <- bin>>, do: f
  end

  @doc false
  def int64s_to_binary(list) when is_list(list) do
    for i <- list, into: <<>>, do: <<i::signed-native-64>>
  end

  @doc false
  def binary_to_int64s(bin) when is_binary(bin) do
    for <<i::signed-native-64 <- bin>>, do: i
  end

  @doc false
  @spec encode_ids!([integer()]) :: {non_neg_integer(), binary()}
  def encode_ids!(ids) when is_list(ids), do: encode_ids(ids, 0, <<>>)

  defp encode_ids([], n, acc), do: {n, acc}

  defp encode_ids([i | rest], n, acc) when is_integer(i) do
    encode_ids(rest, n + 1, <<acc::binary, i::signed-native-64>>)
  end

  defp encode_ids([bad | _], n, _acc) do
    raise ArgumentError, "id #{n} is not an integer: #{inspect(bad)}"
  end

  @doc false
  @spec binary_to_float_rows(binary(), pos_integer()) :: [[float()]]
  def binary_to_float_rows(bin, row_len) when is_binary(bin) do
    decode_rows(bin, row_len * 4, &binary_to_floats/1, [])
  end

  @doc false
  @spec binary_to_int64_rows(binary(), pos_integer()) :: [[integer()]]
  def binary_to_int64_rows(bin, row_len) when is_binary(bin) do
    decode_rows(bin, row_len * 8, &binary_to_int64s/1, [])
  end

  defp decode_rows(<<>>, _row_size, _decode, acc), do: Enum.reverse(acc)

  defp decode_rows(bin, row_size, decode, acc) do
    <<row::binary-size(^row_size), rest::binary>> = bin
    decode_rows(rest, row_size, decode, [decode.(row) | acc])
  end

  @doc false
  @spec encode_vectors!([number()] | [[number()]], pos_integer()) ::
          {non_neg_integer(), binary()}
  def encode_vectors!(data, dim)

  def encode_vectors!([row | _] = rows, dim) when is_list(row) do
    encode_batch(rows, dim, 0, [])
  end

  def encode_vectors!([_ | _] = vector, dim) do
    {1, encode_row!(vector, dim, 0)}
  end

  def encode_vectors!([], _dim), do: {0, <<>>}

  defp encode_batch([], _dim, n, acc) do
    {n, acc |> Enum.reverse() |> IO.iodata_to_binary()}
  end

  defp encode_batch([row | rest], dim, n, acc) when is_list(row) do
    encode_batch(rest, dim, n + 1, [encode_row!(row, dim, n) | acc])
  end

  defp encode_batch([bad | _], _dim, n, _acc) do
    raise ArgumentError, "row #{n} is not a list of numbers: #{inspect(bad)}"
  end

  defp encode_row!(row, dim, row_index), do: encode_row(row, dim, row_index, 0, <<>>)

  defp encode_row([], dim, row_index, count, acc) do
    if count == dim do
      acc
    else
      raise ArgumentError, "row #{row_index} has #{count} elements, expected #{dim}"
    end
  end

  defp encode_row([f | rest], dim, row_index, count, acc) when is_number(f) do
    encode_row(rest, dim, row_index, count + 1, <<acc::binary, f::float-32-native>>)
  end

  defp encode_row([bad | _], _dim, row_index, _count, _acc) do
    raise ArgumentError, "row #{row_index} contains a non-number: #{inspect(bad)}"
  end
end
