defmodule FaissEx.Shared do
  @moduledoc false

  def unwrap!({:ok, result}), do: result

  def unwrap!({:error, reason}) when is_binary(reason) do
    raise RuntimeError, reason
  end

  def unwrap!({:error, reason}) do
    raise RuntimeError, inspect(reason)
  end

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
  def to_flat_floats([h | _] = list) when is_list(h) do
    {length(list), List.flatten(list)}
  end

  def to_flat_floats([_ | _] = list) do
    {1, list}
  end
end
