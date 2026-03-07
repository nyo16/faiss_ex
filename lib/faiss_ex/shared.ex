defmodule FaissEx.Shared do
  @moduledoc false

  def validate_type!(%Nx.Tensor{} = tensor, expected_type) do
    actual = Nx.type(tensor)

    if actual != expected_type do
      raise ArgumentError,
            "expected tensor of type #{inspect(expected_type)}, got #{inspect(actual)}"
    end

    tensor
  end

  def unwrap!({:ok, result}), do: result

  def unwrap!({:error, reason}) when is_binary(reason) do
    raise RuntimeError, reason
  end

  def unwrap!({:error, reason}) do
    raise RuntimeError, inspect(reason)
  end
end
