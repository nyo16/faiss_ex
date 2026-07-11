exclude = [:slow]

# :cuda tests need a CUDA build; :no_cuda tests assert the error paths of a
# non-CUDA build — exactly one of the two groups runs.
exclude =
  if System.get_env("USE_CUDA") == "true",
    do: [:no_cuda | exclude],
    else: [:cuda | exclude]

ExUnit.start(exclude: exclude)
