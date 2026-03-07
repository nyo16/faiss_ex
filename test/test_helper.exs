exclude = [:slow]
exclude = if System.get_env("USE_CUDA") == "true", do: exclude, else: [:cuda | exclude]
ExUnit.start(exclude: exclude)
