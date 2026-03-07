defmodule FaissEx.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :faiss_ex,
      version: @version,
      elixir: "~> 1.20-rc",
      start_permanent: Mix.env() == :prod,
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_env: make_env(),
      deps: deps()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:elixir_make, "~> 0.9", runtime: false},
      {:benchee, "~> 1.3", only: :dev, runtime: false}
    ]
  end

  defp make_env do
    %{
      "USE_CUDA" => System.get_env("USE_CUDA", "false"),
      "FAISS_GIT_REPO" =>
        System.get_env("FAISS_GIT_REPO", "https://github.com/facebookresearch/faiss.git"),
      "FAISS_GIT_REV" => System.get_env("FAISS_GIT_REV", "v1.10.0")
    }
  end
end
