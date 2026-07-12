defmodule FaissEx.MixProject do
  use Mix.Project

  @version "0.3.0"
  @source_url "https://github.com/nyo16/faiss_ex"

  # Keep rev+SHA in sync with Makefile. The SHA is what the tag must resolve
  # to post-clone — tags are mutable refs, the commit is the real pin.
  @faiss_git_rev "v1.14.3"
  @faiss_git_sha "0ca9df4792b173d573044ee14ca0704780176e82"

  def project do
    [
      app: :faiss_ex,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_env: make_env(),
      deps: deps(),
      name: "FaissEx",
      description: description(),
      package: package(),
      docs: docs(),
      source_url: @source_url,
      homepage_url: @source_url
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp description do
    "Elixir NIF bindings for FAISS (Facebook AI Similarity Search). " <>
      "Vector similarity search, index factory, k-means clustering, and optional GPU support."
  end

  defp package do
    [
      name: "faiss_ex",
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url},
      files: ~w(
        lib c_src Makefile
        mix.exs README.md LICENSE CHANGELOG.md
      )
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: ["README.md", "CHANGELOG.md"],
      source_ref: "v#{@version}",
      groups_for_modules: [
        "Core API": [FaissEx, FaissEx.Index, FaissEx.Clustering]
      ],
      nest_modules_by_prefix: [FaissEx]
    ]
  end

  defp deps do
    [
      {:elixir_make, "~> 0.9", runtime: false},
      {:ex_doc, "~> 0.35", only: :dev, runtime: false},
      {:benchee, "~> 1.3", only: :dev, runtime: false}
    ]
  end

  defp make_env do
    rev = System.get_env("FAISS_GIT_REV", @faiss_git_rev)

    # A custom rev builds unverified unless FAISS_GIT_SHA is provided with it.
    default_sha = if rev == @faiss_git_rev, do: @faiss_git_sha, else: ""

    env = %{
      "USE_CUDA" => System.get_env("USE_CUDA", "false"),
      "FAISS_OPT_LEVEL" => System.get_env("FAISS_OPT_LEVEL", "generic"),
      "FAISS_GIT_REPO" =>
        System.get_env("FAISS_GIT_REPO", "https://github.com/facebookresearch/faiss.git"),
      "FAISS_GIT_REV" => rev,
      "FAISS_GIT_SHA" => System.get_env("FAISS_GIT_SHA", default_sha)
    }

    case System.get_env("FAISS_PREFIX") do
      nil -> env
      prefix -> Map.put(env, "FAISS_PREFIX", prefix)
    end
  end
end
