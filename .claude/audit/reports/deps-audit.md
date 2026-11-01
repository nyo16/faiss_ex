# Dependency Audit — faiss_ex

Date: 2026-11-01 | Commit: a397362 (v0.3.2, FAISS v1.15.1) | Scope: mix.exs, mix.lock, Makefile, .github/workflows/*.yml, README install section

## Score: 80/100

Rubric adaptation: `mix deps.audit` is unavailable (mix_audit is not a dep), so its 20 pts are reassigned to **FAISS native dep currency + integrity**: is the pin the latest upstream release, does the SHA match, and do the build and docs reliably produce and describe that native dependency. "Major version behind" also covers CI platform currency (runner images, actions), since this library's real dependency surface is the native toolchain, not Hex.

| Criterion | Max | Deductions | Score |
|---|---|---|---|
| No hex.audit vulnerabilities | 40 | none (`mix hex.audit` clean, pre-verified) | 40 |
| FAISS native dep currency + integrity (replaces deps.audit) | 20 | −5 FAISS build cache keyed by rev only, so changing USE_CUDA gives a silently stale or broken build (N1); −5 cmake/C++ toolchain floor stated wrong (N3) | 10 |
| No major version behind (>2) | 20 | Hex: none. −5 `macos-14` runner retired 2026-11-02 (N2); −5 `ubuntu-22.04`/`-arm` runners deprecated (N4) | 10 |
| No unused dependencies | 10 | none (elixir_make: build; ex_doc: docs; benchee: `bench/faiss_ex_bench.exs:41`; credo: dev tool, see N8) | 10 |
| Version pinning appropriate | 10 | none (all `~> X.Y`, lock committed) | 10 |
| **Total** | 100 | 40 + 10 + 10 + 10 + 10 | **80** |

Trend: 92 (2026-07-12) → 80. Every prior finding is closed or narrowed. The drop comes from new findings: GitHub runner images went out of support, and the build cache problem and cmake-floor mismatch weren't reported last time.

Clean areas (one line each): FAISS **v1.15.1 is the latest upstream release** (`git ls-remote` shows no tag newer than v1.15.1; GitHub release published 2026-09-16) · pin in sync: `mix.exs:9-10` `@faiss_git_rev "v1.15.1"` / `@faiss_git_sha "75c755c5a1fc854756aaf3bd1131213aeb21d8d1"` == `Makefile:14` `FAISS_GIT_REV  ?= v1.15.1` / `Makefile:19-20` `ifeq ($(FAISS_GIT_REV),v1.15.1)` → `FAISS_GIT_SHA ?= 75c755c5a1fc854756aaf3bd1131213aeb21d8d1` · SHA verified upstream: `refs/tags/v1.15.1^{}` = `75c755c5…` (annotated tag object `7ea73398…` peels to the pinned commit; `Makefile:164` compares `rev-parse HEAD`, i.e. the commit, so this is correct) and the local cache tree `HEAD` = `75c755c5…` · scoping correct: `{:elixir_make, "~> 0.9", runtime: false}`, ex_doc/benchee `only: :dev, runtime: false`, credo `only: [:dev, :test], runtime: false` (mix.exs:63-67) · Hex package (`mix hex.build`) ships exactly lib/, `c_src/faiss_ex_nif.c`, Makefile, mix.exs, README, LICENSE, CHANGELOG, with no junk; the only declared package dep is `elixir_make ~> 0.9` (needed by consumers to build) · NIF API used is old and stable (`enif_open_resource_type`, `ERL_NIF_INIT` at c_src:983-1052), with no OTP-version-gated calls.

### `mix hex.outdated`

| Dep | Only | Current | Latest | Major? | Note |
|---|---|---|---|---|---|
| elixir_make | (all) | 0.9.0 | 0.10.1 | No (0.x minor; allowed by `~> 0.9`) | 0.10.0 changelog: requires Elixir 1.14+, proxy-var fix; compatible with `elixir: "~> 1.18"` |
| ex_doc | dev | 0.40.1 | 0.40.4 | No (patch) | |
| benchee | dev | 1.5.0 | 1.5.1 | No (patch) | |
| credo | dev,test | 1.7.19 | 1.7.19 | — | up to date |

## Prior findings status

1. **FAISS pinned to a mutable tag, no SHA — RESOLVED.** `Makefile:19-20` sets the default SHA. `Makefile:162-171` clones, then `rev-parse HEAD` and compares against `FAISS_GIT_SHA`; on mismatch it `rm -rf`s the tree and exits 1. `mix.exs:73-82` passes the SHA through `make_env/0` and leaves it empty only for a custom rev. Remaining gap (verification runs only at clone time) is new finding N6.
2. **Minor version lag on Hex deps — STILL OPEN.** Same three deps lag (table above). elixir_make moved from 0.10.0 to 0.10.1 since the last audit and ex_doc from 0.40.3 to 0.40.4. All are within the existing requirements: `mix deps.update elixir_make ex_doc benchee`.
3. **No OTP minimum declared — PARTIALLY RESOLVED / STILL OPEN.** README:19 now documents "Erlang/OTP 27+". Nothing enforces or tests it; see N5.
4. **(Prior tooling recommendation) run `mix hex.audit` in CI — STILL OPEN.** No `hex.audit` step anywhere in `.github/workflows/ci.yml` or `publish.yml`; see N7.

## New findings

### N2. CI `macos-14` runner is retired on 2026-11-02, so the macOS arm64 job stops working — High
- `.github/workflows/ci.yml:135` — `- { runner: macos-14, label: macOS arm64 }`; the comment at ci.yml:34 also names macos-14.
- GitHub changelog 2026-10-01 ("macOS 14 runner image retirement"): retired November 2, 2026, with brownouts through Oct 30–31. The only Apple Silicon coverage of the darwin-only Makefile path (`Makefile:106-116` libomp discovery, the `install_name_tool` rewrites) disappears tomorrow. From then on the job fails, or hangs queued, on every push and PR.
- Fix: change to `macos-15` (or `macos-26`/`macos-latest`) in ci.yml:135 and update the ci.yml:34 comment. The FAISS cache key already includes `runner.arch`, so no cache-key change is needed.

### N1. FAISS build cache is keyed by rev only, so `USE_CUDA` changes silently reuse a stale or wrong build — Medium
- `Makefile:28` `FAISS_SRC := $(CACHE_DIR)/faiss-$(subst /,_,$(FAISS_GIT_REV))`, with the build in `$(FAISS_SRC)/build`. `Makefile:176` `$(FAISS_BUILD)/Makefile: $(FAISS_SRC)/.cloned` means cmake reconfigures only when the clone stamp changes, never when `-DFAISS_ENABLE_GPU` (Makefile:91-95) or `FAISS_OPT_LEVEL` changes.
- `Makefile:210` `$(NIF_SO): c_src/faiss_ex_nif.c $(LIB_DIR)/libfaiss_c… $(LIB_DIR)/libfaiss…` does not depend on `NIF_CFLAGS`, so flipping `USE_CUDA=true` (which adds `-DFAISS_GPU_ENABLED`, Makefile:69) does not recompile an existing NIF.
- Effects: after a CPU build, `USE_CUDA=true mix compile` (README:528, and CLAUDE.md's `USE_CUDA=true mix test --include cuda`) keeps the CPU-only NIF, and `FaissEx.num_gpus()` quietly returns 0 (README:563). In a fresh `_build` with a warm CPU cache, the NIF is compiled with GPU entry points against a GPU-less `libfaiss_c`, which links (macOS `-undefined dynamic_lookup`, Makefile:60; Linux shared objects allow undefined symbols by default) and then fails at NIF load with an undefined `faiss_*Gpu*` symbol [INFERENCE: not reproduced, no CUDA host]. README:143-144 documents `rm -rf ~/.cache/faiss_ex` for `FAISS_OPT_LEVEL` only; nothing covers USE_CUDA.
- Fix: put the configuration in the build dir name, e.g. `FAISS_BUILD := $(FAISS_SRC)/build-$(FAISS_OPT_LEVEL)-$(if $(filter true,$(USE_CUDA)),gpu,cpu)`, and add a flags stamp (`$(PRIV_DIR)/.nif_flags`, rewritten only when its content differs from `$(NIF_CFLAGS)`) as a prerequisite of `$(NIF_SO)` at Makefile:148 and :210. The README:143-144 workaround can then be dropped.

### N3. Documented build-toolchain floor is wrong: FAISS needs CMake ≥ 3.24 and a C++20 compiler — Medium
- README:22 says "CMake 3.17+". The pinned source's `CMakeLists.txt:22` has `cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)` (v1.14.3 and v1.10.0 have the same line, so this is long-standing), and `CMakeLists.txt:83` has `set(CMAKE_CXX_STANDARD 20)`. README:21 gives no compiler version ("A C/C++ compiler (clang or gcc)").
- README:46 and :51 present `apt-get install -y build-essential cmake libopenblas-dev` as "the exact recipe this project's CI uses (ubuntu-22.04)". Stock Ubuntu 22.04's archive cmake is 3.22.x [INFERENCE: Ubuntu jammy package version], so on a non-GitHub 22.04 host the first `mix compile` fails at FAISS configure. CI passes only because GitHub runner images put a newer cmake ahead of apt's on PATH [INFERENCE: runner image tooling], so the `apt-get install cmake` at ci.yml:31/98 and publish.yml:27 is effectively a no-op.
- Fix: README:22 → "CMake 3.24+", and README:21 → "a C++20 compiler (GCC 10+ / Clang 12+ / AppleClang 14+)". Add a note to the Ubuntu 22.04 recipe: install cmake from Kitware's apt repo or `pip install cmake`. Optionally fail fast in the Makefile by checking `cmake --version` before Step 2 (Makefile:176).

### N4. `ubuntu-22.04` / `ubuntu-22.04-arm` runners are deprecated (removal 2027-04-17) — Medium
- ci.yml:18, :73, :87 (`runs-on: ubuntu-22.04`), ci.yml:137 (`ubuntu-22.04-arm`), publish.yml:15. actions/runner-images#14254: deprecation began 2026-09-17 (longer queue times), fully unsupported 2027-04-17. README:51 ties the documented recipe to 22.04.
- Fix: move to `ubuntu-24.04` / `ubuntu-24.04-arm` (apt cmake 3.28 there also satisfies N3's 3.24 floor), then update README:51. When publish.yml moves, update its `restore-keys: Linux-X64-faiss-` comment (publish.yml:34-37); the arch-qualified key format is unchanged.

### N5. Elixir/OTP floor declared but never tested, and inconsistent with mix.exs — Low
- mix.exs:16 has `elixir: "~> 1.18"` and no OTP constraint. README:19 says "Erlang/OTP 27+", README:20 says "Elixir 1.18+". Elixir 1.18 also runs on OTP 25/26, which mix.exs admits and README forbids. CI and publish test only `ELIXIR_VERSION: "1.19"` / `OTP_VERSION: "28.0"` (ci.yml:12-13, publish.yml:9-10). Neither the declared floor (1.18/27) nor the current release the maintainer develops on (local `Elixir 1.20.2 (compiled with Erlang/OTP 29)`; README:581 benchmarks on 1.20/OTP 29) is exercised.
- Fix: add a small `{elixir, otp}` matrix to the Linux test job, `{1.18, 27}` (floor) and `{1.20, 29}` (latest), keeping 1.19/28 if desired. Bump `OTP_VERSION: "28.0"` to `"28"` so it tracks patch releases. Either drop "27+" from README or say the floor is what CI tests.

### N6. FAISS SHA verified only at clone time; reused or restored cache trees are trusted blindly — Low
- The SHA check lives inside the `$(FAISS_SRC)/.cloned` recipe (Makefile:162-171). Once the stamp exists, nothing re-verifies. That covers a `~/.cache/faiss_ex` restored by `actions/cache`, including publish.yml:36-37's prefix `restore-keys: Linux-X64-faiss-`, which restores the newest entry for any rev, and a tree modified locally after cloning.
- Fix: make Step 2 (Makefile:176) re-run the same `rev-parse HEAD` vs `$(FAISS_GIT_SHA)` comparison before cmake, by factoring the check into a `define`. It costs milliseconds and makes the publish build provably the pinned commit.

### N7. Supply-chain hygiene in workflows: floating action tags, one major behind, no `hex.audit` gate — Low
- Every action is pinned by mutable major tag: `actions/checkout@v6`, `erlef/setup-beam@v1`, `actions/cache@v5` / `cache/restore@v5` (ci.yml:23,25,51,58,100,106; publish.yml:19,21,30). Latest upstream: checkout `v7.0.1`, cache `v6.1.0`, setup-beam `v1.24.1`. That is one major behind, not more than two, so no deduction. publish.yml runs third-party `erlef/setup-beam@v1` in the job that later receives `HEX_API_KEY`. The secret is scoped to the final step only and the token is `contents: read`, so the exposure is bounded.
- No workflow runs `mix hex.audit` (prior recommendation still open).
- Fix: bump to `actions/checkout@v7` / `actions/cache@v6` after checking their release notes. Pin actions in publish.yml by commit SHA (e.g. `erlef/setup-beam@<sha> # v1.24.1`) or adopt GitHub workflow lockfiles. Add `- run: mix hex.audit` after `mix deps.get` in the setup job (ci.yml) and before `mix hex.publish` (publish.yml).

### N8. Smaller doc drift and pin-sync hazards — Low
- **Stale libomp instructions:** README:39-40 says the build "looks for libomp at `$HOMEBREW_PREFIX/opt/libomp`… set `HOMEBREW_PREFIX`". The Makefile now uses `brew --prefix libomp` and `LIBOMP_PREFIX` (Makefile:9-10, 106-111) and ignores `HOMEBREW_PREFIX`. Fix: replace with "auto-detected via `brew --prefix libomp`; override with `LIBOMP_PREFIX`".
- **Unenforced CUDA floor:** README:61 and :533 say CUDA 12.x is required, but neither the Makefile (no nvcc/version check; `USE_CUDA` only toggles flags at Makefile:68-70, 91-95) nor FAISS (`faiss/gpu/CMakeLists.txt:365` `find_package(CUDAToolkit REQUIRED)`, no version) enforces it, so CUDA 11 fails deep in the nvcc build. Fix: when `USE_CUDA=true`, check `nvcc --version` in the Makefile and `$(error)` below 12.
- **Pin-sync hazard:** the pin lives in three literals: mix.exs:9-10, Makefile:14, and the separate `ifeq` literal at Makefile:19. The CI cache key parses only Makefile:14 (`sed -n 's/^FAISS_GIT_REV *?= *//p' Makefile`). Bumping Makefile:14 without :19 silently disables SHA verification for raw `make`. Nothing checks that they agree. Fix: derive Makefile:19 from :14 (`FAISS_DEFAULT_REV := v1.15.1`, then `FAISS_GIT_REV ?= $(FAISS_DEFAULT_REV)` and `ifeq ($(FAISS_GIT_REV),$(FAISS_DEFAULT_REV))`), and add a CI step that greps mix.exs and Makefile and fails if rev or SHA differ.
- **credo not run anywhere:** it was added as a dev/test dep (CHANGELOG.md:76), but there is no `.credo.exs` and no CI step runs it, so it's an unenforced dev tool (not deducted as unused). Fix: add `mix credo --strict` to the ci.yml format job, or drop the dep.
