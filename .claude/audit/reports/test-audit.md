# Test Health Audit — faiss_ex @ a397362 (v0.3.2, FAISS v1.15.1)

Scope: `test/**`, `.github/workflows/ci.yml` (+ `publish.yml` test step), against `lib/**` and `c_src/faiss_ex_nif.c`.

## Score: 89/100 (B) — prior 62/100 (2026-07-12), +27

Rubric: `scoring-methodology.md` → Test Quality. Adaptations for an NIF library (no Ecto/LiveView/Mox):
- `verify_on_exit! in Mox tests` (15) is N/A → reassigned to **NIF error branches reachable from the public API that a test exercises**. Score = 15 × tested/reachable.
- `No flaky test patterns` (20): `-5 per Process.sleep` as written; additionally `-2` per other flake or timing pattern (approximate-search assertion, timeout mismatch), stated so the deduction shows.

| Criterion | Pts | Evidence | Deduction | Score |
|---|---|---|---|---|
| Coverage >70% | 30 | `mix test --cover` (Main ran it, 85 passed / 3 excluded): total **79.66%**. Index 95.16%, Clustering/Shared/FaissEx 100%, NIF 12.50% (`:erlang.nif_error` stubs, unreachable once the NIF loads). Excluding the NIF stubs: about 96%. C is **not** measured: `--cover` is BEAM-only and there is no gcov/llvm-cov step. | 0 (above 70%) | 30 |
| No flaky patterns | 20 | 0 `Process.sleep`/`:timer.sleep` in `test/`. HNSW self-hit assertion `index_test.exs:302` (-2). `Task.await_many(…, 120_000)` above ExUnit's 60 s default timeout at `concurrency_test.exs:35,73` (-2). | -4 | 16 |
| async: true where possible | 15 | `index_test`, `clustering_test`, `shared_test` and `concurrency_smoke_test` are all `async: true`. `concurrency_test.exs:5` is `async: false` with a documented reason. | 0 | 15 |
| NIF error branches covered *(replaces Mox)* | 15 | 18 of 33 public-reachable branches are tested (table below). 15 × 18/33 = 8.2 | -7 | 8 |
| Test duration | 10 | 0.4 s by default | 0 | 10 |
| Error paths tested | 10 | Many error paths have tests, and the messages are asserted exactly. | 0 | 10 |
| **Total** | | 30 + 16 + 15 + 8 + 10 + 10 | | **89** |

N/A, no deduction: Ecto sandbox, factories, LiveView, Oban, Mox. No test mocks anything.

---

## NIF error-branch inventory (`c_src/faiss_ex_nif.c`)

- **93 `{:error, _}` branches in total.**
  - 74 `return make_error_msg(...)` sites. Two of them carry several messages: `:927` returns "size overflow" or "data binary size mismatch", and `:972` returns "no centroids available", "size overflow" or "failed to allocate centroids binary". That makes 77 distinct branches.
  - 16 `make_faiss_error(...)` sites, at `:241, 265, 309, 355, 425, 467, 485, 573, 633, 661, 688, 754, 763, 799, 856, 928`.
- **85 branches compile in the non-CUDA build CI uses.** The `FAISS_GPU_ENABLED` block `:741-808` holds 8 branches. The non-CUDA stubs `:822` and `:828` replace it.
- **Not reachable from the public API.** These are excluded from the score.
  - The Elixir encoders already guarantee these conditions: "invalid index reference", "n must be non-negative", size mismatches when n=0 or n>0, and "size overflow" on the add/train/reconstruct/residuals inputs.
  - Allocation or out-of-memory failures that need fault injection.
  - FAISS calls with no practical CPU trigger: clone `:265`, reset `:485`, `faiss_Clustering_new` `:856` [INFERENCE].
- **33 branches are reachable through the public API. 18 are tested:**

| # | Branch (c_src line) | Public trigger | Tested? |
|---|---|---|---|
| 1 | new_index "invalid arguments" :227 | `Index.new(4, ~c"Flat")`, `Index.new(2**40, "Flat")` | ✗ |
| 2 | "dim must be positive" :230 | `new(0, …)` | ✓ index_test:550 |
| 3 | NUL in description :231 | | ✓ index_test:555 |
| 4 | **factory failure :241** | `Index.new(4, "NotAnIndex")` | ✗ |
| 5 | add FAISS failure :309 | untrained IVF | ✓ index_test:428 |
| 6 | add_with_ids n=0 mismatch :332 | `add_with_ids(idx, [], [1])` | ✗ |
| 7 | add_with_ids size mismatch :345 | | ✓ index_test:539, 675 |
| 8 | **add_with_ids FAISS failure :355** | `add_with_ids` on plain `"Flat"` | ✗ |
| 9 | search "invalid arguments" :372 | `search(idx, v, 2.0)` | ✗ |
| 10 | **"k must be positive" :376** | `search(idx, v, 0)` / `search_binary(…, 0)` | ✗ |
| 11 | search n·k "size overflow" :404 | `k = 2**62` | ✗ |
| 12 | search result alloc failure :409 | very large `k` | ✗ |
| 13 | search FAISS failure :425 | untrained IVF | ✓ index_test:437 |
| 14 | train FAISS failure :467 | `Index.train(IVF4, 2 vectors)` (n < nlist) [INFERENCE: same FAISS check that clustering_test:91 hits] | ✗ |
| 15 | reconstruct negative key :523 | | ✓ index_test:713, 717, 730 |
| 16 | reconstruct int64-overflow range :543 | | ✓ index_test:723 |
| 17 | reconstruct FAISS failure :573 | key ≥ ntotal | ✓ index_test:739 (contiguous path only) |
| 18 | residuals negative key :617 | | ✓ index_test:244 |
| 19 | residuals FAISS failure :633 | `compute_residuals(idx, xs, [99])` | ✗ |
| 20 | write "invalid arguments" :647 | `to_file(idx, ~c"/tmp/x")` | ✗ |
| 21 | write NUL :650 | | ✓ index_test:559 |
| 22 | write FAISS failure :661 | nonexistent dir | ✓ index_test:326 (`{:error, _}` only) |
| 23 | read "invalid arguments" :675 | `from_file(p, io_flags: :x)` | ✗ |
| 24 | read NUL :678 | | ✓ index_test:559 |
| 25 | read FAISS failure :688 | missing file | ✓ index_test:331 (corrupt file ✗) |
| 26 | cpu_to_gpu not compiled :822 | | ✓ index_test:308 |
| 27 | gpu_to_cpu not compiled :828 | forged `%Index{device: {:cuda, 0}}` | ✗ |
| 28 | new_clustering "invalid arguments" :847 | `Clustering.new(4.0, 2)` | ✗ |
| 29 | "d must be positive" :850 | | ✓ clustering_test:79 |
| 30 | "k must be positive" :851 | | ✓ clustering_test:84 |
| 31 | **train_clustering "n must be positive" :894** | `Clustering.train(c, [])` | ✗ |
| 32 | train_clustering FAISS failure :928 | n < k | ✓ clustering_test:91 (`is_binary` only) |
| 33 | "no centroids available" :955 | | ✓ clustering_test:109 |

---

## Prior findings status (old report, 2026-07-12)

| # | Prior finding | Status | Evidence |
|---|---|---|---|
| 1 | Concurrency suite only runs under `:slow` and never in default runs (Critical) | **RESOLVED** | `concurrency_smoke_test.exs:6,14-39` has no tag and is `async: true`, so it always runs. CI runs `mix test --include slow` (`ci.yml:114,175`; `publish.yml:55`). |
| 2 | No GC/destructor test (High) | **RESOLVED** (CPU) | `index_test.exs:445-492`: the owner process dies (446) and resources churn across 25 Task exits (465). GPU-resource destructor still untested (CUDA never runs, see M4). |
| 3 | Untrained-index add/search lacks a unit test (High) | **RESOLVED** | `index_test.exs:427-443` |
| 4 | `get_centroids/1` before train (High) | **RESOLVED** | `clustering_test.exs:109-112` |
| 5 | tmp paths built with `:rand.uniform(100_000)` (Medium) | **RESOLVED** | `@tag :tmp_dir` at `index_test.exs:168,335,388` |
| 6 | `gpu_to_cpu/1` on a host index | **RESOLVED** | The clause now returns an error (`index.ex:449-451`). Tested at `index_test.exs:319-322`. |
| 7 | Invalid factory string untested (Medium) | **STILL OPEN** | No test exercises `c_src:241`. See table #4 and M3. |
| 8 | `add_with_ids/3` on plain Flat (Medium) | **STILL OPEN** | Only `"IDMap,Flat"` is used (`index_test.exs:192,540,646,661,676`). See table #8 and M3. |
| 9 | `reconstruct/2` on an index type without direct reconstruction (IVF without a direct map) (Medium) | **STILL OPEN** | Every reconstruct test uses `"Flat"` (`index_test.exs:125-165,699-742`). See M3. |
| 10 | `compute_residuals/3` batch case (Medium) | **PARTIALLY RESOLVED** | Non-zero residual added (`index_test.exs:232-242`). A successful n>1 batch is still untested. The only n=2 call raises in Elixir (`:252`). |
| 11 | Wrong-dim inputs for search/train/residuals (Low) | **STILL OPEN** | Wrong dim is tested only for `add` (list `:501,509`, binary `:652`). There is no wrong-dim test for `search/3`, `search_binary/3`, `train/2` or `compute_residuals/3`. |
| 12 | Empty input for search/train (Low) | **PARTIALLY RESOLVED** | `search_binary(<<>>)` tested (`:683`). `search(idx, [], k)`, `train(idx, [])`, `reconstruct(idx, [])` and `Clustering.train(c, [])` are not (see M1). |
| 13 | `:io_flags` option untested (Low) | **STILL OPEN** | No `io_flags` anywhere in `test/`. |
| 14 | Weak `is_binary(msg)` on the clustering failure (Low) | **STILL OPEN** | `clustering_test.exs:96` (folded into H1). |
| 15 | `num_gpus` test is near-tautological (Low) | **STILL OPEN** | `index_test.exs:693-696`. The non-CUDA C stub always returns 0 (`c_src:831-834`). Under `:no_cuda` assert `== 0`. |

---

## New findings

### High

**H1. `clustering_test.exs:91-100`: the "stays untrained" assertion is a tautology.**
- The test calls `Clustering.get_cluster_assignment(clustering, …)` on the **original** immutable struct. Its `index` is still `nil`, so the `index: nil` clause at `clustering.ex:103-105` returns `"clustering not trained"` without touching the NIF.
- The assertion would pass even if the failed `train/2` had left the C-side clustering half-trained. That C-side state is exactly what the test name claims to check.
- Fix: replace lines 98-99 with `assert {:error, "no centroids available"} = Clustering.get_centroids(clustering)`. This reads the C-side state through `nif_get_clustering_centroids` (`c_src:954-956`). Also tighten line 96 to a substring match on FAISS's "at least as large as number of clusters" message (e.g. `assert msg =~ "clusters"`).

**H2. Concurrency tests cover only Flat/IVF add/search/reset/ntotal/train. The two-lock clustering path and every other read-locked op are never run concurrently with a writer.**
- `concurrency_smoke_test.exs:21-28` and `concurrency_test.exs:23-28,56-59` call only `add`, `search`, `ntotal`, `reset` and `train`.
- Never run concurrently with a writer, in any suite:
  - `nif_train_clustering`, which holds the index lock then the clustering lock (`c_src:907-908`)
  - `nif_get_clustering_centroids` (`:948`)
  - `reconstruct` (`:560`)
  - `compute_residuals` (`:628`)
  - `clone` (`:261`)
  - `to_file` (`:655`)
  - `add_with_ids` (`:348`)
- `reconstruct` and `compute_residuals` racing `reset` is the use-after-free case the read lock exists to prevent.
- Fix:
  - Add to `concurrency_test.exs` (`:slow`) a test where workers race `Index.reset` / `Index.add` against `Index.reconstruct(idx, [0])`, `compute_residuals`, `clone` and `to_file` on one Flat index. Accept `{:error, msg}` with a binary `msg`, because a key can legitimately fall past `ntotal` after a reset.
  - Add a second test that races `Clustering.train(c, data)` against `Clustering.get_centroids(c)` and `Index.search(trained.index, …)` on a shared clustering/quantizer.
  - Put a cheap two-task version in the smoke file.

### Medium

**M1. `Index.train(idx, [])` and `train(idx, <<>>)` silently return `:ok` and leave the index untrained. No test pins this, and it contradicts `Clustering.train/2`.**
- `nif_train_index` short-circuits n=0 to `atom_ok` (`c_src:448-451`).
- `nif_train_clustering` rejects n≤0 with "n must be positive" (`c_src:894`).
- Neither path has a test (inventory #31; no `train(…, [])` in `test/`).
- Fix: add tests that pin the intended contract, for example `assert :ok = Index.train(ivf, []); assert {:ok, false} = Index.trained?(ivf)` and `assert {:error, "n must be positive"} = Clustering.train(c, [])`. Then decide whether to document or unify the asymmetry. That decision is outside a test audit.

**M2. `k <= 0`, non-integer `k` and huge `k` are untested on both search entry points.**
- `c_src:372,376,404,409` (inventory #9–12). `Index.search/3` and `search_binary/3` pass `k` straight through (`index.ex:238,261`).
- A huge `k` is the only user-controlled allocation size. It should come back as an error, not an OOM.
- Fix: in `index_test.exs` `describe "edge cases"`, add:
  - `assert {:error, "k must be positive"} = Index.search(idx, v, 0)`, plus the same for `search_binary`
  - `assert {:error, "invalid arguments"} = Index.search(idx, v, 1.5)`
  - `assert {:error, _} = Index.search(idx, v, 2**62)` (size overflow)

**M3. FAISS-raised errors on misused index types are untested (old #7, #8, #9 plus new ones).**
- `c_src:241` (bad factory string), `:355` (`add_with_ids` on `"Flat"`), `:467` (IVF train with n < nlist), `:573` on IVF without a direct map, and `:633` (residual with key ≥ ntotal).
- These are the paths where a FAISS C++ exception becomes `{:error, _}` through `make_faiss_error`. A regression there aborts the VM, so a wrong return value is not the main risk.
- Fix: one test each, asserting `{:error, msg}` with a substring:
  - `Index.new(4, "NotAnIndex")`
  - `Index.add_with_ids(flat, [v], [1])`
  - `Index.train(ivf4, two_vectors)`
  - `Index.reconstruct(trained_ivf_with_data, [0])`
  - `Index.compute_residuals(idx, [v], [99])`

**M4. CUDA code is never compiled or tested in any workflow.**
- `ci.yml` and `publish.yml` never set `USE_CUDA`, so `test_helper.exs:5-8` always excludes `:cuda`.
- Effects:
  - The `FAISS_GPU_ENABLED` block (`c_src:741-808`, 8 error branches plus the GPU resource destructor) is dead code as far as CI knows.
  - The 3 uncovered Index lines from the cover report are all GPU paths (`index.ex:178` `new(…, device: {:cuda, _})`, `:413` the `cpu_to_gpu/1` default-arg head, `:438` the `gpu_to_cpu` body).
  - The only `:cuda` test (`index_test.exs:205-218`) is happy-path only.
- Fix:
  - Add a `workflow_dispatch` job (self-hosted or GPU runner) that runs `USE_CUDA=true mix test --include cuda --include slow`. At minimum, add a compile-only CUDA job in an `nvidia/cuda:12.x-devel` container so the `#ifdef` block can't bit-rot.
  - Cheap immediate win: add a `:no_cuda` test asserting `{:error, "GPU support not compiled"}` for both `Index.new(4, "Flat", device: {:cuda, 0})` and the 1-arity `Index.cpu_to_gpu(index)`. That covers `index.ex:178` and `:413`.

**M5. The CI FAISS cache key ignores the FAISS build configuration.**
- The key is `${RUNNER_OS}-${RUNNER_ARCH}-faiss-${FAISS_REV}` (`ci.yml:46,163`). The `CMAKE_FLAGS` (`Makefile:88-95`, e.g. `FAISS_OPT_LEVEL`, `FAISS_ENABLE_GPU`) are not part of it.
- The Makefile configure step depends only on `.cloned` (`Makefile:176`), so it never re-runs either.
- A cmake-flag change in the Makefile is never built or tested in CI until someone bumps the FAISS rev.
- Fix: append a hash of the FAISS build inputs to the key, e.g. `hashFiles('Makefile')`. That costs about 10 min of rebuild per Makefile edit. Alternatively, move `CMAKE_FLAGS` into a dedicated include file and hash only that.

**M6. CI pins one toolchain (`ELIXIR_VERSION: "1.19"`, `OTP_VERSION: "28.0"`, `ci.yml:12-13`) while `mix.exs:16` declares `elixir: "~> 1.18"`.**
- The minimum supported Elixir/OTP pair is never built. For a NIF the OTP side matters: the NIF API version and the dirty-scheduler/RW-lock enif surface.
- Fix: add a `{elixir: "1.18", otp: "27"}` row to the Linux test job, or to the `test-native` matrix as a 4th entry.

### Low

**L1. HNSW self-hit assertion: `index_test.exs:302` `assert hd(hd(labels)) == 0`.**
- This is an exact-recall assertion on approximate search, over data from `:rand.uniform` (`:294`). The data differs on every `--seed`.
- In practice it is near-certain for 100 points with M=32, but it is an approximate-search assertion all the same.
- Fix: use fixed data (e.g. `for i <- 0..99, do: List.duplicate(i * 1.0, 8)`), or assert `0 in hd(labels)`.

**L2. `concurrency_test.exs:35,73`: `Task.await_many(tasks, 120_000)` exceeds ExUnit's default 60 s test timeout.**
- There is no `@moduletag timeout:`. On a slow runner (macos-15-intel) the test dies at 60 s with a generic timeout, never at the intended 120 s.
- Fix: add `@moduletag timeout: 180_000`, or lower the awaits below 60 s.

**L3. `concurrency_test.exs:23`: `Enum.random` runs inside `Task` processes.**
- [INFERENCE] ExUnit seeds `:rand` only in the test process from `--seed`. Spawned Tasks get fresh, unseeded state, so a failing interleaving can't be replayed with `mix test --seed N`.
- Fix: build each worker's op list in the test process (`ops = for _ <- 1..@iterations, do: Enum.random(...)`) and pass it into the Task.

**L4. `concurrency_smoke_test.exs:18-33`: no start barrier.**
- Tasks are spawned one by one and start work at once. For a 1-vector add on dim 8 the first worker can finish a meaningful share before the last one starts, so overlap is likely but not guaranteed.
- Fix: have each task `receive do :go -> :ok end` first, then `Enum.each(tasks, &send(&1.pid, :go))`.

**L5. `index_test.exs:365-374`: "search with n=0 returns empty results via NIF" calls the private `FaissEx.NIF.nif_search_index/4` (`@moduledoc false`).**
- The public `search_binary(idx, <<>>, 2)` test at `:683-689` already covers the same C branch (`c_src:378-387`).
- Repo policy says implementation-detail tests should be deleted. Delete this one.

**L6. Redundant and weak shape-only tests.**
- `clustering_test.exs:30-46` checks only list lengths and is subsumed by `:48-76` (argmin correctness). Delete it.
- `clustering_test.exs:13-27` asserts only shapes, although the data has two exact clusters. Assert `Enum.sort(centroids)` ≈ `[[0,0,0,1],[1,0,0,0]]` within delta.
- `index_test.exs:64-80` uses `Enum.take(labels, 1)` plus a second match. Replace with `assert [[0], [2]] = labels`.

**L7. File I/O error assertions are `{:error, _}` (`index_test.exs:328,332`), and corrupt-file input is untested.**
- Fix:
  - Assert `{:error, msg}` with a substring.
  - Add a `@tag :tmp_dir` test that writes `"not an index"` to a file and asserts `{:error, _} = Index.from_file(path)`. That exercises FAISS's parser path, `c_src:684-688`.
  - Add an `io_flags:` round-trip (old #13).

**L8. Coverage is not configured or gated.**
- `mix.exs` has no `test_coverage` key, so `mix test --cover` exits non-zero ("threshold not met", 79.66% vs the default 90%). The cause is the `FaissEx.NIF` stub module.
- CI never runs `--cover`.
- Fix: `test_coverage: [ignore_modules: [FaissEx.NIF], summary: [threshold: 90]]` in `project/0`, so it measures about 96% and passes. Then switch the Linux job to `mix test --include slow --cover`.

**L9. CI jobs set no `timeout-minutes` (none in `ci.yml`).**
- [INFERENCE] A dirty-NIF deadlock in a lock regression can outlive ExUnit's per-test timeout, because the stuck Task can't be killed while it is inside the NIF. The job then hangs until GitHub's 6 h limit.
- Fix: `timeout-minutes: 30` on `test` and `test-native`, and 60 on `setup` for the FAISS build.

**L10. `ci.yml:66-69` skips `deps.get`/`compile` on a deps-cache hit, even when the FAISS cache step (`ci.yml:50-54`, no `id`) missed.**
- [INFERENCE] If only the FAISS entry was evicted, nothing rebuilds `~/.cache/faiss_ex`. The `test` job's `fail-on-cache-miss: true` restore (`:104`) then fails the run.
- Fix: give the FAISS cache step `id: faiss` and run compile when `steps.cache.outputs.cache-hit != 'true' || steps.faiss.outputs.cache-hit != 'true'`.

---

Clean areas (one line each): no `Process.sleep`; tmp paths go through `@tag :tmp_dir`; tags follow the CLAUDE.md conventions (`@tag :cuda` per test, `:no_cuda` gating in `test_helper.exs`); `ArgumentError` assertions include exact messages; the Linux and macOS (x86_64 and arm64) matrix runs `--include slow`.
