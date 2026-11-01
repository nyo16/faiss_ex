# Code Review: v0.4.0 audit remediation — Elixir API, docs, tests (a397362..master)

Scope: `lib/`, `test/`, `bench/faiss_ex_bench.exs`, plus the README and CHANGELOG sections named in the brief.
I reviewed the code by reading it, and also read the OTP and Elixir sources installed under `~/.asdf` to check boot-order claims. I ran nothing: this session had no shell.

## Summary
- **Status**: ⚠️ Changes Requested
- **Issues Found**: 1 BLOCKER, 4 WARNING, 9 SUGGESTION

---

## BLOCKER

### B1. `config :faiss_ex, omp_threads:` in `config/runtime.exs` is silently ignored in a default Mix release
- `lib/faiss_ex/nif.ex:13,20-21`: `:erlang.load_nif(path, omp_threads_config())` and `Application.get_env(:faiss_ex, :omp_threads, 0)` are read inside `@on_load`.
- `README.md:496-497`: `# config/runtime.exs — many concurrent small requests` followed by `config :faiss_ex, omp_threads: 1`.
- `CHANGELOG.md:40` and `lib/faiss_ex.ex:58` advertise the config key but don't say which config file it must go in.

**Why.** Mix releases default to `RELEASE_MODE=embedded`. Source: `elixir/lib/mix/lib/mix/tasks/release.ex:824-825`, "Defaults to \"embedded\"". In embedded mode the boot order is:
1. Every module is `primLoad`ed up front.
2. `on_load` handlers are deferred. They run while kernel's supervisor starts, during `application:start_boot(kernel)` (`kernel.erl:228-233`, `init(on_load) -> init:run_on_load_handlers()`).
3. Mix inserts `Config.Provider.boot`, which evaluates `runtime.exs`, only after `start_boot(stdlib)` (`mix/lib/mix/release.ex:735-749`, `post_stdlib_applies/2`).
4. When `runtime.exs` is used, `reboot_system_after_config` defaults to `false` (`tasks/release.ex:1260`). No second boot picks the value up.

So `FaissEx.NIF.load_nif/0` runs before `runtime.exs` is applied. `get_env` returns the default `0`, and the documented example does nothing in a standard production release. No error or warning is raised.

Two cases do work:
- Values in `config.exs`/`prod.exs` reach `sys.config` and are applied, because `application:load` runs before kernel starts.
- Interactive `mix run`, `iex` and `mix test` all work, which is why the test suite cannot catch this.

**Fix (pick one):**
- **(a)** Recommended: apply the value after configuration, from an application callback. Add `mod: {FaissEx.Application, []}`. In `start/2`, call `FaissEx.set_omp_threads(validated_config)` and start an empty supervisor. Then either keep `load_info` as the early default or drop it, so only one source remains.
- **(b)** Docs only. Change the README, CHANGELOG and `set_omp_threads` doc to say: use `config/config.exs` (compile-time config); for `runtime.exs` values, call `FaissEx.set_omp_threads/1` from your `Application.start/2`. Also fix the file name in the README example's comment.

---

## WARNING

### W1. `set_omp_threads/1` guard accepts values its NIF rejects (MatchError), and has no sanity cap
- `lib/faiss_ex.ex:77-79`:
  ```elixir
  @spec set_omp_threads(non_neg_integer()) :: :ok
  def set_omp_threads(threads) when is_integer(threads) and threads >= 0 do
    :ok = FaissEx.NIF.nif_set_omp_threads(threads)
  ```
- `c_src/faiss_ex_nif.c:1222-1223`: `if (!enif_get_int(env, argv[0], &threads) || threads < 0) return make_error_msg(env, "invalid arguments");`
- Compare `lib/faiss_ex/nif.ex:22`: the boot path validates `n <= 0x7FFFFFFF`.

**Why.**
- `FaissEx.set_omp_threads(2**31)` passes the guard. `enif_get_int` then fails, the NIF returns `{:error, "invalid arguments"}`, and the `:ok =` match crashes with `MatchError`. That breaks the `:: :ok` spec. It also breaks the `FunctionClauseError` contract asserted at `omp_threads_test.exs:11-14`.
- The runtime path and the config path accept different ranges.
- UNVERIFIED: a huge value that is still in range (e.g. `1_000_000`) is stored and later passed to `omp_set_num_threads`. On Linux, libgomp treats a `pthread_create` failure while forming a team as fatal (`gomp_fatal` → `exit`). That would take the VM down on the next FAISS call. LLVM libomp clamps the value with a warning instead.

**Fix.**
- Use the same upper bound in both places, ideally through one shared guard: `defguardp valid_threads(n) when is_integer(n) and n >= 0 and n <= @max_threads`.
- Consider a practical cap such as `<= 4 * System.schedulers_online()`. At minimum, document that values above the core count oversubscribe the CPU.
- Add `2**31` to the bad-values loop in `omp_threads_test.exs:12`.

### W2. Reconstruct/residual docs list the wrong error for keys ≥ `ntotal` (the common case)
- `lib/faiss_ex/index.ex:351-353` (reconstruct/2): "`{:error, "key out of range"}` for a negative key or a contiguous range that cannot address any index."
- `lib/faiss_ex/index.ex:371` (reconstruct_binary/2): "Returns `{:error, "key out of range"}` for a negative key."
- `lib/faiss_ex/index.ex:385` (compute_residuals/3): "`{:error, "key out of range"}` for a negative key."

What the test suite asserts instead:
- `index_test.exs:765-767`: `Index.reconstruct(index, [5])` on a 2-vector Flat index returns FAISS's message (`=~ "ntotal"`), not `"key out of range"`.
- `index_test.exs:921-926`: `compute_residuals(..., [5])` returns a message matching `=~ "ntotal"`.
- `index_test.exs:895-900`: on `IDMap,Flat`, the call returns `"reconstruct not implemented..."`.
- `index_test.exs:944-955`: IVF with sparse ids returns `"key out of range"` for an id range with unwritten rows (the `UNWRITTEN_ROW_TAG` path).
- "A contiguous range that cannot address any index" really means the INT64_MAX-overflow guard. A caller cannot tell that from the wording.

**Why.** A caller who matches on the documented `"key out of range"` misses the most frequent failure: a key past `ntotal`.

**Fix.** Document both groups of cases in all three docs (`compute_residuals_binary/3` inherits them):
- negative keys, INT64_MAX-overflow keys, and IVF ids with no stored vector → `{:error, "key out of range"}`;
- keys ≥ `ntotal`, and index types without `reconstruct` → `{:error, faiss_message}`.

### W3. Starvation regression test: the timing margin depends on the machine, and the test never checks its own setup
- `test/faiss_ex/concurrency_test.exs:195-214`:
  ```elixir
  # ~1.7 s of k-means on an M-series CPU, longer on CI runners
  training = for _ <- 1..(80_000 * dim), ...
  trainer = Task.async(fn -> Index.train(busy, training) end)
  Process.sleep(200)
  ...
  Process.sleep(200)
  assert {:ok, %{labels: [[0]]}} = Index.search(other, query, 1)
  assert Process.alive?(trainer.pid), ...
  ```

**Why.**
- **Spurious failure.** Training IVF1024 on 80k×64 is about 1e11 FLOP of k-means, run by the default OpenMP team (all cores). On a many-core x86 box (32+ cores, AVX-512) it can finish inside the 400 ms of sleeps. `Process.alive?` is then false, and the failure message wrongly blames the search ("search … waited for training").
- **False pass.** The waiters may not have reached the lock by the end of the second 200 ms sleep. B's search can then run before the dirty-CPU pool is saturated, and the test passes even with the starvation bug present.
- **Confusing failure.** On a loaded CI runner the first sleep can be too short. A waiter then searches the untrained IVF index first and gets `{:error, "...not trained..."}`. That only shows up as `Enum.all?(...) == false` at `:217`.

**Fix.**
- Run the trainer with `FaissEx.set_omp_threads(1)`. The module is already `async: false`; restore the setting with `on_exit`. Train duration then scales with single-core speed rather than core count. Shrink `80_000` if the runtime gets too long.
- `assert Process.alive?(trainer.pid)` before spawning the waiters.
- Just before B's search, assert that every waiter is still pending: `Enum.all?(waiters, &Process.alive?(&1.pid))`.
- `bench/faiss_ex_bench.exs` repeats this pattern in its starvation block. There it only affects the printed numbers, but the same `set_omp_threads(1)` change applies.

### W4. Shared-index race tests never let a reconstruct/residual read succeed while writers run
- `test/faiss_ex/concurrency_test.exs:103` builds `{:ok, index} = Index.new(dim, "IDMap,Flat")`. It races `reconstruct(index, [0])` and `compute_residuals(index, [vec], [0])` (`:157-161`) against `reset`/`add`/`add_with_ids`.
- `test/faiss_ex/concurrency_smoke_test.exs:51` uses the same IDMap index and the same reader ops.

**Why.** On `IDMap,Flat`:
- `reconstruct` and `compute_residual` always throw "reconstruct not implemented" (`index_test.exs:899-900`);
- plain `add` always errors.

So the readers only ever take the error branch, and the path the remediation changed is never raced. That path is the Flat batched `reconstruct_n` after the under-lock `ntotal` check, with `reset` shrinking `ntotal` between the Elixir call and the lock.

Plan 6.2 specified "on one Flat index". Using IDMap covers the former abort path but drops the success path. The `check` closures accept `{:error, _}`, so these tests cannot distinguish a read that raced correctly from a read that never worked.

**Fix.**
- Parameterize the slow test over `["Flat", "IDMap,Flat"]`. On Flat, `add` and `reconstruct([0])` succeed whenever `ntotal > 0`.
- To check the success path, assert that any returned row equals one of the vectors written.
- In the smoke test, either switch to `"Flat"` (writer: `reset` + `add`) or add a Flat reader next to the existing one.

---

## SUGGESTION

### S1. Detectable type mistakes are handled inconsistently: some raise, some return `{:error, _}`
- `index_test.exs:358-359` locks in `{:error, "invalid arguments"}` for `to_file(index, ~c"/x")` and `from_file(path, io_flags: :x)`.
- `Index.new/3` raises `ArgumentError` for a bad `:metric`/`:device` (`index.ex:181-191,208-216`). CLAUDE.md says "caller mistakes the library can detect (bad option/metric/device …) raise `ArgumentError`", and a non-integer `:io_flags` is a bad option value just like a bad `:metric`.
- `search/3` with `k = 1.5` returns an error tuple, while a non-integer key raises `ArgumentError` (`shared.ex:24-26`).
- **Fix.** Either validate `is_integer(io_flags)` in `from_file/2` and raise, or state the narrower rule in the moduledoc: option names and `:metric`/`:device` raise; other bad values return `{:error, "invalid arguments"}`. Either way, document the outcomes for `k <= 0` and a non-integer `k` in the `search/3` doc. Today `index.ex:278` only says "`k` must be a positive integer".

### S2. `cpu_to_gpu/2` and `new/3` raise different exceptions for the same bad device id
- `cpu_to_gpu/2` guards with `is_integer(device_id) and device_id >= 0` (`index.ex:505-506,518-519`), so a negative id raises `FunctionClauseError`. Test: `index_test.exs:803-805`.
- `new/3` goes through `validate_device!` (`index.ex:212-216`) and raises `ArgumentError` with a message.
- **Fix.** Add a fallback clause, `def cpu_to_gpu(%__MODULE__{}, device_id), do: raise ArgumentError, "invalid device_id ..."`, that reuses the `validate_device!` message.

### S3. The Clustering "self-consistent" claim ignores that `:index` is a public, mutable, shared quantizer
- `clustering.ex:22-29` says: "every returned struct stays self-consistent — its centroids and its quantizer always come from the same training run".
- `clustering.ex:51-52` exposes `:index` as a plain `Index.t()`.
- `Index.add(trained.index, ...)` or `Index.reset(trained.index)` mutates the quantizer in place. That changes `get_cluster_assignment/2` for every copy of the struct, while `get_centroids/1` stays the same.
- **Fix.** Add a typedoc or moduledoc note: treat `:index` as read-only, and `Index.clone/1` it before mutating.

### S4. `reconstruct/2` is the only list-returning function that rejects binary keys
- `index.ex:355-356` has `@spec reconstruct(t(), [integer()])` and `when is_list(keys)`.
- `compute_residuals/3` (`index.ex:390-392`) accepts `id_data()` keys and also returns lists.
- **Fix.** Switch to `Shared.encode_id_data!/1` and `id_data()`, a one-line change. Otherwise, call out the exception in the Binary API bullet at `index.ex:55-58`.

### S5. Key-path error messages say "id"
- `shared.ex:25` raises `"id #{n} is not an integer"`, and `shared.ex:59-61` raises `"ids binary has … bytes"`. Both also fire for keys passed to `reconstruct*` and `compute_residuals*`.
- **Fix.** Use neutral wording ("id/key"), or pass a label in, so key errors don't mention ids.

### S6. README "Concurrency tuning" inaccuracies
- `README.md:518` lists `KMP_BLOCKTIME=0 (LLVM libomp)` under "On Linux, also:". On Linux, the NIF and the default FAISS build link libgomp (`Makefile:198-200`); `KMP_BLOCKTIME` is the knob for macOS (Homebrew libomp). Move it to a macOS line.
- `README.md:503` says "Measured on an Apple M1 Max", but the Benchmarks section (`README.md:646`) says "Apple M4 Max".
- The speedup is quoted three different ways: README 1529→586 ms (2.6×), CHANGELOG `:42` "1.5 s → 0.57 s", and `faiss_ex.ex:66` "~2.8×". Pick one run and quote it everywhere.

### S7. The README's `FAISS_GIT_SHA` default only holds for the default rev
- `README.md:130` gives the default as "commit of `v1.15.1`". `Makefile:26-29` applies that default only when `FAISS_GIT_REV` is the default. A custom rev builds unverified (with a warning) unless the SHA is supplied.
- **Fix.** Reword the default to: "commit of `v1.15.1` when `FAISS_GIT_REV` is the default; otherwise empty (unverified, warns)".

### S8. The race test writes about 300 index files per run
- `concurrency_test.exs:167` uses `Path.join(tmp_dir, "race-#{w}-#{i}.index")`, one file per `:to_file` op. That is 4 readers × 300 iterations × ¼ ≈ 300 files, and ExUnit leaves `tmp_dir` in place until the next run.
- **Fix.** Use one path per reader (`"race-#{w}.index"`) and overwrite it each time. Concurrency is still exercised, and the file count stays at 4.

### S9. Test names and assertions that claim more than they observe
- `clustering_test.exs:125` is named "rejects empty training data before allocating anything", but nothing observes allocation. Rename it ("…returns n must be positive") or drop the claim.
- `index_test.exs:1032` (misaligned input): arm64 and x86-64 both tolerate misaligned `float`/`int64` loads, so this test passes with or without `aligned_input`. It is a smoke test, not a regression test. Say so in its comment so nobody relies on it to catch removal of the copy.
- `index_test.exs:306-310` and `:1007-1010` both assert that `cpu_to_gpu` returns "GPU support not compiled" on a non-CUDA build. Merge them into the `@describetag :no_cuda` block.

---

## Clean areas
- `Keyword.validate!` in `Index.new/3` and `from_file/2` has correct defaults, and unknown keys raise. Tested at `index_test.exs:785-789` and `:362-366`.
- `validate_device!/1` rejects `:gpu`, `{:cuda, -1}`, `{:cuda, 0.0}` and `:cuda` before any NIF call. Tested at `:791-797`.
- `add_with_ids/3` raises on a count mismatch, including binary forms and `[]` vs `[1]` (`index.ex:259-261`; tests `:561-573`, `:700-706`).
- Binary input is accepted by `search/3`, `compute_residuals/3`, `compute_residuals_binary/3`, and Clustering train/assignment. Result shapes and decoding match the list paths, and tests cover each.
- Clustering value semantics: `train/2` allocates a fresh clustering and quantizer, never touches the input `ref`, and failures leave the input untrained. Resources are freed by GC only, and nothing keeps a pointer from one resource to another after training. Tests: `clustering_test.exs:38-73`.
- `FaissEx.NIF` validates `load_info` (`nif.ex:20-28`) with the same range as the C `enif_get_int`. Raising inside `@on_load` fails the load with a readable reason.
- Every changed function's `@spec` matches its behavior, except `set_omp_threads/1` (W1).
- The README "Binary API" and "Saving and loading" trust-boundary text match the `from_file/2` docs and the NIF behavior.
- Async safety:
  - `omp_threads_test` and `concurrency_test` are `async: false`, because they touch global OMP state and saturate the dirty pool.
  - `on_exit` restores the thread count.
  - Random op lists are drawn in the test process, so `--seed` reproduces them.
- Bench: `Index.new` moved to `before_each`; IVF, HNSW and shuffled-key scenarios are present; `set_omp_threads` is restored in `after_scenario`.
