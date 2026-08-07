# Test Health Audit — faiss_ex

**Test health score: 62/100** — solid happy-path + error-message coverage and a genuinely useful (if opt-in) concurrency stress suite, but no test exercises NIF-resource GC/destructor paths, several error paths around untrained/unsupported operations are unverified, and the concurrency suite is excluded from every default `mix test` run so the riskiest code path (per-resource RW locks) is effectively unverified in CI.

## Coverage Gaps

- **`Index.reconstruct/2` on an index type that doesn't support direct reconstruction** (e.g. `IVF` without `make_direct_map`, or `HNSW`) — no test. `reconstruct` is only exercised against `"Flat"` (index_test.exs:124-164). If FAISS raises/returns an error for unsupported reconstruction, that C error-mapping path is untested. **Medium.**
- **`Index.compute_residuals/3` batch case** — only a single-vector, zero-residual case is tested (index_test.exs:223-234). No test with a batch (`n > 1`) or non-zero residuals (i.e. actually verifying `xs - reconstruct(keys)` arithmetic against distinct values). **Medium.**
- **`Index.add_with_ids/3` on a plain `"Flat"` index** (no `IDMap`) — only tested against `"IDMap,Flat"` (index_test.exs:193-205). FAISS's `add_with_ids` on a bare Flat index should error; this rejection path is unverified. **Medium.**
- **Invalid/garbage factory description string** — no test that `Index.new(dim, "NotARealFactoryString")` returns `{:error, _}`. Only NUL-byte rejection is tested (index_test.exs:452-454). This is a plausible common user mistake. **Medium.**
- **`from_file/2` with `:io_flags` option** — the option exists (index.ex:293-299) but no test passes anything other than the default. **Low.**
- **`Clustering.get_centroids/1` before `train/2`** — untrained-clustering error path is tested for `get_cluster_assignment/2` (clustering_test.exs:102-107) but not for `get_centroids/1`. Calling `nif_get_clustering_centroids` on an untrained ref is unverified — could be a crash risk if the C side doesn't guard it. **High** (untested C-level guard on possibly-uninitialized data).
- **`Index.search/3` with a query vector of the wrong dimension** — dimension-mismatch `ArgumentError`s are tested for `add/2` (index_test.exs:398-420) but not for `search/3`, `train/2`, `add_with_ids/3`, `compute_residuals/3`, or `reconstruct/2` (keys/xs length mismatch). Since all route through the same `Shared.encode_vectors!/2`, risk is low, but zero direct evidence exists for these call sites. **Low.**
- **`gpu_to_cpu/1` called on a host index** — the function only has a clause for `device: {:cuda, _}` (index.ex:371); calling it on a host index raises `FunctionClauseError`. No test documents/asserts this behavior. **Low.**
- **GC/destructor / resource-lifecycle test** — nothing exercises an index or clustering resource going out of scope and being collected while a lock is (or isn't) held, nor verifies no crash/leak when a `%Index{}` reference is dropped mid-use. Given the module docs make explicit claims about `ErlNifRWLock` + destructor safety (index.ex:39-53), there is zero test evidence for this contract. **High.**

## Error-Path Coverage

- Dimension mismatches: covered well for `add/2`/`add_with_ids/3` (index_test.exs:391-465), not for `search/3`, `train/2`, `compute_residuals/3`, `reconstruct/2` (see above).
- Empty vectors: `add(index, [])` no-op tested (index_test.exs:430-434); no equivalent test for `search(index, [], k)` or `train(index, [])`.
- Untrained-index operations: `Clustering` has an explicit untrained test (clustering_test.exs:102-107), but `Index` has none — e.g. `search`/`add` on a freshly-created `"IVF256,Flat"` index before `train/2` is only implicitly exercised inside the concurrency test (concurrency_test.exs:43-80), which is `:slow`-tagged and excluded by default. No dedicated, always-run unit test asserts `Index.add`/`Index.search` on an untrained IVF index returns `{:error, _}` rather than crashing. **High** — this is the single biggest "untrained state" gap and currently only has coverage behind an excluded tag.
- Invalid factory strings: not tested (see above).
- Bad file paths for `read_index`: covered (`from_file` on nonexistent path, index_test.exs:289-291).

## Concurrency

- A dedicated `concurrency_test.exs` exists and is well-designed (randomized interleavings of add/search/reset/ntotal, and train racing search/add on IVF, asserting the index remains functional afterward). This is good.
- **Critical process gap**: the entire file is tagged `@moduletag :slow`, and `test/test_helper.exs:1` excludes `:slow` by default. `mix test` (the command documented in CLAUDE.md and run in CI-equivalent workflows) **never executes the concurrency suite**. The RW-lock code path described as the core safety contract in `index.ex`'s moduledoc has no default-run coverage. This should either be split into a fast smoke-level concurrency test that runs by default (small `@iterations`/`@workers`) or CI must be confirmed to run `mix test --include slow`. No evidence of the latter was found in the repo (no `.github/workflows` inspected as part of this scope, but nothing in `test/` or `CLAUDE.md` mandates it beyond documentation of the flag). **Critical.**

## Quality

- **Assertion strength**: generally good — most tests use pattern matching (`assert {:ok, %{...}} = ...`) rather than loose `{:ok, _}`, and `index_test.exs:57-61`/`145-164` verify exact labels/distances with `assert_in_delta`. A few exceptions:
  - `clustering_test.exs:95-96`: `assert {:error, msg} = ...; assert is_binary(msg)` — accepts any error message. Acceptable for a message that may change across FAISS versions, but weaker than the exact-string assertions used elsewhere in this same file (clustering_test.exs:80-86) — inconsistent rigor. **Low.**
  - `index_test.exs:468-471`: `FaissEx.num_gpus/0` test only asserts non-negative integer — fine given no-GPU CI, but provides no regression protection for the actual GPU count on `:cuda`-tagged runs (no `:cuda`-tagged assertion of a specific expected count exists elsewhere either). **Low.**
- **Test independence / tmp files**: `index_test.exs:172`, `296`, `359` and similarly-patterned tests build tmp file names via `:rand.uniform(100_000)` instead of `System.unique_integer([:positive])` or ExUnit's `tmp_dir: true` setup tag. With `async: true` (index_test.exs:2) and a 100k-value space, collision probability is low but non-zero, and this is exactly the kind of intermittent, hard-to-reproduce flake `tmp_dir: true` (or `unique_integer`) eliminates entirely. Three separate tests each mint their own tmp path independently rather than sharing a `setup` helper — duplicated logic, `Medium` for the collision risk, `Low` for the duplication.
- **Tag usage**: matches CLAUDE.md conventions correctly — `@tag :cuda` before individual tests (index_test.exs:208, 270), `@moduletag :slow` inside a describe-free module for concurrency (concurrency_test.exs:9). No violations found.
- **Flaky-risk patterns**: no `Process.sleep`, no ordering assumptions beyond FAISS's own deterministic exact-search guarantees (reasonable to rely on for `"Flat"` L2/IP). Concurrency tests correctly avoid asserting on interleaving outcomes and only assert "didn't crash + still functional," which is the right level of rigor for this kind of test — no flakiness identified there.

## Resource Cleanup

- No test exercises the destructor/GC path at all — e.g., no test that drops a reference (`index = nil; :erlang.garbage_collect()`) and confirms no crash, no test for GC of a GPU-resource-holding struct, no test for concurrent GC-during-use. Given the module explicitly documents destructor-based safety as part of its concurrency contract, this is a meaningful blind spot for a NIF-resource library. **High.**

## Clean Areas (no issues)

- Factory-string / index-type diversity (Flat, IVF, HNSW, IDMap) is well covered for the happy path.
- File I/O round-trip (write/read, search-result equivalence, description nullness after load) is thoroughly tested.
- `clone/1` independence is tested correctly (mutation of one doesn't affect the other).
- ArgumentError paths from `Shared.encode_vectors!/2` (ragged batches, non-numbers, wrong length, non-list rows) are all covered with exact message assertions.
- GPU error paths for non-CUDA builds are explicitly tagged and tested (`:no_cuda`).
