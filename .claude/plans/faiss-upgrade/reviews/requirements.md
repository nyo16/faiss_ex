## Requirements Coverage (from Plan file .claude/plans/faiss-upgrade/plan.md)

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 1.1 | Bump `FAISS_GIT_REV` v1.10.0 → v1.14.3 in Makefile | MET | `Makefile:11` |
| 1.2 | Bump `FAISS_GIT_REV` default in mix.exs `make_env/0` | MET | `mix.exs:72` |
| 1.3 | `mix clean && mix compile` builds clean on v1.14.3 | UNCLEAR | build-verification claim, no artifact in diff to confirm |
| 1.4 | `mix test` green on v1.14.3 | UNCLEAR | test-run claim, not verifiable from diff alone |
| 1.5 | Verify `faiss_last_exception` thread_local at v1.14.3 | UNCLEAR | verification of upstream source, not part of this repo's diff |
| 2.1 | rwlocks on all index alloc sites; write lock on mutations, read lock on reads; documented lock order | MET | `c_src/faiss_ex_nif.c:16-31` header, `wrap_index_resource` (~L143), rwlock calls throughout (add/train/reset/search/reconstruct/clone/write/gpu/getters) |
| 2.2 | `nif_reset_index` → dirty CPU | MET | `c_src/faiss_ex_nif.c` nif_funcs entry `{"nif_reset_index", 1, nif_reset_index, ERL_NIF_DIRTY_JOB_CPU_BOUND}` |
| 2.3 | `nif_get_clustering_centroids` → dirty CPU | MET | nif_funcs entry `ERL_NIF_DIRTY_JOB_CPU_BOUND` for `nif_get_clustering_centroids` |
| 2.4 | NUL-byte rejection via `contains_nul` in new_index/write_index/read_index | MET | `contains_nul` helper + calls in `nif_new_index`, `nif_write_index`, `nif_read_index` |
| 2.5 | reconstruct fast path: contiguity check + `faiss_Index_reconstruct_n` + fallback loop | MET | `nif_reconstruct_batch`: contiguous check loop, `faiss_Index_reconstruct_n` call, else per-key loop |
| 2.6 | dim/d/k <= 0 validation with clear messages | MET | `if (dim <= 0) return make_error_msg(env, "dim must be positive");` in nif_new_index; `d <= 0`/`k <= 0` in nif_new_clustering |
| 2.7 | atoms cached via `init_atoms` (on_load + on_upgrade) | MET | `static void init_atoms(...)`, called in both `on_load` and `on_upgrade` |
| 2.8 | n==0 paths use `enif_make_new_binary`; search/residuals validate empty binaries | MET | `nif_search_index` n==0 branch checks `data_bin.size != 0`; `nif_compute_residuals` n==0 branch checks both bins |
| 2.9 | Makefile: `-lc++` Darwin / `-lstdc++` else | MET | `Makefile` diff shows conditional linkage split by `UNAME_S` |
| 2.10 | load_nif handles `{:error, :bad_name}` | MET | `lib/faiss_ex/nif.ex:6-13` |
| 2.11 | concurrency_test.exs :slow (8 workers × 300 ops); NUL tests; dim/d/k tests | MET | `test/faiss_ex/concurrency_test.exs` new file (8 workers, 300 iterations, `@moduletag :slow`); NUL/dim tests present in `test/faiss_ex/index_test.exs` diff (not fully quoted but file diff shows +99 lines) |
| 3.1 | get_cluster_assignment uses `Index.search` on trained quantizer | MET | `lib/faiss_ex/clustering.ex` `get_cluster_assignment/2` now calls `Index.search(quantizer, data, 1)` directly on `clustering.index` |
| 3.2 | `Shared.encode_vectors!/2` single-pass with row-indexed ArgumentError; handles `[]`; old helpers deleted; call sites updated | MET | `lib/faiss_ex/shared.ex` `encode_vectors!/2`, `encode_batch`, `encode_row`; `to_flat_floats`/`validate_length!` removed from `index.ex`/`shared.ex`; all 6 Index call sites (add, add_with_ids, search, train, compute_residuals) + Clustering.train updated |
| 3.3 | explicit metric case in `Index.new/3`, `@metric_map` removed | MET | `lib/faiss_ex/index.ex` case statement replacing `@metric_map`; `@metric_map` attribute deleted from diff |
| 3.4 | `Clustering.train/2` uses `with`; get_cluster_assignment no chain | MET | `lib/faiss_ex/clustering.ex` `train/2` uses `with {:ok, quantizer} <- ... , :ok <- ...` |
| 3.5 | `cpu_to_gpu` clause for `{:cuda, _}` → error | MET | `lib/faiss_ex/index.ex` new `def cpu_to_gpu(%__MODULE__{device: {:cuda, _}}, _device_id)` clause |
| 3.6 | @spec on all public functions + @typedoc; new types `vectors/0`, `error/0` | MET | `@spec` added throughout `index.ex`/`clustering.ex`; `@type vectors`, `@type error` defined in `index.ex`; `@typedoc` on structs |
| 3.7 | `from_file` → `description: nil`; struct type widened | MET | `lib/faiss_ex/index.ex` from_file sets `description: nil`; `@type t` field widened to `String.t() \| nil` |
| 3.8 | `FaissEx.num_gpus/0` returns plain integer | MET | `lib/faiss_ex.ex` new `def num_gpus`, unwraps `{:ok, n}` |
| 3.9 | `Shared.unwrap!/1` deleted | MET | removed from `lib/faiss_ex/shared.ex` diff; no occurrences found via grep |
| 3.10 | tests added incl. `:no_cuda` tag, from_file nil test | MET | `test/test_helper.exs` handles `:no_cuda` exclusion; `test/faiss_ex/index_test.exs` diff (+99 lines, not individually itemized here) |
| 4.1 | Concurrency section in Index moduledoc; k>ntotal padding documented | MET | `lib/faiss_ex/index.ex` moduledoc `## Concurrency` section; `search/3` doc notes k > ntotal padding with label -1 |
| 4.2 | README build table v1.14.3; CUDA 12.x note; Thread Safety rewrite; num_gpus example | MET | `README.md` diff: version table, CUDA 12.x note + v1.11.0 escape hatch, rewritten Thread Safety section, `num_gpus = FaissEx.num_gpus()` example |
| 4.3 | Bench table refreshed with version label + explanatory note | MET | `README.md` diff replaces bench table, adds note on add-1000/add-10000 encode/GC dominance and kmeans assign scenario |
| 4.4 | CHANGELOG v0.2.0 entry; @version 0.2.0 | MET | `CHANGELOG.md` new `## v0.2.0` section; `mix.exs` `@version "0.2.0"` |
| 4.5 | CLAUDE.md updated: pinned version note, locking convention, dirty-scheduler list, Shared description, :no_cuda tag | MET | `CLAUDE.md` diff adds pinned-version note, locking convention paragraph, updated dirty-scheduler list, Shared module description, `:no_cuda` tag doc |

### Deferred table (confirm NOT implemented)

| Deferred item | Status |
|---|---|
| Shared `StandardGpuResources` across indexes | Confirmed not implemented — `nif_index_cpu_to_gpu` still allocates a fresh `FaissStandardGpuResources` per call (`c_src/faiss_ex_nif.c` ~L690-720) |
| `search_with_params`, `remove_ids`, `range_search` | Confirmed absent — no matches in `lib/` or `c_src/` |
| badarg vs `{:error, ...}` for malformed NIF args | Confirmed unchanged — error-tuple style retained throughout |
| Alignment hardening for sub-binary inputs | Confirmed not implemented — no alignment checks added |
| Binary-in/binary-out fast-path API | Confirmed not implemented — public API still list-based |
| decode-then-chunk double allocation | Confirmed unchanged — accepted tradeoff, no change found |

**Summary**: 25 MET · 0 PARTIAL · 0 UNMET · 3 UNCLEAR
