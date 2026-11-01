# Architecture Review — faiss_ex (2026-11-01, a397362, v0.3.2 / FAISS v1.15.1)

Scope: `lib/**`, `c_src/faiss_ex_nif.c`, `mix.exs`, `Makefile`. Spec = `CLAUDE.md`
(locking, lock-unwinding, dirty-scheduler, data-flow contracts).

Not a Phoenix app: Phoenix contexts / Scopes / verified routes / FallbackController /
PubSub / LiveView / Ecto criteria are N/A (no deduction).

## Score: 71/100

Rubric adaptation (Architecture, scoring-methodology.md):

| Criterion (adapted) | Max | Deductions | Score |
|---|---|---|---|
| Context boundaries → **layer boundaries** (Index/Clustering → NIF → C → libfaiss_c; each concern owned by the right layer) | 25 | −5 × 1 (GPU-resources lifetime owned by the Elixir struct, not the C resource — M1) | 20 |
| Module naming → **naming & convention consistency** across public modules | 15 | −3 × 3 (`d` vs `dim`, `get_` prefix split, raise-vs-`{:error,_}` split for the same mistake — L3, L4) | 6 |
| Fan-out < 5 per module | 15 | 0 (max fan-out 3: `Clustering` → `NIF`, `Shared`, `Index`) | 15 |
| API surface < 30 funcs **per public module**, incl. uniform input forms | 15 | −5 × 1 (list/binary input handled non-uniformly across Index/Clustering — L2). Sizes OK: Index 18, Clustering 4, FaissEx 1, NIF 20 internal | 10 |
| No compile-time cycles | 15 | 0 (`mix xref graph --format stats` → 0 cycles, pre-verified) | 15 |
| Folder structure → **project/build layout conventions** (single source of build config, correct rebuild keys) | 15 | −5 (rev/SHA/defaults duplicated Makefile↔mix.exs, no drift guard — L1) −5 (build artifacts not keyed by `USE_CUDA`/`FAISS_OPT_LEVEL` — M3) | 5 |
| **Total** | 100 | | **71** |

`20 + 6 + 15 + 10 + 15 + 5 = 71`. Trend: 88 → 71. The drop comes from the wider scope
(build config, input uniformity, struct semantics — not checked last time) plus the
partial input-form asymmetry the binary API introduced (b5127f1). All three prior
findings are resolved or accepted, and the C layer matches the spec exactly (table below).

## Prior findings status

1. **Dead `Shared.floats_to_binary/1`**: **RESOLVED.** It is gone from `lib/faiss_ex/shared.ex`
   (no definition in lines 1–93). The test-only encoders are now private to the tests
   (`test/faiss_ex/shared_test.exs:82-84`). `binary_to_floats/1` and `binary_to_int64s/1`
   (`shared.ex:5,10`) are still public, but `decode_rows` uses them (`shared.ex:32,39`), so
   they are not dead.
2. **`nif_new_index`/`nif_new_clustering` on normal schedulers**: **STILL OPEN, ACCEPTED.**
   They are still flag `0` (`c_src/faiss_ex_nif.c:1028,1047`). CLAUDE.md now writes this down as
   spec, and neither NIF takes a lock, so this is no longer a spec deviation and costs no
   points. A related item that is not accepted is `nif_get_num_gpus` (L7).
3. **Metric enum duplicated as magic numbers**: **RESOLVED (drift guarded).** The literals
   `:l2 → 1` and `:inner_product → 0` (`lib/faiss_ex/index.ex:152-166`) are pinned by
   `_Static_assert(METRIC_INNER_PRODUCT == 0 …)` and `_Static_assert(METRIC_L2 == 1 …)`
   (`c_src/faiss_ex_nif.c:22-23`). If FAISS renumbers the enum, the build fails. Remaining gap,
   not scored: the NIF casts any int, `(FaissMetricType)metric` (`c_src/faiss_ex_nif.c:237`),
   so the Elixir `case` is the only range check. Passing an atom and mapping it in C would
   remove the duplication entirely.

## C layer vs CLAUDE.md spec

All 20 entries in `nif_funcs` (`c_src/faiss_ex_nif.c:1028-1049`) match the 20 stubs in
`lib/faiss_ex/nif.ex:18-41` by name and arity.

| NIF | Lock (lines) | Scheduler | Validation + alloc above lock / terms after unlock | Matches spec |
|---|---|---|---|---|
| `nif_new_index` | none | normal (1028) | n/a | ✅ |
| `nif_clone_index` | R 261–263 | CPU (1029) | wrap/term after unlock | ✅ |
| `nif_add_to_index` | W 304–307 | CPU (1030) | size check 296–302 above | ✅ |
| `nif_add_with_ids_to_index` | W 348–353 | CPU (1031) | size checks above | ✅ |
| `nif_search_index` | R 416–421 | CPU (1032) | checks + both `enif_alloc_binary` 408–414 above | ✅ |
| `nif_train_index` | W 462–465 | CPU (1033) | size check above | ✅ |
| `nif_reset_index` | W 481–483 | CPU (1034) | n/a | ✅ |
| `nif_reconstruct_batch` | R 560–569 | CPU (1035) | key/contiguity checks + alloc 553 above | ✅ |
| `nif_compute_residuals` | R 628–630 | CPU (1036) | checks + alloc 621 above | ✅ |
| `nif_write_index` | R 655–657 | IO (1037) | `dup_cstr` above, free after | ✅ |
| `nif_read_index` | none | IO (1038) | n/a | ✅ |
| `nif_get_index_dim` | none (cached `res->dim`, 209) | normal (1039) | n/a | ✅ |
| `nif_get_index_ntotal` | R 717–719 | CPU (1040) | term after unlock | ✅ |
| `nif_get_index_is_trained` | R 730–732 | CPU (1041) | term after unlock | ✅ |
| `nif_index_cpu_to_gpu` | R 758–760 | IO (1043) | `StandardGpuResources_new` above | ✅ |
| `nif_index_gpu_to_cpu` | R 795–797 | IO (1044) | wrap after unlock | ✅ |
| `nif_get_num_gpus` | none | normal (1045) | n/a | ✅ (see L7) |
| `nif_new_clustering` | none | normal (1047) | n/a | ✅ |
| `nif_train_clustering` | W idx 907 → W clust 908; single `unlock:` 923–925, mirror order | CPU (1048) | Validation under lock is spec-sanctioned (reads `faiss_Clustering_d`); status-only (`err`/`ret`), terms 927–929 after unlock | ✅ (see L6) |
| `nif_get_clustering_centroids` | R 948; single `unlock:` 969–970 | CPU (1049) | Alloc 962 under lock is spec-sanctioned (size comes from locked state); terms after unlock | ✅ |

No hand-written per-branch unlocks. No term construction or `make_*error` under a lock.
The only two-lock function takes the locks quantizer → clustering. Every locking NIF is dirty.

## New findings

### M1 — Medium: GPU-resources lifetime is owned by the Elixir struct, not the C index resource
- **Where:** `c_src/faiss_ex_nif.c:62-71` (`IndexResource` has no GPU-resources field),
  `:766-776` (`gpu_res_resource` is returned as a separate term). `lib/faiss_ex/index.ex:85,415-425`
  (the struct field `gpu_resources_ref` is the only thing keeping it alive).
- **Evidence:** The `GpuIndex` points into `StandardGpuResources`, but the C side keeps no
  reference from the index resource to the GPU-resources resource. Every hot-path function
  binds only `ref` (e.g. `search/3` at `index.ex:235`, `add/2` at `:199`). If the caller passes
  its last copy of the struct (`Index.search(make_gpu_index(), q, k)`), or rebuilds the struct
  without `gpu_resources_ref`, the GPU resources can be collected while the index resource is
  still live. That is a use-after-free in the destructor at `faiss_ex_nif.c:113-120`.
  [INFERENCE] I have not reproduced the GC timing; CUDA tests are excluded locally.
- **Fix:** Add `GpuResourcesResource *gpu_res` to `IndexResource`. Call `enif_keep_resource`
  in `nif_index_cpu_to_gpu`, and `enif_release_resource` in `index_resource_destructor`
  *after* `faiss_Index_free`. Return only the index ref, and remove the `gpu_resources_ref`
  field and its typedoc from `Index` (clean cutover).

### M2 — Medium: `Clustering` mixes value-struct semantics with a shared mutable resource
- **Where:** `lib/faiss_ex/clustering.ex:73-81`, `:88-91`, `:104-111`.
- **Evidence:** `train/2` mutates the shared `ref` in place but returns a *new* struct with a
  fresh quantizer and `trained?: true`. `get_centroids/1` reads the shared `ref`, while
  `get_cluster_assignment/2` reads the struct's own quantizer. After
  `{:ok, a} = train(c, x1); {:ok, b} = train(c, x2)`:
  - `get_centroids(a)` returns x2-trained centroids.
  - `get_cluster_assignment(a, …)` searches x1's quantizer.
  - `c.trained?` is `false`, yet `get_centroids(c)` succeeds.

  `Index` avoids this by making `train/2` return `:ok` and reading `trained?` from C
  (`index.ex:272-275,403-405`). The struct also exposes the quantizer publicly (`:index`,
  `clustering.ex:35-37`), so `Index.add(clustering.index, …)` silently corrupts assignments.
  `Clustering` also reaches into `quantizer.ref` (`clustering.ex:79`), which is an opaque
  `Index` field.
- **Fix:** Pick one model.
  - *Value model:* allocate a fresh `nif_new_clustering` resource inside `train/2`.
  - *Resource model (matches `Index`):* store the quantizer in `ClusteringResource` on the C
    side, via `enif_keep_resource`. `train/2` returns `:ok`, `trained?/1` reads C, and
    `get_cluster_assignment/2` searches the C-held quantizer.

  Either way, drop the `:index` and `:trained?` struct fields.

### M3 — Medium: build cache and NIF artifact are not keyed by `USE_CUDA` / `FAISS_OPT_LEVEL`
- **Where:** `Makefile:28-29` (the source and in-tree `build/` dir are keyed only by
  `FAISS_GIT_REV`), `:176-178` (cmake configures only when `build/Makefile` is missing),
  `:69,89,92` (config-dependent flags), `:210` (`$(NIF_SO)` depends only on the `.c` file and
  the libs).
- **Evidence:** Flip `USE_CUDA=true` after a CPU build in the same `MIX_ENV`, and nothing is
  reconfigured or rebuilt:
  - cmake keeps `FAISS_ENABLE_GPU=OFF`.
  - The existing `priv/libfaiss_ex.so` (built without `-DFAISS_GPU_ENABLED`) is up to date
    by mtime, so `FaissEx.num_gpus/0` silently returns 0 and `cpu_to_gpu` says
    "GPU support not compiled".
  - If the NIF were rebuilt, it would include `c_api/gpu` headers against a CPU-only
    `libfaiss_c`, and on macOS `-undefined dynamic_lookup` (`Makefile:57`) defers that failure
    to load time.

  [INFERENCE] I traced this from the make rules; I did not reproduce it. `FAISS_OPT_LEVEL`
  has the same problem.
- **Fix:** Key the build on the full config, e.g.
  `FAISS_BUILD := $(FAISS_SRC)/build-$(USE_CUDA)-$(FAISS_OPT_LEVEL)` with `cmake -B $(FAISS_BUILD)`.
  Write a `$(PRIV_DIR)/.build_config` stamp file whose content is those vars; rewrite it only
  when the content changes, and make `$(NIF_SO)` and the lib copies depend on it.

### L1 — Low: FAISS rev/SHA and every build default are stated twice (three times for the rev), with no drift guard
- **Where:** `Makefile:13-24` (rev at :14 *and* again in `ifeq` at :19, SHA at :20,
  repo/CUDA/opt defaults). `mix.exs:9-10,72-85` (the same rev, SHA, repo URL, `USE_CUDA`, and
  `FAISS_OPT_LEVEL` defaults, plus the "default SHA only for default rev" logic duplicated
  at :76 vs `Makefile:19`).
- **Evidence:** `make_env/0` always passes every variable, so under `mix compile` the
  Makefile defaults are dead. CI derives its cache key from the *Makefile* rev
  (`.github/workflows/ci.yml:44,162`), but the build uses the *mix.exs* rev, so a half-done
  bump builds one FAISS under another's cache key. If you bump `Makefile:14` but not `:19`,
  direct `make` silently builds unverified (empty SHA, `Makefile:22`). Nothing checks that
  the two files agree; CLAUDE.md just says "keep them in sync".
- **Fix:** Make the Makefile the single source. `make_env/0` should forward only variables
  that are actually set (the way `FAISS_PREFIX` is handled at `mix.exs:87-90`), and the
  `@faiss_git_rev`/`@faiss_git_sha` attributes should go. In the Makefile, introduce
  `FAISS_DEFAULT_REV := v1.15.1`, used by both `FAISS_GIT_REV ?=` and the `ifeq`. Update the
  `sed` in ci.yml and the CLAUDE.md sync note to match.

### L2 — Low: list/binary input forms are handled non-uniformly
- **Where:** `lib/faiss_ex/index.ex:41-44` claims "Every hot-path function also works directly
  with raw binaries". But:
  - `search/3` is list-only (`:235`, a binary raises `FunctionClauseError`).
  - `compute_residuals/3` is list-only for both args (`:337-338`).
  - `reconstruct/2` takes list keys only (`:303`), while `reconstruct_binary/2` takes both
    (`:321-322`).
  - `Clustering.train/2` and `get_cluster_assignment/2` are list-only (`clustering.ex:73,108-109`).

  The binary validators `encode_vector_data!/2` and `encode_id_data!/1` are private to `Index`
  (`index.ex:453-481`), so `Clustering` cannot reuse them without duplicating them.
- **Fix:** Move `encode_vector_data!/2` and `encode_id_data!/1` into `Shared` next to
  `encode_vectors!/2` and `encode_ids!/1`. Accept `vector_data()`/`id_data()` in `search/3`
  input, `reconstruct/2` keys, `compute_residuals/3`, and `Clustering.train/2`. Output shape
  stays keyed by the function, not the input. Fix the moduledoc claim.

### L3 — Low: the same mistake raises in one function and returns `{:error,_}` in another
- **Where:** `lib/faiss_ex/index.ex:136-139` (the docs admit it), `:221` (`add_with_ids/3`
  computes and then discards `_n_ids`, leaving the mismatch to the NIF's
  `"binary size mismatch"`), `:342-345` (`compute_residuals/3` raises `ArgumentError`).
- **Fix:** In `add_with_ids/3`, compare `n_ids` to `n` and raise the same `ArgumentError` as
  `compute_residuals/3`. Every caller-shape error then raises, as `encode_*!` already does,
  and only FAISS-reported failures are `{:error,_}`. Delete the "not uniform" paragraph.

### L4 — Low: naming drift between `Index` and `Clustering`
- **Where:** The struct field and `new/2` arg are `d` (`clustering.ex:27,59`), vs `dim`
  (`index.ex:85,148`). `get_centroids/1` and `get_cluster_assignment/2`
  (`clustering.ex:88,104`) carry a `get_` prefix, vs the bare `ntotal/1`, `dim/1`, `trained?/1`
  (`index.ex:391,397,403`).
- **Fix:** In a 0.x minor bump, rename to `dim`, `centroids/1`, and `assign/2`
  (or `cluster_assignment/2`) with a CHANGELOG entry and no aliases.

### L5 — Low: options are unvalidated
- **Where:** `lib/faiss_ex/index.ex:149-150,372` use `Keyword.get`, so a typo such as
  `metrc: :inner_product` is silently ignored. A bad `:device` value hits the `case` at
  `:176-179` and raises `CaseClauseError`, while a bad `:metric` gets an `ArgumentError`
  (`:164`).
- **Fix:** Use `Keyword.validate!(opts, metric: :l2, device: :host)` (and `io_flags: 0`), plus
  an explicit `other -> raise ArgumentError` clause for `:device`.

### L6 — Low: `ClusteringResource` doesn't cache `d`, so `nif_train_clustering` validates under the lock and never checks the quantizer's dim
- **Where:** `c_src/faiss_ex_nif.c:73-76` (no `d` field; compare `IndexResource.dim` at
  :62-71), `:907-918` (size check inside the critical section), `:894`
  (`n <= 0` is an error, while `nif_train_index` returns `:ok` for `n == 0`, :447-451).
  Through Elixir, `Clustering.train(c, [])` therefore allocates a quantizer
  (`clustering.ex:78`) and then fails.
- **Fix:** Cache `d` at `nif_new_clustering`, since it is immutable like `Index::d`. Move the
  size check above the locks, and add `if (idx_res->dim != clust_res->d) return error`. The
  critical section then holds only `faiss_Clustering_train`, with no `goto`. Treat `n == 0`
  the same way across both train NIFs.

### L7 — Low: `nif_get_num_gpus` runs on a normal scheduler and calls into CUDA
- **Where:** `c_src/faiss_ex_nif.c:811-815,1045`.
- **Evidence:** Under `USE_CUDA=true` it calls `faiss_get_num_gpus()` → `cudaGetDeviceCount`,
  which initializes the CUDA driver on first call. [INFERENCE] That can take from ~100 ms to
  seconds. The spec's dirty rule only covers lock waits, but this is a long-running call on
  a normal scheduler thread.
- **Fix:** Flag it `ERL_NIF_DIRTY_JOB_IO_BOUND` and update CLAUDE.md's dirty-scheduler list.

### L8 — Low: per-NIF size-validation block is copy-pasted 6×
- **Where:** `c_src/faiss_ex_nif.c:289-302, 329-345, 375-397, 447-460, 504-512, 593-611`.
  Each repeats the `n < 0` check, the `n == 0` special case, and the
  `safe_mul(n, dim) → safe_mul(·, sizeof(float))` check against `bin.size`.
- **Fix:** Add one `static const char *check_rows(size_t n, size_t row_bytes, const ErlNifBinary *bin)`
  helper that returns an error message or NULL. It keeps validation above the lock and
  shrinks each NIF to the lock plus the `faiss_*` call.

### L9 — Low: stale "lists only" docs after the binary API
- **Where:** `lib/faiss_ex.ex:6-7` ("All data flows as plain Elixir lists"),
  `lib/faiss_ex/index.ex:6` ("Vectors are passed as flat lists … or lists of lists").
- **Fix:** Mention the binary forms, or link to `FaissEx.Index`'s *Binary API* section.

Other areas, one line each: there is no layer bypass (only `FaissEx`, `Index`, and
`Clustering` call `FaissEx.NIF`; `bench/` uses `Shared` only to pre-encode). There are no
compile cycles. The duplicated install recipes for the system-FAISS and source paths
(`Makefile:132-146` vs `:191-207`) are cosmetic.
