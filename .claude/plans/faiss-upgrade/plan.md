# Plan: FAISS v1.14.3 Upgrade + Code Review Fixes + Docs

**Status:** COMPLETE (2026-07-11) — all 31 tasks done + release prep; 4 commits on master; awaiting user decision on review/publish
**Depth:** standard
**Created:** 2026-07-11
**Research:** `.claude/plans/faiss-upgrade/research/` (faiss-release-research.md, c-nif-review.md, elixir-review.md)

## Goal

Upgrade the pinned FAISS from v1.10.0 to v1.14.3, fix the correctness and
performance issues found in code review (one critical, one high), and improve
docs (@spec coverage, concurrency semantics, README accuracy). Release as v0.2.0.

## Key research findings

- **Latest FAISS tag: v1.14.3** (2026-06-13). **No breaking C API changes**
  since v1.10.0 — every signature the NIF uses was verified identical on
  current `main`. Upgrade is config-only for the build.
- **CUDA 12.x required** as of FAISS 1.12.0 (CUDA 11 dropped). Cmake flag
  `FAISS_ENABLE_RAFT` → `FAISS_ENABLE_CUVS` rename doesn't affect us (we set neither).
- **[C1/critical]** No locking in the NIF: concurrent `add`/`train`/`reset` racing
  `search` on the same index is undefined behavior in FAISS — two plain Elixir
  processes can segfault the VM. Needs `ErlNifRWLock` per resource.
- **[H1/high]** `nif_reset_index` runs on a normal scheduler; O(ntotal) dealloc
  blocks the VM. Must be dirty CPU.
- **[Elixir/high]** `Clustering.get_cluster_assignment/2` rebuilds and re-populates
  a Flat quantizer on **every call**, ignoring the already-trained
  `clustering.index` — the trained quantizer already contains the centroids.
- Memory-safety audit of the C layer came back clean (error-path frees, binary
  ownership, overflow guards all verified sound).

---

## Phase 1 — FAISS v1.14.3 upgrade [build]

- [x] 1.1 Bump `FAISS_GIT_REV ?= v1.10.0` → `v1.14.3` in `Makefile:11`
- [x] 1.2 Bump the `FAISS_GIT_REV` default in `mix.exs:72` (`make_env/0`) to `v1.14.3`
- [x] 1.3 `mix clean && mix compile` — built clean. Metal claim CONFIRMED:
      `FAISS_ENABLE_METAL:BOOL=ON` by default on Apple Silicon in v1.14.3, but
      our `faiss`/`faiss_c` targets get NO Metal linkage (otool: only libomp,
      Accelerate, libSystem, libc++) — it only adds OBJCXX language + a separate
      backend target we don't build. No cmake pin needed. One benign upstream
      warning (C-linkage in FAISS's own Clustering_c.cpp). OpenMP hints kept.
- [x] 1.4 `mix test` — 27 passed, 1 excluded (cuda), 0 failures on v1.14.3
- [x] 1.5 Verified: `thread_local std::exception_ptr faiss_last_exception` in
      `c_api/error_impl.cpp:15` (+ extern in error_impl.h) at tag v1.14.3

**Verify:** `mix compile` clean, `mix test` green.

## Phase 2 — C NIF fixes (`c_src/faiss_ex_nif.c`) [c]

- [x] 2.1 **[C1/critical]** rwlocks added — `wrap_index_resource` helper covers
      all index alloc sites; write lock on add/add_with_ids/train/reset/
      train_clustering (lock order: quantizer index → clustering, documented in
      file header); read lock on search/reconstruct/residuals/write/clone/
      gpu transfers/centroids/getters. NOTE beyond plan: dim/ntotal/is_trained
      getters moved to DIRTY_JOB_CPU_BOUND — they now block on the rwlock and a
      writer can hold it for seconds (same rationale as H1).
- [x] 2.2 **[H1]** `nif_reset_index` → dirty CPU
- [x] 2.3 **[M4]** `nif_get_clustering_centroids` → dirty CPU
- [x] 2.4 **[M3]** NUL-byte rejection via `contains_nul` helper in new_index /
      write_index / read_index (guards `size > 0` before memchr)
- [x] 2.5 **[M2]** reconstruct fast path — contiguity check `keys[i] ==
      keys[i-1]+1`, batched `faiss_Index_reconstruct_n`, per-key loop fallback
- [x] 2.6 **[L4]** `dim <= 0` → "dim must be positive"; `d`/`k` → "d must be
      positive" / "k must be positive"
- [x] 2.7 **[L2]** atoms cached in statics via `init_atoms` (on_load + on_upgrade)
- [x] 2.8 **[L8]** n==0 paths use `enif_make_new_binary`; search/residuals also
      validate the input binaries are empty when n==0
- [x] 2.9 **[L5]** Makefile: `-lc++` Darwin / `-lstdc++` else (kept explicit
      rather than dropped; Darwin link verified locally, Linux path untestable
      here — flag is the same as before on Linux so no regression risk)
- [x] 2.10 **[L7]** load_nif handles `{:error, :bad_name}` with clear message
- [x] 2.11 [test] concurrency_test.exs (:slow, 8 workers × 300 mixed
      add/search/reset/ntotal ops + post-check); NUL tests (desc + both paths);
      dim/d/k <= 0 tests. `mix test --include slow`: 33 passed.

**Verify:** `mix compile` (no new warnings), `mix test`, `mix test --include slow`.

## Phase 3 — Elixir fixes (`lib/`) [elixir]

- [x] 3.1 **[high]** get_cluster_assignment now `Index.search(quantizer, data, 1)`
      on the trained quantizer; regression test asserts `ntotal == k` and
      label == manual argmin over centroid distances
- [x] 3.2 `Shared.encode_vectors!(data, dim) :: {n, binary}` — single pass,
      iodata per row, ArgumentError with row index ("row 1 has 2 elements,
      expected 3"); also handles `[]` as `{0, <<>>}` (NIF no-ops on n=0).
      All 6 call sites updated; `to_flat_floats` + `validate_length!` deleted.
      `floats_to_binary` kept (public-ish decode symmetry) though lib no
      longer calls it.
- [x] 3.3 explicit metric case in `Index.new/3`, raises ArgumentError;
      `@metric_map` attribute removed
- [x] 3.4 `Clustering.train/2` uses `with` for Index.new + nif_train_clustering;
      get_cluster_assignment needs no chain anymore (single search call)
- [x] 3.5 `cpu_to_gpu` clause for `{:cuda, _}` → `{:error, "index already on GPU"}`
      (added function head for the default arg)
- [x] 3.6 @spec on all public functions in Index/Clustering + FaissEx.num_gpus/0;
      @typedoc on both structs; new public types `Index.vectors/0`, `error/0`
- [x] 3.7 `from_file` → `description: nil`; struct type widened; documented
- [x] 3.8 `FaissEx.num_gpus/0` returns a plain integer (unwraps the NIF tuple)
- [x] 3.9 `Shared.unwrap!/1` deleted
- [x] 3.10 [test] all added; plus `:no_cuda` tag (excluded when USE_CUDA=true)
      for the "GPU support not compiled" assertion; from_file description nil
      test. Suite: 43 passed incl. :slow.

**Verify:** `mix format`, `mix compile --warnings-as-errors`, `mix test`.

## Phase 4 — Docs [docs]

- [x] 4.1 Concurrency section added to Index moduledoc; k>ntotal -1 padding
      documented in search/3 (done during the 3.6 rewrite)
- [x] 4.2 README: build table v1.14.3; CUDA 12.x note (+ v1.11.0 escape hatch
      for CUDA 11 users); Thread Safety rewritten around the rwlock (incl.
      shard-for-parallel-ingest tip); num_gpus/0 example; bench version label
      pending 4.3
- [x] 4.3 Table refreshed from run 2 (runs 1+2 consistent). Investigated the
      odd add-1000 ≈ add-10000 result with a direct probe: NIF-side add of
      10k vectors is ~0.4ms; batch adds are dominated by list encode + BEAM
      major-GC pauses (~10ms) on the 20MB corpus — bimodal, honest for a
      list-API workload. Added README note explaining it. Added
      "kmeans assign 100 vectors" scenario (258μs median). kmeans train ~2x
      faster on v1.14.3 (9.2→4.5ms); add-10k 56→14.5ms (encode win).
- [x] 4.4 CHANGELOG v0.2.0 entry (Changed/Added/Fixed); @version 0.2.0
- [x] 4.5 CLAUDE.md: pinned version + keep-in-sync note, locking convention,
      updated dirty-scheduler list, Shared module description, :no_cuda tag

**Verify:** `mix docs` builds without warnings; README examples compile as written.

## Deferred (needs user decision — not dropped)

| Finding | Why deferred |
|---|---|
| [M1] Shared `StandardGpuResources` across indexes (new NIF arity) | Needs CUDA hardware to validate; additive API change |
| New API surface: `search_with_params` (nprobe/efSearch), `remove_ids`, `range_search` | Feature expansion beyond "update + optimize"; suggest a follow-up plan |
| [L1] badarg vs `{:error, "invalid arguments"}` for malformed NIF args | Public behavior change; unreachable via the Elixir wrapper today |
| [L3] Alignment hardening for sub-binary inputs | Unreachable via public API; add code comment only |
| Binary-in/binary-out fast-path API (skip list encode/decode entirely) | New API design; pairs well with the follow-up features plan |
| `Index` decode-then-chunk double allocation on results | Accepted tradeoff for list API; superseded if binary API lands |

## Risks

1. **Could the v1.14.3 build fail on this machine?** Possible (new Metal
   backend / cmake defaults on Apple Silicon). Mitigation: task 1.3 verifies
   before any code changes; cache dir scheme isolates versions so rollback is
   just reverting the rev.
2. **Could the rwlock deadlock?** Only multi-lock site is `train_clustering`
   (clustering + quantizer). Fixed lock order documented in 2.1. Everything
   else takes exactly one lock.
3. **Does the trained quantizer really hold the centroids (3.1)?** FAISS
   `Clustering::train` adds centroids to the passed index — verified in FAISS
   source; regression test in 3.10 asserts `ntotal == k` before relying on it.
4. **Bench numbers change meaning** — table mixes FAISS upgrade + our perf
   fixes. Acceptable: label the table with version; optional before/after run
   noted in scratchpad if wanted.

## Execution notes

- Phases 2 and 3 are independent of each other but both depend on Phase 1
  (test against v1.14.3 baseline first).
- Suggested commits: one per phase (build bump / C fixes / Elixir fixes / docs+release prep).
