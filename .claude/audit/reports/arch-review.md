# Architecture Review — faiss_ex

Scope: `lib/faiss_ex.ex`, `lib/faiss_ex/index.ex`, `lib/faiss_ex/clustering.ex`,
`lib/faiss_ex/nif.ex`, `lib/faiss_ex/shared.ex`, `c_src/faiss_ex_nif.c` (structure only).

Note: this is a pure Elixir NIF library, not a Phoenix app — evaluated as a
library (module cohesion, public API design, NIF layering), not against
Phoenix context/scope conventions.

**Architecture health score: 88/100** — clean single-purpose layering (public
API → NIF stubs → single C file → libfaiss_c) with consistent return
conventions and correct lock discipline; the only real defects are one dead
public function and a couple of documentation/consistency nits.

## Issues

### Low

1. **Dead code: unused public function** — `lib/faiss_ex/shared.ex:5`
   `floats_to_binary/1` is defined but never called anywhere in `lib/` or
   `test/` (`encode_vectors!/2` is the only encoder actually used by
   `Index`/`Clustering`). Either remove it or mark it `@doc false` +
   justify why it's kept as public API (it currently has `@doc false` but
   still ships as a callable public function with no test coverage).

2. **`nif_new_index`/`nif_new_clustering` run on normal (non-dirty)
   schedulers** — `c_src/faiss_ex_nif.c:977`, `:996`. `faiss_index_factory`
   (called from `nif_new_index`, `c_src/faiss_ex_nif.c:193`) can allocate
   nontrivial structures for composite factory strings (e.g.
   `"IVF4096,PQ64"`, `"HNSW32"`) — this is CPU work that isn't guaranteed
   O(1), yet the NIF is registered with flags `0` (normal scheduler) rather
   than `ERL_NIF_DIRTY_JOB_CPU_BOUND`. This is called out as intentional in
   `CLAUDE.md`, but for factory strings that build large graph/quantizer
   structures it risks briefly blocking a normal scheduler thread. Worth
   re-confirming the assumption holds for all supported factory strings, or
   moving these two to dirty schedulers for consistency with every other
   allocating/mutating NIF in the file.

3. **Metric-type mapping duplicated as magic numbers on both sides of the
   NIF boundary** — `lib/faiss_ex/index.ex:104-115` hardcodes
   `:l2 -> 1`, `:inner_product -> 0`, passed as a bare integer through
   `NIF.nif_new_index/3` to `c_src/faiss_ex_nif.c:193`
   (`(FaissMetricType)metric`). Nothing enforces that this mapping stays in
   sync with FAISS's `MetricType` C enum across a `FAISS_GIT_REV` bump —
   a silent metric mismatch (e.g. L2 silently becomes IP) would not raise,
   it would just search with the wrong metric. Consider a shared
   comment/constant referencing the enum values, or a compile-time
   assertion in the C file.

### Informational (not scored, no action required)

- `lib/faiss_ex/index.ex:39-53` documents the read/write lock split
  precisely, and it was cross-checked against `c_src/faiss_ex_nif.c`
  (`nif_add_to_index`, `nif_add_with_ids_to_index`, `nif_train_index`,
  `nif_reset_index` use `enif_rwlock_rwlock`/`rwunlock`; `nif_search_index`,
  `nif_reconstruct_batch`, `nif_compute_residuals`, `nif_clone_index`,
  `nif_write_index`, the property getters, and GPU transfer use
  `enif_rwlock_rlock`/`runlock`) — locking matches the documented contract
  exactly, including the two-resource lock order in
  `nif_train_clustering` (`c_src/faiss_ex_nif.c:858-861`, index lock
  acquired before clustering lock, matching the module doc). No defect.

## Clean areas (one-liner each)

- **Module cohesion**: each module has a single clear responsibility
  (`FaissEx` top-level convenience, `Index` CRUD/search, `Clustering`
  k-means, `NIF` raw stubs, `Shared` encode/decode) — no god modules, all
  under 400 lines.
- **Public API consistency**: uniform `{:ok, _} | {:error, String.t()}` /
  `:ok | {:error, String.t()}` return convention across every public
  function in `Index` and `Clustering`; struct-as-first-arg calling
  convention is consistent.
- **NIF/Elixir separation**: `NIF` module is a pure, undocumented
  (`@moduledoc false`) stub layer with no business logic; encoding/decoding
  is isolated in `Shared`; higher-level modules never touch binaries
  directly except via `Shared`.
- **Typespec coverage**: `@spec` present on effectively every public
  function in `Index`, `Clustering`, and `FaissEx`; `@type t()` defined for
  both structs with field-level doc.
- **Documentation**: every public module and function has a `@moduledoc`/
  `@doc` with runnable examples; GPU/error/edge-case behavior (e.g. `k`
  exceeding `ntotal`) is explicitly documented.
- **Error handling consistency**: uniform pattern of `case`/`with` on NIF
  results, `{:error, _} = err -> err` pass-through, no swallowed errors,
  no bare rescue.
- **C-file size**: 1001 lines for ~19 NIF entry points plus resource
  lifecycle/locking/helpers is reasonable for a single translation unit at
  this API surface; not yet at a size that forces a split (would
  reconsider past ~1500-2000 lines or if a second bounded concern, e.g. a
  new resource family, is added).
- **Resource/lock discipline**: every resource type has a destructor that
  frees the FAISS object and destroys its lock; lock acquire/release is
  paired on every exit path checked (including early-return validation
  failures).
- **Struct design**: `Index.t()` and `Clustering.t()` carry only handle/
  metadata fields (no leaked NIF internals beyond the opaque `ref`),
  `device` modeled as a proper `:host | {:cuda, id}` tagged tuple rather
  than booleans/flags.
