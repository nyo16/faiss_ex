# Changelog

## v0.4.0 (2026-11-01)

Remediation release for the 2026-11-01 health audit: VM-abort and
scheduler-starvation fixes, OpenMP control, a working `FAISS_OPT_LEVEL`,
and a hardened build and CI. Contains behavior changes — read *Changed*
before upgrading.

### Changed (behavior)

- **`add_with_ids/3` raises `ArgumentError`** when the id count does not
  match the vector count (was `{:error, "binary size mismatch"}`), matching
  `compute_residuals/3`.
- **Options are validated.** Unknown keys in `Index.new/3` and
  `Index.from_file/2` raise `ArgumentError` (a typo like `metrc:` used to be
  ignored), and an invalid `:device` raises `ArgumentError` instead of
  `CaseClauseError`. `cpu_to_gpu/2` requires a non-negative integer device.
- **`Index.train(index, [])` reports untrained indexes**: still `:ok` for
  indexes that are already trained (Flat, HNSW), but now
  `{:error, "no training vectors"}` for indexes that need training (IVF, PQ,
  PCA) — it used to return `:ok` without training anything.
- **`Clustering.train/2` no longer mutates its input.** Each call builds a
  fresh FAISS clustering object and quantizer; the struct you pass in stays
  untrained and usable, and two trains from one struct are independent.
  Empty data returns `{:error, "n must be positive"}` before allocating.
- **`FAISS_OPT_LEVEL` defaults to `dd`** (runtime SIMD dispatch: AVX2 /
  AVX-512 on x86-64, NEON / SVE on aarch64). Only `dd` and `generic` are
  accepted. Upgrading triggers one FAISS rebuild; the old
  `~/.cache/faiss_ex/faiss-v1.15.1/build/` directory is no longer used and
  can be deleted.
- **Locked NIFs wait on dirty-IO schedulers under contention.** A call
  that finds its index locked moves to the dirty-IO pool to wait (size it
  with `+SDio`) instead of occupying a dirty-CPU thread.
- Elixir 1.18 / OTP 27 is the CI-tested floor (also tested: 1.19/28,
  1.20/29). Building from source needs CMake 3.24+ and, for GPU builds, the
  CUDA 12+ toolkit; both are checked before configuring.

### Added

- `FaissEx.set_omp_threads/1` and `config :faiss_ex, omp_threads: n`: the
  OpenMP thread count FAISS uses per call. Measured on a 100k × 128 Flat
  index: 10 processes × 200 single-query searches 1.5 s → 0.57 s with `1`;
  one 1000-query batch 78 ms → 123 ms (keep the default for big batches).
- Binary API for the remaining paths: `Index.search/3` and
  `compute_residuals/3` accept binaries; `Index.compute_residuals_binary/3`;
  `Clustering.train/2` and `get_cluster_assignment/2` accept binaries;
  `Clustering.get_centroids_binary/1`.
- Factory descriptions are capped at 256 bytes
  (`{:error, "description too long"}`).

### Fixed

- **VM abort on reconstruct/residual errors.** FAISS runs
  `compute_residual_n` and the generic `reconstruct_n` inside OpenMP
  parallel regions; an exception there (a key past `ntotal`, or an index
  type without `reconstruct` such as `IDMap,Flat`) cannot reach the C API's
  error handling and terminated the whole BEAM. `compute_residuals(index,
  [v], [5])` on a one-vector Flat index was enough. These now return
  `{:error, reason}`. Present since `compute_residuals` and `reconstruct`
  were added.
- **Dirty-scheduler starvation.** One waiter per dirty-CPU scheduler on a
  busy index (e.g. searches during a long `train`) pinned the whole
  dirty-CPU pool, so work on *unrelated* indexes queued for seconds. A
  search on an idle index during such a train now takes ~0.04 ms.
- **`FAISS_OPT_LEVEL=avx2|avx512` was a silent no-op** (FAISS builds those
  as separately named libraries the build never installed).
- **Stale FAISS builds.** Switching `FAISS_OPT_LEVEL` or `USE_CUDA` reused
  the previous build. Each configuration now has its own build directory,
  and a configuration stamp re-copies libraries and re-links the NIF.
- **`FaissEx.num_gpus/0` on CUDA builds** called `faiss_get_num_gpus`
  without its output parameter and returned the status code (0); GCC 14
  also rejected the implicit declaration, breaking CUDA builds.
- **Negative GPU device ids** aborted inside CUDA; now rejected.
- **Misaligned binary input** (sub-binaries from `binary_part/3` or pattern
  matches) was passed to FAISS as misaligned `float*`/`int64*` — undefined
  behavior. Misaligned input is now copied to an aligned buffer.
- **Uninitialized memory returned by IVF reconstruction.** For an IVF index
  filled with `add_with_ids` and sparse ids, reconstructing a contiguous key
  range returned heap contents for keys with no stored vector (FAISS's IVF
  batched reconstruct skips them). Those keys now return
  `{:error, "key out of range"}`. Present since v0.1.0.
- `INT64_MAX` reconstruction keys on the per-key path signed-overflowed
  FAISS's bound check; now `{:error, "key out of range"}`.
- `nif_new_index` ran on a normal scheduler; factory parsing and large
  quantizer allocation now run dirty.

### Security

- `Index.from_file/2` documents its trust boundary: an on-disk IVF index
  file can make the loader memory-map an arbitrary path it names, and
  malformed files can still crash the VM — only load trusted files.
  `io_flags` accepts only `0`, `2`, `4`, `6` (`8`, skip IVF data, produced
  indexes that dereference unmapped memory). Index files with a
  non-positive dimension are rejected (they corrupted the heap on
  `reconstruct`).
- Factory descriptions are documented as trusted input (they size
  allocations).
- Build: `FAISS_GIT_REV`/`FAISS_GIT_REPO` are allow-listed (`https://`
  only), `FAISS_GIT_SHA` must be a 40-character hex commit id, recipe
  paths are quoted, and the pinned FAISS commit is re-verified on every
  build, so a warm cache checked out at another commit fails closed.
- CI: actions pinned by commit SHA and the CUDA container by digest,
  Dependabot for actions and Hex deps (7-day cooldown), `contents: read`
  permissions, no persisted checkout credentials, job timeouts,
  `mix hex.audit`, and releases only from validated `v*` tags on commits
  that are on master.

## v0.3.2 (2026-10-31)

### Changed

- **Pinned FAISS upgraded from v1.14.3 to v1.15.1** (commit
  `75c755c5a1fc854756aaf3bd1131213aeb21d8d1`). Upstream C API changes are
  additive only (new ScalarQuantizer, BinaryIVF, and DirectMap entry points);
  nothing this library calls changed. Existing saved indexes load unchanged.
  GPU builds still require CUDA 12.x.

## v0.3.1 (2026-08-07)

Lock-discipline hardening and API consistency. No new features.

### Fixed

- **Out-of-bounds read from negative reconstruction keys (memory safety).**
  FAISS bounds-checks only the upper end of a key: `IndexFlat::reconstruct`
  tests `key < ntotal`, which every negative key passes, and
  `IndexFlatCodes::reconstruct_n`'s `i0 + ni <= ntotal` signed-overflows to a
  negative value — and so passes — once `i0` nears `INT64_MAX`. Either way
  FAISS read outside the codes array and returned heap contents as vector
  data. `Index.reconstruct/2`, `reconstruct_binary/2`, and
  `compute_residuals/3` all reached it from ordinary Elixir; a plain `-1` was
  enough, and results were non-deterministic between runs. The NIF now
  rejects negative keys and overflowing contiguous ranges up front with
  `{:error, "key out of range"}`. Keys past `ntotal` are unaffected — FAISS
  still reports those itself. Present since v0.1.0.
- **The NIF lock can no longer be leaked on an error path.** Sixteen early
  returns each re-typed their own `enif_rwlock_*unlock`; a single omission
  would have wedged that index forever and blocked a dirty-scheduler thread
  on every later call, with nothing in the build or test suite able to catch
  it. Eleven of them are gone — `add`, `add_with_ids`, `train`,
  `reconstruct`, and `residuals` now validate argument sizes (and allocate
  the result binary) *before* taking the lock, so their critical section is
  the bare FAISS call. The remaining five, in `train_clustering` and
  `get_clustering_centroids`, must read locked state to validate and now use
  a single `unlock:` exit instead of per-branch unwinds.
- Shorter critical sections as a side effect: `train` no longer holds the
  write lock across overflow checks and error formatting, a size-mismatch
  error now returns immediately rather than queueing behind a contended
  lock, and the two `unlock:` functions build their result and error terms
  after releasing, so no lock covers term construction.
- `Index.gpu_to_cpu/1` on an index already on the host returns
  `{:error, "index already on host"}` instead of raising
  `FunctionClauseError` — matching `cpu_to_gpu/2` and the `@spec` that
  already advertised an error tuple.
- `Index.compute_residuals/3` validates its keys with the same encoder as
  `reconstruct/2`. A non-integer key raised an opaque bitstring-construction
  error and a key/vector count mismatch degraded into the C layer's
  `{:error, "binary size mismatch"}`; both now raise `ArgumentError` naming
  the offending value or the two counts.

### Changed

- `Index.to_file/2`, `Index.from_file/2`, and `Index.new/3` no longer accept
  charlist paths and factory strings. This was undocumented and untested —
  every `@spec` involved already said `String.t()`. Pass a binary, or
  `List.to_string/1` a charlist at the call site.
- The ten `case`-with-passthrough-error-clause wrappers in
  `FaissEx.Index`/`FaissEx.Clustering` are now `with` expressions. Behavior
  is identical; note that `Index.new/3`'s inferred return type is now
  `dynamic()` rather than a closed union, so struct-update syntax on its
  result needs an explicit `%Index{} = index` match to satisfy the type
  checker.
- `Index.new/3` documents its error discipline: an unknown `:metric` raises
  `ArgumentError`, while a FAISS-rejected `dim` or `description` returns
  `{:error, reason}`. The note is explicit that the raise/tuple split is
  *not* uniform across the module — `add_with_ids/3` returns a tuple for an
  id count mismatch where `compute_residuals/3` raises.

### Internal

- `credo` added as a dev/test dependency.
- `FaissEx.Shared.floats_to_binary/1` and `int64s_to_binary/1` removed —
  they had no production callers and shipped in the hex package as test
  fixtures. They are now private helpers in the test files. `Shared` is
  `@moduledoc false`, so this is not a public API change.
- C: `dup_cstr/1` replaces three copies of the alloc/memcpy/NUL-terminate
  block; `make_ok_atom/1` deleted in favour of the `atom_ok` global.

## v0.3.0 (2026-07-12)

Binary fast-path API and encode/decode optimizations.

### Added

- **Binary API.** Vectors already held as raw `f32-native` binaries (HTTP
  responses, database blobs, files) can skip list encoding/decoding
  entirely — adding 10,000×128 vectors drops from ~14.4 ms to ~0.35 ms.
  - `add/2`, `train/2`, and `add_with_ids/3` accept row-major `f32-native`
    vector binaries and `s64-native` ID binaries anywhere they accept
    lists (mixing forms is fine); the vector count is inferred from
    `byte_size`, and a non-divisible binary raises `ArgumentError`.
  - `search_binary/3` and `reconstruct_binary/2` return raw result
    binaries for zero-decode pipelines. The list-returning functions never
    change shape based on input type.

### Changed

- `dim/1` no longer takes the index lock and runs on a normal scheduler —
  the dimension is cached in the NIF resource at creation (it is immutable
  in FAISS). `ntotal/1` and `trained?/1` still read live state under the
  lock on dirty schedulers.
- Search result decoding is single-pass per row (no flat-decode-then-chunk),
  and the NIF's search critical section no longer covers size checks or
  result allocation — the lock is held only for the FAISS call.

## v0.2.0 (2026-07-11)

FAISS upgrade, thread safety, and performance fixes.

### Changed

- **Pinned FAISS upgraded from v1.10.0 to v1.14.3.** No breaking C API
  changes; existing saved indexes load unchanged. GPU builds now require
  CUDA 12.x (FAISS 1.12.0 dropped CUDA 11).
- **Indexes are now safe to share across processes.** Every index and
  clustering object is guarded by a read-write lock in the NIF layer:
  mutations (`add`, `add_with_ids`, `train`, `reset`) serialize against
  all other operations, while searches and other reads still run
  concurrently. Previously a mutation racing a search on the same index
  was undefined behavior that could crash the VM.
- `reset/1`, `get_centroids/1`, and the `dim`/`ntotal`/`trained?` getters
  now run on dirty schedulers.
- `FaissEx.Index.from_file/2` sets `description: nil` (was `"from_file"`).
- Invalid `:metric` options raise a descriptive `ArgumentError`.
- Ragged or wrong-length vector batches raise `ArgumentError` naming the
  offending row (was a generic count mismatch message).

### Added

- `FaissEx.num_gpus/0`.
- `@spec` for every public function and `@typedoc` for the structs.
- Concurrency stress test (`mix test --include slow`).

### Fixed

- `Clustering.get_cluster_assignment/2` reuses the quantizer index that
  training populated instead of rebuilding a Flat index and re-adding all
  centroids on every call.
- Vector list encoding is single-pass (was three list traversals per call).
- `reconstruct/2` uses FAISS's batched `reconstruct_n` when keys are a
  contiguous ascending range.
- Factory strings and file paths containing NUL bytes are rejected instead
  of being silently truncated.
- `Index.new/3` and `Clustering.new/2` validate `dim`/`d`/`k > 0` with
  clear errors.
- `Clustering.train/2` propagates quantizer creation failures instead of
  raising a `MatchError`.
- `Index.cpu_to_gpu/2` on an index already on the GPU returns
  `{:error, "index already on GPU"}` instead of raising
  `FunctionClauseError`.
- NIF loading reports a clear error when the `:faiss_ex` priv dir cannot
  be found.

## v0.1.0 (2026-03-07)

Initial release.

- Flat, IVF, HNSW, PQ index support via FAISS index factory
- Vector add, search, train, clone, reset, reconstruct
- Compute residuals (batch API)
- Add vectors with custom IDs (`IDMap`)
- File I/O (save/load indexes)
- K-means clustering with cluster assignment
- Optional CUDA GPU support (`cpu_to_gpu`, `gpu_to_cpu`)
- Inner product and L2 distance metrics
- Plain list API — no external dependencies beyond `elixir_make`
- NIF resource types with BEAM GC cleanup
- Dirty schedulers for CPU-bound and IO-bound operations
