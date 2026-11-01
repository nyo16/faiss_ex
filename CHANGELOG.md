# Changelog

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
