# Changelog

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
