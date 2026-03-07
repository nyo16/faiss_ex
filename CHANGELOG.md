# Changelog

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
