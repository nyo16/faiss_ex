---
module: "FaissEx"
date: "2026-11-01"
problem_type: performance_issue
component: nif
symptoms:
  - "Many concurrent single-query searches are ~3x slower than the same searches with OMP_NUM_THREADS=1"
  - "A benchmark on a 10k-vector index shows no OpenMP effect at all, hiding the problem"
root_cause: "Each FAISS call opens an OpenMP team of one thread per core even for one query on a large index; with N dirty-CPU schedulers each running a call, up to N x cores runnable threads compete for the cores. The ICV set by omp_set_num_threads is per thread, and dirty scheduler threads are reused."
severity: high
tags: [openmp, dirty-scheduler, oversubscription, faiss, benchmark, omp_set_num_threads]
---

# OpenMP teams multiply across dirty schedulers

## Symptoms

Flat index, 100k × 128, k = 10, Apple M1 Max:

| Workload | default threads | 1 thread |
|---|---|---|
| 10 procs × 200 single-query searches | 1529 / 1635 ms | 586 / 557 ms |
| one 1000-query batch | 78 / 88 ms | 123 / 126 ms |

On a **10k** index both columns were equal — FAISS does not split a single
query over a small database, so small benchmarks hide the effect.

## Root Cause

Two thread pools stacked: BEAM dirty-CPU schedulers (one per core) each run
a FAISS call that starts its own OpenMP team (one per core).

## Solution

- `FaissEx.set_omp_threads/1` / `config :faiss_ex, omp_threads: n` store a
  count in a C11 atomic; `apply_omp_threads()` calls `omp_set_num_threads`
  on the *calling dirty thread* before every FAISS compute call (the ICV is
  per thread; setting it once at load only affects the loading thread).
- The NIF links the **same** OpenMP runtime as libfaiss (macOS: Homebrew
  libomp via `LIBOMP_PREFIX`; Linux: `-fopenmp` → libgomp). Two runtimes in
  one process hang or crash — verify with `otool -L` / `ldd`.
- libiomp5-linked system FAISS → `-DFAISS_EX_NO_OMP`, setting ignored.

## Prevention

- Concurrency benchmarks must use a database large enough for FAISS to
  parallelize a single query (≥ 100k rows for Flat).
- Never assume a per-thread runtime setting made once applies to pooled
  threads.
