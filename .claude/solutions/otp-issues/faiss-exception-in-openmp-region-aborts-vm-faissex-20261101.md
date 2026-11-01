---
module: "FaissEx.NIF"
date: "2026-11-01"
problem_type: otp_issue
component: nif
symptoms:
  - "The whole BEAM exits with 'libc++abi: terminating due to uncaught exception of type faiss::FaissException' (exit 134) — no error tuple, no crash report"
  - "Index.compute_residuals(flat, [v], [5]) on a 1-vector index kills the VM; Index.reconstruct on an \"IDMap,Flat\" index kills the VM even for one key"
  - "Contiguous reconstruct on an IVF index filled with add_with_ids returns denormals / huge floats (uninitialized heap) instead of an error"
root_cause: "FAISS runs reconstruct inside `#pragma omp parallel for` in Index::compute_residual_n and the base Index::reconstruct_n. A C++ exception cannot leave an OpenMP parallel region — even one serialized by an `if` clause — so the C API's try/catch never sees it and the runtime calls std::terminate. Separately, IndexIVF::reconstruct_n writes only rows whose id lies in the key range, leaving other rows of the caller's buffer untouched."
severity: critical
tags: [nif, openmp, std-terminate, faiss, c-api, exception, uninitialized-memory, reconstruct]
---

# A FAISS exception inside an OpenMP region aborts the VM — the C API's try/catch cannot help

## Symptoms

- BEAM process exits with `libc++abi: terminating due to uncaught exception
  of type faiss::FaissException: ... 'key < ntotal' failed` (exit 134).
- Repros (FAISS v1.15.1, all from plain Elixir):
  - `compute_residuals(flat_with_1_vector, [[0,0,0,0]], [5])`
  - `reconstruct(idmap_flat, [0])` ("reconstruct not implemented")
  - `reconstruct_binary(hnsw_2000, Enum.to_list(1500..2600))`
- Related, not an abort: IVF + `add_with_ids([1000, 2000, 3000])` then
  `reconstruct([0, 1, 2])` → `{:ok, garbage}` (heap contents).

## Investigation

1. **"The C API catches everything"** — wrong for parallel code.
   `faiss_Index_compute_residual_n` wraps the call in try/CATCH_AND_HANDLE
   (`c_api/Index_c.cpp:189`), but the throw happens on an OpenMP worker
   inside `Index::compute_residual_n` (`faiss/Index.cpp:142`).
2. **Chunk reconstruct_n to ≤1000 keys** so `omp parallel for if (ni > 1000)`
   stays serial — **dead end**: `IDMap,Flat` `reconstruct([0])` (n = 1)
   still aborted. A serialized region is still a region; the outlined
   function cannot propagate exceptions.
3. Checked which overrides are safe: `IndexFlatCodes::reconstruct_n` and
   `IndexIVF::reconstruct_n` range-check up front and contain no parallel
   region. IVF/HNSW *search* paths capture exceptions per thread
   (`omp_capture_exception`) and rethrow outside — safe.
4. While there: `IndexIVF::reconstruct_n` iterates the inverted lists and
   writes only rows whose id ∈ [i0, i0+ni); with sparse custom ids, other
   rows are never written.

## Root Cause

C++ exceptions and OpenMP regions do not compose: any FAISS entry point
that calls a throwing virtual (`reconstruct`) inside `#pragma omp` turns an
ordinary input error into `std::terminate`, and a NIF cannot catch that.

## Solution

`c_src/faiss_ex_nif.c`:

- Residuals: never `faiss_Index_compute_residual_n`; loop
  `faiss_Index_compute_residual` per vector (each call is guarded, runs on
  the calling thread).
- Reconstruct: batched `faiss_Index_reconstruct_n` only when
  `has_safe_reconstruct_n` (`faiss_IndexFlat_cast || faiss_IndexIVF_cast`)
  **and** `keys[0] + n <= ntotal` checked under the lock; otherwise per-key
  `faiss_Index_reconstruct`.
- Unwritten rows: tag every row's first float with a signaling-NaN bit
  pattern (`UNWRITTEN_ROW_TAG = 0x7FA5A5A5`) before the batched call; if a
  row is still tagged afterwards → `{:error, "key out of range"}`.
- `INT64_MAX` keys rejected (`IndexFlatCodes::reconstruct(key)` =
  `reconstruct_n(key, 1)`, whose `key + 1` overflows).

Tests: `index_test.exs` describe "reconstruction errors never abort the VM"
(each case aborted the test VM before the fix — verified with throwaway
`mix run` scripts, exit 134).

## Prevention

- Before calling any batched FAISS C entry point (`*_n`, `search_and_*`,
  `range_search`, …), read its implementation: `#pragma omp` around code
  that can throw without `omp_capture_exception` = VM abort. CLAUDE.md
  records the rule.
- Never return an `enif_alloc_binary` buffer that FAISS might not have fully
  written; pre-fill or tag it.
- Write error-path tests by *probing every index type* (Flat, IVF, HNSW,
  IDMap, IDMap2) with bad keys — the abort only appeared on non-Flat types
  or out-of-range keys, which the happy-path suite never exercised.

## Related

- `.claude/solutions/otp-issues/nif-rwlock-getters-must-be-dirty-faissex-20260711.md`
