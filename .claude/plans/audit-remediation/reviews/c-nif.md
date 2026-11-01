# Review: c_src/faiss_ex_nif.c + Makefile OpenMP/link section (a397362..master)

Reviewer scope: LOCKED_NIF try-lock/reschedule paths, lock pairing, aligned_input ownership,
reconstruct overflow/sentinel/has_safe_reconstruct_n, OpenMP control, Makefile OpenMP/link flags.
Probes run: `mix run --no-compile --no-start -e ...` (built NIF, macOS arm64, Accelerate BLAS),
`cc -fsyntax-only -DFAISS_EX_NO_OMP`, `/usr/bin/make` (GNU Make 3.81) `-n -p clean` parses.

## Findings

### 1. BLOCKER — `train/2` with zero vectors kills the VM for PCA / RQ / PRQ indexes
- **Where:** `c_src/faiss_ex_nif.c:601-603` (contract comment), `:633` (`faiss_Index_train` now reached with `n == 0`).
- **Evidence:** before this patch, `nif_train_index` returned `:ok` for `n == 0` without calling FAISS
  (`a397362:c_src/faiss_ex_nif.c:448-451`). It now forwards `n == 0` with the `empty_input` pointer.
  Probe (`{:ok, i} = FaissEx.Index.new(16, desc); FaissEx.Index.train(i, <<>>)` / `train(i, [])`):
  ```
  PCA8,Flat   -> "LDC must be >= MAX(N,1): LDC=0 N=0 BLAS error: Parameter number 11 passed to cblas_ssyrk had an invalid value"   exit=255
  RQ4x4       -> same BLAS error, exit=255
  PRQ2x2x4    -> same BLAS error, exit=255
  PCAR8,SQ8   -> same BLAS error, exit=255
  ```
  (IVF/PQ/OPQ/SQ/HNSW-SQ return `{:error, ...}`; Flat/LSH/LSQ/RaBitQ/SQfp16 return `:ok`.)
  `PCAMatrix::train` / the RQ codebook path call `cblas_ssyrk` with `n = 0`; Accelerate's
  BLAS error handler terminates the process. Not an exception, so no try/catch can save it.
- **Why it matters:** a public-API call with an empty list or `<<>>` (e.g. an empty batch from
  upstream code) takes down the whole node on macOS. "FAISS decides" (plan 3.7/5.8) assumed FAISS
  either no-ops or throws for an empty training set; for BLAS-backed trainers it does neither.
- **Fix:** do not hand FAISS an empty training set. Keep the no-op-for-trained contract by
  deciding under the write lock, recording only a status:
  ```c
  int ret;
  if (n == 0) {
      ret = faiss_Index_is_trained(res->index) ? 0 : -2;   /* no FAISS call */
  } else {
      ret = faiss_Index_train(res->index, (idx_t)n, (const float *)data);
  }
  enif_rwlock_rwunlock(res->lock);
  free_owned(owned);
  if (ret == -2) return make_error_msg(env, "training requires at least one vector");
  if (ret != 0) return make_faiss_error(env, "training failed");
  ```
  (or reject `n == 0` with an untrained index in `FaissEx.Index.train/2` before the NIF).
  Update `test/faiss_ex/index_test.exs:871-872` (IVF n=0 currently asserts FAISS's
  "Number of training points (0)" text), the comment at `:601-603`, the `train/2` doc (plan 5.8),
  and CHANGELOG. Add a regression test with `"PCA8,Flat"` + `train(index, [])`.

### 2. WARNING — OpenMP runtime match is only negative-detected (iomp5); other mismatches link a second runtime
- **Where:** `Makefile:177-201`.
- **Evidence:**
  ```make
  ifneq ($(findstring iomp5,$(FAISS_LINKED_LIBS)),)
    NIF_OMP_DISABLED := true
  ...
  else ifeq ($(UNAME_S),Darwin)
    NIF_CFLAGS += -Xclang -fopenmp -I$(LIBOMP_PREFIX)/include
    NIF_LDFLAGS += -L$(LIBOMP_PREFIX)/lib -lomp
  else
    NIF_CFLAGS += -fopenmp
  ```
  With `FAISS_PREFIX`, any libfaiss whose OpenMP runtime is neither iomp5 nor the one the NIF
  is about to link still gets OpenMP linked into the NIF:
  - macOS: a system FAISS linked against a libomp at another install name (MacPorts
    `/opt/local/lib/libomp/libomp.dylib`, a custom LLVM) while `brew --prefix libomp` is also
    installed → the NIF loads brew's libomp and libfaiss loads its own. LLVM libomp aborts on the
    second copy (`OMP: Error #15 ... already initialized`); `on_load` calls `omp_get_max_threads()`
    so the NIF's copy initializes first and the first FAISS parallel region aborts the VM.
    Before this patch the NIF linked no OpenMP, so this config worked.
  - Linux: a clang-built libfaiss (`NEEDED libomp.so.5`) with the default gcc NIF build → libgomp
    and libomp both loaded; `omp_set_num_threads` binds to libgomp, so `set_omp_threads/1` is a
    silent no-op on FAISS's libomp regions (the exact "two runtimes" case the comment at
    `:166-176` says must not happen).
  The plan's own risk #2 ("links against the exact `LIBOMP_PREFIX` FAISS was configured with")
  holds for source builds only.
- **Fix:** for `FAISS_PREFIX`, positively match the runtime the NIF will link and disable
  otherwise:
  ```make
  ifdef FAISS_PREFIX
    ifeq ($(UNAME_S),Darwin)
      NIF_OMP_RUNTIME := $(LIBOMP_PREFIX)/lib/libomp.dylib
    else
      NIF_OMP_RUNTIME := libgomp.so
    endif
    ifeq ($(LIBOMP_PREFIX)$(UNAME_S),Darwin)
      NIF_OMP_DISABLED := true
    else ifeq ($(findstring $(NIF_OMP_RUNTIME),$(FAISS_LINKED_LIBS)),)
      NIF_OMP_DISABLED := true
    endif
  endif
  ```
  (keeps the iomp5 case disabled as a consequence) and update the comment at `:166-176`.

### 3. SUGGESTION — `FAISS_EX_NO_OMP` build emits `-Wunused-variable`
- **Where:** `c_src/faiss_ex_nif.c:283`.
- **Evidence:** `cc -std=c11 -fsyntax-only -Wall -Wextra -DFAISS_EX_NO_OMP ...` →
  `faiss_ex_nif.c:283:12: warning: unused variable 'g_omp_default' [-Wunused-variable]`.
  The CI syntax job (`.github/workflows/ci.yml:191`) uses `-Werror`; it does not cover
  `FAISS_EX_NO_OMP` today, but any job that adds it will fail.
- **Fix:** move `static int g_omp_default = 1;` inside the existing `#ifndef FAISS_EX_NO_OMP`
  block (both readers/writers — `apply_omp_threads` and `init_omp` — already are).

## Clean areas
- (1) LOCK_TRY contention paths: every LOCKED_NIF releases `owned` copies and result binaries
  (search: both; reconstruct/residuals/centroids: result) before `RESCHEDULE_BLOCKING`; no lock held;
  `train_clustering` rwunlocks the index lock before rescheduling when the clustering trylock fails.
- (2) Each successful acquire has exactly one unlock, no return between acquire and unlock,
  `free_owned` after unlock; centroids records only a status inside the critical section.
- (3) `enif_schedule_nif` from a dirty NIF with the original `argc`/`argv` and a static `#name`
  is valid (erl_nif docs: dirty jobs "can be reclassified and rescheduled using enif_schedule_nif");
  the IO twin re-inspects terms and re-validates.
- (4) `aligned_input` sets `*owned = NULL` before any branch; the `||`-chained second call never
  leaves `owned_*` uninitialized on any path (short-circuit leaves it NULL-initialized at declaration).
- (5) `keys[0] + n <= ntotal` cannot overflow: `keys[0] <= INT64_MAX - n` is enforced for contiguous
  keys before the lock. Tag pre-fill only when `batched`, check only when `used_batch`; the per-key
  fallback overwrites or errors. `has_safe_reconstruct_n`: IndexFlat1D/L2/IP inherit
  `IndexFlatCodes::reconstruct_n` (range check, `sa_decode`, no region); IndexFlatPanorama overrides
  with a full up-front range check before `Index::reconstruct_n`, and `Panorama::reconstruct` cannot
  throw; IndexRefineFlat is not an IndexFlat (IndexRefine) so it takes the per-key path. No IVF
  subclass overrides `reconstruct_n`; `reconstruct_from_offset` throws (FlatDedup, base) occur
  outside any region.
- (6) Atomic: seq_cst store / relaxed load on `_Atomic int` is sufficient; `apply_omp_threads` runs on
  the executing thread in both CPU and IO entries before every compute call; `init_omp` on upgrade
  revalidates `load_info` and recaptures the default.
- (7) `else ifeq` parses and selects correctly on GNU Make 3.81 (`/usr/bin/make -n -p clean`: brew
  libomp → `-Xclang -fopenmp ... -lomp`; `LIBOMP_PREFIX=` → `-DFAISS_EX_NO_OMP`; `UNAME_S=Linux` →
  `-fopenmp` in both CFLAGS/LDFLAGS); built NIF and priv/lib/libfaiss.dylib both reference
  `/opt/homebrew/opt/libomp/lib/libomp.dylib` (otool -L).
