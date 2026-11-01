# Performance Audit — faiss_ex

Date: 2026-11-01 · Commit: a397362 (v0.3.2, FAISS v1.15.1) · Prior: 2026-07-12 @ 3d0a7e1 (84/100)
Scope: `lib/faiss_ex/{index,shared,clustering}.ex`, `c_src/faiss_ex_nif.c`, `Makefile`, `bench/faiss_ex_bench.exs`

Measurements are from throwaway probe scripts run with `mix run` on this host (Apple M-series, 10 cores, 10 dirty CPU schedulers, libomp + Accelerate confirmed via `otool -L priv/lib/libfaiss.dylib`). The full Benchee suite was not run: 18 scenarios × 9 s ≈ 162 s plus data generation is too close to the 3-minute limit.

## Score: 48/100

Rubric adaptation (non-Phoenix library; the N+1, index, preload, LiveView and SELECT * criteria are N/A and were replaced with the categories below). Each deduction uses the closest original rubric value. A VM-wide throughput bottleneck counts like a GenServer bottleneck (−10). A hot-path inefficiency counts like an N+1 or missing index (−5). Minor issues count like a missing preload or a SELECT * (−3/−2). No criterion goes below 0.

| Criterion | Max | Deductions | Score |
|---|---|---|---|
| Redundant passes/copies in hot paths | 30 | none (the `encode_batch` iodata copy is deliberate; it was measured 2.4× faster than the alternative per CLAUDE.md) | 30 |
| Lock / CPU contention, critical-section size | 20 | −10 OpenMP oversubscription (N2) · −3 whole-batch write lock, no chunking (P6) · −2 centroids alloc+copy under lock (N7) | 5 |
| Scheduler misuse | 15 | −10 dirty-CPU pool exhaustion by lock waiters (N1) · −5 `nif_new_index` on a normal scheduler (N4) · −2 `nif_get_num_gpus` (P8) = −17 → floored | 0 |
| Memory amplification | 15 | −5 Binary API gap for Clustering/`compute_residuals` (N5) · −5 one `StandardGpuResources` per GPU index (N6) | 5 |
| Build optimization defaults | 10 | −5 `FAISS_OPT_LEVEL=avx2/avx512` is a silent no-op (N3, supersedes P9) | 5 |
| Benchmark coverage of hot paths | 10 | −5 no concurrency scenario (N8) · −2 reconstruct/decode paths missing (N8) | 3 |
| **Total** | **100** | 30 + 5 + 0 + 5 + 5 + 3 | **48** |

The drop from 84 to 48 is not a code regression. Most prior Elixir- and C-level findings are resolved. The new deductions come from three issues that were measured this time and were already present at 3d0a7e1 [INFERENCE: lock/scheduler/build design unchanged since]: N1, N2 and N3.

## Prior findings status

| # | Prior finding | Status | Evidence |
|---|---|---|---|
| P1 | List-only API, no raw-binary passthrough | **RESOLVED** for Index | `search_binary/3` `index.ex:258-264`, `reconstruct_binary/2` `:321-325`, binary input on `add/train/add_with_ids` `:199-223,272-275` via `encode_vector_data!` `:453-467` (zero-copy, size check only). The Clustering and `compute_residuals` remainder is tracked as N5. |
| P2 | Decode used two passes (`binary_to_floats` + `chunk_every`) | **RESOLVED** | Single-pass `decode_rows/4` `shared.ex:42-47`; row sub-binaries feed a comprehension and are not retained. Measured cost: 3.6 ms to decode a 1000×100 result vs 102 ms for the search itself. |
| P3 | Result binaries allocated under the read lock | **RESOLVED** | Allocation now happens before the lock: search `faiss_ex_nif.c:407-416`, reconstruct `:552-560`, residuals `:620-628`. The critical section is the `faiss_*` call only. |
| P4 | `encode_batch` makes an extra full-payload copy | **WONTFIX (measured)** | `shared.ex:64-66` is unchanged on purpose: CLAUDE.md records the flat-accumulator rewrite as 2.4× slower. No deduction. |
| P5 | Property getters on dirty CPU schedulers | **STILL OPEN**, now worse than first thought | `faiss_ex_nif.c:1040-1041`. Prior impact was "µs per call"; the real impact is N1 (getters waiting on a lock hold dirty CPU threads). |
| P6 | Bulk `add`/`train` holds the write lock for the whole batch, no chunking | **STILL OPEN** | `faiss_ex_nif.c:304-307,462-465`; `index.ex:199-202` has no `chunk_size:`. README `:454` documents only serialization, not search stalls. |
| P7 | `reconstruct/2` walked `keys` twice | **RESOLVED** | `index.ex:304` uses single-pass `Shared.encode_ids!/1` (`shared.ex:16-22`). |
| P8 | `nif_get_num_gpus` on a normal scheduler (CUDA init can take tens of ms) | **STILL OPEN** | `faiss_ex_nif.c:1045` flag `0`; the GPU implementation is at `:811-815`. |
| P9 | Default `FAISS_OPT_LEVEL=generic` leaves AVX2 unused | **STILL OPEN, and the override does not work** | See N3. The documented override builds the wrong CMake target. |

## New findings

### N1 — High — Requests waiting on one index's lock can fill the dirty-CPU pool and stall every other index
- **Where:** all lock-taking NIFs are `ERL_NIF_DIRTY_JOB_CPU_BOUND` (`c_src/faiss_ex_nif.c:1029-1036,1040-1041,1048-1049`). They block in `enif_rwlock_rlock`/`rwlock` (e.g. `:416`, `:560`, `:717`, `:730`) while holding a dirty CPU scheduler thread. The doc claim at `index.ex:78-79` covers normal schedulers only.
- **Evidence (measured):** `train/2` on `IVF2048,Flat` (100k×64, 3.9 s) takes A's write lock. Then 10 tasks call `Index.ntotal(A)`, one per dirty CPU scheduler.
  - A one-query `search_binary` on a **different** index B then took **3643.8 ms**.
  - The baseline is 1.7 ms. The control (train running, no waiters) is **0.1 ms**.
  - So the stall is caused by the pool queue, not by CPU load. The same thing happens with searches against an index that is being trained or bulk-added.
  - The VM's own dirty-CPU work (e.g. large-heap GC) shares the same pool [INFERENCE].
- **Fix:**
  1. Move the pure getters (`ntotal`, `is_trained`) to `ERL_NIF_DIRTY_JOB_IO_BOUND`. They compute nothing, and the IO pool exists for blocking waits.
  2. For heavy reads (search, reconstruct, residuals), try `enif_rwlock_tryrlock` first on the CPU-bound entry. If it fails, `enif_schedule_nif` an IO-bound variant that blocks on the lock and does the whole operation there. Do not hold a pthread rwlock across a reschedule: an unlock from a different thread is UB.
  3. Update the Concurrency section in CLAUDE.md and the moduledoc.

### N2 — High — FAISS's OpenMP teams multiply across dirty schedulers (≈3× throughput loss under concurrency)
- **Where:**
  - The NIF never limits OpenMP threads. There is no `omp_set_num_threads` call anywhere in `c_src/faiss_ex_nif.c`, and the FAISS C API offers no equivalent (grep of `c_api/` for `omp_|num_threads`: none).
  - FAISS opens an `omp parallel num_threads(nt)` team even for a single query (`faiss/utils/distances.cpp:301,342,652`).
  - Each of the 10 dirty CPU schedulers can therefore start a ~10-thread team, so up to ~100 runnable threads compete for 10 cores.
- **Evidence (measured):** Flat index, 100k×128, `search_binary` with k=10, two runs each:

  | Scenario | Default OpenMP | `OMP_NUM_THREADS=1` |
  |---|---|---|
  | 10 processes × 200 single-query searches | 1845 / 1634 ms | 622 / 572 ms (~3× faster) |
  | Serial single-query latency (200 calls) | 225 ms | 215 ms (OpenMP adds nothing here) |
  | One 1000-query batch | 102 ms | 164 ms (OpenMP helps here) |

  - libomp's default 200 ms spin-wait after a parallel region (`KMP_BLOCKTIME`) likely adds to the loss [INFERENCE].
  - On Linux, FAISS links whatever `libopenblas.so` resolves to (`faiss/CMakeLists.txt:751-762`). On Ubuntu that is usually the pthreads OpenBLAS, which nests its own thread pool inside OpenMP [INFERENCE].
- **Fix:**
  1. Add a per-call `:threads` option (or an app-env default). The NIF calls `omp_set_num_threads(n)` on the dirty thread before the `faiss_*` call; that setting is per-thread, so concurrent callers do not interfere. Link the NIF against libomp/libgomp.
  2. Document the trade-off in README "Concurrency": many concurrent small queries → `OMP_NUM_THREADS=1` (with `KMP_BLOCKTIME=0` / `OMP_WAIT_POLICY=passive`); few large batches → default.
  3. Recommend `libopenblas-openmp-dev` in the Linux setup.

### N3 — High — `FAISS_OPT_LEVEL=avx2|avx512` has no effect: the Makefile builds and ships the generic `faiss` target
- **Where:** `Makefile:183` (`$(MAKE) ... faiss`), `:188` (`faiss_c`), `:191` and `:200` (copy `faiss/libfaiss` and `c_api/libfaiss_c`). README `:130-145` documents `FAISS_OPT_LEVEL=avx2 mix compile` as the way to get AVX2.
- **Evidence:**
  - In FAISS v1.15.1 the AVX2/AVX-512 code goes into **separate** targets: `faiss_avx2`, `faiss_avx512` and `faiss_avx512_spr`, with `-mavx2 -mfma ...` set only on them (`faiss/CMakeLists.txt:454-495`). The C API equivalents are `faiss_c_avx2` and friends (`c_api/CMakeLists.txt:44-83`).
  - The main `faiss` target (`faiss/CMakeLists.txt:452`) gets SIMD flags only when `FAISS_OPT_LEVEL=dd` (`:545-602`).
  - Result: an x86-64 build with `avx2` still ships the scalar/SSE `libfaiss`. Linux x86 is the usual production target and the CI target (`.github/workflows/ci.yml`, ubuntu).
  - arm64 is not affected: NEON is always compiled into `faiss` (`:629-632`).
  - A changed `FAISS_OPT_LEVEL` also never reconfigures an existing build: `$(FAISS_BUILD)/Makefile` depends only on `.cloned` (`Makefile:176`). README `:144`'s `rm -rf` works around this.
- **Fix:**
  - Preferred: default `FAISS_OPT_LEVEL` to `dd` (runtime SIMD dispatch into the main `faiss`/`faiss_c` targets). One portable binary then uses AVX2/AVX-512 when the CPU has it, and the target-name problem goes away. Change both `Makefile:24` and `mix.exs:80`.
  - If you keep explicit levels: build `faiss_$(LEVEL)` + `faiss_c_$(LEVEL)`, copy them as `libfaiss`/`libfaiss_c`, and fix the `install_name_tool`/soname handling.
  - Either way: put the opt level in the build-dir path (e.g. `build-$(FAISS_OPT_LEVEL)`) so changing it triggers a reconfigure, and update README `:130-147`.

### N4 — Medium — `nif_new_index` runs `faiss_index_factory` on a normal scheduler; it measures 4–176 ms
- **Where:** `c_src/faiss_ex_nif.c:1028` (flag `0`), with the factory call at `:237`.
- **Evidence (measured `Index.new/2`):**

  | Factory string | Time |
  |---|---|
  | `IVF262144,Flat` (d=128) | 4.3 ms |
  | `IVF1048576,PQ16` (d=128) | 16.4 ms |
  | `LSHr` (d=512) | 21.0 ms |
  | `LSHr` (d=2048) | **175.7 ms** |
  | `HNSW32` | 0.28 ms |

  - The LSH rotation is built in the constructor: `IndexLSH.cpp:37-38`, `rrot.init(5)`.
  - All of these exceed the 1 ms budget for normal schedulers, and the factory string is caller-controlled.
  - The moduledoc promise "heavy operations run on dirty schedulers" (`index.ex:78-79`) is therefore broken for `new/3`.
- **Fix:** flag `nif_new_index` as `ERL_NIF_DIRTY_JOB_CPU_BOUND`. Index creation is rare, so the ~µs dirty hop costs nothing. Update the CLAUDE.md dirty-scheduler list. `nif_new_clustering` (`:1047`) is just a constructor and can stay on a normal scheduler.

### N5 — Medium — The Binary API stops at `Index`: Clustering and `compute_residuals` only accept lists in and return lists out
- **Where:**
  - `clustering.ex:73` (`train/2` guards `is_list(data)`), `:88-91` (`get_centroids/1` always decodes), `:108-110` (`get_cluster_assignment/2` list-only).
  - `index.ex:336-349` (`compute_residuals/3`: `is_list` guards, always decodes).
  - Moduledoc `index.ex:41` claims "Every hot-path function also works directly with raw binaries".
- **Evidence:**
  - k-means training is the biggest input in the library. A caller holding a binary (Nx, file, HTTP) must build a list of lists (~32 B per float on the heap) only for `encode_vectors!/2` to turn it straight back into a binary.
  - On the output side, the measured decode amplification is 8× for f32 rows (boxed float + cons cell per element). For a 1000×100 result: 1.2 MB of binaries became 4.83 MB of terms (`:erts_debug.flat_size`).
- **Fix:**
  - Route `Clustering.train/2` through the same `encode_vector_data!` binary clause as `Index.train/2` (move that helper to `Shared`).
  - Add `get_centroids_binary/1` (the NIF already returns `{k, d, bin}`), and let `get_cluster_assignment/2` accept a binary by delegating to `Index.search_binary/3`.
  - Either accept binary `xs`/`keys` in `compute_residuals/3` or add `compute_residuals_binary/3`.
  - Fix the moduledoc claim.

### N6 — Medium (CUDA builds only) — Every `cpu_to_gpu` creates its own `StandardGpuResources`, reserving up to 1.5 GiB of GPU temp memory per index
- **Where:** `c_src/faiss_ex_nif.c:751-752` (`faiss_StandardGpuResources_new` on every call). The resource is owned per index at `:766-775`.
- **Evidence:**
  - Each resource reserves temp memory per device on first use: up to 512 MiB / 1 GiB / **1.5 GiB** depending on GPU size (`faiss/gpu/StandardGpuResources.cpp:51-58,180-202,452-455`).
  - It also creates its own cuBLAS handle and streams [INFERENCE: FAISS GPU design].
  - N GPU indexes therefore reserve N× that memory. FAISS recommends sharing one resource object across indexes.
- **Fix:**
  - Create one `GpuResourcesResource` per device: lazily in a NIF-global table guarded by a mutex, or as an explicit `FaissEx.GpuResources.new/0` passed to `cpu_to_gpu/3`.
  - Keep the per-index `gpu_resources_ref` as a refcount holder.
  - Optionally expose `faiss_StandardGpuResources_setTempMemory` (C API `StandardGpuResources_c.h:36`).

### N7 — Low — `nif_get_clustering_centroids` allocates and copies the result while holding the read lock
- **Where:** `c_src/faiss_ex_nif.c:948-970`. `enif_alloc_binary` at `:962` and `memcpy` at `:967` both run under the read lock.
- **Evidence:** the allocation size is `k*d`. Both values are constructor arguments that FAISS never changes, so they could be cached at wrap time, the same way `IndexResource.dim` is (`:66-70`). The copy itself has to stay inside the lock because FAISS owns the buffer. The cost is small (k×d floats), but it is the only remaining allocation inside a critical section.
- **Fix:** cache `k`/`d` in `ClusteringResource` and call `enif_alloc_binary(k*d*4)` before the lock. That also lets `nif_train_clustering` (`:910-918`) validate size before taking both write locks, so its `goto unlock` validation branches disappear.

### N8 — Medium/Low — The benchmarks do not exercise the paths where the cost actually is
- **Where:** `bench/faiss_ex_bench.exs:41-112`.
- **Evidence:**
  - Every scenario is single-process. N1 and N2, which show 3× and >1000× effects, cannot appear in it.
  - `reconstruct` only uses contiguous keys (`:24-26`), so the per-key loop (`faiss_ex_nif.c:563-567`) is never measured.
  - There is no `reconstruct_binary` scenario and no list-vs-binary decode comparison at large k (`k = 10` only, `:7`).
  - Only `Flat` is tested; there is no IVF/HNSW search.
  - The `add` scenarios include `Index.new` in the timed function (`:45,50,56,60`).
- **Fix:**
  - Add a `parallel: System.schedulers_online()` Benchee run (or a Task-fanout scenario) for small searches.
  - Add scenarios for shuffled-key reconstruct and `reconstruct_binary`, and for k=100 with list vs `search_binary`.
  - Add one IVF/HNSW index.
  - Move index creation into `before_each`.

Clean areas: the C side uses zero-copy `enif_inspect_binary` inputs and `enif_alloc_binary`/`enif_make_binary` outputs with no extra copies; validation and allocation happen before the lock in every single-lock NIF; there is no O(n²) pattern; the decoders produce no sub-binary retention; and the NIF is built with `-O2`, FAISS with `Release`. macOS uses Accelerate + Homebrew libomp, as intended.
