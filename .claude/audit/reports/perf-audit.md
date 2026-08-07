# Performance Audit — faiss_ex

Date: 2026-07-12
Scope: `lib/faiss_ex/shared.ex`, `lib/faiss_ex/index.ex`, `lib/faiss_ex/clustering.ex`, `c_src/faiss_ex_nif.c`, `Makefile`/`mix.exs` (flags only)

## Performance health score: 84/100

The native layer is close to optimal (zero-copy binary inspection, per-resource RW locks, correct dirty-scheduler placement); the remaining cost is concentrated in the list-based public API, which forces O(n) term-building and ~4-10x memory blowup on encode/decode at large batch sizes.

---

## Findings

### 1. [HIGH — at scale] List-only API forces full decode of results; no raw-binary passthrough
- **Where:** `/Users/nmaroulis/Source/faiss_ex/lib/faiss_ex/index.ex:184-196` (`search/3`), `:243-254` (`reconstruct/2`), `:262-274` (`compute_residuals/3`); `/Users/nmaroulis/Source/faiss_ex/lib/faiss_ex/clustering.ex:92-100`
- **Issue:** The NIF already returns compact f32/s64 binaries, but the public API unconditionally decodes them into lists of lists (`Shared.binary_to_floats/1` + `Enum.chunk_every/2`). Each decoded float is a boxed 8-byte float plus cons cell — roughly 16 bytes/element vs 4 bytes in the binary, so a `search(index, batch_10k, k: 100)` result materializes ~32 MB of heap terms from an 8 MB binary, all on the calling process heap (GC pressure). Same asymmetry on input: users holding vectors as binaries (e.g., from Nx, files, or a previous FAISS call) must convert binary → list → binary.
- **Impact:** For large `n × k` or large `dim`, decode dominates end-to-end latency and memory; the FAISS search itself is often cheaper than the term-building. This is the single biggest scalability ceiling the library imposes on users.
- **Suggestion:** Add an opt-in raw path — e.g., `search(index, data, k, output: :binary)` returning `{distances_bin, labels_bin}` and accepting a pre-encoded `{n, binary}` input — leaving the list API as the friendly default. No C changes required.

### 2. [MEDIUM] Result decode does two full passes plus an intermediate list (`decode` then `chunk_every`)
- **Where:** `/Users/nmaroulis/Source/faiss_ex/lib/faiss_ex/index.ex:189-190`, `:249`, `:269`; `/Users/nmaroulis/Source/faiss_ex/lib/faiss_ex/clustering.ex:95`
- **Issue:** `binary_to_floats(bin) |> Enum.chunk_every(k)` builds one flat list of n×k elements, then a second pass re-consumes it into row sublists, allocating a second full list structure and discarding the first. A single recursive pass that pattern-matches `k` (or `dim`) elements per iteration (`<<row::binary-size(k*4), rest::binary>>` + per-row comprehension) halves list allocation and passes.
- **Impact:** ~2x decode allocation/GC on every search/reconstruct/residual/centroid call; noticeable at large n×k, invisible at small ones.

### 3. [MEDIUM] Search result binaries are allocated inside the read-lock critical section
- **Where:** `/Users/nmaroulis/Source/faiss_ex/c_src/faiss_ex_nif.c:348-377` (`nif_search_index`)
- **Issue:** `enif_rwlock_rlock` is taken at line 348, but the two `enif_alloc_binary` calls (lines 370-377) depend only on `n` and `k` — not on any index state — and could execute before locking. Also `dim` is immutable after index creation, so the `data_bin.size` validation (lines 350-359) could move out too (Elixir already validated row length against `dim` during encode; alternatively pass `dim` down). For large `n*k` these allocations can page-fault/zero multi-MB buffers while holding the lock.
- **Impact:** Readers don't block each other, but the lock hold time is extended, delaying pending writers (`add`/`train`) and — because `enif_rwlock` writer-preference then queues subsequent readers — can convoy searches behind a writer on mixed workloads. Cheap fix, pure win.
- **Same pattern (lower stakes):** `nif_reconstruct_batch` (`:499-511`) and `nif_compute_residuals` (`:560-579`) also allocate result binaries under the read lock.

### 4. [MEDIUM] Batch encode builds per-row binaries, reverses, then copies again via `IO.iodata_to_binary`
- **Where:** `/Users/nmaroulis/Source/faiss_ex/lib/faiss_ex/shared.ex:39-45` (`encode_batch/4`), `:51-63` (`encode_row/5`)
- **Issue:** Each row is appended element-by-element into its own binary (the append is BEAM-optimized, fine), but rows are then collected into a list, `Enum.reverse`d, and concatenated with `IO.iodata_to_binary/1` — one extra full copy of the entire payload plus a transient list of n refc binaries. Appending rows directly into a single accumulator binary (the same writable-binary optimization already used inside `encode_row`) eliminates the row list, the reverse, and the final copy — one pass, one buffer.
- **Impact:** ~1 extra full-payload copy + n binary headers per `add`/`train`/`search` batch call. For a 1M × 128 f32 add (512 MB payload) that is a spare 512 MB copy.

### 5. [LOW] Property getters run on dirty CPU schedulers, paying migration cost per call
- **Where:** `/Users/nmaroulis/Source/faiss_ex/c_src/faiss_ex_nif.c:988-990` (`nif_get_index_dim/ntotal/is_trained` flagged `ERL_NIF_DIRTY_JOB_CPU_BOUND`)
- **Issue:** The dirty placement is deliberate and correct (they block on the rwlock, and a writer may hold it for seconds during `train`), but it means every `ntotal/1` call pays the ~1-3 µs dirty-scheduler round trip for a nanosecond read. An `enif_rwlock_tryrlock` fast path is not available for these as NIFs can't reschedule themselves to dirty conditionally without `enif_schedule_nif`; that (try-rlock on a normal scheduler, `enif_schedule_nif` to dirty on contention) would make the common uncontended case ~free.
- **Impact:** Only matters if callers poll `ntotal` in hot loops; correctness-first trade-off, acceptable as-is.

### 6. [LOW] Bulk `add` holds the write lock for the entire batch, starving searches
- **Where:** `/Users/nmaroulis/Source/faiss_ex/c_src/faiss_ex_nif.c:251-265` (`nif_add_to_index`); Elixir entry `/Users/nmaroulis/Source/faiss_ex/lib/faiss_ex/index.ex:146-153`
- **Issue:** A single `add/2` with a very large batch (or `train/2`) holds the write lock for the full FAISS call, blocking all concurrent searches on that index for its duration. This is inherent to the (correct) locking model, but the Elixir layer offers no chunking, so users doing live ingest + serve on one index get multi-second search stalls.
- **Impact:** Latency spikes on mixed read/write workloads with large batches. Mitigation is documentation plus an optional `chunk_size:` option on `add/2` that splits the NIF calls, letting searches interleave between chunks.

### 7. [LOW] `reconstruct/2` traverses `keys` twice (`length/1` then encode)
- **Where:** `/Users/nmaroulis/Source/faiss_ex/lib/faiss_ex/index.ex:244-245`
- **Issue:** `length(keys)` is one full pass; `Shared.int64s_to_binary(keys)` is a second. A combined count-and-encode (as `encode_vectors!/2` already does for vectors) does it in one. Same micro-pattern applies to `add_with_ids/3` ids (`:166`), where ids length is never counted in Elixir at all (fine — C validates), so only `reconstruct` double-passes.
- **Impact:** Negligible except for very long key lists; trivial fix.

### 8. [LOW, GPU builds only] `nif_get_num_gpus` runs on a normal scheduler but may initialize the CUDA driver
- **Where:** `/Users/nmaroulis/Source/faiss_ex/c_src/faiss_ex_nif.c:773-777`, scheduled at `:994` with flag `0`
- **Issue:** `faiss_get_num_gpus()` calls `cudaGetDeviceCount`, whose first invocation in a process loads/initializes the CUDA driver and can take tens of milliseconds — well over the 1 ms normal-scheduler budget.
- **Impact:** One-time normal-scheduler stall on first call in CUDA builds. Flag it `ERL_NIF_DIRTY_JOB_IO_BOUND`.

### 9. [LOW] Default `FAISS_OPT_LEVEL=generic` leaves x86 SIMD performance unused
- **Where:** `/Users/nmaroulis/Source/faiss_ex/Makefile:13`; `/Users/nmaroulis/Source/faiss_ex/mix.exs:69`
- **Issue:** The default builds FAISS without `avx2`/`avx512` dispatch. On x86-64 hosts, `avx2` typically yields 2-4x on flat/PQ distance computations. The env override exists (good, deliberate portability default), but nothing auto-detects the host or surfaces the knob at build time, so most users silently run the slow kernels.
- **Impact:** Up to several-x search/train throughput on x86; no effect on Apple Silicon. Consider auto-detecting (`sysctl`/`/proc/cpuinfo`) with `generic` fallback, or at least printing a build-time hint.

---

## Clean areas (one line each)

- C binary handling: inputs via `enif_inspect_binary` (zero-copy) and outputs via `enif_alloc_binary`/`enif_make_binary` (no double copy; the one `memcpy` in centroids is unavoidable since FAISS owns that buffer).
- Locking: per-resource `ErlNifRWLock`, reads share, writes exclusive, two-lock ordering in `nif_train_clustering` fixed and documented — no global lock, no ordering hazard.
- Dirty scheduler assignments for the heavy ops (CPU-bound vs IO-bound split) are correct; only items 5 and 8 above are debatable.
- `nif_reconstruct_batch` contiguous-key fast path (batched `reconstruct_n`) is a genuine optimization and the O(n) contiguity scan runs outside the lock.
- Decode helpers use binary comprehensions (not `:binary.bin_to_list`) — the efficient construct.
- Per-call Elixir overhead is otherwise minimal: single-pass `encode_vectors!` with inline dim validation, no re-validation, no re-encoding, batch and single-vector share one path.
- NIF compiled `-O2 -fPIC`; FAISS built `CMAKE_BUILD_TYPE=Release` — no missing optimization flags (aside from item 9's opt-level default).
