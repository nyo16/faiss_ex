# Security Audit: faiss_ex (a397362, v0.3.2, FAISS v1.15.1)

Scope: `c_src/faiss_ex_nif.c` (primary), `lib/faiss_ex/*.ex`, `Makefile`, `mix.exs`,
`.github/workflows/{ci,publish}.yml`. Upstream behaviour was checked against the
cached FAISS v1.15.1 tree at `~/.cache/faiss_ex/faiss-v1.15.1/` (cited as `faiss/…`).
Not applicable here (no web layer): auth, sessions, CSRF, XSS, SQL, LiveView authz.
`sobelow` was not run because it is not a dependency and has nothing to scan in a NIF library.

## Score: 74/100

How the rubric was adapted (the Phoenix criteria become their NIF-library equivalents).
Per-finding deductions: Critical −15, Medium −5, Low −2. Each category has a floor of 0.

| Criterion (pts) | Findings | Deduction | Score |
|---|---|---|---|
| Critical memory-safety bugs (30) | none | 0 | 30 |
| High/Medium NIF issues (20) | N1 GPU negative device → `abort()` (M), N2 untrusted factory string → VM DoS (M), N5 unaligned binary pointers (L) | −5 −5 −2 | 8 |
| Lock / resource lifetime (15) | none | 0 | 15 |
| Atom exhaustion (10) | none | 0 | 10 |
| Untrusted-input deserialization (10) | N3 `from_file/2` on an untrusted file → arbitrary-path read-write mmap (M) | −5 | 5 |
| Supply chain / secrets (15) | N4 actions pinned to mutable tags in the job that holds the secret (M), P2 unquoted Makefile env interpolation (L, still open), N6 workflow hardening (L) | −5 −2 −2 | 6 |
| **Total** | | | **74** |

The previous score was 85. No code regressed. The drop comes from new findings: the
earlier audit did not trace the GPU device path, factory-string parsing, or on-disk
inverted lists into FAISS.

Clean areas, one line: every `enif_get_*` and `enif_inspect_binary` return value is
checked. Every size product goes through `safe_mul`, and binary sizes are validated
against the cached `res->dim` before the lock. Lock unwinding is correct: there is no
return path that still holds a lock, and the two functions that must validate locked
state each use a single `unlock:` label (`c_src/faiss_ex_nif.c:923-925`, `:969-970`).
Dirty-scheduler placement matches CLAUDE.md. Destructors NULL their pointers, and
resources stay alive through `argv` for the length of each call. GPU index lifetime is
safe: `GpuIndex` holds a `shared_ptr` to the impl
(`faiss/gpu/StandardGpuResources.h:204,265`; `faiss/gpu/GpuIndex.cu:63`), so freeing
the `GpuResources` resource first cannot cause a use-after-free. Only 4 static atoms
are created (`:1000-1005`). The Elixir layer has no `String.to_atom`, `binary_to_term`,
`System.cmd`, or `Code.eval`.

## Prior findings status

| # | Prior finding | Status | Evidence |
|---|---|---|---|
| P1 | FAISS pulled from a mutable tag with no SHA check (Medium) | **RESOLVED** | `Makefile:15-21` pins `FAISS_GIT_SHA` for `v1.15.1`. `Makefile:163-172` fails closed: on mismatch it runs `rm -rf` and `exit 1`, and the `.cloned` stamp is never touched. `mix.exs:9-10,76-86` mirrors the pin. Residual gap: a custom `FAISS_GIT_REV`, or an explicitly empty `FAISS_GIT_SHA`, builds unverified without any warning (`Makefile:16-22`, `mix.exs:79`). Folded into P2's fix. |
| P2 | Build-time shell interpolation of env vars (Low) | **STILL OPEN** | Expansions are still unquoted: `git clone --depth 1 --branch $(FAISS_GIT_REV) $(FAISS_GIT_REPO) $(FAISS_SRC)` at `Makefile:162`, `git -C $(FAISS_SRC)` at `:164`, `rm -rf $(FAISS_SRC)` at `:169`, `cp $(SYSTEM_LIB_DIR)/…` at `:134,142`. `FAISS_SRC` is built from `$(HOME)` and the env-controlled `FAISS_GIT_REV` (`:27-28`). |
| P3 | Unbounded result allocation from caller `n`/`k` (Low/info) | **STILL OPEN (accepted)** | `c_src/faiss_ex_nif.c:398-412`: `safe_mul` plus a checked `enif_alloc_binary`, which is memory-safe. A huge `k` is still a caller-driven allocation. No change needed unless `k` comes from untrusted input. |
| P4 | NIF trusts FAISS to bounds-check reconstruct keys (Low/info) | **RESOLVED** | The prior premise was wrong: negative keys were an out-of-bounds read, fixed in 1f82d4b. The NIF now rejects negative keys on every path: `keys_non_negative` at `:185-190`, used in reconstruct at `:522` and residuals at `:616`. It also rejects contiguous ranges whose end overflows int64 (`:542`). Tests cover the batched, per-key, binary and residual paths (`test/faiss_ex/index_test.exs:244-249,713-731`). Upper bounds are enforced by FAISS v1.15.1: `IndexFlat.cpp:285`, `IndexFlatCodes.cpp:79`, `invlists/DirectMap.cpp:75-76`, `IndexFastScan.cpp:619-621`, and `IndexIDMap.cpp:388+` (`rev_map` lookup). |

## New findings

### N1. A negative GPU device id aborts the whole BEAM (CUDA builds)
- **Severity**: Medium (VM-wide DoS from a single bad argument; `USE_CUDA=true` builds only)
- **Location**: `c_src/faiss_ex_nif.c:746-759` (`nif_index_cpu_to_gpu`); `lib/faiss_ex/index.ex:413-416` and `:178` (`Index.new(..., device: {:cuda, id})`)
- **Evidence**: The NIF passes `device` from `enif_get_int` straight to `faiss_index_cpu_to_gpu` with no sign check. The Elixir clauses have no guard; `non_neg_integer()` exists only in the `@spec`. Inside FAISS:
  - `GpuIndex::GpuIndex` checks only the upper bound, `config_.device < getNumDevices()` (`faiss/gpu/GpuIndex.cu:65-68`), so `-1` passes.
  - `initializeForDevice` checks `FAISS_ASSERT(device < getNumDevices())` (`faiss/gpu/StandardGpuResources.cpp:342`), which `-1` also passes.
  - `getDeviceProperties(-1)` then calls `CUDA_VERIFY(cudaGetDeviceProperties(&prop, device))` (`faiss/gpu/utils/DeviceUtils.cu:68`). That fails with an invalid-device error and reaches `FAISS_ASSERT_FMT` → `abort()` (`faiss/impl/FaissAssert.h:52-64`).
  - The C API's catch-all cannot intercept `abort()`, so the node dies.
- **Fix**: Reject the value in the NIF before any FAISS call: `if (device < 0) return make_error_msg(env, "device must be non-negative");` after `:749`. Also add `when is_integer(device_id) and device_id >= 0` to `cpu_to_gpu/2`, and add a `:cuda`-tagged test for `cpu_to_gpu(index, -1)`.

### N2. `Index.new/3` docs invite untrusted factory strings, which can OOM or crash the VM from a normal scheduler
- **Severity**: Medium (DoS)
- **Location**: `lib/faiss_ex/index.ex:133-134` ("`dim` or `description` … routinely come from configuration or user input"); `c_src/faiss_ex_nif.c:237`; `:1028` (`nif_new_index` on a normal scheduler)
- **Evidence**: The factory string controls allocation size at construction time, with no upper bound:
  - `IVF<n>[kM]` sets `nlist` (`faiss/index_factory.cpp:291-292`). `parse_nlist` returns `std::stoi(s) * multiplier` (`:274`), and that `int` multiply overflows for `IVF3000M`.
  - `IMI2x<nbit>` computes `nlist = (size_t)1 << (2 * nbit)` (`:300`), which is undefined behaviour for `nbit ≥ 32`, and builds a `MultiIndexQuantizer(d, 2, nbit)` with `2^nbit` centroids per sub-quantizer.
  - Something like `"IVF100M,Flat"` makes the inverted lists value-initialise about 1e8 × 2 `std::vector` headers. With Linux overcommit that commits memory page by page, and the OOM killer can take the BEAM. This runs inside `nif_new_index`, which is registered on a **normal** scheduler (`:1028`). [INFERENCE: exact peak memory per factory string not measured]
  - FAISS parses descriptions with `std::regex` (`index_factory.cpp:213` and others). libstdc++'s regex executor is recursive, so very long inputs are a known stack-exhaustion risk. [INFERENCE: not reproduced]
- **Fix**: Remove "or user input" from the doc and state that `description` must be trusted or chosen from an allow-list. Defensively, cap `byte_size(description)` (for example 256) in `Index.new/3`, and move `nif_new_index` to `ERL_NIF_DIRTY_JOB_CPU_BOUND`, since construction can allocate gigabytes. Update CLAUDE.md's dirty-scheduler list to match.

### N3. `from_file/2` on an untrusted index file can open and **write** to an arbitrary path chosen by the file
- **Severity**: Medium (untrusted-input deserialization; matters for any app that loads uploaded or downloaded `.index` files)
- **Location**: `lib/faiss_ex/index.ex:362-372` (`io_flags` defaults to `0`, and the doc gives no trust warning); `c_src/faiss_ex_nif.c:684`; README `:243-244` (no warning)
- **Evidence**: FAISS registers `OnDiskInvertedListsIOHook` by default (`faiss/invlists/InvertedListsIOHook.cpp:37`). An IVF file with on-disk lists carries its own `filename` (`faiss/invlists/OnDiskInvertedLists.cpp:736-737`). With `io_flags = 0`, `read_only` is false (`:719`), so `do_mmap()` runs `fopen(filename, "r+")` and `mmap(..., PROT_READ|PROT_WRITE, MAP_SHARED, ...)` with a `totsize` taken from the file (`:270-283`, `:764-766`). Consequences:
  - A crafted index makes the BEAM map any file the OS user can open read-write.
  - `search`/`reconstruct` disclose that file's bytes as vector data.
  - `add/2` writes attacker-shaped data into it.
  - A `totsize` larger than the real file causes SIGBUS on access, which crashes the VM.
  - FAISS v1.15.1 does validate many other fields on read (`faiss/impl/index_read.cpp:1757-1760`, `validate_HNSW` at `:1307`, the NSG range checks at `:1340-1360`). The on-disk filename is not one of them.
- **Fix**: Add a "Security" note to `from_file/2` and the README: index files are trusted input, never load user-supplied files, and on-disk IVF indexes reference external paths. Optionally expose `:io_flags` symbolically and recommend `IO_FLAG_READ_ONLY` (16) for read-only use. That blocks the write primitive but not the read.

### N4. GitHub Actions pinned to mutable tags in the job that holds `HEX_API_KEY`
- **Severity**: Medium (supply chain)
- **Location**: `.github/workflows/publish.yml:19,21,30` (`actions/checkout@v6`, `erlef/setup-beam@v1`, `actions/cache@v5`); `:79` (`HEX_API_KEY`). The same pattern appears throughout `ci.yml:23,25,51,58,76,78,90,92,100,106,139,141,166`.
- **Evidence**: `HEX_API_KEY` is scoped to the final step, but earlier steps run third-party action code in the same job. A moved `erlef/setup-beam@v1` tag could swap the toolchain (`mix`/`erl`) on PATH, and the later `mix hex.publish` step would then run the attacker's binary with the key in its environment.
- **Fix**: Pin every `uses:` to a full commit SHA, with the version in a comment (`erlef/setup-beam@<40-char-sha> # v1.x.y`), and let Dependabot (`package-ecosystem: github-actions`) bump them.

### N5. Binary API hands possibly unaligned pointers to `float*`/`idx_t*` (undefined behaviour)
- **Severity**: Low
- **Location**: `c_src/faiss_ex_nif.c:306,351-352,418,464,520-522,615-616,625,921`; exposed by the binary pass-through in `lib/faiss_ex/index.ex:455-466` (added in b5127f1)
- **Evidence**: Before b5127f1, every input binary was freshly built from iodata. Now a caller binary passes through as-is (`encode_vector_data!/2` only checks `rem(byte_size, dim*4)`). A sub-binary such as `binary_part(body, 1, n)` or `<<_hdr::binary-size(3), rest::binary>>` is not copied by `enif_inspect_binary` when its bit offset is 0. Its `data` pointer can therefore sit at any byte offset, and the NIF dereferences it as `idx_t*` itself (`keys_non_negative`, `:186-187`, and the contiguity loop) before handing it to FAISS. On x86-64 and arm64, scalar and `loadu` accesses tolerate this. It is still undefined behaviour, and any aligned-SIMD path or alignment-assuming vectoriser would fault. [INFERENCE: no crash reproduced on current targets]
- **Fix**: In the NIF, check `((uintptr_t)bin.data % _Alignof(float|int64_t)) != 0`, and when misaligned copy into an `enif_alloc` buffer (or reject). Alternatively, do the cheap check in Elixir and `:binary.copy/1` only when needed. Add a test using `binary_part/3` at offset 1.

### N6. Workflow hardening gaps
- **Severity**: Low
- **Location / Evidence**:
  - `ci.yml` has no `permissions:` block (`ci.yml:1-14`), so `push` runs on `master` get the repository-default `GITHUB_TOKEN` scope, which can be write-all on older repos.
  - `publish.yml:41-42` interpolates the tag name, unvalidated, into a GNU `sed` program. Git ref names may contain `/` and `;`, so a tag like `v1/;e<cmd>;#` becomes a `sed e` command. This requires tag-push rights. [INFERENCE: not executed]
  - `publish.yml` has no `environment:` with required reviewers, so anyone who can push a `v*` tag can publish to Hex.
- **Fix**:
  - Add top-level `permissions: contents: read` to `ci.yml`.
  - In `publish.yml`, validate first: `[[ "$TAG_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$ ]] || exit 1`.
  - Move the publish job into a protected `environment: hex` that holds `HEX_API_KEY`, and add a tag-protection rule for `v*`.

## Recommendations (priority order)
1. N1: add a `device < 0` guard in the NIF and Elixir. This is a one-line fix for a VM `abort()`.
2. N3 and N2: document that index files and factory strings are trusted input. Cap description length and move `nif_new_index` to a dirty scheduler.
3. N4: pin actions by SHA, especially in `publish.yml`.
4. P2 + P1 residual: quote every Makefile expansion (`"$(FAISS_SRC)"`, `"$(FAISS_GIT_REV)"`, `"$(FAISS_GIT_REPO)"`, `"$(SYSTEM_LIB_DIR)"`), validate `FAISS_GIT_REV` against `^[A-Za-z0-9._/-]+$`, and print a `$(warning …)` when `FAISS_GIT_SHA` is empty.
5. N5, N6: alignment copy-or-reject; workflow permissions, tag validation, and a protected environment.

## Tools to recommend (run manually)
- `mix hex.audit` (already clean). Add `mix_audit` as a `:dev` dep if `mix deps.audit` is wanted.
- Build the NIF with `-fsanitize=address,undefined,alignment` and fuzz each NIF with malformed `n`/`k`/binary/sub-binary inputs. N5 would show up immediately.
- `zizmor` or `actionlint` on `.github/workflows/` for N4 and N6.
