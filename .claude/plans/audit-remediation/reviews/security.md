# Security Review: faiss_ex v0.4.0 audit remediation (`a397362..master`)

Scope: `Makefile`, `.github/workflows/{ci,publish}.yml`, `.github/dependabot.yml`,
NIF input validation in `c_src/faiss_ex_nif.c`, and the `Index.from_file/2` trust docs.
Method: static read of the repo, plus the pinned FAISS source at
`~/.cache/faiss_ex/faiss-v1.15.1`. Action and FAISS pins were checked against the GitHub API.
No probes were executed: this agent has no shell. Every runtime claim below is
code-derived, and claims marked **[INFERENCE]** were not executed.

## Executive summary

No blockers. There are 4 warnings:

- 2 are memory-safety holes reachable through `from_file/2`. One is reachable with a
  *legitimate* file and a documented flag.
- 1 is a docs claim that overstates what `io_flags: 6` protects against.
- 1 is a release gate that does not exist yet.

The Makefile/CI hardening is sound. What remains there is defense-in-depth and comment accuracy.

---

## WARNING 1: `from_file` docs present `io_flags: 6` as making untrusted files safe; it does not

- **Location:** `lib/faiss_ex/index.ex:453-457`, `README.md:259-264`, `CHANGELOG.md:89-91`
- **Evidence:**
  > `For files from an untrusted source, pass io_flags: 6 (IO_FLAG_READ_ONLY ||| IO_FLAG_ONDISK_SAME_DIR): the stored path is replaced by the index file's own directory and the mapping is read-only.`

  README: *"For untrusted files pass `io_flags: 6`"*.
- **Why it matters:** Flag 6 only constrains *which* file the on-disk IVF loader maps. FAISS
  deserialization is not hardened against crafted input, and several crash or disclosure
  paths survive flag 6. All of these come from FAISS v1.15.1 source:
  1. **Unchecked mmap length and offsets.** `OnDiskInvertedListsIOHook::read` (`invlists/OnDiskInvertedLists.cpp:716-767`) takes `totsize`, `lists[].offset` and `lists[].size` from the file. `do_mmap` maps `totsize` bytes (`:281`) and never compares it to the real file size. `get_codes`/`get_ids` return `ptr + lists[i].offset` without bounds (`:386-401`). As a result:
     - A `totsize` larger than the file means SIGBUS on first touch.
     - An out-of-range offset means a SIGSEGV, or arbitrary process memory returned as floats by `search`/`reconstruct`.
     - Either way, a crafted file crashes the VM.
  2. **`ONDISK_SAME_DIR` only takes the basename** (`:739-761`). A crafted file can still map any *sibling* file in the index's directory, such as another tenant's upload in a shared upload dir. Combined with (1), its bytes come back through `reconstruct`.
  3. **Null inverted lists.** The `il00` invlists tag yields `invlists == nullptr` (`impl/index_read.cpp:533-539`). `IndexIVF::search` guards against this (`IndexIVF.cpp:330,429`). `IndexIVF::reset` (`:1272-1274`) and `IndexIVF::reconstruct_n` (`:1088-1091`) do not, and both are reachable through `Index.reset/1` and the batched `reconstruct`. The result is a null dereference, which kills the VM.
  4. **A zero-dimension index** passes FAISS validation and corrupts the heap inside our NIF (see WARNING 2).

  Following the docs' advice therefore still lets an untrusted file crash the BEAM or leak memory.
- **Fix:** This is a docs fix only. D4's default (`io_flags: 0`) and the documentation approach stay as decided. Change the wording in all three places to something like:
  > Index files are not a safe interchange format: FAISS does not validate crafted files, and a malicious file can crash the VM or expose memory regardless of `:io_flags`. Load only files you produced or fully trust. `io_flags: 6` additionally stops a trusted-but-relocated on-disk IVF index from mapping the absolute path stored inside it (it resolves the lists file next to the index file, read-only).

  Also update the CHANGELOG bullet ("Use `io_flags: 6` for untrusted files").

## WARNING 2: zero-dimension index from `from_file` gets a 4-byte heap overwrite in `nif_reconstruct_batch`

- **Location:** `c_src/faiss_ex_nif.c:364` (`res->dim = faiss_Index_d(index);`), `:740`, `:771`
- **Evidence:**
  ```c
  if (batched) {
      for (ErlNifSInt64 i = 0; i < n; i++) {
          memcpy(result + i * dim, &UNWRITTEN_ROW_TAG, sizeof(UNWRITTEN_ROW_TAG));
  ```
  FAISS accepts `d == 0`: `read_index_header` uses `FAISS_CHECK_RANGE(idx.d, 0, (1 << 20) + 1)` (`impl/index_read.cpp:286`), and the range includes 0. For `IxF2`, the only follow-up check is `codes.size() == ntotal * code_size` (`:1756-1761`), which is `0 == 0`. Then:
  - `Index.reconstruct(idx, [0])` (or `reconstruct_binary`) reaches the NIF with `n = 1`.
  - `result_size = 1 * 0 * 4 = 0`, so `enif_alloc_binary(0, ...)` succeeds.
  - A single key counts as `contiguous`, and IndexFlat makes it `batched`.
  - The tagging loop then writes 4 bytes into a 0-byte binary. The `memcmp` at `:771` reads them back.

  This happens before the lock and before any `ntotal` check, so the file needs no vectors.
  - Nothing else blocks the path: `Index.new` rejects `dim <= 0` and `Clustering.new` rejects `d <= 0`, so `from_file` is the only entry point. `reconstruct_binary` never consults `dim` on the Elixir side.
  - **[INFERENCE]** Repro, not executed: `to_file` a `Flat` index with `dim=4` and no vectors, overwrite bytes 4..7 (the `int32 d` after the fourcc) with `<<0::32>>`, then `from_file` + `reconstruct(idx, [0])`.
- **Why it matters:** This is memory corruption in the BEAM heap allocator from a 40-odd-byte file. It needs no on-disk IVF and is unaffected by `io_flags`. It also breaks the `%Index{dim: pos_integer()}` typespec, which every Elixir caller relies on.
- **Fix:** Reject a non-positive dimension at the boundary, so the cached `res->dim` is always positive. In `nif_read_index`, after a successful `faiss_read_index_fname`:
  ```c
  if (faiss_Index_d(index) <= 0) {
      faiss_Index_free(index);
      return make_error_msg(env, "index dimension must be positive");
  }
  ```
  `clone` and `gpu_to_cpu` inherit `d` from an already-validated source. Optionally, also make `wrap_index_resource` refuse `d <= 0` as an invariant guard. Add the patched-header test above to `index_test.exs`.

## WARNING 3: documented `io_flags` value `8` crashes the VM on legitimate on-disk IVF files; arbitrary flag bits pass through

- **Location:** `lib/faiss_ex/index.ex:442` (table row), `:461` (no value validation)
- **Evidence:**
  > `| IO_FLAG_SKIP_IVF_DATA | 8 | Load an IVF index's structure without its inverted lists |`

  The two inverted-list types behave differently under flag 8:
  - **On-disk inverted lists (`ilod`):** `OnDiskInvertedListsIOHook::read` skips `do_mmap()` (`OnDiskInvertedLists.cpp:765-767`), so `ptr` stays `nullptr` (`:357`). `invlists` is non-null, which passes the `IndexIVF` null guards. `search`/`reconstruct` then dereference `nullptr + lists[i].offset` through `get_codes`/`get_ids` (`:391,399`), which is a SIGSEGV. This happens with a genuine index written by Python FAISS, with no crafting.
  - **In-memory `ilar` lists** (the only kind `to_file` writes): flag 8 computes `h2 = (io_flags & 0xffff0000) | "il"` and looks up a hook that does not exist (`impl/index_read.cpp:697-711`). Every `from_file(p, io_flags: 8)` of a faiss_ex-written IVF therefore returns an error. The documented effect never happens.

  `io_flags` is otherwise passed through unchecked. The high 16 bits select an `InvertedListsIOHook` (same lines), and `IO_FLAG_MMAP`/`IO_FLAG_MMAP_IFC` turn on reader modes the docs never mention.
- **Why it matters:** A documented option turns a trusted file into a VM crash.
- **Fix:** Pick one:
  - (a) Drop the `8` row and validate in `from_file/2`, raising `ArgumentError` otherwise:
    ```elixir
    unless is_integer(io_flags) and Bitwise.band(io_flags, Bitwise.bnot(0b110)) == 0
    ```
    This matches the existing "unknown options raise" contract.
  - (b) Keep `8` but document that the resulting index supports only `ntotal`/`dim`/`trained?`, that `search`/`reconstruct`/`reset` on an on-disk IVF loaded this way crash the VM, and that it fails outright for in-memory IVF.

  (a) is the safer cutover. Note it under "Changed" in the CHANGELOG.

## WARNING 4: `publish.yml` claims a reviewer gate that does not exist; first `v*` tag publishes ungated

- **Location:** `.github/workflows/publish.yml:17-18`
- **Evidence:**
  > `# HEX_API_KEY lives in this environment, which requires reviewer approval`
  > `environment: hex`

  The `hex` environment does not exist yet, and the token cannot create it (assignment note). GitHub auto-creates a referenced environment on the first job run *with no protection rules*. `secrets.HEX_API_KEY` resolves to the repository-level secret when the environment has none.
- **Why it matters:** Until the plan 2.8 user action is done, any `v*` tag push (any write-access account, from any commit) runs the publish job with the key and no approval. The comment tells reviewers the opposite.
- **Fix:** Complete these **before the next `v*` tag push**:
  1. Create the `hex` environment with required reviewers. Set *Deployment branches and tags* to "Selected", with the tag rule `v*`.
  2. Add `HEX_API_KEY` as an environment secret and **delete the repository-level secret**. Otherwise a future workflow without `environment:` can still read it.
  3. Restrict `v*` tag creation with a **repository ruleset**. Legacy tag-protection rules are deprecated.

  Until then, either avoid tagging, or reword the comment ("must require reviewer approval — see plan 2.8") so it states a requirement rather than a fact.

---

## SUGGESTION 1: `FAISS_GIT_SHA` is not validated, and an empty value silently drops the default pin

- **Location:** `Makefile:27-29`, `:127-129`, `:258`, `:260`
- **Evidence:**
  - Default pin: `FAISS_GIT_SHA ?= $(FAISS_DEFAULT_SHA)` … `FAISS_GIT_SHA ?=`
  - Recipe: `if [ "$$actual" != "$(FAISS_GIT_SHA)" ]` / `echo "… expects $(FAISS_GIT_SHA)."`
- **Why it matters:**
  - Unlike REV/REPO, SHA has no allow-list. It is expanded inside double-quoted shell, where `"`, `` ` `` and `$(` are live.
  - `?=` treats an environment variable that is *set but empty* as defined. So something like `FAISS_GIT_SHA: ${{ vars.X }}` with an unset variable disables verification of the default rev, and the only signal is a `$(warning)`. The documented opt-out is easy to trigger by accident.
- **Fix:**
  - Validate a non-empty SHA as exactly 40 hex characters: `$(words)` = 1, and `strip_chars` over `0-9a-f` is empty. Otherwise `$(error)`. This also turns an abbreviated SHA into a clear error instead of a permanent mismatch.
  - Optionally, require an explicit `FAISS_GIT_SHA=none` to opt out for the default rev.

## SUGGESTION 2: `FAISS_PREFIX` / `LIBOMP_PREFIX` / `MIX_APP_PATH` are unquoted in compiler flags

- **Location:** `Makefile:64` (`-L$(LIB_DIR)`), `:86` (`-I$(FAISS_PREFIX)/include`), `:163-165` (`"-I$(LIBOMP_PREFIX)/include"` inside cmake args), `:179/181` (`$(shell otool -L "$(SYSTEM_LIB_DIR)/libfaiss.dylib")`), `:196-197`
- **Why it matters:** These values come from the caller's environment, which is the same trust domain as `CC`/`PATH`, so this is not an exploitable boundary. But:
  - A project checked out under a path with a space (`~/My Projects/app`) breaks `-L$(LIB_DIR)` at link time.
  - A prefix with a space breaks `-I`.
  - A `"`/`$`/`` ` `` in a prefix escapes the `$(shell …)` quoting.

  The CHANGELOG line "recipe paths are quoted" is only half true: `cp`/`mkdir` paths are quoted, flags are not.
- **Fix:** Move `ALNUM`/`strip_chars` above `ifdef FAISS_PREFIX`; today they are only defined in the from-source branch. Then reject these paths with a clear `$(error)` when they contain whitespace or characters outside `[A-Za-z0-9._/~+@-]`. Quoting each flag is the alternative, but it is noisier.

## SUGGESTION 3: the REV/REPO allow-list checks the *expanded* value

- **Location:** `Makefile:35`, `:105`, `:114`
- **Why it matters:** Environment variables become recursively expanded make variables. A `FAISS_GIT_REV='$(shell …)'` therefore runs at `FAISS_SRC :=` (`:35`), before the check on `:105`. This is harmless under the threat model: whoever controls the environment already controls `MAKEFLAGS`/`CC`/`PATH`.
- **Fix:** No code change needed. Optionally, reword the comment at `:91-93` to say the allow-list is for robustness, not a security boundary. Alternatively, test `$(findstring $$,$(value FAISS_GIT_REV))` to reject the unexpanded text.

## SUGGESTION 4: `verify-faiss` checks `HEAD`, not the tree, despite the comment

- **Location:** `Makefile:254-263`
- **Evidence:**
  > `# … so a warm cache whose source tree no longer matches the pin also fails closed.`

  The check is only `git rev-parse HEAD`.
- **Why it matters:** Locally modified tracked files in `~/.cache/faiss_ex/faiss-v1.15.1` pass verification.
- **Fix:** Append `git -C "$(FAISS_SRC)" diff --quiet HEAD -- || { echo "ERROR: FAISS tree modified"; exit 1; }`. It checks tracked files only, so the untracked `build-*` directories are unaffected. Otherwise, reword the comment to "whose checkout is no longer the pinned commit".

## SUGGESTION 5: `cuda-syntax` container is pinned by mutable tag; Dependabot covers Actions only

- **Location:** `.github/workflows/ci.yml` (`container: nvidia/cuda:12.6.3-devel-ubuntu24.04`), `.github/dependabot.yml`
- **Why it matters:** This is low impact: the job has a read-only token, no secrets, and `persist-credentials: false`. Still, the image tag can be repointed. **[INFERENCE]** The `github-actions` ecosystem does not bump workflow `container:` images. Hex dependencies (`mix.lock`) also get no Dependabot updates.
- **Fix:**
  - Pin `nvidia/cuda:12.6.3-devel-ubuntu24.04@sha256:<digest>`.
  - Add a `package-ecosystem: mix` entry with the same `cooldown`.

## SUGGESTION 6: a tag on a commit not on master is publishable

- **Location:** `.github/workflows/publish.yml` ("Release gate" block)
- **Why it matters:** The gate re-runs format checks and tests on the tagged commit, but never checks that the commit was reviewed or merged. After WARNING 4 is fixed, the human approver is the only control.
- **Fix:** Use `fetch-depth: 0` on checkout, then add a step:
  ```sh
  git merge-base --is-ancestor "$GITHUB_SHA" origin/master || { echo "::error::tag not on master"; exit 1; }
  ```

## SUGGESTION 7: `nif_new_index` casts an unchecked int to `FaissMetricType`

- **Location:** `c_src/faiss_ex_nif.c:397`
- **Evidence:**
  > `faiss_index_factory(&index, dim, desc, (FaissMetricType)metric);`
- **Why it matters:** `Index.new/3` validates the metric. But every other NIF argument (`dim`, `k`, `n`, `device`, thread count) is also range-checked in C, and an out-of-range value converted to the C++ `faiss::MetricType` (no fixed underlying type) is unspecified/UB. `FaissEx.NIF` is `@moduledoc false`, so this is defense in depth only.
- **Fix:** Add `if (metric != METRIC_L2 && metric != METRIC_INNER_PRODUCT) return make_error_msg(env, "invalid metric");`. The values are already `_Static_assert`ed at `:30-31`.

## SUGGESTION 8 (correctness, not security): `brew --prefix libomp` defeats the no-OpenMP fallback

- **Location:** `Makefile:155-160`, `:190-193`
- **Why it matters:** **[INFERENCE]** `brew --prefix <formula>` prints the keg path even when the formula is not installed. `--installed` is the variant that fails instead. So `LIBOMP_PREFIX` is never empty on a Homebrew machine. Both the "libomp not found" warning and the `FAISS_EX_NO_OMP` fallback for `FAISS_PREFIX` builds are unreachable, and the link fails with `-lomp` instead.
- **Fix:** `$(shell brew --prefix --installed libomp 2>/dev/null)`.

## SUGGESTION 9 (doc drift): CLAUDE.md still says the pin lives in `mix.exs`

- **Location:** `CLAUDE.md` ("both are defined in BOTH `Makefile` and `mix.exs` (`make_env/0`); keep them in sync"); `Makefile:3-4` ("mix.exs passes nothing"); `mix.exs` (no `make_env`)
- **Fix:** Say the pin is single-sourced in the `Makefile` (`FAISS_DEFAULT_REV`/`FAISS_DEFAULT_SHA`).

---

## Clean areas

- **Action pins:** all three are verified against upstream tags via the GitHub API: `actions/checkout@3d3c42e5…` = v7.0.1, `erlef/setup-beam@54075bcc…` = v1.24.1, `actions/cache@55cc8345…` = v6.1.0. The FAISS `v1.15.1` tag resolves to `75c755c5a1fc…` = `FAISS_DEFAULT_SHA`.
- **Token scope:**
  - `ci.yml` sets top-level `contents: read`, and `publish.yml` sets job-level `contents: read`.
  - Every checkout uses `persist-credentials: false`.
  - There is no `pull_request_target`/`workflow_run`.
- **`${{ }}` script injection:** no expression appears inside any `run:`. Runner, arch and Makefile hash reach the shell through `env:`. Matrix and label values are static.
- **Tag validation:** the semver regex runs before `sed`. Its charset excludes `/`, `&`, `\` and newline, so the `sed` substitution cannot be injected.
- **Cache poisoning:**
  - PR caches are ref-scoped, and only master pushes write default-branch caches.
  - `publish.yml` only restores.
  - The Hex package ships source only (`mix.exs` `files:`), so a cached `libfaiss` cannot reach the release artifact.
- **REV/REPO allow-list:**
  - `strip_chars` logic is correct; commas and `%` in the value are treated literally.
  - `subst /,_` prevents path traversal in `FAISS_SRC`.
  - `--branch` consumes values with a leading `-`.
  - The `https://`-only rule blocks `ext::`/`file://` transports.
- **SHA verification fails closed:**
  - After a clone mismatch, the clone is deleted (`rm -rf`).
  - `verify-faiss` is phony and an order-only prerequisite of both cmake configure and the NIF link, so it runs on every build, before configure and the link.
- **Stamp `printf`:** the `'\''` escaping is correct, and `$ORIGIN` stays inside single quotes.
- **NIF arithmetic:**
  - `check_rows`/`rows_bytes` cover every n×dim and n×k product, and every `i * dim` offset is bounded by an overflow-checked allocation.
  - `keys_in_range` plus the `INT64_MAX - n` guard close the signed-overflow paths.
  - The batched reconstruct is gated on `ntotal` under the lock.
  - String arguments have NUL and length checks; `device < 0`, thread count < 0, and n/k bounds are all rejected.

## Tools to run manually

- `mix hex.audit`; `mix deps.audit` (requires `mix_audit`, not currently a dependency)
- WARNING 2 regression test: the patched-header repro described there.
