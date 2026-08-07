# Requirements Verifier Review — techdebt-lock-discipline

**REQ_SOURCE:** `plan techdebt-cleanup` (`.claude/plans/techdebt-cleanup/plan.md`, 22 tasks / 5 phases)
**Implementation:** `techdebt-lock-discipline` @ `fd1afd4` (3 commits: `4f210ff` C, `9118d27` Elixir, `fd1afd4` release)
**Method:** every verdict below is grounded in the working-tree source or in `git show master:<file>` for the "was removed" claims. Plan checkboxes and `**Result:**` notes were not treated as evidence.

## Verdict

PASS WITH WARNINGS

All 22 tasks are MET. One unauthorized side effect of the Phase 2 single-exit refactor widens two critical sections relative to `master` (WARNING). No task was skipped, no deferred item was implemented, and no Elixir/C change in the diff is unauthorized.

## Requirements Coverage

| Task | Verdict | Evidence (file:line) |
|---|---|---|
| 0.1 credo dep + baseline recorded, nothing fixed | MET | `mix.exs:68` (dep), `mix.lock` has 1 credo entry, `scratchpad.md:111-130` (verbatim baseline), baseline finding still live at `lib/faiss_ex/clustering.ex:23` (alias `NIF` before `Index`) |
| 1.1 `nif_add_to_index` hoist; drop master `:268`,`:272` | MET | `c_src/faiss_ex_nif.c:278-285` checks above lock at `:287`; crit section = `:289-290` only. Master `:268`/`:272` absent from HEAD unlock inventory |
| 1.2 `nif_add_with_ids_to_index` hoist both `safe_mul` groups; drop `:313`,`:318` | MET | `faiss_ex_nif.c:320-329` (both groups + combined check) above lock at `:331`; crit section `:333-336` |
| 1.3 `nif_train_index` hoist; drop `:430`,`:434` | MET | `faiss_ex_nif.c:436-443` above lock at `:445`; crit section `:447-448` |
| 1.4 `nif_reconstruct_batch` hoist sizes **and** `enif_alloc_binary`; drop `:513`,`:519`; leave the post-unlock release path | MET | sizes `faiss_ex_nif.c:517-521`, **`enif_alloc_binary` hoisted** `:524`, lock at `:531`, unlock `:540`, FAISS-error `enif_release_binary` still after the unlock at `:542-545` |
| 1.5 `nif_compute_residuals` hoist sizes **and** `enif_alloc_binary`; drop `:575`,`:580`,`:586` | MET | sizes `faiss_ex_nif.c:575-584`, **`enif_alloc_binary` hoisted** `:587`, lock `:595`, unlock `:597` |
| 1.6 zero error-path unlocks in those five; slow suite | MET (suite not re-run) | HEAD has 16 `enif_rwlock_*unlock` total, all column-4 success-path or label body (`faiss_ex_nif.c:290,336,448,540,597`); master had 34 with 18 in error branches |
| 2.1 `nif_train_clustering` single `unlock:`, **mirror-order** release | MET | acquire index→clustering `faiss_ex_nif.c:871-872`; release clustering `:889` → index `:890` under label `:888` — exact mirror; both error branches `goto unlock` at `:877`,`:881` |
| 2.2 `nif_get_clustering_centroids` 3 error sites → `goto`; **`memcpy` must NOT leave the rlock** | MET | `goto unlock` at `:918`,`:922`,`:926`; label `:935`, runlock `:936`; `memcpy` at `:929` is *inside* rlock (`:910`–`:936`) — non-goal honoured, no "helpful" hoist |
| 2.3 extend the locking-convention block comment | MET | `faiss_ex_nif.c:37-47` — validate-before-lock + single-`unlock:` rule + mirror-order + the "never hand-write an unlock per branch" rationale |
| 2.4 grep audit: every unlock is success-path or label body | MET (suite not re-run) | full HEAD inventory `:246,290,336,404,448,466,540,597,624,686,699,727,764,889,890,936` — `889/890` and `936` are the two label bodies; the other 14 immediately follow their `faiss_*` call |
| 3.1 exactly the 10 `case`→`with` sites; `from_file` hard match kept | MET | 8 in `index.ex:235,258,279,302,341,368,410,432` + 2 in `clustering.ex:60,89` = **10**; zero `{:error, _} = err` clauses remain in `lib/`; pre-existing `with` at `index.ex:165` / `clustering.ex:78` untouched; **hard match preserved** at `index.ex:371` with rationale comment `:369-370` |
| 3.2 `encode_ids!` + `ArgumentError` naming both counts | MET | `index.ex:334` (`{n_keys, keys_bin} = Shared.encode_ids!`), raise at `:336-339` naming `n_keys` and `n` |
| 3.3 `gpu_to_cpu(%{device: :host})` → `{:error, "index already on host"}` | MET | `index.ex:443-445`; doc updated `:428` |
| 3.4 two new tests: untagged GPU mirror + residual raises | MET | (a) `test/faiss_ex/index_test.exs:299-302`, inside `describe "GPU error paths (non-CUDA build)"` (`:286`), **untagged** — `@tag :no_cuda` at `:287` binds only the test at `:288`. (b) `index_test.exs:232-241` (count mismatch) and `:243-250` (non-integer key), both `assert_raise ArgumentError` with exact messages |
| 4.1 `dup_cstr` at exactly 3 sites; callers keep guard/OOM/free | MET | definition `faiss_ex_nif.c:167-173`; calls `:216`, `:619`, `:647` (3 — "exactly three strikes"); each caller keeps `contains_nul` (`:214,:617,:645`), `"out of memory"` (`:217,:620,:648`) and `enif_free` (`:221,:625,:652`) |
| 4.2 delete `make_ok_atom`, use `atom_ok` | MET | `git grep make_ok_atom -- c_src` → **0 lines** at HEAD vs **10** at master (9 calls + definition). Scratchpad's "8 + definition converted in 4.2, 1 absorbed by 2.1" is consistent: `nif_train_clustering:886` now yields `atom_ok` inline |
| 4.3 both helpers out of `lib/`, local test helpers, no `Shared` regression | MET | `lib/faiss_ex/shared.ex` diff removes only those two `@doc false` functions; `grep floats_to_binary\|int64s_to_binary lib/` → none; helpers now `shared_test.exs:82,86` and `index_test.exs:553,557` (`encode/1`, `encode_ids/1`); reference-impl assertion kept at `shared_test.exs:67`; no `test/support`, no `elixirc_paths` added |
| 4.4 delete `Index.to_binary/1`, pass through at 3 sites | MET | function gone; `grep -rn to_binary lib test c_src` → only `IO.iodata_to_binary` at `shared.ex:65`. Pass-through at `index.ex:165` (`description`), `:351` (`to_file` path), `:368` (`from_file` path) |
| 4.5 document `Index.new/3` error discipline (docs only) | MET | `index.ex:128-136` `## Errors` section; no behavioral change to `new/3` in the diff beyond the 4.4 pass-through |
| 4.6 CHANGELOG + version bump + **both** CLAUDE.md edits | MET | `CHANGELOG.md:3-59` (v0.3.1, incl. the charlist-narrowing and `dynamic()` call-outs); `mix.exs:4` `@version "0.3.1"`; CLAUDE.md **Lock unwinding (C layer)** paragraph added after the locking-convention paragraph, **and** the `shared.ex` bullet now ends "…now live as private helpers in the test files — do not re-add them here" |
| 4.7 final gate | MET (self-verified in part) | `mix format --check-formatted` → clean (run here). Suite/credo/docs/hex.build not re-run per instruction; relied on the orchestrator's clean `mix clean --force` rebuild (79 passed/1 excluded, credo at the 1-finding baseline) |

**MET: 22, PARTIAL: 0, UNMET: 0, UNCLEAR: 0**

### Deferred items — genuinely not actioned

- **P3-f** `"invalid index reference"` ×6 → still 6 literal copies (`faiss_ex_nif.c:240,461,672,682,695,758`), and `grep "#define"` in the file returns nothing: no `GET_INDEX_OR_ERROR` macro was sneaked in. Correctly deferred.
- **P4-b** test setup repetition → 61 `= Index.new(` call sites across `test/`, no shared setup helper, no `test/support/`, no `elixirc_paths` in `mix.exs`. Correctly deferred. (The two new `defp` helpers in `index_test.exs` are task 4.3's fixture relocation, not a setup factory.)

## Findings

### [WARNING] Phase 2's single-exit shape moved term/error construction *into* the critical sections
**Location:** `c_src/faiss_ex_nif.c:886` (`nif_train_clustering`), `c_src/faiss_ex_nif.c:929-933` (`nif_get_clustering_centroids`)
**Issue:** No task authorized widening a critical section, and Phase 1's stated principle is the opposite — "the critical section shrinks to the bare `faiss_*` call plus its unlock" (plan.md:75), with task 1.3 naming *error formatting under the write lock* as part of the defect being fixed. The `result = ret == 0 ? atom_ok : make_faiss_error(...)` at `:886` now runs `faiss_get_last_error()` + `enif_make_new_binary` **while holding both write locks**; on master (`:885-889`) both locks were released *before* `make_faiss_error`. Likewise, `enif_make_binary` + `enif_make_tuple3` + `make_ok` at `:930-933` now run under the read lock; master did them after `runlock`. Neither is a bug — `faiss_get_last_error` is thread-local (`:140-142`) and the centroids `memcpy` legitimately stays inside — but both are unrequested regressions in lock hold time, in the two functions this branch exists to harden.
**Why it matters:** `nif_train_clustering` holds the quantizer index's write lock, so every concurrent operation on that index waits through the error-message allocation. It also makes the branch's own convention comment (`:39-40`, "the critical section then holds nothing but the `faiss_*` call and its one unlock") false for the two functions it was written to describe — exactly the doc/code drift task 2.3 was meant to prevent.
**Fix:** keep the `unlock:` label, but carry the FAISS return code out instead of the term: store `ret` (already declared at `:868`), set `result = atom_ok` on success, `goto unlock`, and build the error term *after* both unlocks (`if (ret != 0) return make_faiss_error(...)`). Same pattern for centroids: `memcpy` under the lock, then `runlock`, then construct the tuple from the already-owned `bin`.

### [SUGGESTION] The release commit swept in 31 unrelated `.claude/**` documents
**Location:** commit `fd1afd4` (`.claude/audit/**`, `.claude/plans/faiss-upgrade/**`, `.claude/plans/perf-analysis/**`, `.claude/plans/perf-fast-path/**`, 3 solution docs)
**Issue:** No task authorized these. `4f210ff` and `9118d27` are clean and phase-scoped; the v0.3.1 commit mixes a 56-line CHANGELOG + version bump with ~3.4k lines of previously-untracked planning artifacts from earlier, unrelated efforts.
**Why it matters:** Cosmetic only — `mix.exs:44-47` ships an explicit `files:` allowlist (`lib c_src Makefile mix.exs README.md LICENSE CHANGELOG.md`), so nothing leaks into the hex package. But it makes the release commit unreviewable by diff and buries the two lines that actually cut the release.
**Fix:** split the doc import into its own commit before tagging; leave `fd1afd4` as CHANGELOG + `@version` + `mix.exs`/`mix.lock`.

### [SUGGESTION] `nif_train_clustering`'s `ret` is now function-scoped and unset on the error paths
**Location:** `c_src/faiss_ex_nif.c:868`
**Issue:** Hoisting `int d, ret;` above the lock (required, since a `goto` may not jump over an initializer) leaves `ret` uninitialized whenever either validation branch takes `goto unlock`. It is never read there, so this is well-defined today, but the guarantee is now positional rather than structural.
**Why it matters:** A future edit that adds a post-label `if (ret != 0)` reads an indeterminate value, and `-Wall -Wextra` will not flag it because the success path does assign it.
**Fix:** `int d, ret = 0;` — free, and it makes the invariant local. (Adopting the WARNING's fix makes this mandatory rather than optional.)

## Risks — which materialized

1. **Error-ordering change under contention (Phase 1)** — **materialized as designed.** Size-mismatch errors now return without queueing on the lock (`faiss_ex_nif.c:283-285` vs master `:268-274`). Returned terms are byte-identical and error *precedence* is preserved in all five functions (arg-inspect → `n < 0` → `n == 0` → overflow → size), so no test could observe it. Benign.
2. **`goto` is a new pattern in this file** — **materialized, mitigated as planned.** Two labels exist and task 2.3 documents the rule at `:37-47` plus CLAUDE.md, so it reads as intentional.
3. **`with` swallowing a third return shape** — **did not materialize.** Every C return path in the file is `atom_ok`, `make_ok`, `make_error_msg`, or `make_faiss_error`, i.e. `:ok | {:ok, _} | {:error, binary}`; `lib/faiss_ex/nif.ex:18-41` adds no others.
4. **Charlist narrowing (4.4)** — **materialized, disclosed.** `CHANGELOG.md:35-38` names the three affected functions and gives the `List.to_string/1` workaround. No stale charlist reference survives in `lib/` or `README.md`.
5. **A wall of pre-existing credo findings** — **did not materialize.** Baseline is the single alias-order finding, still unfixed at `clustering.ex:23` as task 0.1 required.

**Risk the plan did not list, which did materialize:** the `with` conversion of `cpu_to_gpu/2` widened `Index.new/3`'s inferred return to `dynamic()`, breaking struct-update type checking downstream. Confirmed present and handled: `index_test.exs:294` now matches `{:ok, %Index{} = index}`, and `CHANGELOG.md:39-45` warns downstream callers. Recorded in `scratchpad.md:86-105`. This is disclosed scope, not creep — but note the new test at `index_test.exs:299-302` deliberately does *not* need the annotation (it passes the struct to a function rather than updating it), so the fix is minimal rather than defensive.

## Pre-existing (untouched, informational)

- `lib/faiss_ex/clustering.ex:23` — alias `FaissEx.NIF` not alphabetically ordered in its group; the one credo `--strict` finding, deliberately unfixed per task 0.1. PRE-EXISTING.
