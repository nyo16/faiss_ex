# Review — techdebt-lock-discipline (v0.3.1)

**Date:** 2026-08-07
**Branch:** `techdebt-lock-discipline` (3 commits on `master`)
**Diff reviewed:** `git diff master -- c_src lib test` — 266 insertions, 191 deletions across 6 code files
**Panel:** 5 agents — C lock audit, Elixir reviewer, testing reviewer, security analyzer, requirements verifier

## Verdict

**PASS WITH WARNINGS**

Zero blockers. All 22 plan requirements MET with independent source citations. The
load-bearing change — eliminating 16 hand-unwound lock paths — is correct and
machine-verified. Four warnings are worth addressing before merge; none block.

Note: three specialist agents (`elixir-reviewer`, `testing-reviewer`,
`requirements-verifier`) failed to launch on the first attempt with
`404 model: claude-sonnet-4-0` and were re-run on the generic worker with
identical briefs. Infra failure, not a review signal.

## Requirements Coverage

**Source:** `.claude/plans/techdebt-cleanup/plan.md` — `MET: 22, PARTIAL: 0, UNMET: 0, UNCLEAR: 0`

Verified independently of the plan's own checkboxes. Trap checks the verifier
was asked to police, all confirmed:

| Check | Result |
|---|---|
| 1.1–1.5 named unlock lines removed | All 11 gone; master 34 unlocks → HEAD 16 |
| 1.4 / 1.5 `enif_alloc_binary` genuinely hoisted | Yes — `:524`, `:587`, not just size checks |
| 2.1 mirror-order release | acquire idx `:871` → clust `:872`; release clust `:889` → idx `:890` |
| 2.2 centroids `memcpy` stayed under rlock | Yes, `:929` inside `:910`–`:936`. Non-goal honoured |
| 3.1 exactly 10 conversions | 10 found; zero `{:error, _} = err` clauses left in `lib/` |
| 3.1 `from_file` hard match preserved | Yes, `index.ex:371` with rationale comment |
| 4.2 zero `make_ok_atom` references | `git grep` → 0 at HEAD vs 10 at master |
| 4.3 both helpers left `lib/` | Yes; local `defp`s in both test files |
| 4.6 CHANGELOG + version + **both** CLAUDE.md edits | All three landed |
| Deferred P3-f NOT actioned | 6 literal copies remain, no `#define` sneaked in |
| Deferred P4-b NOT actioned | 61 `= Index.new(` sites, no shared helper, no `test/support/` |

**Scope creep:** none in `c_src`, `lib`, or `test` — every hunk maps to a task.

## Findings

### [WARNING — HIGH CONFIDENCE] The new `## Errors` block documents a convention the module does not follow

Flagged independently by the Elixir reviewer **and** the testing reviewer.

**Location:** `lib/faiss_ex/index.ex:128-136`

**Issue:** Task 4.5 added a doc block asserting a module-wide rule — "developer
errors raise, FAISS and data errors return error tuples". The same commit
violates it:

| call | 2 vectors, 1 id | contract |
|---|---|---|
| `Index.compute_residuals/3` | raises `ArgumentError` (new) | developer error |
| `Index.add_with_ids/3` | `{:error, "binary size mismatch"}` | runtime error |

`add_with_ids/3` already computes the id count at `index.ex:218` and discards it
as `_n_ids`, letting the C size check produce a tuple. Separately, a charlist
`description` — unambiguously a developer error — returns
`{:error, "invalid arguments"}` from `c_src/faiss_ex_nif.c:210`.

The block's stated rationale is also wrong: it claims `:metric` is "a
compile-time constant in the caller", but `metric` is read from `opts` at
`index.ex:146` exactly like `dim` and `description`.

**Why it matters:** Both contracts are now pinned by green tests
(`index_test.exs:232` and `:519`), so the inconsistency is sticky — whoever
later aligns them must change a passing test, which reads like a regression.

**Fix:** Drop the generalizing sentence and the compile-time-constant rationale,
scoping the block to `new/3`. Do **not** revert `compute_residuals/3` — raising
there is correct and matches `reconstruct/2`. Symmetric `add_with_ids/3` raising
is a 0.4.0 breaking follow-up, not this branch.

### [WARNING] Phase 2 widened the very critical sections the branch set out to shrink

**Location:** `c_src/faiss_ex_nif.c:886`, `:929-933`

**Issue:** The single-exit rewrite moved term construction inside the locks.
`nif_train_clustering` now runs `make_faiss_error` — `faiss_get_last_error()`
plus `enif_make_new_binary` — while holding **both** write locks; master
released both first. `nif_get_clustering_centroids` now runs
`enif_make_binary`/`enif_make_tuple3`/`make_ok` under the rlock; master did them
after `runlock`.

**Why it matters:** Not a correctness bug — `faiss_get_last_error` is
thread-local (`:140-142`). But it contradicts the branch's own thesis twice
over: task 1.3 explicitly named "error formatting under the write lock" as part
of the defect being fixed, and the new convention comment this branch wrote at
`:37-47` states the critical section should hold "nothing but the `faiss_*` call
and its one unlock". The two functions the comment describes are the two that
violate it.

**Fix:** Carry `ret` past the `unlock:` label and build the term after both
unlocks. Keeps single-exit, restores the narrow critical section.

### [WARNING] `compute_residuals` happy path cannot detect a key-encoding bug

**Location:** `test/faiss_ex/index_test.exs:221-230`

**Issue:** The only success-path test uses key `0` on a one-vector index and
asserts every residual component is `≈ 0.0`. Key `0` is byte-order invariant
(eight zero bytes either way), so the assertion still passes if the function
byte-swaps keys, ignores `keys` entirely, or returns a correctly-sized zero
buffer.

**Why it matters:** This diff switched that function's key encoder from
`int64s_to_binary/1` to `encode_ids!/1` (`lib/faiss_ex/index.ex:334`) — precisely
the change a key-blind test cannot police. `reconstruct/2` is covered
(`index_test.exs:611` uses `[0, 1]`), `compute_residuals` is not.

**Fix:** Add a nonzero-residual case (agent verified live against the built NIF:
two vectors, key `1` → `[[-3.0, -3.0, -3.0]]`). Keep the identity case.

### [WARNING] `defp` inside `describe` is module-scoped — the branch ships two opposite conventions

**Location:** `test/faiss_ex/index_test.exs:551-559` vs `test/faiss_ex/shared_test.exs:78-88`

**Issue:** `alias` and `defp` sit in the same visual block but scope oppositely:
`describe/2` expands to a plain `quote`, so `alias` is block-scoped while `defp`
compiles into the module and is callable from every other `describe`. Verified
empirically with a standalone ExUnit probe. `shared_test.exs` in this same diff
places the equivalent helpers correctly at module level.

**Why it matters:** A cleanup branch shipping two opposite conventions for the
identical construct, in two files, is the exact debt this branch exists to
remove. A second `defp encode/1` added under a later `describe` would be a
redefinition, not a shadow.

**Fix:** Move `encode/1` and `encode_ids/1` to module level, mirroring
`shared_test.exs`. Leave the `alias` where it is.

### [WARNING — PRE-EXISTING] Caller-controlled `int64` keys reach an overflowing FAISS bounds check

**Location:** `c_src/faiss_ex_nif.c:505-514` (pre-scan), FAISS `IndexFlatCodes.cpp:79`

**Not introduced by this branch** — the contiguity pre-scan is unchanged from
master. Reported because it is a live memory-safety hole in reachable code.

FAISS guards `reconstruct_n` with `i0 >= 0 && i0 + ni <= ntotal`. With
`i0 = INT64_MAX`, `i0 + ni` signed-overflows to a negative value and the guard
passes.

**Reproduced empirically on this branch** (2-vector `Flat` index):

```
Index.reconstruct(idx, [9_223_372_036_854_775_807])
#=> {:ok, [[2.8146685221692822e23, 3662495350784.0, 9.388699710976274e-44]]}
```

Heap contents outside the index, returned as float data to safe Elixir — an
out-of-bounds read and an information-disclosure primitive. A larger contiguous
range starting near `INT64_MAX` did not segfault in testing, but the read is
undefined behaviour either way.

**Fix:** Reject `keys[i] < 0` and `keys[0] > INT64_MAX - n` in the existing
pre-scan at `:505-514`. Cheap — the loop already walks every key. Recommend a
separate commit, since it is a real bug fix rather than debt cleanup.

### [SUGGESTION] The new `unlock:` invariant is comment-enforced, not build-enforced

**Location:** `Makefile:48`

The branch's thesis is that hand-written unlocks are unsafe because nothing in
the build can catch a missed one. The replacement pattern has its own unenforced
precondition: every variable read at or after the label must be declared above
the first `goto`, and `result` must be assigned on every path. Adding
`-Wjump-misses-init` (gcc) / `-Wconditional-uninitialized` (clang) makes the
compiler check it. Both verified clean on this branch already — zero cost today.

### [SUGGESTION] Grouped minor items

1. `c_src/faiss_ex_nif.c:868` — `ret` is indeterminate on both `goto` paths.
   Never read today and `-Wall -Wextra` will not flag a future post-label read.
   Initialize `ret = 0`.
2. `lib/faiss_ex/shared.ex:4-12` — `binary_to_floats/1` and `binary_to_int64s/1`
   are now the only public functions without `@spec`. Cosmetic.
3. `test/faiss_ex/shared_test.exs:65` — add `±INT64` boundary values to the
   `ids` fixture so the differential oracle also covers the s64 edge.
4. Commit `fd1afd4` swept in ~3.4k lines of previously-untracked `.claude/**`
   planning artifacts alongside the CHANGELOG and version bump. Harmless to
   consumers (`mix.exs:44-47` is an explicit allowlist) but makes the release
   commit unreviewable.

## Verified, no finding

- **C lock audit, 7/7 checks PASS.** 16 acquisitions / 16 releases pairing 1:1
  with matching read/write flavour; no return inside any critical section; no
  double-unlock; mirror-order release confirmed; no `goto` crosses a
  declaration; `res->dim` written exactly once at `:192` before publication at
  `:193`, so the hoisting-safety argument holds with no TOCTOU; binary ownership
  exact at every new boundary; `dup_cstr` guarded and freed on all paths with no
  new overflow surface; `atom_ok` initialized in `on_load` before any NIF can
  dispatch. Machine-verified: `gcc-16 -Wjump-misses-init -Wmaybe-uninitialized
  -Wclobbered`, `clang -Wconditional-uninitialized -Wunreachable-code`, and
  `clang --analyze` all clean.
- **All 10 `with` conversions behavior-preserving**, checked against actual C
  return shapes rather than the stubs.
- **Relocated test encoders are character-identical** to the deleted originals —
  same endianness, same widths.
- **Untagged `gpu_to_cpu` test placement is correct** and would be a bug if
  tagged `:no_cuda`; confirmed by reproducing the `USE_CUDA=true` exclusion set.
- **All three new tests are mutation-sensitive** — none pass trivially.
- **The `encode_ids!` reference-oracle assertion did not become a tautology.**
  `encode_ids!/1` is a tail-recursive accumulator; the helper is a `for/into`
  comprehension — structurally distinct, so it remains a genuine differential
  oracle. Its unique value is catching symmetric encoder/decoder drift.
- **Elixir-side validation is strictly stricter than master** — no check was
  removed; a forged struct `dim` cannot bypass the C re-validation.

## Risks from the plan

| Risk | Outcome |
|---|---|
| 1. Error ordering under contention | Materialized as designed, benign; precedence preserved |
| 2. `goto` new to this file | Mitigated by task 2.3 doc + CLAUDE.md |
| 3. `with` swallowing a third shape | Did not materialize — no NIF returns one |
| 4. Charlist narrowing | Materialized, disclosed in CHANGELOG; no stale refs |
| 5. Credo wall | Did not materialize — baseline is 1 finding |
| *(unlisted)* `Index.new/3` inferred type widened to `dynamic()` | Materialized; handled and disclosed |
