# Plan: Tech debt — C lock discipline, Elixir API consistency, cleanup

**Status:** COMPLETE (22/22 tasks)
**Depth:** standard
**Created:** 2026-08-07
**Input:** `/phx:techdebt` report (this session). All 13 findings covered:
11 as tasks, 2 explicitly deferred (see Deferred section). No research
agents spawned — the report IS the research (Iron Law #7).
**Target version:** v0.3.1 (no additive API; one raise→error-tuple fix)

## Goal

Eliminate the manual lock-unwind hazard in the C NIF and the drifted
conventions in the Elixir layer. The load-bearing item is P1: **16 early
returns each hand-retyping their own `enif_rwlock_*unlock`**. A single
forgotten unlock does not crash — it wedges that index forever and blocks
a dirty scheduler thread on every subsequent call. Nothing in the build or
test suite can catch an omission.

Everything else is consistency debt: the correct pattern already exists in
this codebase and 10–15 sites drifted from it.

## Verified baseline (do not re-derive)

- `74 passed, 3 excluded` · `mix format --check-formatted` clean
- C compiles warning-free under `-Wall -Wextra`
- `mix credo` **not installed** (`** (Mix) The task "credo" could not be found`)
- No `elixirc_paths` in mix.exs; no `test/support/` directory
- Probe results (2026-08-07):
  - `gpu_to_cpu` on host index → `{:raised, "no function clause matching in FaissEx.Index.gpu_to_cpu/1"}`
  - `int64s_to_binary([1, :a])` → `"construction of binary failed: segment 1 of type 'integer': ..."`
  - `encode_ids!([1, :a])` → `"id 1 is not an integer: :a"`
  - `compute_residuals` key/vector count mismatch → `{:error, "binary size mismatch"}` (C-level, **not** pinned by any test)
  - `Index.new(3, "Flat", metric: :cosine)` → raises `ArgumentError`

## Non-goals (verified NOT waste by the perf-fast-path plan — do not "fix")

Carried forward from `.claude/plans/perf-fast-path/plan.md:24-33`:

- Centroids `memcpy` under rlock — the pointer belongs to the locked
  object. **Task 2.2 touches only the unwind paths, never the memcpy.**
- `reconstruct` contiguity pre-scan (already outside the lock, 500-506)
- `ntotal`/`is_trained` stay dirty + locked (mutable state, unlike `dim`)
- `enif_alloc_binary` → `enif_make_binary` ownership-transfer pattern

The `res->dim` immutability invariant (struct comment `faiss_ex_nif.c:51-55`)
was established and verified against FAISS v1.14.3 source in that plan's
task 2.1. **Phase 1 relies on it. Do not re-litigate.**

---

## Phase 0 — Tooling gap [tooling]

- [x] 0.1 **[P4-a]** Add `{:credo, "~> 1.7", only: [:dev, :test], runtime: false}`
      to `mix.exs:63-69`, `mix deps.get`, run `mix credo --strict`, and record
      the **baseline** finding list verbatim in `scratchpad.md`. Do NOT fix
      pre-existing findings — that is separate triage, out of scope here. The
      point is that later phases can assert *no NEW findings*.

**Verify:** `mix credo --strict` runs and its output is captured in scratchpad.
**Result:** baseline is exactly **1** finding (alias order, `clustering.ex:23`),
recorded verbatim in `scratchpad.md`. Not fixed.

## Phase 1 — Validate before locking [c] (P1-2 — DO FIRST)

Five NIFs take the lock and *then* do pure arithmetic on `res->dim`, which
is the cached immutable field. `nif_search_index` already does it correctly
and says so at `:360-362`. Hoisting the checks above the lock in all five
**dissolves 11 of P1-1's 16 unwind paths outright**, which is why this
phase precedes Phase 2.

In each task: move `int dim = res->dim;`, the `safe_mul` calls, the byte-size
comparison, and (where present) the result `enif_alloc_binary` to *above* the
lock acquisition. The error branches then need no unlock at all. The critical
section shrinks to the bare `faiss_*` call plus its unlock.

- [x] 1.1 **[P1-2]** `nif_add_to_index` (`c_src/faiss_ex_nif.c:263-274`) —
      hoist above the `enif_rwlock_rwlock` at `:263`. Removes the unlocks at
      `:268` and `:272`. Write-lock critical section becomes `faiss_Index_add`
      only.
- [x] 1.2 **[P1-2]** `nif_add_with_ids_to_index` (`:307-320`) — hoist both
      `safe_mul` groups (`expected_data`, `expected_ids`) and the combined
      size check above `:307`. Removes unlocks at `:313`, `:318`.
- [x] 1.3 **[P1-2]** `nif_train_index` (`:425-436`) — same shape as 1.1.
      Removes unlocks at `:430`, `:434`. This is the biggest win: `train` can
      hold the write lock for seconds, and it currently holds it across
      `safe_mul` and error formatting.
- [x] 1.4 **[P1-2]** `nif_reconstruct_batch` (`:508-521`) — hoist the dim /
      `safe_mul` / `result_size` computation **and** the
      `enif_alloc_binary(result_size, &result_bin)` above the rlock at `:508`
      (mirrors what perf-fast-path 2.2 did for `search`). Removes unlocks at
      `:513`, `:519`. The FAISS-error release path at `:536-539` is already
      after the unlock — leave it.
- [x] 1.5 **[P1-2]** `nif_compute_residuals` (`:569-588`) — hoist dim /
      `safe_mul` / size checks **and** `enif_alloc_binary(expected_data, ...)`
      above the rlock at `:569`. Removes unlocks at `:575`, `:580`, `:586`.
- [x] 1.6 [test] Confirm zero error-path unlocks remain in these five
      functions (every `*unlock` should be immediately after the `faiss_*`
      call). Run `mix test --include slow` — the stress tests in
      `concurrency_test.exs` and `concurrency_smoke_test.exs` directly
      exercise the changed lock scopes.

**Verify:** `mix compile --warnings-as-errors`, `mix test --include slow`.
**Result:** compiles warning-free under `-Wall -Wextra`; `76 passed, 1 excluded`
(74 baseline + 2 slow). All 11 error-path unlocks gone; each of the five
functions now holds its lock across the `faiss_*` call only.

## Phase 2 — Single-exit for the genuinely-locked functions [c] (P1-1 remainder)

After Phase 1, five unwind sites remain — in the two functions whose
validation reads *locked* state and therefore cannot be hoisted.

- [x] 2.1 **[P1-1]** `nif_train_clustering` (`:867-886`) — validation calls
      `faiss_Clustering_d(clust_res->clustering)` at `:870`, which reads the
      locked object, so it must stay inside. Introduce a single `unlock:`
      label; both error branches `goto unlock`. This collapses 4 hand-typed
      unlock statements (2 sites × 2 locks) into one exit. **Preserve the
      documented lock order** — acquire index→clustering (`:865-867`), release
      in the mirror order at the label.
- [x] 2.2 **[P1-1]** `nif_get_clustering_centroids` (`:902-928`) — same
      `goto unlock` treatment for the 3 error sites at `:912`, `:918`, `:924`.
      **Non-goal:** do NOT move the `memcpy` at `:927` out of the read lock —
      the `centroids` pointer belongs to the locked object (verified in
      perf-fast-path).
- [x] 2.3 Extend the locking-convention block comment
      (`c_src/faiss_ex_nif.c:25-40`) with the rule this phase establishes:
      *validate before acquiring the lock wherever the inputs allow it; where
      validation must read locked state, use the single `unlock:` exit —
      never hand-unwind per branch.* This is the durable fix; the code change
      alone will drift back without it.
- [x] 2.4 [test] Full suite incl. slow. Grep the file to confirm every
      remaining `*unlock` is either the single success-path release or the
      `unlock:` label body.

**Verify:** `mix compile --warnings-as-errors`, `mix test --include slow`.
**Result:** both functions are single-exit (`unlock:` label; train_clustering
releases clustering→index, mirroring the index→clustering acquisition).
Grep audit: every `enif_rwlock_*unlock` in the file is now either the
success-path release immediately after the `faiss_*` call, or the `unlock:`
label body. `memcpy` at centroids stayed inside the read lock as required.

## Phase 3 — Elixir API consistency [elixir]

- [x] 3.1 **[P2-1]** Convert the 10 passthrough `case` blocks to `with`.
      Sites — `index.ex`: `:227-235` (search), `:254-260` (search_binary),
      `:279-285` (clone), `:306-312` (reconstruct), `:341-347`
      (compute_residuals), `:372-386` (from_file), `:416-428` (cpu_to_gpu),
      `:440-452` (gpu_to_cpu); `clustering.ex`: `:60-66` (new), `:93-99`
      (get_centroids). A `with` without `else` returns the non-matching value
      unchanged — exactly what `{:error, _} = err -> err` did, so this is
      behavior-preserving. Matches the idiom already used at
      `clustering.ex:82` and `index.ex:157`.
      **Care — `from_file` (`:372-386`)** contains a nested hard match
      `{:ok, dim} = NIF.nif_get_index_dim(ref)` at `:374`. Keep it a hard
      match. Folding it into the `with` would convert a MatchError (invariant
      violation: we just opened this index) into a silent `{:error, _}`
      return. That is a behavior change, not a refactor.
- [x] 3.2 **[P2-2]** `compute_residuals/3` (`index.ex:336-348`): replace
      `Shared.int64s_to_binary(keys)` at `:339` with
      `{n_keys, keys_bin} = Shared.encode_ids!(keys)` and raise
      `ArgumentError` when `n_keys != n`, naming both counts. Today a
      non-integer key produces an opaque bitstring-construction message and a
      count mismatch degrades to the C-level `{:error, "binary size mismatch"}`
      (both probe-verified). Aligns with `reconstruct/2:304`, which already
      uses `encode_ids!`. Raising (not `{:error, _}`) matches the established
      convention: developer errors raise, FAISS/data errors return tuples
      (ragged lists, non-divisible binaries — see perf-fast-path
      `scratchpad.md:20-21`).
- [x] 3.3 **[P2-3]** Add the missing symmetric clause to `Index.gpu_to_cpu/1`
      (`index.ex:439`): `def gpu_to_cpu(%__MODULE__{device: :host}), do:
      {:error, "index already on host"}`. Today it raises FunctionClauseError
      while its counterpart `cpu_to_gpu/2` returns a clean error tuple
      (`:431-433`) — and `@spec` already advertises `{:ok, t()} | error()`.
- [x] 3.4 [test] Two new tests.
      (a) For 3.3: mirror of `index_test.exs:273-277` in the same
      `describe "GPU error paths (non-CUDA build)"` block —
      **untagged**, like its neighbour, because it fakes the struct and never
      enters the NIF (`@tag :no_cuda` at `:267` applies only to the test at
      `:268`).
      (b) For 3.2: key/vector count mismatch and non-integer key both raise
      `ArgumentError` with the expected message. No existing test pins the old
      `{:error, "binary size mismatch"}` behavior for this function (verified:
      the two assertions at `:497` and `:636` are `add_with_ids`), so nothing
      needs updating — but the new behavior is currently uncovered.

**Verify:** `mix format`, `mix compile --warnings-as-errors`, `mix test`,
`mix credo --strict` (no new findings vs the 0.1 baseline).
**Result:** all 10 converted; `from_file`'s nested hard match kept as a hard
match. `77 passed`, credo still at the 1-finding baseline.
**Unplanned side effect (see scratchpad):** the `with` conversion of
`cpu_to_gpu/2` widened `Index.new/3`'s *inferred* return type to `dynamic()`,
which broke the type checker's struct-update proof at `index_test.exs:294`.
Fixed by matching `{:ok, %Index{} = index}` there; noted in the CHANGELOG
because downstream struct-update callers hit the same warning.

## Phase 4 — Cleanup, docs, changelog [c] [elixir] [docs]

- [x] 4.1 **[P3-a]** Extract `static char *dup_cstr(const ErlNifBinary *bin)`
      (returns NULL on OOM) for the 3 identical
      `enif_alloc(size+1)` → `memcpy` → NUL-terminate blocks:
      `nif_new_index` (`:198-206`), `nif_write_index` (`:615-625`),
      `nif_read_index` (`:645-654`). Callers keep their own `contains_nul`
      guard, their `"out of memory"` error, and their `enif_free`.
      Exactly three strikes.
- [x] 4.2 **[P3-c]** Delete `make_ok_atom/1` and use the `atom_ok` global at
      its **9** call sites (`:260, :282, :304, :330, :422, :444, :462, :631,
      :891`). The function `(void)env`s its only parameter and returns a
      global. (The techdebt report said 8 — recount confirms 9.)
- [x] 4.3 **[P3-b]** `Shared.floats_to_binary/1` has zero production callers —
      it is a test fixture in `lib/`, and `mix.exs:45` ships `lib` in the hex
      package. Used by `index_test.exs:528` and `shared_test.exs:12,28,39`.
      After 3.2, `int64s_to_binary/1` also becomes test-only
      (`shared_test.exs:54,67`, `index_test.exs:583,593`).
      **Recommendation: define both locally in the test files** rather than
      adding `elixirc_paths(:test)` + a `test/support/` module — verified
      neither exists today, and each helper is a one-line comprehension. Do
      not add build wiring for two one-liners. `Shared` is `@moduledoc false`
      (internal), so removal is not a breaking change.
      Note `shared_test.exs:64-68` uses `int64s_to_binary` as the *reference
      implementation* against which `encode_ids!` is asserted — that test
      still works with a local helper; keep it.
- [x] 4.4 **[P3-d]** Delete `Index.to_binary/1` (`index.ex:455-456`). Its
      charlist clause is unreachable given every caller's `@spec` says
      `String.t()`, and no test covers it — which leaves the function an
      identity on binaries. Pass `description`/`path` straight through at the
      3 call sites (`:155`, `:355`, `:372`).
      **Decision required at execution:** this narrows the (undocumented,
      unspecced) charlist tolerance. Alternative is to keep the clause and
      widen the specs to `String.t() | charlist()`. Recommend deletion —
      the specs are the contract and they already say `String.t()`.
      **Decided at execution: deleted**, per the recommendation. Called out
      in the CHANGELOG under Changed as the one potentially-breaking item.
- [x] 4.5 **[P3-e]** Document, don't change, `Index.new/3`'s error discipline
      (`index.ex:120-170`): it raises `ArgumentError` on a bad `:metric` but
      returns `{:error, _}` for a bad `dim`. That split is *correct* under the
      house convention (developer error raises; FAISS/data error returns a
      tuple) but is undocumented. Add a `## Errors` note to the `@doc`.
- [x] 4.6 CHANGELOG entry for v0.3.1; bump `@version` in `mix.exs:4`. Update
      the CLAUDE.md locking-convention paragraph (`CLAUDE.md:48`) with the
      validate-before-lock + single-exit rule from 2.3, and the `Shared`
      module description (`CLAUDE.md:43`) if 4.3 removes helpers from it.
- [x] 4.7 [test] Final gate: `mix format --check-formatted`,
      `mix compile --warnings-as-errors`, `mix test --include slow`,
      `mix credo --strict` (no new findings), `mix docs` clean,
      `mix hex.build` clean.

**Verify:** all of 4.7 green.
**Result:** all green. `mix format --check-formatted` clean; C compiles
warning-free under `-Wall -Wextra`; `79 passed, 1 excluded` with `--include
slow`; `mix credo --strict` still exactly the 1 baseline finding; `mix docs`
and `mix hex.build` (v0.3.1) clean.

---

## Deferred — input findings deliberately NOT actioned

Recorded per Iron Law #5. Both were flagged **"recommend leaving"** in the
source report; they are listed here so they are not silently dropped.

1. **[P3-f] `"invalid index reference"` boilerplate ×6** (`:225, :452, :674,
   :684, :697, :760`). A `GET_INDEX_OR_ERROR(argv[0])` macro would collapse
   it, but a macro that hides a `return` costs more readability than the six
   explicit lines cost. **Deferred indefinitely** — not a debt item, a taste
   call already made.
2. **[P4-b] Test setup repetition** — `{:ok, index} = Index.new(...)` appears
   ~30× across the suite. The factory string differs per test (`"Flat"`,
   `"IVF4,Flat"`, `"HNSW32"`, `"IDMap,Flat"`), so a shared helper would
   parameterize the one thing each test actually varies. This is idiomatic
   ExUnit, not duplication. **Deferred indefinitely.**

## Risks

1. **Hoisting validation changes error *ordering* (Phase 1).** Today a
   size-mismatch error on a contended index blocks until the lock is free,
   then returns. After hoisting it returns immediately. No test asserts
   ordering, and the returned term is identical — but it is a real (benign)
   observable change under concurrency. Mitigation: `mix test --include slow`
   covers the contended paths.
2. **`goto` in NIF code (Phase 2).** Single-exit `goto unlock` is the
   idiomatic C answer and is strictly safer than 5 hand-typed unwinds, but it
   is a new pattern in this file. Mitigation: task 2.3 documents the rule in
   the convention block so it reads as intentional, not as a one-off.
3. **`with`-conversion silently swallowing a non-error non-match (3.1).** If
   any NIF ever returns a third shape, the old `case` raised CaseClauseError
   and the new `with` returns it verbatim to the caller. All NIF stubs return
   `{:ok, _} | :ok | {:error, binary}` (`nif.ex`), so the third shape does not
   exist today. Accepted; noted so a future NIF addition does not get a free
   silent passthrough.
4. **4.4 narrows charlist tolerance.** Pre-1.0 library, `@spec` already says
   `String.t()`, zero test coverage, zero known callers. Low risk, but it is
   the only task in this plan that could break a downstream user who relied
   on undocumented behavior. Call it out in the CHANGELOG.
5. **Phase 0 may surface a wall of pre-existing credo findings.** Explicitly
   out of scope — record the baseline, fix nothing, assert no *new* findings
   in later phases. Do not let credo triage derail the lock work.

## Execution notes

- **Phase 1 must precede Phase 2** (it removes 11 of the 16 sites Phase 2
  would otherwise have to handle).
- Phase 0 is independent and should land first so every later phase can
  verify against the credo baseline.
- Phase 3 is independent of Phases 1–2 (different language, different files)
  and could run in parallel on a separate branch if desired.
- Phase 4 last. 4.3 depends on 3.2 (which orphans `int64s_to_binary`).
- Suggested commits: one per phase. Branch + PR (never push master).
- Suggested branch: `techdebt-lock-discipline`.
