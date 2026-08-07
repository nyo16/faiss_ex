# Scratchpad — techdebt-cleanup

## Status (2026-08-07)

**COMPLETE — 22/22 tasks** on branch `techdebt-lock-discipline`.
Final gate all green: format clean, C warning-free under `-Wall -Wextra`,
`79 passed, 1 excluded` (`--include slow`), credo at the 1-finding baseline,
`mix docs` and `mix hex.build` (v0.3.1) clean.

## Findings → tasks map

| Finding | Task(s) |
|---|---|
| P1-1 16 hand-unwound lock paths | 1.1–1.5 (removes 11), 2.1–2.2 (remaining 5) |
| P1-2 5 NIFs validate under lock | 1.1–1.5 |
| P2-1 10× passthrough `case` | 3.1 |
| P2-2 `compute_residuals` unvalidated id encoder | 3.2 |
| P2-3 `gpu_to_cpu` raises, counterpart doesn't | 3.3 |
| P3-a C string dup ×3 | 4.1 |
| P3-b `floats_to_binary` dead in lib | 4.3 |
| P3-c `make_ok_atom` ×9 | 4.2 |
| P3-d `to_binary` charlist clause | 4.4 |
| P3-e `Index.new` mixed error discipline | 4.5 (docs only) |
| P3-f `"invalid index reference"` ×6 | **DEFERRED** |
| P4-a no credo | 0.1 |
| P4-b test setup repetition | **DEFERRED** |

## Decisions

- **Phase 1 before Phase 2.** Hoisting validation above the lock dissolves
  11 of the 16 unwind sites, so the `goto unlock` work in Phase 2 only has
  to cover the two functions that genuinely read locked state
  (`faiss_Clustering_d`, `faiss_Clustering_centroids`). Doing Phase 2 first
  would mean writing labels into five functions that are about to stop
  needing them.
- **`goto unlock` over per-branch unwinds.** Single-exit is the idiomatic C
  answer. Task 2.3 writes the rule into the convention block at
  `faiss_ex_nif.c:25-40` — the code fix alone drifts back without the
  documented rule, which is exactly how the current 16 sites accumulated.
- **3.2 raises rather than returning `{:error, _}`.** House convention:
  developer errors raise (ragged lists, non-divisible binaries — see
  `perf-fast-path/scratchpad.md:20-21`), FAISS/data errors return tuples.
  A key/vector count mismatch is a developer error.
- **4.3 defines test helpers locally, no `test/support/`.** Verified: no
  `elixirc_paths` in mix.exs and no `test/support/` today. Adding build
  wiring plus a support module to relocate two one-line comprehensions
  costs more than it removes.
- **4.5 is a docs task, not a code change.** `Index.new/3`'s raise-on-metric
  / error-tuple-on-dim split is already correct under the house convention;
  it is just undocumented.
- **P3-f and P4-b deferred, not dropped.** Both were "recommend leaving" in
  the source report. Recorded in the plan's Deferred section so a future
  re-audit does not rediscover them as new.

## Corrections to the source report

- `make_ok_atom` has **9** call sites, not 8 (`:260, :282, :304, :330, :422,
  :444, :462, :631, :891`).

## Facts verified while planning (do not re-derive)

- No test pins `compute_residuals`' key-count mismatch. The two
  `{:error, "binary size mismatch"}` assertions (`index_test.exs:497, :636`)
  are both `add_with_ids`. So 3.2 breaks nothing — but the new behavior is
  uncovered and needs the test in 3.4(b).
- `index_test.exs:273` ("cpu_to_gpu on an index already on GPU") carries
  **no** tag; `@tag :no_cuda` at `:267` applies only to the test at `:268`.
  The 3.3 mirror test is therefore untagged and goes in the same describe.
- `mix credo` is not installed. `mix format --check-formatted` passes.
  Baseline suite: 74 passed, 3 excluded. C compiles clean under
  `-Wall -Wextra`.

## Open questions — RESOLVED at execution

- **4.4 charlist narrowing** — **deleted `to_binary/1`**, per the plan's
  recommendation. The three call sites now pass `description`/`path`
  straight through. Flagged in the CHANGELOG under *Changed* as the only
  potentially-breaking item in the release.
- **3.3 error wording** — went with **`"index already on host"`**, matching
  the `device: :host` atom and symmetric with `"index already on GPU"`.
- **Phase 0 credo baseline size** — one finding. No backlog, nothing to
  triage. See below.

## Discovered during execution (not in the plan)

- **3.1 has a type-inference cost the plan did not anticipate.** Risk #3
  called out the *runtime* passthrough semantics, which held. What it missed:
  `with` without `else` returns the NIF's term, which Elixir's type checker
  sees as `dynamic()`. Converting `cpu_to_gpu/2` therefore widened
  `Index.new/3`'s inferred return from a closed
  `{:ok, %Index{}} | {:error, term()}` union to `dynamic()`, and struct-update
  syntax on the result stopped type-checking:

  ```
  test/faiss_ex/index_test.exs:295: when defining the variable "index",
  you must also pattern match on "%FaissEx.Index{}"
  ```

  Verified NEW by stashing `lib/` + `test/` and recompiling — baseline is
  warning-free. Fixed at the one affected site by matching
  `{:ok, %Index{} = index}`, which is better test code anyway. Recorded in
  the CHANGELOG because any downstream caller doing struct-update on a
  `new/3` result hits the same warning. **If this becomes a real complaint,
  the fix is an explicit `else {:error, _} = err -> err` on `cpu_to_gpu/2`
  only — not a wholesale revert of 3.1.**
- **`make_ok_atom` call-site count.** The scratchpad's recount of 9 was
  right, but one of them (`nif_train_clustering`) was rewritten to `atom_ok`
  as part of task 2.1's single-exit refactor, so task 4.2 converted the
  remaining 8 plus the function definition.

## Credo baseline (task 0.1, 2026-08-07)

`{:credo, "~> 1.7", only: [:dev, :test], runtime: false}` added to
`mix.exs`. `mix credo --strict` output, verbatim:

```
Checking 11 source files ...

  Code Readability
┃
┃ [R] ↘ The alias `FaissEx.NIF` is not alphabetically ordered among its group.
┃       lib/faiss_ex/clustering.ex:23:9 #(FaissEx.Clustering)

Analysis took 0.06 seconds (0.00s to load, 0.06s running 69 checks on 11 files)
87 mods/funs, found 1 code readability issue.
```

**Baseline = exactly 1 finding.** Not fixed (out of scope per task 0.1).
Later phases assert *no new* findings, i.e. `--strict` still reports only
this one.

## Dead-ends

_(none yet)_
