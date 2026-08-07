# Elixir Reviewer Review — techdebt-lock-discipline

Scope: `lib/faiss_ex/index.ex`, `lib/faiss_ex/clustering.ex`, `lib/faiss_ex/shared.ex`
(full files read, not just the diff). `c_src/` consulted read-only for return-shape
evidence only. Test files out of scope.

## Verdict

PASS WITH WARNINGS

The cleanup is idiomatic and, at every site I checked, behavior-preserving. One
documentation claim in the new `## Errors` block is inaccurate against the module's
own code; everything else is stylistic.

## Findings

### [WARNING] New `## Errors` block claims a module-wide convention the module does not follow

**Location:** `lib/faiss_ex/index.ex:128-136` (claim); `lib/faiss_ex/index.ex:205-207`
and `:215-220` (counter-example); `lib/faiss_ex/index.ex:336-339` (the new raise)

**Issue:** The block asserts *"The same split applies across this module: developer
errors raise, FAISS and data errors return error tuples."* Two code paths contradict it:

1. **The identical error class is handled two opposite ways.** After this branch,
   `compute_residuals/3` raises `ArgumentError` on an id/vector count mismatch
   (`index.ex:336-339`), while `add_with_ids/3` returns `{:error, "binary size
   mismatch"}` for exactly the same mismatch — documented as the contract at
   `index.ex:205-207` and produced by `c_src/faiss_ex_nif.c:315,328`. `add_with_ids/3`
   even computes the id count at `index.ex:218` and then discards it as `_n_ids`, so
   the information needed to raise is already in hand. Two sibling functions, same
   caller mistake, two different failure shapes.
2. **A developer error returns a tuple.** After the `to_binary/1` deletion, a charlist
   `description` (or a non-integer `dim`) fails `enif_inspect_binary` and returns
   `{:error, "invalid arguments"}` (`c_src/faiss_ex_nif.c:207-211`). That is squarely a
   developer error returning a tuple, which the block says does not happen.

Secondary: the block's justification — *"the atom is a compile-time constant in the
caller"* — is not generally true. `metric` is read from `opts` at `index.ex:146` and can
just as easily come from `Application.get_env/2` as `dim` and `description` can, so the
stated rationale does not actually separate the two cases.

**Why it matters:** This is the one place in the library that tells a reader how to
write error handling against it. A caller who reads it and wraps `add_with_ids/3` in
`try/rescue ArgumentError` (or, conversely, matches `{:error, _}` around
`compute_residuals/3`) gets a crash from the path they thought they had covered. It
also sets a maintenance trap: the next person adding a validation has a written
convention that the surrounding code does not obey, so either choice looks wrong.

**Fix (pick one, in preference order):**

- **Now, non-breaking — scope the claim to `new/3`.** Drop the final sentence
  (`index.ex:134-136`) and the compile-time-constant rationale, leaving the block
  describing only why `new/3` itself has two failure shapes. Nothing else in the branch
  depends on the generalization.
- **Follow-up (0.4.0, breaking) — actually unify.** Make `add_with_ids/3` raise on
  count mismatch using the `_n_ids` already returned at `index.ex:218`; note
  `encode_id_data!/1` returns a count for the binary path too (`index.ex:463-474`), so
  all four list/binary combinations can be checked symmetrically. This changes a
  documented `{:error, _}` return into a raise, so it needs a CHANGELOG entry under
  *Changed*, not *Internal*.

Do **not** revert `compute_residuals/3` — raising there is the better discipline and
matches `reconstruct/2`'s use of `encode_ids!/1` (`index.ex:300`). The doc is what is
wrong, not the code.

### [SUGGESTION] `with`-without-`else` trades a closed return type for brevity

**Location:** all 10 converted sites — `index.ex:235`, `:258`, `:279`, `:302`, `:341`,
`:368`, `:410`, `:432`; `clustering.ex:60`, `:89`

**Issue:** A `with` with no `else` returns the non-matching term verbatim. The old
`case` blocks matched only the happy shape plus `{:error, _} = err`, so any third shape
raised `CaseClauseError` at the NIF boundary; now it is handed to the caller unchanged.
Given the NIF contract (`{:ok, _} | :ok | {:error, binary}`, `lib/faiss_ex/nif.ex`) this
is unreachable today, and I found no site where the reachable set differs — see
Confirmations. But the failure mode has moved from "crash at the boundary" to "return a
term that violates the `@spec`, one stack frame away from where it can be diagnosed."
This is also the mechanism behind the already-known `Index.new/3` → `dynamic()`
inference widening (`new/3` at `index.ex:165-177` tail-calls `cpu_to_gpu/2` at `:175`,
whose `with` at `:410` no longer has a closed return).

**Why it matters:** Low, and the tradeoff appears to have been made knowingly (the
`dynamic()` consequence is in the CHANGELOG at `CHANGELOG.md:39-43`). Recording it so
the next person who hits a type-checker complaint downstream of these functions knows
the cause and the lever.

**Fix:** None required. If the inference widening becomes a nuisance at more call
sites, adding `else {:error, _} = err -> err` to the offending `with` restores both the
closed union and the old raise-on-surprise behavior at the cost of the three lines the
branch just removed — apply it selectively, not across all ten.

### [SUGGESTION] `Shared` spec coverage is now uneven

**Location:** `lib/faiss_ex/shared.ex:4-12`

**Issue:** After the two deletions, `binary_to_floats/1` (`:5`) and `binary_to_int64s/1`
(`:10`) are the only remaining public functions in the module without a `@spec`; every
other one has had one since 0.3.0 (`:15`, `:29`, `:36`, `:50`). Both are still live —
called from `binary_to_float_rows/2` at `:32` and `binary_to_int64_rows/2` at `:39` — so
this is not dead code, just an inconsistency the deletions made more visible.

**Why it matters:** Cosmetic. `@moduledoc false`, so no doc output is affected.

**Fix:** `@spec binary_to_floats(binary()) :: [float()]` and
`@spec binary_to_int64s(binary()) :: [integer()]`. Two lines, or leave it.

## Confirmations

Each item the assignment asked me to verify, with the evidence:

1. **`case` → `with`, 10 sites — behavior-preserving.** Every converted `case` had
   exactly two clauses: the happy shape and `{:error, _} = err -> err`. The `with`
   reproduces both (the error tuple falls through verbatim) and differs only on terms
   neither clause matched, which previously raised `CaseClauseError`. Checked against
   the C returns: `nif_search_index` (`c_src:355-380`), `nif_reconstruct_batch`
   (`:484-494`), `nif_compute_residuals` (`:561-583`), `nif_new_index` (`:210`),
   `nif_read_index` (`:642`), `nif_index_cpu_to_gpu` (`:715`) all return
   `{:ok, _} | {:error, binary}` only. No site swallows something the `case` raised on
   for any input reachable through the public API. `index.ex:165` and
   `clustering.ex:78` were already `with` on master and are unchanged.
2. **`from_file/2` hard match — confirmed and correct.** `index.ex:371` keeps
   `{:ok, dim} = NIF.nif_get_index_dim(ref)` inside the `with` body with the rationale
   in a comment at `:369-370`. The reasoning holds: `nif_get_index_dim`
   (`c_src:668-675`) has exactly one failure branch — `enif_get_resource` on a
   non-index term — and `ref` came from a successful `nif_read_index` one line earlier,
   so it is provably an index resource. A `MatchError` there is an invariant violation,
   not a caller error, and folding it into the `with` chain would convert a library bug
   into an `{:error, _}` the caller cannot act on. Correct as written.
3. **`compute_residuals/3`.** The raise at `index.ex:336-339` is before the NIF call at
   `:341`; the message names both counts (`"got #{n_keys} keys for #{n} vectors"`); the
   `@doc` at `:327-328` documents all three raise conditions. `reconstruct/2` uses the
   same `Shared.encode_ids!/1` at `:300`. Consistent with `reconstruct/2` and with
   `encode_vectors!/2`'s existing raise discipline — but see the WARNING above for the
   `add_with_ids/3` divergence.
4. **`gpu_to_cpu/1`.** Clause order is correct: specific `{:cuda, _}` at `index.ex:431`,
   fallback `:host` at `:443` — mirroring `cpu_to_gpu/2` (`:409` / `:421`). The two
   clauses are exhaustive over `t()`'s `device` field (`:100`). `@spec` at `:430`
   (`{:ok, t()} | error()`, with `error :: {:error, String.t()}` at `:118`) covers the
   new return, and the `@doc` at `:428` states it. Symmetric with `cpu_to_gpu/2` in
   shape, message style, and doc placement.
5. **`to_binary/1` deletion.** All three former call sites pass through correctly:
   `new/3` at `index.ex:165` (`description`), `to_file/2` at `:351` (`path`),
   `from_file/2` at `:368` (`path`). All three `@spec`s already said `String.t()`
   (`:144`, `:349`, `:364`). No remaining `@doc`, `@spec`, `@typedoc`, moduledoc, or
   README text implies charlist support — grepped `lib/` and `README.md` for
   `charlist` / `to_charlist` / `~c"`, zero hits. Behavior narrowing confirmed:
   a charlist now yields `{:error, "invalid arguments"}` (`c_src:210`), which is
   called out in `CHANGELOG.md:33-37`.
6. **`Shared` deletions.** Zero remaining callers of `floats_to_binary/1` or
   `int64s_to_binary/1` anywhere in `lib/` — the only `to_binary(` hit in `lib/` is
   `IO.iodata_to_binary/1` at `shared.ex:65`. The last production caller,
   `compute_residuals/3`, moved to `encode_ids!/1` in this same branch. Nothing else in
   `lib/` regressed; `binary_to_floats/1` and `binary_to_int64s/1` remain reachable via
   the row decoders. `CLAUDE.md:43` was updated with a do-not-re-add note.
7. **Type specs.** Every `@spec` still describes the real return after the conversions.
   Spot-checked all ten: `search/3` `:230-231`, `search_binary/3` `:253-254`,
   `clone/1` `:277`, `reconstruct/2` `:298`, `compute_residuals/3` `:330`,
   `from_file/2` `:364`, `cpu_to_gpu/2` `:406`, `gpu_to_cpu/1` `:430`,
   `Clustering.new/2` `:58`, `Clustering.get_centroids/1` `:87`. The `with`
   pass-through can in principle return a term outside the spec, but only for NIF
   returns the C layer does not produce (see the second SUGGESTION).
8. **`## Errors` block accuracy.** The `new/3`-specific half is accurate: unknown
   `:metric` raises at `index.ex:160-163`; a non-positive `dim` returns
   `{:error, "dim must be positive"}` (`c_src:213`) and a factory string FAISS rejects
   returns `{:error, "failed to create index"}` (`c_src:224`). The generalization to the
   whole module is not — see the WARNING.

**Also confirmed (already-known findings):** the `Index.new/3` → `dynamic()` widening is
real and its cause is `cpu_to_gpu/2`'s new `with` at `index.ex:410` reached from
`new/3`'s `{:cuda, _}` branch at `:175`; documented at `CHANGELOG.md:39-43`. The
`to_binary/1` charlist narrowing is documented at `CHANGELOG.md:33-37`.

## Pre-existing

- `lib/faiss_ex/clustering.ex:23-25` — alias order (`NIF`, `Shared`, `Index`) is not
  alphabetical. PRE-EXISTING, known credo baseline, deliberately unfixed.
- `lib/faiss_ex/index.ex:331-332` — `compute_residuals/3` is list-only
  (`when is_list(xs) and is_list(keys)`) while `add_with_ids/3` and
  `reconstruct_binary/2` accept binaries. PRE-EXISTING gap in the Binary API surface,
  not touched by this branch.
