# ExUnit Test-Quality Review — techdebt-lock-discipline

Scope: `test/faiss_ex/index_test.exs`, `test/faiss_ex/shared_test.exs`.
Evidence: `mix test test/faiss_ex/shared_test.exs test/faiss_ex/index_test.exs` → 67 passed / 1 excluded, zero warnings.
`mix test --exclude no_cuda test/faiss_ex/index_test.exs` → 56 passed / 2 excluded (tag-arithmetic probe, see WARNING-free note under Findings).

## Verdict

PASS WITH WARNINGS

The three new tests are real (each fails on a plausible mutation, verified below), the relocated
encoders are byte-identical reimplementations, and the untagged `gpu_to_cpu` placement is correct.
Three items are worth fixing before merge, one of which is a coverage hole the diff walked past.

## Findings

### [WARNING] `compute_residuals` happy path cannot detect a key-encoding bug

**Location:** `test/faiss_ex/index_test.exs:221-230`

**Issue:** The only success-path test adds one vector `[1.0, 2.0, 3.0]`, asks for the residual of
that same vector against key `0`, and asserts every component is `≈ 0.0`. Key `0` is byte-order
invariant (`<<0::signed-native-64>>` is eight zero bytes either way) and it is the only valid key
in a one-vector index. The assertion therefore passes if `compute_residuals` byte-swaps keys,
ignores `keys` entirely, or returns a zero-filled buffer of the right size.

This matters *specifically because of this diff*: `compute_residuals/3` switched its key encoder
from `Shared.int64s_to_binary/1` to `Shared.encode_ids!/1` (`lib/faiss_ex/index.ex:334`). That is
exactly the kind of swap a key-blind test cannot police. `reconstruct/2` is covered here
(`index_test.exs:611` uses keys `[0, 1]`, and a swapped `1` becomes `0x0100…` → FAISS error), but
`compute_residuals` is not.

**Why it matters:** The branch's stated goal is de-risking, and it modified the key path of this
function while leaving the function's only positive test unable to observe the change. A silently
mis-encoded key yields plausible-looking floats, not a crash.

**Fix:** Make the existing test key-sensitive with a nonzero expected residual — verified live
against the built NIF:

```elixir
{:ok, index} = Index.new(3, "Flat")
:ok = Index.add(index, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# xs - reconstruct(key 1) = [1,2,3] - [4,5,6]
assert {:ok, [[a, b, c]]} = Index.compute_residuals(index, [[1.0, 2.0, 3.0]], [1])
Enum.each([a, b, c], &assert_in_delta(&1, -3.0, 1.0e-6))
```

(Observed: key `1` → `[[-3.0, -3.0, -3.0]]`, key `0` → `[[0.0, 0.0, 0.0]]`. Keep the existing
zero-residual case too; it documents the identity, it just cannot stand alone.)

### [WARNING] `defp` inside a `describe` block is module-scoped, unlike the `alias` beside it

**Location:** `test/faiss_ex/index_test.exs:551-559`

**Issue:** Lines 551–559 put `alias FaissEx.Shared`, `defp encode/1`, and `defp encode_ids/1` in the
same visual block, but the two constructs have opposite scoping. Verified empirically with a
standalone ExUnit probe: `describe/2` expands into a plain `quote` block in the test module, so
`alias` is lexically scoped to the block and does **not** leak, while `defp` compiles straight into
the module and **is** callable from every other `describe` in the file.

**Why it matters:** Two failure modes, both invisible today. (1) A reader trusts the visual grouping
and assumes `encode/1` is local to "binary API"; it is not, and a second `defp encode/1` added under
`describe "num_gpus"` (line 680, already after this block) is a redefinition, not a shadow.
(2) It contradicts `test/faiss_ex/shared_test.exs:78-88` in this very diff, which places the
equivalent helpers correctly at module level with a comment. A cleanup branch shipping two opposite
conventions for the identical construct, in two files, is the specific debt this branch exists to
remove.

**Fix:** Move `encode/1` and `encode_ids/1` to module level at the bottom of `FaissEx.IndexTest`,
mirroring `shared_test.exs`. Leave the `alias` where it is (it is correctly block-scoped and only
used inside that block). Do not reach for `test/support/` — the project has no `elixirc_paths`
override for it and two four-line helpers do not justify adding one.

### [WARNING] Two contradictory contracts for the same developer error, now both pinned by tests

**Location:** `test/faiss_ex/index_test.exs:232-241` vs `test/faiss_ex/index_test.exs:519-528`

**Issue:** A key/vector count mismatch now behaves differently depending on which function you call:

| call | 2 vectors, 1 id | contract |
|---|---|---|
| `Index.compute_residuals/3` | raises `ArgumentError` (new, line 232) | developer error |
| `Index.add_with_ids/3` | `{:error, "binary size mismatch"}` (line 519) | runtime error |

This diff added *both* the raising branch (`lib/faiss_ex/index.ex:336-339`) and the `## Errors`
moduledoc that states the rule — "developer errors raise, FAISS and data errors return error
tuples" (`lib/faiss_ex/index.ex:128-135`). `add_with_ids/3` discards its id count
(`{_n_ids, ids_bin} = encode_id_data!(ids)`, line 218) and lets the C size check produce a tuple, so
the list-input path violates the rule the same commit wrote down.

**Why it matters:** The test suite is now the authoritative record of an inconsistency, which makes
it sticky: whoever later aligns `add_with_ids/3` with the documented rule has to change a green
test, which reads like a regression in review.

**Fix:** Either (a) make `add_with_ids/3` raise on a *list* count mismatch for symmetry (the binary
path at line 663 legitimately keeps returning `{:error, "binary size mismatch"}`, since binary sizes
are data, not a literal), or (b) if the divergence is deliberate, say so in the `## Errors` doc and
rename the test at line 519 to `"add_with_ids with mismatched list id count returns error (binary
sizing, not a raise)"` so the asymmetry is stated rather than merely observed.

### [SUGGESTION] The `encode_ids!` reference-oracle assertion — not a tautology, but its weakest witness

**Location:** `test/faiss_ex/shared_test.exs:64-69` and `78-88`

**Answering the question directly: it did not become a tautology.** `Shared.encode_ids!/1` is a
tail-recursive accumulator (`<<acc::binary, i::signed-native-64>>`, `lib/faiss_ex/shared.ex`), while
`int64s_to_binary/1` is a `for … into: <<>>` comprehension. Two structurally different hand-written
implementations of the same wire format. `assert bin == int64s_to_binary(ids)` compares distinct
code paths, so it is a genuine differential oracle, not `f(x) == f(x)`.

What actually changed is the *contract under test*, not the logic. Previously both sides shipped in
`lib/`, so the assertion defended a cross-module invariant ("the library's two encoders agree").
That invariant no longer exists — one side stopped shipping. The assertion narrowed to "the shipped
encoder agrees with an in-file reference."

Is the narrowed version still worth its line? Yes, marginally, and the margin is precise. Line 68
(`Shared.binary_to_int64s(bin) == ids`) already pins `encode_ids!`'s bytes through a production
decoder. The *only* class of bug line 67 catches that line 68 misses is **symmetric drift** — an
encoder and decoder changed the same wrong way together (both flipped to big-endian, both to
unsigned). A same-file reference oracle is exactly the right instrument for that, and the fixture
`[100, -5, 0, 9_999_999_999]` is well chosen: the `-5` makes a signed→unsigned regression
detectable, and `100`/`9_999_999_999` are not byte-order palindromes. The helper is itself anchored
independently at line 54-55, where `int64s_to_binary/1` is decoded by production
`Shared.binary_to_int64_rows/2`. Three witnesses, none circular. The test rename to
"…the same binary as a naive s64 comprehension" is honest and correct.

The residual risk is human, not logical: the fix-up path for a failing line 67 is now to edit the
helper eight lines below, which reads as a test change. Before, the same shortcut meant editing
`lib/`, which reads as a production change.

**Fix (optional, cheap):** extend the comment at `shared_test.exs:78-81` with one sentence —
"these are the fixed reference encodings; if a test fails against them, the production encoder
changed, not these" — and add `-9_223_372_036_854_775_808` / `9_223_372_036_854_775_807` to the `ids`
fixture at line 65 so the differential also covers the s64 boundary (currently only
`binary_to_int64_rows` at line 52 exercises those).

### [SUGGESTION] Grouped minor observations

**Location:** `test/faiss_ex/index_test.exs:243-250`, `232-241`, `553-559`; `test/faiss_ex/shared_test.exs:82-88`

1. **Exact-message assertions are appropriately tight, not brittle.** `assert_raise ArgumentError,
   "got 1 keys for 2 vectors, expected one key per vector"` and `"id 0 is not an integer: :nope"`
   are full-string matches, which is the convention already used at lines 476, 484, 492, 500, 508,
   644, 653 and `shared_test.exs:72`. Both new messages carry the interpolated values that are the
   actual point of the message (the counts; the index and the offending term), so a regex would
   *lose* coverage. No change recommended — noted because the question was asked.
2. **The non-integer-key test is thin but not redundant.** `index_test.exs:243` duplicates
   `shared_test.exs:71` at the message level. Its residual value is pinning that
   `compute_residuals/3` routes keys through `encode_ids!/1` rather than raw binary construction
   (which on `master` produced a different, useless `ArgumentError` from `<<:nope::signed-64>>`).
   That is worth one test; just be aware it is a delegation test, not a behavior test.
3. **Four encoders, two names, two files.** `index_test.exs` calls them `encode/1` / `encode_ids/1`;
   `shared_test.exs` calls the identical bodies `floats_to_binary/1` / `int64s_to_binary/1`. Given
   there is no `test/support/`, per-file duplication is the right call, but aligning the names would
   make the two files greppable as one concept.

## Verified, no finding

- **Helper fidelity.** Diffed against `git show master:lib/faiss_ex/shared.ex:5-17`. Both relocated
  bodies are character-identical to the deleted originals — `<<f::float-32-native>>` and
  `<<i::signed-native-64>>`, same endianness, same widths — differing only by the dropped
  `when is_list(list)` guard, which is immaterial in a test fixture. `index_test.exs`'s `encode/1`
  additionally inlines the `List.flatten/1` its old call site performed in the pipe; net behavior
  identical. Both index_test helpers are further anchored end-to-end through FAISS
  (`index_test.exs:636` searches an index populated via `encode/1` + `encode_ids/1` and asserts the
  label is `100`), so a silent endianness error in them would fail loudly.
- **Untagged `gpu_to_cpu` placement is correct.** `test/test_helper.exs:5-8` excludes `:no_cuda`
  when `USE_CUDA=true`, else `:cuda`. `CLAUDE.md:61` documents that `@tag` binds to the next `test`,
  not the `describe`, so the `@tag :no_cuda` at line 287 covers only line 288. The new test must run
  in *both* builds, because `gpu_to_cpu(%Index{device: :host})` (`lib/faiss_ex/index.ex:443-445`)
  matches before any NIF call and is build-independent. Untagged is right, and tagging it
  `:no_cuda` would be a bug (it would stop running under CUDA where it is equally valid).
  Confirmed empirically: `mix test --exclude no_cuda test/faiss_ex/index_test.exs` — which reproduces
  the `USE_CUDA=true` exclusion set — runs the new test and passes (56 passed, 2 excluded).
- **Mutation sensitivity of the three new tests.** (a) Delete the `device: :host` clause →
  `FunctionClauseError`, test fails. (b) Delete the `if n_keys != n` raise → the C layer's
  `keys_bin.size != expected_keys` check (`c_src/faiss_ex_nif.c:582`) returns
  `{:error, "binary size mismatch"}`, so `assert_raise` fails. (c) Revert to
  `int64s_to_binary(keys)` → different `ArgumentError` message, exact match fails. None pass
  trivially.
- **`index_test.exs:294` (`{:ok, %Index{} = index}`).** Correct and load-bearing; consistent with
  line 8, which already pattern-matches `%Index{}` out of `Index.new/3`. No test-quality objection.

## Coverage gaps

- `Index.new/3`, `to_file/2`, `from_file/2` no longer accept charlists (`to_binary/1` deleted).
  Nothing in `test/` ever passed a charlist — that absence is the evidence the tolerance was dead,
  and it is why the deletion is safe. **No test recommended**: pinning the new charlist behavior
  would enshrine an accidental error shape for an input the `@spec`s never allowed. The CHANGELOG
  note is the correct artifact.
- The `case` → `with` conversions drop the implicit `CaseClauseError` on an unexpected NIF return
  shape (a bare `with` passes the value through). Unreachable — every NIF returns `{:ok, _}` or
  `{:error, _}`. No test warranted.
- `c_src/faiss_ex_nif.c:582`'s `keys_bin.size` branch is unreachable from `compute_residuals/3`
  (list-only API raises first). Defense-in-depth at the NIF boundary; the equivalent branch for
  `add_with_ids/3` *is* covered at `index_test.exs:663`. Acceptable.

## Pre-existing

- `test/faiss_ex/index_test.exs:190-202` — `add_with_ids/3` list happy path asserts only `ntotal`,
  never that the custom ids survive a search; the binary-API sibling at line 636 does. PRE-EXISTING.
