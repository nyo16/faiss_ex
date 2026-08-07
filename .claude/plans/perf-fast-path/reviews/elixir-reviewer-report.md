# Code Review: perf-binary-fast-path (Elixir)

## Summary
- **Status**: ⚠️ Changes Requested (one item needs compile verification; rest are minor)
- **Issues Found**: 4

## Critical Issues

None confirmed. See "Needs Verification" below for one item that could be
critical if it doesn't compile as written.

## Needs Verification

1. **`lib/faiss_ex/shared.ex:53`** — `decode_rows/4` uses a pinned variable
   as a bit-string size specifier:
   ```elixir
   defp decode_rows(bin, row_size, decode, acc) do
     <<row::binary-size(^row_size), rest::binary>> = bin
     ...
   ```
   UNVERIFIED: I could not run `mix compile` in this environment to confirm
   `size(^row_size)` is accepted syntax vs. the idiomatic unpinned
   `size(row_size)` (referencing an already-bound outer variable in a size
   specifier does not require `^` — pin is for binding-position
   disambiguation, and `row_size` is never a binding target here). Since
   `shared_test.exs` and `index_test.exs` both exercise
   `binary_to_float_rows/2` end-to-end, this most likely already compiles
   and passes as-is, but please have verification-runner confirm
   `mix compile --warnings-as-errors` and `mix test` pass so this doesn't
   silently regress. If it does compile, the `^` is redundant — consider
   dropping it for clarity (LOW, style only, contingent on the above).

## Warnings

1. **`lib/faiss_ex/shared.ex:39-48` (`binary_to_float_rows/2`,
   `binary_to_int64_rows/2`)** — `@spec` declares `row_len :: pos_integer()`
   but there is no guard/runtime check enforcing `row_len > 0`. If `row_len`
   is ever `0`, `row_size` becomes `0` and `decode_rows/4` matches a
   zero-width row on every call (`<<row::binary-size(0), rest::binary>> = bin`
   binds `row = <<>>`, `rest = bin` unchanged), so the base case
   (`bin == <<>>`) is never reached for a non-empty input — infinite
   recursion / hang rather than a clean error. Currently unreachable from
   the public API (`dim` and `k` are always `pos_integer()` upstream), so
   this is defense-in-depth rather than a live bug, but worth a guard
   (`when is_binary(bin) and row_len > 0`) given these are the functions
   the new binary fast-path leans on most heavily.

## Suggestions

1. **`lib/faiss_ex/shared.ex:26,34-36`** — `encode_ids!/1`'s error clause
   `encode_ids([bad | _], n, _acc)` matches any non-integer, but the guard
   on the prior clause (`is_integer(i)`) means floats, atoms, tuples, etc.
   all funnel through the same generic message. That's fine and consistent
   with `encode_row!`'s style in the same module — no change needed, just
   confirming parity (not a finding, noted for completeness).

2. **`lib/faiss_ex/index.ex:480-508` (`encode_vector_data!/2`,
   `encode_id_data!/2`)** — both private helpers duplicate the
   `rem(byte_size, unit)` divisibility-check pattern almost verbatim
   (differ only in unit size and error text). Could factor into one
   `validate_binary_len!(bin, unit, label)` helper to avoid drift if the
   error message wording needs to change later. Minor, not blocking.

3. **`lib/faiss_ex/index.ex` moduledoc (Binary API section) and
   `add_with_ids/3` doc (lines 194-206)** — doc text is accurate against
   the implementation I read (guard clauses, error tuples, combos all
   match `add_with_ids accepts mixed list/binary forms` test). No
   discrepancies found between `@spec`/`@doc` and behavior for any of
   `add/2`, `train/2`, `add_with_ids/3`, `search_binary/3`,
   `reconstruct_binary/2`.

## Test Coverage

Coverage for the new surface is solid:
- `test/faiss_ex/shared_test.exs` — row decoders (empty/single/many rows,
  round-trip, parity with flat-decode+chunk), `encode_ids!/1` (empty,
  round-trip, non-integer raise with position).
- `test/faiss_ex/index_test.exs` ("binary API" describe block) — list vs.
  binary parity for add/train/search, `search_binary` vs `search` parity,
  `reconstruct_binary` list/binary key parity, mixed list/binary combos
  for `add_with_ids`, divisibility-error messages (vectors and ids),
  empty-binary no-op for `add`, empty-query `search_binary`, and the
  documented `add_with_ids` size-mismatch `{:error, "binary size
  mismatch"}` path.

No coverage gap found for the new functions themselves. One gap worth
adding: no test exercises `binary_to_float_rows/2` or
`binary_to_int64_rows/2` with `row_len` derived from a runtime value that
could be `0` (see Warning #1) — low priority since it's not reachable
today.

## Verdict

Implementation is idiomatic (tail-recursive single-pass decoders/encoders,
tagged-tuple returns, `ArgumentError` for programmer errors vs.
`{:error, _}` for FAISS/NIF-layer failures, ArgumentError messages include
position/size context). The only item needing action before merge is
confirming `size(^row_size)` compiles cleanly under
`mix compile --warnings-as-errors` — everything else here is optional
polish (Warning #1 guard, Suggestion #2 dedup).
