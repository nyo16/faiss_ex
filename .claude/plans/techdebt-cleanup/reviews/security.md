# Security Review — techdebt-lock-discipline

## Scope note

This is a NIF binding library: no HTTP, no session, no database, no
template rendering, no auth surface. The OWASP web categories in the
standard checklist (authn/authz, SQL injection, XSS, CSRF, secrets,
security headers) have no applicable code in this repo and are not
reported. The real attack surface is caller-controlled data crossing the
BEAM/C boundary, and that is what was audited.

## Verdict

PASS WITH WARNINGS

The lock-discipline refactor is memory-safe and validation-complete. Every
hoisted check survived the move, every `safe_mul` guard is intact and still
precedes the `ErlNifBinary.size` comparison, every `enif_rwlock_*lock` has
exactly one matching unlock on every path, and `dup_cstr` neither leaks nor
double-frees. No BLOCKER. The findings below are pre-existing hazards on the
audited surface, not regressions introduced by this branch.

## Verification performed

Before-image: `git show master:c_src/faiss_ex_nif.c` saved to
`/tmp/master_nif.c` (1010 lines), compared function-by-function against the
working tree (1013 lines) and against `/tmp/faissex_review.diff`.

### 1. Hoisted validation is complete

For each of the five NIFs, the set of checks between argument decode and the
FAISS call is identical to master; the only removed lines in each hunk are
`enif_rwlock_*lock` / `enif_rwlock_*unlock` and `make_ok_atom`. No `-` line
in the diff carries a check absent from the post-image.

| NIF | current | master | checks preserved |
|---|---|---|---|
| `nif_add_to_index` | `c_src/faiss_ex_nif.c:272-285` | `/tmp/master_nif.c:265-277` | `n<0`, `n==0 && size!=0`, 2×`safe_mul`, `size!=expected` |
| `nif_add_with_ids_to_index` | `c_src/faiss_ex_nif.c:312-329` | `/tmp/master_nif.c:309-321` | `n<0`, `n==0` dual-size, 3×`safe_mul`, dual `size!=expected` |
| `nif_train_index` | `c_src/faiss_ex_nif.c:430-443` | `/tmp/master_nif.c:427-440` | `n<0`, `n==0 && size!=0`, 2×`safe_mul`, `size!=expected` |
| `nif_reconstruct_batch` | `c_src/faiss_ex_nif.c:487-526` | `/tmp/master_nif.c:510-523` | `n<0`, `safe_mul` keys, `keys size!=expected`, 2×`safe_mul` result, alloc |
| `nif_compute_residuals` | `c_src/faiss_ex_nif.c:564-589` | `/tmp/master_nif.c:571-590` | `n<0`, `n==0` dual-size, 3×`safe_mul`, dual `size!=expected`, alloc |

The hoisting premise holds. `res->dim` is written exactly once, in
`wrap_index_resource` at `c_src/faiss_ex_nif.c:192`, before
`enif_make_resource` at `:193` publishes the resource; grep for `->dim`
finds no other write and reads only at `:278, :320, :374, :436, :517, :575,
:674`. Reading it unlocked introduces no TOCTOU — the other input,
`ErlNifBinary.size`, is fixed for the duration of the call, so validation
outside the lock is decided on exactly the same two values the locked
version used.

### 2. `safe_mul` coverage is intact

Every allocation and every size comparison is fed by a `safe_mul` result:
`:280`, `:322-323`, `:376`, `:384-386`, `:438`, `:490`, `:519`, `:577-578`,
`:875`, `:920`. In all five hoisted NIFs the `size != expected` comparison
happens before any pointer cast or pointer arithmetic:

- `nif_add_to_index` — `safe_mul` `:280`, compare `:283`, cast `:289`.
- `nif_add_with_ids_to_index` — `safe_mul` `:322-323`, compare `:327`, casts `:334-335`.
- `nif_train_index` — `safe_mul` `:438`, compare `:441`, cast `:447`.
- `nif_reconstruct_batch` — `safe_mul` `:490`, compare `:493`, cast `:503`.
- `nif_compute_residuals` — `safe_mul` `:577-578`, compare `:582`, casts `:591-593`.

`(size_t)n` casts are safe because `n < 0` is rejected first (`:272`,
`:312`, `:358`, `:430`, `:487`, `:564`; `n <= 0` at `:861`).

### 3. `dup_cstr` is correct at all 3 call sites

Each call is immediately preceded by a `contains_nul` guard:
`:214`→`:216` (`nif_new_index`), `:617`→`:619` (`nif_write_index`),
`:645`→`:647` (`nif_read_index`). Interior-NUL truncation of a factory
string or a path is therefore still rejected on every path into
`dup_cstr`, exactly as the helper's contract comment at `:164-166`
requires.

Exactly one `enif_free` per allocation, on every path: `:221`, `:625`,
`:652`. Each is unconditional after the `!ptr` early return, and there is
no `return` between the allocation and the free in any of the three
functions. No double free, no leak. In `nif_new_index` the
`wrap_index_resource` failure path at `:228` is also clean — `desc` was
already freed at `:221`, and `wrap_index_resource` owns and frees the index
itself (`:176-177`).

`bin->size + 1` at `:168` cannot overflow: `size` is a `size_t` from
`enif_inspect_binary`, and wrapping would require a binary occupying the
entire address space, which is not an allocatable term on any target.

### 4. `nif_reconstruct_batch` — unsigned reasoning survived

The contiguity pre-scan at `:505-514` is byte-identical to master
(`/tmp/master_nif.c:496-506`); it was already above the lock, and the diff
only moved the `rlock` from master `:508` down to current `:531`, past it.
`(uint64_t)keys[i] != (uint64_t)keys[i - 1] + 1u` at `:511` keeps the
increment in unsigned arithmetic, so `INT64_MAX + 1` wraps rather than being
signed-overflow UB. Confirmed intact.

The scan is also correctly ordered relative to validation: `keys_bin.size !=
expected_keys` at `:493` precedes the `(const idx_t *)` cast at `:503`, so
the loop can only read inside the inspected binary.

`result + i * dim` at `:537` cannot index outside `result_bin`:
`safe_mul((size_t)n, (size_t)dim, &nd)` and `safe_mul(nd, sizeof(float),
&result_size)` at `:519` establish `n * dim * 4 <= SIZE_MAX`, hence
`n * dim <= SIZE_MAX / 4 < INT64_MAX`, so the `int64` product `i * dim` does
not signed-overflow, and the highest element written is
`result[(n-1)*dim + dim - 1] == result[n*dim - 1]` — exactly the last float
of the `result_size`-byte allocation at `:524`.

### 5. `nif_compute_residuals`

Both size checks are present after hoisting: `data_bin.size != expected_data
|| keys_bin.size != expected_keys` at `:582`, identical to master
`/tmp/master_nif.c:580`. The allocation at `:587` is
`enif_alloc_binary(expected_data, &result_bin)` — i.e. exactly
`n * dim * sizeof(float)`, which is what `faiss_Index_compute_residual_n`
writes (`n` residuals of `d` floats each). `expected_data` is what gets
allocated; there is no path where `expected_keys` or a stale value is used.

### 6. Lock/unlock balance and the `goto` conversions

All 15 acquisitions pair with exactly one release with no intervening
`return`: `244/246`, `287/290`, `331/336`, `399/404`, `445/448`, `464/466`,
`531/540`, `595/597`, `622/624`, `684/686`, `697/699`, `725/727`, `762/764`,
`871+872 / 889+890` (mirror order preserved), `910/936`.

Both `goto`-converted functions are sound. Every declaration precedes the
lock — `:866-869` in `nif_train_clustering`, `:905-908` in
`nif_get_clustering_centroids` — so no `goto` jumps over an initialization.
`result` is assigned on every path reaching the label: `:875-882` branches
plus the success path `:886` before `unlock:` at `:888`; and `:916-927`
branches plus the success path `:931-934` before `unlock:` at `:935`. No
allocated `ErlNifBinary` escapes unreleased on any error path — `bin` at
`:924` is only reached on success and is transferred by `enif_make_binary`
at `:934`.

### 7. Elixir-side validation: nothing bypasses a previously-existing check

`compute_residuals/3` (`lib/faiss_ex/index.ex:331-341`) is strictly stricter
than before. The new `n_keys != n` raise at `:336` fires on a case that
previously reached C and returned `{:error, "binary size mismatch"}`.
`Shared.encode_ids!/1` raises on non-integers (`lib/faiss_ex/shared.ex:22-26`),
where the deleted `Shared.int64s_to_binary/1` raised the same `ArgumentError`
from binary construction. No check was removed anywhere.

The four binary-input paths remain fully defended, because the byte-size
divisibility check on the Elixir side is a convenience, not the security
boundary — C re-derives and re-validates every size against `res->dim`:

- `add/2` `lib/faiss_ex/index.ex:196-199` → `encode_vector_data!` `:449-461` (`rem(size, dim*4)`) → C re-checks `size == n*dim*4` at `c_src/faiss_ex_nif.c:283`.
- `train/2` `lib/faiss_ex/index.ex:269-272` → same helper → C `:441`.
- `add_with_ids/3` `lib/faiss_ex/index.ex:215-219` → `encode_id_data!` `:465-475` (`rem(size, 8)`) → C `:327` re-checks both `n*dim*4` and `n*8`. The discarded `_n_ids` at `:218` is safe: a vector/ID count mismatch is caught there and returned as `{:error, "binary size mismatch"}`, as the `@doc` at `:205-208` states.
- `reconstruct_binary/2` `lib/faiss_ex/index.ex:316-320` → `encode_id_data!` → C `:493`.

A forged struct cannot produce a mismatched buffer either:
`%Index{index | dim: 999}` makes the Elixir side compute a wrong `n`, but C
validates against `res->dim` obtained from FAISS, not against the Elixir
struct, so the call fails with a size-mismatch error rather than reading out
of bounds.

Deleting `Index.to_binary/1` is confirmed benign: `new/3` `:145`,
`to_file/2` `:350` and `from_file/2` `:365` now hand the term straight to
`enif_inspect_binary`, which returns false for a charlist and yields
`{:error, "invalid arguments"}`. Narrower, but not unsafe.

## Findings

### [WARNING] Out-of-range reconstruct keys reach FAISS's overflowing bounds check — PRE-EXISTING, not introduced by this branch

**Location:** `c_src/faiss_ex_nif.c:533` and `:537` (and `:596` via
`compute_residual_n`); upstream guard at
`~/.cache/faiss_ex/faiss-v1.14.3/faiss/IndexFlatCodes.cpp:79`

**Issue:** Keys are fully caller-controlled `int64` and the NIF validates
only `keys_bin.size`, never the key *values*. FAISS's own guard is
`FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal))`. With
`i0 == INT64_MAX` the `i0 + ni` addition is signed-overflow UB. It is
reachable from pure Elixir with a one-element list —
`Index.reconstruct(index, [9_223_372_036_854_775_807])`: `n == 1`, the
contiguity loop at `:510` never executes, `contiguous` stays `1`, and `:533`
calls `faiss_Index_reconstruct_n(index, INT64_MAX, 1, result)`. The list
`[9_223_372_036_854_775_807, -9_223_372_036_854_775_808]` reaches the same
call via the unsigned wrap at `:511`, which is precisely the wrap the
comment at `:505-508` documents as deliberate.

**Why it matters:** [INFERENCE] on two's-complement targets the overflowed
sum wraps to a negative value that satisfies `<= ntotal`, the guard passes,
and `codes.data() + i0 * code_size` (`IndexFlatCodes.cpp:80`) computes a
wildly out-of-range pointer that `sa_decode` then reads — a VM-crashing heap
overread, or heap bytes leaking into the returned result binary. A caller
passing untrusted vector IDs (a database key, a request parameter) turns
that into a remote crash. This is not fast-path-specific: master's per-key
branch and `compute_residuals/3` funnel into the same
`reconstruct_n(key, 1, ...)`, so the branch does not introduce it — but the
audit of item 4 surfaces it, and the existing tests
(`test/faiss_ex/index_test.exs:124-165`) only cover in-range keys.

**Fix:** Reject the values in the pre-scan, which already walks the array
for free:

```c
int contiguous = 1;
for (ErlNifSInt64 i = 0; i < n; i++) {
    if (keys[i] < 0) return make_error_msg(env, "keys must be non-negative");
    if (i > 0 && (uint64_t)keys[i] != (uint64_t)keys[i - 1] + 1u) contiguous = 0;
}
/* reconstruct_n computes i0 + ni internally; keep it in range */
if (contiguous && keys[0] > INT64_MAX - n) contiguous = 0;
```

Negative keys are meaningless for every index type exposed here, so
rejecting them costs nothing and removes both the UB and the `INT64_MIN`
wrap. Worth a follow-up ticket; not a reason to hold this branch.

### [SUGGESTION] Grouped: three minor points

- **`c_src/faiss_ex_nif.c:931-934`** — term construction moved *inside* the
  read lock. Master built the tuple after `runlock`
  (`/tmp/master_nif.c:928-933`); the single-exit rewrite now runs
  `enif_make_int64` twice and `enif_make_binary` once while holding the
  lock. No correctness or deadlock risk (none of these touch a resource
  lock), but it lengthens a critical section that writers block on, for no
  benefit. If it matters: stash `k`, `d` and `bin` and build the tuple after
  the label returns. Marginal, and it is the natural cost of the constraint
  that the `memcpy` stays inside the lock.
- **`lib/faiss_ex/clustering.ex:89-90`** — the `with` rewrite preserves an
  existing quirk: the NIF returns the authoritative `d` and it is discarded
  as `_d` in favour of the struct's `d`. If the two ever disagree, the
  centroid binary is chunked at the wrong stride and
  `Shared.binary_to_float_rows/2` fails with a `MatchError` out of
  `decode_rows` (`lib/faiss_ex/shared.ex:44-47`) rather than a clear error.
  Using the returned `_d` costs nothing and keeps the NIF as the single
  source of truth.
- **`lib/faiss_ex/index.ex:218` vs `:336`** — `add_with_ids/3` discards
  `_n_ids` and lets C report a count mismatch as `{:error, "binary size
  mismatch"}`, while `compute_residuals/3` now raises `ArgumentError` for
  the equivalent caller bug. Both are documented and both are safe; noted
  only because the two now differ in shape for the same class of mistake.

## Pre-existing, one line each

- `c_src/faiss_ex_nif.c:289, :334-335, :400-403, :503, :591-593` — `(const float *)` / `(const idx_t *)` casts of `ErlNifBinary.data` assume alignment; a byte-offset sub-binary (`Index.add(index, binary_part(bin, 1, dim*4))`) passes the `rem(byte_size, dim*4)` check at `lib/faiss_ex/index.ex:452` and yields an unaligned pointer. Technically UB, harmless on the x86-64/aarch64 targets this builds for since FAISS uses unaligned SIMD loads.
- `lib/faiss_ex/shared.ex:22-24` — `<<i::signed-native-64>>` silently truncates integers outside the int64 range (`2**64 + 5` encodes as `5`) instead of raising; unchanged by this diff.
- `lib/faiss_ex/index.ex:350, :365` — `to_file/2` and `from_file/2` pass arbitrary paths to FAISS with no traversal restriction. Correct for a library (the caller owns path policy) and `contains_nul` at `c_src/faiss_ex_nif.c:617, :645` blocks NUL truncation; recorded so it reads as a conscious boundary rather than an oversight.
- `lib/faiss_ex/index.ex:172-175` — a `:device` option other than `:host` or `{:cuda, id}` raises `CaseClauseError`, unlike the `ArgumentError` the adjacent `:metric` handling raises at `:158-160`.

## Already-known findings — confirmed

- **Confirmed.** `test/faiss_ex/index_test.exs:294` now binds `{:ok, %Index{} = index}`; the `with` conversion of `cpu_to_gpu/2` at `lib/faiss_ex/index.ex:409-419` widening `new/3`'s inferred return is the correct explanation, and re-binding the struct is the minimal fix. No security consequence — the type checker never guarded a memory-safety property here.
- **Confirmed.** Deleting `Index.to_binary/1` narrows charlist tolerance to `{:error, "invalid arguments"}` at the three call sites (`lib/faiss_ex/index.ex:145`, `:350`, `:365`). Behavioural narrowing only; `enif_inspect_binary` rejects the term safely rather than misreading it.

## Tools

`mix sobelow` targets Phoenix and has nothing to analyse here; the OWASP
checklist categories do not apply to this codebase. The tools that would
actually add signal are dynamic:

- An ASan + UBSan build of the NIF driven by the existing suite —
  `-fsanitize=signed-integer-overflow` is what catches the `reconstruct_n`
  warning above; `-fsanitize=alignment` catches the sub-binary cast.
- `mix hex.audit` for the single Hex dependency (`elixir_make`).
