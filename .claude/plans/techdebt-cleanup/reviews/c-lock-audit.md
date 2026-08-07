# C Lock Audit Review — techdebt-lock-discipline

Scope: `c_src/faiss_ex_nif.c` only (1013 lines, read in full). Diff reviewed
against `master`. All line numbers below are post-change.

## Verdict

PASS

The lock-discipline rewrite is correct. Every one of the 7 required checks
passes with no correctness, safety, or ownership defect found. One
SUGGESTION is recorded — it strengthens the build so the new invariant is
machine-enforced rather than comment-enforced.

## Check 1 — Lock/unlock pairing

**CORRECT.** The file contains exactly 16 acquisitions and 16 releases, and
they pair 1:1 inside a single function with matching read/write flavour
(no `rlock` released via `rwunlock` or vice versa):

Read pairs (10): `244/246` (clone), `399/404` (search), `531/540`
(reconstruct_batch), `595/597` (compute_residuals), `622/624` (write_index),
`684/686` (ntotal), `697/699` (is_trained), `725/727` (cpu_to_gpu),
`762/764` (gpu_to_cpu), `910/936` (get_clustering_centroids).

Write pairs (6): `287/290` (add), `331/336` (add_with_ids), `445/448`
(train_index), `464/466` (reset), `871/890` + `872/889` (train_clustering).

- **No path returns while holding a lock.** Every critical section is
  straight-line between acquire and release. The only non-straight-line one
  is `nif_reconstruct_batch:532-539`, whose loop exits via `break`, not
  `return`. The two `unlock:`-label functions have zero `return` between
  acquisition and label (`847-891`, `895-937`).
- **No double-unlock.** Each `unlock:` label is reached either by
  fall-through or by exactly one `goto`, never both:
  `nif_train_clustering:877, 881` + fall-through from `886`;
  `nif_get_clustering_centroids:918, 922, 926` + fall-through from `933`.
- The 11 dissolved sites are genuinely dissolved, not moved: `add:278-285`,
  `add_with_ids:320-329`, `train_index:436-443`, `reconstruct_batch:517-526`,
  `compute_residuals:575-589` all now sit entirely above their acquisition
  (`287`, `331`, `445`, `531`, `595`), matching the pattern
  `nif_search_index:371-397` already used.
- Lock lifecycle is also sound: `enif_rwlock_create` at `178`/`826` is
  destroyed on the allocation-failure path (`186`, `834`) and in the
  destructors (`91`, `104`). Unchanged by this diff.

## Check 2 — Lock ordering

**CORRECT.** `nif_train_clustering` acquires index then clustering
(`871` `enif_rwlock_rwlock(idx_res->lock)`, `872`
`enif_rwlock_rwlock(clust_res->lock)`) and the `unlock:` label releases in
exact mirror order — clustering first (`889`), then index (`890`). This
matches the documented convention at `27-35`.

`nif_train_clustering` is the **only** two-lock function. Confirmed by
enumerating all 16 acquisition sites (Check 1): every other function has
exactly one acquire/release pair, including `nif_index_cpu_to_gpu`, which
touches two FAISS objects but only locks one (`725`). The two resources in
`train_clustering` are distinct types (`CLUSTERING_RESOURCE_TYPE` at `854`,
`INDEX_RESOURCE_TYPE` at `857`), so self-deadlock by aliasing is impossible,
and `faiss_Clustering_train` (`884`) does not re-enter the NIF layer.

## Check 3 — `goto` correctness

**CORRECT.** In both label functions every declaration is hoisted *above*
the first `goto`, so no jump crosses a declaration at all:

- `nif_train_clustering:867-869` declares `result`, `d`, `ret`, `nd`,
  `expected` before the locks at `871-872`; the first `goto` is at `877`.
- `nif_get_clustering_centroids:905-908` declares `result`, `k`, `d`,
  `centroids_size`, `bin_size`, `centroids`, `bin` before the lock at `910`;
  the first `goto` is at `918`.

`result` is assigned on **every** path reaching each label:
`train_clustering` — `876`, `880`, `886` (ternary covers both `ret` outcomes);
`get_clustering_centroids` — `917`, `921`, `925`, `930`.

Variables read only after their producing call succeeded, so no
uninitialised read: `nd`/`expected` (`875`, short-circuit `||` means the
second `safe_mul` is skipped when the first fails, and both are dead on the
`goto` path); `bin_size` is written only by a successful `safe_mul` (`920`)
and read only at `924`/`929`; `bin` is read only after `enif_alloc_binary`
returned true (`924`); `ret` is never read after the label. `centroids` and
`centroids_size` are pre-initialised at `906-907` before being passed by
address to `faiss_Clustering_centroids` (`914`).

Machine-verified, not just eyeballed:

```
gcc-16 -fsyntax-only -std=c11 -Wall -Wextra -Wno-unused-parameter \
  -Wjump-misses-init -Wmaybe-uninitialized -Wclobbered   → clean (exit 0)
clang -fsyntax-only -std=c11 -Wall -Wextra \
  -Wconditional-uninitialized -Wuninitialized -Wunreachable-code → clean (exit 0)
clang --analyze (default checkers)                       → no diagnostics
```

`-Wjump-misses-init` is precisely the diagnostic for a `goto` skipping an
initialiser, and `-Wconditional-uninitialized` is clang's path-sensitive
check that `result` is set on every path to the label. Both silent.

## Check 4 — The hoisting safety argument

**CORRECT — no BLOCKER.** `res->dim` is assigned in exactly one place in the
whole file: `wrap_index_resource:192`, `res->dim = faiss_Index_d(index)`.
`res->index` is assigned in exactly two: `wrap_index_resource:190` and the
destructor's `res->index = NULL` at `88`, which only runs at GC after the
last reference is gone, when no NIF can be executing against it. There is no
other `->dim =` or `->index =` anywhere in the file.

The five hoisted reads (`278`, `320`, `436`, `517`, `575`) plus the
pre-existing `374` and the lock-free getter `674` are therefore reads of a
write-once field.

Publication is properly ordered: `192` writes `dim` *before* `193`
`enif_make_resource` creates the term and before the term is returned to
Erlang, so any other scheduler thread that can reach the pointer must first
have received the term through the VM, which supplies the barrier. This is a
safely-published immutable field, not an unsynchronised race.

The three functions that could conceivably rotate an index — `nif_clone_index`
(`243-256`), `nif_index_cpu_to_gpu` (`733-750`), `nif_index_gpu_to_cpu`
(`761-774`) — all build a **new** resource via `wrap_index_resource` and
leave the source resource untouched; the Elixir side likewise swaps in a new
`ref` rather than mutating one. `faiss_Index_train` and `faiss_Index_reset`
do not alter `Index::d`.

The invariant comment at `63-66` claims verification against a specific FAISS
revision; `Makefile:12` pins `FAISS_GIT_REV ?= v1.14.3`, which matches. No
drift.

Note there is also no TOCTOU introduced by hoisting: the other input to the
size checks is `data_bin.size` / `keys_bin.size`, which are fixed for the
duration of the NIF call by the caller's term reference.

## Check 5 — Binary ownership across the new boundary

**CORRECT — no leak, no double-release, no use-after-transfer.**

`nif_reconstruct_batch`: allocated at `524`, now above the lock. Between
`524` and the acquisition at `531` there are only two assignments
(`528-529`) and no return, so no path escapes with the binary orphaned.
Exactly two exits follow: `543` `enif_release_binary` on FAISS failure
(after the unlock at `540`, correctly outside the critical section), or `547`
`enif_make_binary` transfer. Mutually exclusive, one each.

`nif_compute_residuals`: allocated at `587`. Lines `591-593` are pointer
initialisations with no return before the lock at `595`. Exits: `599`
release, or `603` transfer. Mutually exclusive.

For completeness, the two untouched sites are also correct:
`nif_search_index` releases `distances_bin` at `395` when the second
allocation fails, releases both at `406-407` on FAISS failure, and transfers
both at `411-412`; `nif_get_clustering_centroids` allocates at `924` and, on
the only path where the allocation succeeded, transfers at `933` — the three
`goto unlock` branches (`918`, `922`, `926`) all precede or are the failure
of the allocation, so `bin` is never live-and-untransferred at the label.

## Check 6 — `dup_cstr`

**CORRECT.** Three call sites, all still `contains_nul`-guarded immediately
before the call, and all `enif_free` unconditionally before branching on the
FAISS return code — so error paths free too:

| site | guard | call | free |
|---|---|---|---|
| `nif_new_index` | `214` | `216` | `221` (before the `ret != 0` check at `223`) |
| `nif_write_index` | `617` | `619` | `625` (before the check at `627`) |
| `nif_read_index` | `645` | `647` | `652` (before the check at `654`) |

The only early return between call and free is the `!desc`/`!path` OOM check
(`217`, `620`, `648`), where nothing was allocated. `nif_write_index` frees
at `625` *after* `enif_rwlock_runlock` at `624`, so the free is outside the
critical section — correct.

Overflow: `bin->size + 1` at `168` is `size_t + 1`, which can only wrap if
`bin->size == SIZE_MAX`. That is byte-for-byte the same arithmetic the old
inline code performed (`desc_bin.size + 1`, `path_bin.size + 1`), so the
extraction introduces **no** new overflow surface, and an `ErlNifBinary`
obtained from a real Erlang term can never reach `SIZE_MAX`. The
`memcpy(s, bin->data, bin->size)` at `170` for a zero-length binary is also
unchanged from master.

The doc comment at `164-166` correctly states both the ownership contract
(caller frees with `enif_free`) and the precondition (guard with
`contains_nul` first).

## Check 7 — `make_ok_atom` removal

**CORRECT.** `grep -c make_ok_atom c_src/faiss_ex_nif.c` → `0`. All eight
former call sites now yield the global directly: `275`, `295`
(`nif_add_to_index`), `317`, `341` (`nif_add_with_ids_to_index`), `433`,
`453` (`nif_train_index`), `471` (`nif_reset_index`), `631`
(`nif_write_index`), plus `886` in `nif_train_clustering`'s ternary.

`atom_ok` is declared at `122` and initialised in `init_atoms:961`.
`init_atoms` is called from `on_load:970` and `on_upgrade:978`, in both cases
*before* `register_resource_types` and before returning `0`. `ERL_NIF_INIT`
at `1012` wires `on_load` as the load callback and `on_upgrade` as the
upgrade callback (`reload` and `unload` are `NULL`). The VM will not dispatch
any NIF in `nif_funcs` unless the load callback returned `0`, so `atom_ok` is
always initialised before any NIF can observe it. Atom terms are immediates
and remain valid across environments, so caching the value from the load env
in a static is the standard idiom.

## Findings

### [SUGGESTION] The new `unlock:` invariant is comment-enforced, not build-enforced

**Location:** `Makefile:48` (`NIF_CFLAGS := -std=c11 -O2 -fPIC -Wall -Wextra -Wno-unused-parameter`)

**Issue:** The branch's stated thesis is that hand-written per-branch unlocks
are unsafe because "nothing in the build or the test suite can catch" a
missed one (`c_src/faiss_ex_nif.c:45-47`). The `unlock:` pattern that replaces
them has its own unenforced precondition: every variable read at or after the
label must be declared above the first `goto`, and `result` must be assigned
on every path reaching it. Today that holds (Check 3), but the next person to
add a `size_t foo = bar();` between `nif_train_clustering:872` and `877`
silently reintroduces the hazard the branch exists to remove.

**Why it matters:** Low likelihood, but the failure mode is a
read-of-uninitialised-`ERL_NIF_TERM` returned to the VM — undefined
behaviour that would very likely not be caught by the test suite either.
The current warning set (`-Wall -Wextra`) does not diagnose it.

**Fix:** Add the two warnings that do, both of which are already clean on
this branch (verified above, exit 0):

```make
NIF_CFLAGS += $(if $(filter %gcc,$(CC)),-Wjump-misses-init,-Wconditional-uninitialized)
```

or simply enable both unconditionally with `-Wno-unknown-warning-option`.
Zero cost today, and it converts the comment at `44-47` into something the
build actually checks.

## Pre-existing (not analysed further)

- `c_src/faiss_ex_nif.c:906` — `k` and `d` are `size_t` but
  `faiss_Clustering_k`/`_d` return `int`; harmless because
  `nif_new_clustering:817-818` rejects non-positive values. Unchanged by this
  diff. PRE-EXISTING.
- `c_src/faiss_ex_nif.c:489-501` — `nif_reconstruct_batch` validates the keys
  binary before its `n == 0` shortcut while its siblings check `n == 0`
  first; a cosmetic error-precedence inconsistency, not a defect. Unchanged
  by this diff. PRE-EXISTING.
