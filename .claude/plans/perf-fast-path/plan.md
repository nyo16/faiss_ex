# Plan: Performance — binary fast path + encode/decode/lock cleanups

**Status:** COMPLETE (all 18 tasks; 1.1 resolved as measured-slower-reverted)
**Depth:** standard
**Created:** 2026-07-12
**Input:** `.claude/plans/perf-analysis/perf-report.md` (all 5 findings covered; agent research in `.claude/plans/perf-analysis/research/`)
**Target version:** v0.3.0 (additive API)

## Goal

Remove the measured list-API overhead: end-to-end `Index.add` of 10k×128
vectors is ~14.5ms while the NIF-side add is ~0.4ms — the gap is list→binary
encoding plus BEAM major-GC scanning the caller's list corpus. Ship a binary
fast-path API (the only lever for the big win) plus the four low-effort
encode/decode/lock cleanups.

## Measured baseline (do not re-derive)

- NIF add 10k×128 f32: ~0.4ms · end-to-end list `add`: ~14.5ms
- search 100 queries k=10: ~2.2ms end-to-end (already cheap)
- Bimodal encode timings = major-GC pauses (~10ms) on the ~20MB corpus
- See `.claude/solutions/performance-issues/benchee-batch-numbers-gc-dominated-faissex-20260711.md`

## Non-goals (verified NOT waste — do not "fix")

- `enif_alloc_binary` → `enif_make_binary` pattern (ownership transfer, no
  copy; required for the error path)
- Centroids memcpy under rlock (pointer belongs to the locked object)
- reconstruct contiguity pre-scan (negligible vs its win)
- `ntotal`/`is_trained` getters stay dirty + locked (they read MUTABLE state,
  unlike `dim`)
- Deferred API expansion (`search_with_params`, `remove_ids`, `range_search`)
  — separate follow-up plan

---

## Phase 1 — Elixir encode/decode quick wins [elixir]

- [x] 1.1 **[F2/DO FIRST — MEASURED SLOWER, REVERTED]** Flat-accumulator batch encode
      (`lib/faiss_ex/shared.ex:39-67`): thread ONE growing binary accumulator
      across the whole batch instead of per-row binaries + `Enum.reverse` +
      `IO.iodata_to_binary` flatten. Keep the per-row element count so the
      row-indexed ArgumentError messages are unchanged (existing tests pin
      the exact messages — they must pass untouched).
- [x] 1.2 **[F4]** — added `binary_to_float_rows/2` AND `binary_to_int64_rows/2` (labels at index.ex:190 are s64; both search decode paths now single-pass). Single-pass row decoder: add
      `Shared.binary_to_float_rows(bin, row_len)` (binary-split per row, no
      flat-list-then-chunk). Replace `binary_to_floats |> Enum.chunk_every`
      at `index.ex:189-190` (search), `:249` (reconstruct), `:269`
      (compute_residuals), `clustering.ex:95` (get_centroids). Keep
      `binary_to_floats/1` for flat uses.
- [x] 1.3 **[F5]** — done; raises "id N is not an integer: ..." on bad input; C layer already guards ids/n mismatch ("binary size mismatch"). `Shared.encode_ids!(ids) :: {n, binary}` — one traversal
      replacing `length(keys)` + `int64s_to_binary(keys)` in
      `Index.reconstruct/2` (index.ex:244-245); also use in `add_with_ids/3`.
- [x] 1.4 [test] — test/faiss_ex/shared_test.exs added (also covers
      binary_to_int64_rows/2); 58 passed. Round-trip tests for `binary_to_float_rows/2` (empty, one
      row, many rows, f32 precision) and `encode_ids!/1`. Full suite green.

**Verify:** `mix format`, `mix compile --warnings-as-errors`, `mix test`.

## Phase 2 — C NIF quick wins [c]

- [x] 2.1 **[F3]** — done; d-immutability verified in FAISS v1.14.3 source (only prepend_transform/addIndex mutate d, neither reachable via NIF surface). Cache `int dim` in `IndexResource`: set once in
      `wrap_index_resource` (c_src/faiss_ex_nif.c:148) via `faiss_Index_d`
      (covers new/clone/read/gpu paths — all go through the helper). Replace
      the six under-lock `faiss_Index_d(res->index)` calls in
      add/add_with_ids/train/search/reconstruct_batch/compute_residuals with
      `res->dim`. `nif_get_index_dim` reads `res->dim` lock-free → move it
      back to a normal scheduler (flag 0) and update the nif_funcs comment
      (the "getters are dirty" note now applies only to ntotal/is_trained).
- [x] 2.2 **[SKIP-item folded in]** — done; search critical section is now just faiss_Index_search. With `dim` cached, move search's size
      computation + result `enif_alloc_binary` calls BEFORE the rlock so the
      critical section is just `faiss_Index_search` (alloc-failure paths then
      need no unlock).
- [x] 2.3 — done; locking-convention paragraph also updated. Update the CLAUDE.md dirty-scheduler list (dim getter back to
      normal scheduler; rationale: immutable field, no lock).
- [x] 2.4 [test] — 60 passed (incl. 2 slow stress tests). Full suite + `mix test --include slow` (stress tests must
      still pass — they exercise the changed lock scopes).

**Verify:** `mix compile --warnings-as-errors`, `mix test --include slow`.

## Phase 3 — Binary fast-path API [elixir] (F1 — the centerpiece)

**API decision (recorded in scratchpad):** binary input overloads the
existing mutation functions (n inferred from `byte_size / (dim*4)`); binary
OUTPUT gets an explicit new function (`search_binary/3`) so the return shape
of existing functions never changes based on input type.

- [x] 3.1 — done via shared `encode_vector_data!/2` helper. `Index.add(index, bin) when is_binary(bin)`: validate
      `rem(byte_size(bin), dim*4) == 0` (raise ArgumentError with actual
      sizes otherwise), infer n, call NIF directly — zero traversal of user
      data. Same clause for `Index.train/2`.
- [x] 3.2 — done via `encode_id_data!/1`; ids-binary divisibility raises, count mismatch stays a C-layer {:error, "binary size mismatch"}. `Index.add_with_ids(index, data, ids)` binary clauses: data as f32
      binary and/or ids as s64 binary (mixed list/binary allowed; reuse
      1.3's `encode_ids!` for the list side).
- [x] 3.3 — done. `Index.search_binary(index, queries_bin, k)` returning
      `{:ok, %{n: n, distances: distances_bin, labels: labels_bin}}` —
      raw f32/s64 binaries, zero decode. Document row-major layout and how
      to decode (`Shared`/binary comprehension example).
- [x] 3.4 — done, accepts list or s64 binary keys. `Index.reconstruct_binary(index, keys_bin_or_list)` returning the
      raw f32 binary (skips decode-then-chunk entirely for bulk export).
- [x] 3.5 — done; added vector_data/id_data types, Binary API moduledoc section with decode examples. @spec + docs for all new/extended functions; moduledoc gets a
      **Binary API** section (when to use it: embeddings already arrive as
      binaries from HTTP/DB; layout contract: f32-native row-major).
- [x] 3.6 [test] — 10 new tests in "binary API" describe block; 68 passed. Parity tests: list API and binary API produce identical
      results on the same data (add/train/search/reconstruct); divisibility
      ArgumentError; empty binary no-op; add_with_ids mixed forms.

**Verify:** `mix format`, `mix compile --warnings-as-errors`, `mix test`.

## Phase 4 — Bench, docs, changelog [docs]

- [x] 4.1 — done; measured 352µs binary vs 14.4ms list add (41×), search 100q binary 1.99ms vs 2.14ms. Add bench scenarios: "add 10000 vectors (binary)" and
      "search k=10, 100 queries (binary)" alongside the list versions —
      the table then shows the fast path directly (~0.4ms expected vs
      ~14.5ms list). Re-run `mix run bench/faiss_ex_bench.exs`, refresh the
      README table (keep the GC note for the list rows).
- [x] 4.2 — done; also bumped install snippet to ~> 0.3.0. README: Binary API section with an embeddings example (binary from
      an HTTP client straight into `add/2`); update the "no precompiled
      binaries" perf framing if needed.
- [x] 4.3 — done. CHANGELOG: v0.3.0 entry (binary fast path, encode/decode
      optimizations, lock-free dim getter); bump `@version` to "0.3.0".
- [x] 4.4 — done; mix docs + mix hex.build both clean. CLAUDE.md: Shared module description update (new helpers), Binary
      API mention in data-flow line.

**Verify:** `mix docs` no warnings; `mix hex.build` clean.

## Risks

1. **Binary-append optimization actually engaging in 1.1** — if the flat
   accumulator escapes or is matched wrongly, appends degrade to O(n²).
   Mitigation: bench before/after with the existing "add 1000/10000"
   scenarios; revert to per-row iodata if not faster.
2. **`res->dim` staleness (2.1)** — is `d` truly immutable for every index
   type we can wrap (factory/clone/read/gpu)? FAISS sets `d` at construction
   and never mutates it; verify by grep in FAISS source (`d =` in Index.h
   hierarchy) before relying on it. If any doubt: keep the getter locked but
   still cache for the six internal call sites (they were under lock anyway).
3. **API ambiguity (3.1)** — a charlist like ~c"abc" is a list and could hit
   the list clause with garbage; already true today. Binary clause only
   matches `is_binary` — no new ambiguity. Empty binary → n=0 no-op (matches
   list `[]` behavior).
4. **Return-shape stability** — binary-in on add/train returns `:ok` exactly
   like list-in; search stays list-out unless the caller opts into
   `search_binary/3`. No behavior change for existing callers.

## Execution notes

- Phases 1, 2 are independent; Phase 3 depends on 1.3 only (encode_ids!);
  Phase 4 last.
- Suggested commits: one per phase. Branch + PR (never push master).
- All work on a feature branch, e.g. `perf-binary-fast-path`.
