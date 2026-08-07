# Scratchpad — perf-fast-path

## Status (2026-07-12)

All 4 phases done on branch `perf-binary-fast-path` (4 commits, one per
phase). Full suite + slow green; mix docs + hex.build clean. Measured:
binary add 10k×128 = 352µs vs 14.4ms list (41×). Review phase: elixir +
C-NIF specialist reports in `reviews/`. Solution doc captured:
`binary-append-accumulator-slower-than-iodata-faissex-20260712.md`.

## Decisions

- **Binary-in overloads existing add/train; binary-out is a NEW function**
  (`search_binary/3`, `reconstruct_binary/2`). Rationale: inferring n from
  byte_size makes binary input unambiguous and ergonomic, but changing the
  RETURN shape of `search/3` based on input type would be surprising —
  explicit opt-in for binary results.
- **n inferred, not passed**: `n = div(byte_size(bin), dim * 4)` after a
  divisibility check. No `{n, bin}` tuple API — one less thing to get wrong.
- **ArgumentError for non-divisible binary** (developer error), consistent
  with the ragged-list convention from v0.2.0.
- **dim cache is safe**: FAISS `Index::d` is set at construction, never
  mutated (verify in FAISS headers during 2.1 — risk item 2).
- **ntotal/is_trained stay dirty+locked** — mutable state, unlike d.
- Findings→tasks map: F1→Phase 3, F2→1.1, F3→2.1, F4→1.2, F5→1.3,
  folded SKIP item→2.2. Non-goals listed in plan header.

## Open questions / verify during execution

- Does the flat-accumulator encode (1.1) actually beat per-row iodata?
  Bench both; the append optimization must engage (keep accumulator
  unshared, only appended).
- reconstruct_binary keys input: accept both list and s64 binary? Plan says
  yes (keys_bin_or_list) — keep symmetric with add_with_ids.

## Dead-ends

- **[1.1] Flat-accumulator encode is SLOWER, reverted.** Measured encode of
  10k×128 (min/median µs over 10 reps):
  - baseline (per-row binaries + reverse + iodata_to_binary): 12716 / 14066
  - per-element append to one flat accumulator: 31650 / 32087 (2.4× slower)
  - hybrid (per-row binary appended to flat accumulator): 33761 / 47796
  Appending to a large writable refc binary costs more per op than building
  small per-row binaries and flattening once. Plan risk #1 mitigation applied:
  kept the original code. Do NOT retry flat accumulators here.

## API Failure — 2026-07-12 13:29

Turn ended due to API error. Check progress.md for last completed task.
Resume with: /phx:work --continue

## API Failure — 2026-07-12 13:34

Turn ended due to API error. Check progress.md for last completed task.
Resume with: /phx:work --continue

## API Failure — 2026-07-12 13:46

Turn ended due to API error. Check progress.md for last completed task.
Resume with: /phx:work --continue
