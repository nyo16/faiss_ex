---
module: "FaissEx.Shared"
date: "2026-07-12"
problem_type: performance_issue
component: encoding
symptoms:
  - "rewriting a per-row-binaries + IO.iodata_to_binary encoder to thread one growing binary accumulator made it 2.4x SLOWER (13.4ms -> 31.7ms for 10k x 128 floats)"
  - "hybrid variant (per-row binary appended onto one flat accumulator) was worse still: 33.8ms min with GC-heavy 47.8ms median"
root_cause: "the ERTS binary-append optimization avoids O(n^2) copying but each append to a large writable refc binary still costs more per operation than building small per-row binaries on the heap and doing one final iodata_to_binary flatten"
severity: low
tags: [binary-append, iodata, refc-binary, encoding, benchmarking, efficiency-guide]
---

# Threading one growing binary accumulator can be SLOWER than per-row iodata

## Symptoms

- Expected win from "one accumulator, no reverse, no flatten" did not appear;
  both flat-accumulator variants measured 2.4–2.5× slower than the existing
  per-row-binaries + `Enum.reverse` + `IO.iodata_to_binary` design.

## Investigation

Probe: `:timer.tc` × 10 reps of `Shared.encode_vectors!(data, 128)` on a
10k×128 corpus, comparing min/median (min is the GC-free signal):

| variant | min µs | median µs |
|---|---|---|
| per-row binaries + reverse + iodata_to_binary (baseline) | 12,716 | 14,066 |
| per-element append to one flat accumulator | 31,650 | 32,087 |
| per-row binary appended to flat accumulator | 33,761 | 47,796 |

The append optimization DID engage (timings would be seconds, not ms, if it
degraded to O(n²)) — the per-append overhead on a multi-MB writable refc
binary is simply higher than heap-binary construction plus one final flatten.

## Root Cause

"Avoid intermediate lists, thread one binary" is not automatically faster.
Small per-row binaries are cheap heap allocations; `IO.iodata_to_binary` is
one alloc + memcpy. A growing refc binary pays bs_append bookkeeping on every
element and creates GC pressure as it reallocates.

## Solution

Keep the per-row iodata design (see `lib/faiss_ex/shared.ex`,
`encode_vectors!/2`). The comment there and CLAUDE.md now record the
measurement so it is not "optimized" again.

## Prevention

- Never land an encode/accumulator rewrite without a before/after probe on
  realistic sizes; compare MIN times (medians absorb major-GC pauses).
- If lists must be traversed element-wise, the traversal itself (~13ms for
  1.28M floats) is the floor — the real fix for hot paths is accepting
  binaries directly (FaissEx v0.3.0 Binary API), not a cleverer encoder.

## Related

- `.claude/solutions/performance-issues/benchee-batch-numbers-gc-dominated-faissex-20260711.md`
- `.claude/plans/perf-fast-path/scratchpad.md` (dead-end entry with raw numbers)
