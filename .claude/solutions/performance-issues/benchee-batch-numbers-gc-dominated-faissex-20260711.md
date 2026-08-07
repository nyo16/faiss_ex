---
module: "FaissEx.Index"
date: "2026-07-11"
problem_type: performance_issue
component: benchmarking
symptoms:
  - "benchee: 'add 1000 vectors' (11.1ms) nearly as slow as 'add 10000 vectors' (14.5ms) — costs don't scale with n"
  - "±87% deviation on the batch-add scenario, bimodal timings: same call is either ~1.3ms or ~11-21ms across iterations"
root_cause: "the measured time is dominated by BEAM major-GC pauses copying the ~20MB benchmark corpus held on the process heap (a 10k x 128 list-of-lists), not by the NIF or FAISS — iterations that trigger a major GC pay ~10ms regardless of batch size"
severity: medium
tags: [benchee, benchmarking, gc, nif, list-encoding, bimodal, major-gc]
---

# Benchee numbers for list-API NIF calls can be GC artifacts, not code cost

## Symptoms

- `add 1000 vectors` ≈ `add 10000 vectors` in the README benchmark table —
  nonsensical if cost scaled with n.
- Huge deviation (±87%) with bimodal raw timings; re-running the whole suite
  reproduced the same averages, so it was NOT transient machine noise.

## Investigation

1. **Thermal/background noise** — re-ran full suite: same numbers. Ruled out.
2. **FAISS regression in v1.14.3** — probed the NIF directly with
   pre-encoded binaries: `nif_add_to_index` of 10k vectors is 0.15–0.4ms.
   Ruled out.
3. **Root cause found** — a probe script timing each layer separately
   (`:timer.tc` × 10 reps of: full add / encode-only / NIF-only) showed
   encode-only is bimodal: ~1.3ms OR ~11ms for the same 1000-row input.
   The slow iterations coincide with major GCs copying the 20MB corpus list
   that lives on the same process heap.

## Root Cause

Holding a large list-of-lists corpus on the benchmarking process's heap means
every allocation burst (encoding rows to binaries) can trigger a major GC that
copies the whole corpus. That ~10ms pause lands on whichever iteration
triggered it — flat overhead unrelated to the batch size being measured.

## Solution

Don't "fix" the numbers — understand and label them:

```elixir
# Probe pattern: time each layer separately, 10 reps, look at the SPREAD
{t, _} = :timer.tc(fn -> Shared.encode_vectors!(data, dim) end)      # Elixir layer
{t, _} = :timer.tc(fn -> NIF.nif_add_to_index(idx.ref, n, bin) end)  # C layer
```

- README table kept the honest numbers plus a note: batch-add scenarios are
  dominated by list→binary encoding + GC; NIF-side add of 10k vectors ≈ 0.4ms.
- The real perf fix that DID land: single-pass `encode_vectors!/2` (1 traversal
  instead of 3) took `add 10000` from 56ms → 14.5ms.

### Files Changed

- `README.md` — table + explanatory note (commit 603286d)
- probe script pattern: see scratchpad of `.claude/plans/faiss-upgrade/`

## Prevention

- Specific guidance: before publishing benchee numbers for NIF wrappers,
  sanity-check that cost scales with input size; if not, probe encode vs NIF
  vs GC separately before concluding anything about the C side.
- Bimodal timings + high deviation on the BEAM ≈ suspect GC first.
- A future binary-in/binary-out API (deferred feature) would bypass the
  encode+GC cost entirely for hot paths.

## Related

- `.claude/solutions/otp-issues/nif-rwlock-getters-must-be-dirty-faissex-20260711.md`
