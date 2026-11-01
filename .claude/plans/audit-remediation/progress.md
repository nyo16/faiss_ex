# Progress — audit-remediation (/phx:full)

State: COMPLETED (review cycle partial — see below) → COMPOUNDING done
Cycle: 1 of 10

| Phase | Status | PR | CI |
|---|---|---|---|
| 0 CI hotfix (macos-15) | done | #8 → master | 6/6 green |
| 1 Build system | done | #9 → #8 | covered by #10 run |
| 2 CI + supply chain | done | #10 → #9 | 9/9 green |
| 3 NIF correctness | done | #11 → #10 | covered by #12 run |
| 4 Starvation + OpenMP | done | #12 → #11 | 9/9 green |
| 5 Elixir API (+ VM-abort fix) | done | #13 → #12 | covered by #14 run |
| 6 Test hygiene | done | #14 → #13 | 9/9 green |
| 7 Release v0.4.0 (+ IVF unwritten-rows fix) | done | #15 → #14 | 9/9 green (pre-fix push); re-run pending |

Merge order: #8 → #9 → … → #15, then tag `v0.4.0`.

## Review (cycle 1)

Panel: C/NIF reviewer, elixir-reviewer, security-analyzer — **all three hit an
account rate limit (429, retry-after ≈ 2.6 h) before writing reports.** The C
reviewer's last progress line reported one finding (IVF `reconstruct_n` leaks
uninitialized memory); reproduced and fixed in #15. The omp-throw sweep it was
about to run was done by hand (IVF/HNSW search capture exceptions; Clustering
has no omp; Flat omp regions are non-throwing kernels).

Not reviewed by a specialist yet: Elixir API diff, security of Makefile/CI
diff. Re-run `/phx:review` after the limit resets.

## User actions outstanding

- GitHub: create the `hex` environment with required reviewers, move
  `HEX_API_KEY` into it, add a tag-protection rule for `v*` (before tagging).
- Merge the stack in order.
