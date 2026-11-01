# Progress — audit-remediation (/phx:full)

State: COMPLETED — v0.4.0 released (hex.pm 0.4.0, GitHub release v0.4.0)
Cycles: 2 of 10 (cycle 1 implementation; cycle 2 review fixes)

| Phase | PR | Notes |
|---|---|---|
| 0 CI hotfix (macos-15) | #8 | |
| 1 Build system | #9 | carried #10 (Phase 2) after a stack-unwind mishap: deleting a base branch via API closed #9 and merged #10 into #9's branch; #9 reopened against master |
| 2 CI + supply chain | #10 → merged into #9 | |
| 3 NIF correctness | #11 | |
| 4 Starvation + OpenMP | #12 | |
| 5 Elixir API + OpenMP-region VM-abort fix | #13 | |
| 6 Test hygiene | #14 | |
| 7 Release v0.4.0 + IVF unwritten-rows fix | #15 | |
| Review fixes (cycle 2) | #16 | triage in `reviews/triage.md` |

Release: tag `v0.4.0` on `0f0f138` → "Publish to Hex" run 41343150930 succeeded (on-master check, tests, hex.audit, hex.build, publish). Master CI green on `0f0f138`.

## Outstanding (needs repo admin; token gets HTTP 403)

- Create the `hex` environment with required reviewers; move `HEX_API_KEY`
  into it and delete the repo-level secret. (The first publish auto-created an
  unprotected `hex` environment — add the protection rules to it.)
- Repository ruleset protecting `refs/tags/v*` (creation/update/deletion).
