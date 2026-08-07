# Dependency Audit — faiss_ex

Date: 2026-07-12 | Scope: Hex deps, FAISS source pin, toolchain assumptions

## Dependencies Health Score: 92/100

Minimal, correctly-scoped dep tree with the FAISS pin current and in sync; only trivial version lag and a mutable-tag pin keep it off the high 90s.

## Summary of clean areas (one line each)

- Hex deps (3 direct): all actually used, all correctly scoped — `elixir_make` (`runtime: false`, build-only), `ex_doc` and `benchee` (`only: :dev, runtime: false`).
- `mix hex.audit`: no retired packages (established prior to this audit).
- FAISS pin sync: `FAISS_GIT_REV = v1.14.3` matches in `Makefile` (line 11) and `mix.exs` `make_env/0` (line 72), per project convention.
- FAISS currency: v1.14.3 is the **latest upstream release** (June 13, 2026); no known CVEs for FAISS itself in 1.14.x (a Flowise path-traversal CVE mentions Faiss vector stores but does not affect the FAISS library).
- Compile health: clean with `--warnings-as-errors` (established prior to this audit).
- Toolchain: `elixir: "~> 1.18"` requirement vs. local Elixir 1.20 / OTP 29 — consistent; dirty schedulers and `ErlNifRWLock` used by the NIF are available on any OTP an Elixir 1.18+ install implies.

## Issues Found

### 1. FAISS pinned to a mutable git tag, not a commit SHA — Low

`FAISS_GIT_REV` defaults to the tag `v1.14.3`, and the Makefile clones with `git clone --depth 1 --branch $(FAISS_GIT_REV)` (Makefile line 140) with no commit-SHA or checksum verification. Tags are mutable refs: a re-pointed or compromised tag would silently change what gets compiled into the NIF. Practical risk is low (Meta-maintained repo, build cached at `~/.cache/faiss_ex/`), but pinning the commit SHA (or verifying it post-clone) would make the source build reproducible and tamper-evident.

Files: `/Users/nmaroulis/Source/faiss_ex/Makefile`, `/Users/nmaroulis/Source/faiss_ex/mix.exs`

### 2. Minor version lag on all three Hex deps — Low

`mix hex.outdated`:

| Dep | Current | Latest | Notes |
|---|---|---|---|
| elixir_make | 0.9.0 | 0.10.0 | Satisfies existing `~> 0.9` requirement; the only runtime-adjacent (build) dep, worth bumping |
| ex_doc | 0.40.1 | 0.40.3 | dev-only, patch lag |
| benchee | 1.5.0 | 1.5.1 | dev-only, patch lag |

All updates are non-breaking per requirements; a `mix deps.update --all` clears the board.

### 3. No OTP minimum declared — Low

`mix.exs` declares `elixir: "~> 1.18"` but no OTP floor, and the package ships a NIF built against `ERL_EI`/`erts` headers of whatever OTP compiles it. Not a defect (NIFs are compiled per-install, not prebuilt), but a documented OTP minimum in README/mix.exs would preempt support questions.

File: `/Users/nmaroulis/Source/faiss_ex/mix.exs`

## Tooling Recommendation

**mix_audit / sobelow: not worth adding as project deps.** With three dev/build-only Hex deps and no web surface, `mix deps.audit` would scan almost nothing and sobelow (Phoenix-focused) is inapplicable. The real supply-chain surface is the FAISS source build, which neither tool covers — the commit-SHA pin (Issue 1) is the higher-value hardening. If CI belt-and-braces is desired, run `mix hex.audit` in CI (already available, zero deps) rather than adding mix_audit.
