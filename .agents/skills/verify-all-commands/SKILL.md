---
name: verify-all-commands
description: Autonomously orchestrate reproducible scientific validation across a registry of econometric commands using specification, independent code and theory audits, Python-Stata-R execution, comparison, repair loops, and an independent gate. Use for unattended command-by-command verification, resuming interrupted validation, or rebuilding attestations after code changes.
---

# Verify All Commands

Coordinate work; do not collapse specialist roles into one agent. Persist every transition so the pipeline can resume safely.

## Inputs

- `validation/registry.yaml` with command priority, owner, enabled flag, and latest attestation.
- Command specifications under `validation/commands/<command>/`.
- Available specialist agents and engine executables.
- Current Git commit, retry limit, and optional command selection.

## Outputs and Run Layout

Create a unique, immutable `validation/runs/<command>/<run_id>/` containing:

- `state.json` and append-only `events.jsonl`;
- `environment.json` and `inputs.json`;
- code and theory audit artifacts;
- `python/`, `stata/`, and `r/` normalized results;
- comparison artifacts;
- `attestation.json` when gated.

Never reuse a run ID after code, specification, data, package, or environment changes.

Also update `validation/registry.yaml` with the terminal run ID and status. Update a command's latest accepted attestation only when the terminal status is `PASSED`.

## State Machine

Use only these transitions:

```text
DISCOVERED -> SPEC_PENDING
SPEC_PENDING -> SPEC_READY | BLOCKED_SPEC | BLOCKED_ENV
SPEC_READY -> AUDIT_PENDING
AUDIT_PENDING -> AUDITS_READY | NEEDS_REPAIR | BLOCKED_ENV
AUDITS_READY -> ENGINE_PENDING
ENGINE_PENDING -> ENGINES_READY | FAILED_EXECUTION | BLOCKED_ENV
ENGINES_READY -> COMPARE_PENDING
COMPARE_PENDING -> COMPARISON_READY | NEEDS_REPAIR | BLOCKED_ENV
COMPARISON_READY -> GATE_PENDING
GATE_PENDING -> PASSED | FAILED | BLOCKED_ENV
NEEDS_REPAIR -> REPAIR_ANALYSIS
REPAIR_ANALYSIS -> REPAIR_IMPLEMENTATION | FAILED
REPAIR_IMPLEMENTATION -> REPAIRED | FAILED
REPAIRED -> RERUN_PENDING
RERUN_PENDING -> AUDIT_PENDING
FAILED_EXECUTION -> NEEDS_REPAIR | BLOCKED_ENV
```

`PASSED`, `FAILED`, `BLOCKED_SPEC`, and `BLOCKED_ENV` are terminal for that run. Resume environmental work by creating a new run after the missing condition changes.

## Orchestration Procedure

1. Lock one registry command and record Git, spec, data, and environment identities.
2. Run `prepare-command-spec`. Missing R, Stata, required paper, package, or data is `BLOCKED_ENV`; ambiguity in an otherwise available specification is `BLOCKED_SPEC`.
3. Spawn separate `code-auditor` and `theory-auditor` roles. Both must complete; neither may edit product code.
4. Run Python and Stata runners plus `r-runner` against identical cases. Preserve explicit skipped/error records.
5. Send normalized artifacts to `tri-reporter`; do not let runners compare their own outputs.
6. If evidence identifies a fixable defect, send it first to an analysis role and then to a separate repair role. Never let a repair agent approve its work.
7. After any repair, create a new run ID and rerun code audit, theory audit, all three engines, and comparison. Prior evidence remains immutable.
8. Send complete evidence to `gatekeeper` only once per run.
9. Update the registry's latest attestation only from `PASSED`; otherwise record the terminal state and reason.
10. Release the lock and select the next enabled command.

## Retry and Blocking Rules

- Default to three repair attempts per command unless the registry specifies fewer.
- Exhausted attempts end in `FAILED`, not pass.
- A missing R/Stata executable, locked package, required paper, or required data ends in `BLOCKED_ENV` immediately.
- Any skipped case or metric prevents `PASSED`.
- Never silently continue with Python-Stata only or a smaller case set.
- On interruption, read `state.json`, verify artifact hashes, and continue only from a legal transition.

## Failure Conditions

- `BLOCKED_ENV`: a required engine, package, primary paper, dataset, license, hash, or mandatory evidence is unavailable.
- `BLOCKED_SPEC`: available authoritative material does not resolve the command contract.
- `FAILED_EXECUTION`: an available engine or case fails to execute; route only to repair or an evidenced environment block.
- `FAILED`: audits, comparisons, the independent gate, or exhausted repair attempts prove the command cannot be accepted.
- `PASSED`: only the independent gate may assign this state after complete three-layer evidence.

## Role Separation

- `code-auditor`: source and independent-oracle audit; no repairs.
- `theory-auditor`: primary literature and formula mapping; no repairs.
- engine runners: execution and normalization only.
- `tri-reporter`: comparison only; no repairs or gate decision.
- analyzer and repair agent: diagnosis and code changes only.
- `gatekeeper`: immutable final decision only.

## Prohibited Actions

- Do not let one agent define tolerances, repair code, and approve the same run.
- Do not overwrite runs, reports, raw outputs, or attestations.
- Do not treat skip, warning, partial execution, or two-engine agreement as pass.
- Do not alter frozen cases or tolerances after results are visible.
- Do not merge, publish, or change stable status without a `PASSED` attestation and user-authorized workflow.

## Invocation Order

`prepare-command-spec` -> (`audit-command-code` + `audit-command-theory`) -> Python/Stata/R runners -> `compare-three-engines` -> `validation-gate`. Repairs branch only from `NEEDS_REPAIR` and always loop back through both audits.
