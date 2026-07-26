---
name: audit-command-code
description: Independently audit the numerical and statistical implementation of an econometric command. Use after a command specification is frozen, after estimator code changes, or when validation differences may arise from samples, objectives, derivatives, covariance formulas, degrees of freedom, or numerical safeguards.
---

# Audit Command Code

Audit why the Python implementation should be correct without using the implementation itself as its oracle.

## Inputs

- Frozen `validation/commands/<command>/manifest.yaml` and `cases.yaml`.
- Relevant source files and existing tests.
- Data generators and declared mathematical definitions.
- Current Git commit and run ID.

Reject mutable or incomplete specifications. If required data, formulas, dependencies, or source files are unavailable, return `BLOCKED_ENV`.

## Outputs

Write run-scoped evidence:

- `validation/runs/<command>/<run_id>/code_audit.md`
- `validation/runs/<command>/<run_id>/code_findings.json`

When an independent executable oracle is warranted, place it under `validation/commands/<command>/oracles/` and record its SHA256 in the audit. Findings must identify severity, formula or invariant, source location, affected cases, reproduction steps, and disposition.

## Required Audit Surface

1. Trace sample construction, missing values, weights, constants, categorical encoding, sorting, and dropped observations.
2. Trace design matrices, parameter order, constraints, rank checks, and identification failures.
3. Re-derive objective functions, scores, Hessians, estimating equations, and convergence criteria independently.
4. Re-derive conventional, robust, clustered, bootstrap, and model-specific covariance paths plus finite-sample corrections.
5. Check degrees of freedom, absorbed effects, singleton handling, test statistics, p-values, confidence intervals, and returned metadata.
6. Exercise ill-conditioning, scale changes, separation, censoring limits, empty groups, collinearity, overflow, non-convergence, and invalid inputs as applicable.
7. Build independent NumPy/SciPy or direct algebra oracles. The oracle must not import or call Tabra estimator internals.
8. Run relevant unit and oracle tests and preserve commands, versions, seeds, and raw outcomes.

## Decision

Set `code_audit_status` to exactly one of:

- `PASS`: all release-critical invariants and oracle cases pass.
- `FAIL`: at least one confirmed implementation or safety defect exists.
- `BLOCKED_ENV`: a required formula, dataset, dependency, or executable evidence is unavailable.

Warnings cannot hide failed release-critical checks. A skip is not a pass.

## Prohibited Actions

- Do not modify estimator code; a separate repair agent owns fixes.
- Do not copy expected values from Tabra, Stata, or R into an oracle.
- Do not accept visual similarity or rounded console output as evidence.
- Do not relax frozen tolerances or discard non-convergent cases.
- Do not issue the final validation attestation.

## Invocation Order

Run after `prepare-command-spec`, in parallel with `audit-command-theory`. After any repair, rerun this audit from the beginning before engine comparison.
