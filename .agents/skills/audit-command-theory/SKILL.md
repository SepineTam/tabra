---
name: audit-command-theory
description: Audit an econometric command against primary methodological literature and map mathematical formulas to implementation and tests. Use after specification, after behavior changes, or when assumptions, parameterization, inference, identification, or interpretation require scientific verification.
---

# Audit Command Theory

Establish what the estimator is mathematically supposed to compute and where each claim is implemented and tested.

## Inputs

- Frozen command `manifest.yaml` and `cases.yaml`.
- Source files, public documentation, and tests.
- Primary papers, books by the method's authors, official errata, or authoritative software manuals needed to resolve parameterization.
- Current Git commit and run ID.

If a required primary source cannot be accessed or a data-generating assumption cannot be established, return `BLOCKED_ENV`; do not substitute an uncited secondary summary.

## Outputs

Write:

- `validation/commands/<command>/theory.md`
- `validation/commands/<command>/formula_map.csv`
- `validation/runs/<command>/<run_id>/theory_audit.md`
- `validation/runs/<command>/<run_id>/theory_findings.json`

Each `formula_map.csv` row must include `formula_id`, citation and stable URL/DOI, page/equation, assumptions, mathematical expression or precise description, parameterization, code location, test or case ID, implementation status, and limitations.

## Workflow

1. Define the population model, sampling assumptions, identification conditions, objective or likelihood, estimator, and asymptotic target.
2. Map every reported coefficient, marginal quantity, covariance estimator, test statistic, and diagnostic to a source.
3. Resolve sign conventions, link functions, scale normalizations, censoring/truncation definitions, base categories, fixed-effect transformations, and small-sample corrections.
4. Trace each release-critical formula to exact source code and at least one independent test or validation case.
5. Document cases where software implements only a subset or extension of the cited method.
6. State user-facing assumptions and conditions under which causal or structural interpretation is not justified.
7. Record citation metadata precisely; distinguish primary theory from software-compatibility references.

## Decision

Set `theory_audit_status` to:

- `PASS`: every release-critical mathematical claim has primary support and a complete code/test mapping.
- `FAIL`: the implementation conflicts with the cited model, an assumption is misrepresented, or a critical formula is unmapped.
- `BLOCKED_ENV`: a required paper, authoritative definition, licensed data description, or source location is unavailable.

A missing citation or skipped formula can never be marked passing.

## Prohibited Actions

- Do not treat Stata or R output as mathematical proof.
- Do not cite blogs, generated prose, or secondary tutorials as the sole authority.
- Do not invent equation numbers, assumptions, or citations.
- Do not modify estimator code or issue the final gate decision.
- Do not downgrade a discrepancy to stylistic without a mathematical argument.

## Invocation Order

Run after `prepare-command-spec`, independently of `audit-command-code`. Both audits must pass before the result layer may receive a passing gate decision. Rerun after repairs that alter formulas, samples, inference, or output meaning.
