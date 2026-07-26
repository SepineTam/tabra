---
name: compare-three-engines
description: Compare normalized Python, Stata, and R results under one frozen econometric validation specification. Use after all three engines finish, when classifying cross-engine differences, or when deciding whether a mismatch is implementation, parameterization, sample, inference, or environment related.
---

# Compare Three Engines

Compare complete machine-readable artifacts. Three-engine agreement is evidence, not proof; disagreements are resolved with the code and theory audits rather than majority voting.

## Inputs

- Frozen `manifest.yaml`, `cases.yaml`, and their recorded hashes.
- `python/results.json`, `stata/results.json`, and `r/results.json` from the same run ID.
- All three environment files.
- `code_audit.md`, `theory_audit.md`, and `formula_map.csv`.

If any engine, required metric, paper mapping, data hash, or environment record is missing, return `BLOCKED_ENV`.

## Outputs

- `validation/runs/<command>/<run_id>/comparison.json`
- `validation/runs/<command>/<run_id>/comparison.md`

For every case and metric, record all three values, pairwise absolute and relative differences, frozen tolerance, pass/fail, sample membership equality, and a classification with evidence. Include passing metrics; do not report only mismatches.

## Comparison Procedure

1. Validate schema, command, case, specification hash, data hash, seed, and engine versions.
2. Compare included-row hashes first. Stop metric comparison for any case with unequal samples.
3. Require exact equality for counts, labels, ordering, base categories, omitted parameters, masks, degrees of freedom, and convergence state.
4. Apply the predeclared `atol` and `rtol` to floating metrics without modification.
5. Check coefficient and covariance alignment by canonical parameter identity, not display position alone.
6. Classify mismatches as `SAMPLE`, `PARAMETERIZATION`, `OBJECTIVE`, `VCE_DOF`, `NUMERICAL`, `CONVERGENCE`, `MISSING_OUTPUT`, `ENGINE_ENV`, or `UNRESOLVED`.
7. Use theory and independent-oracle evidence to explain likely ownership. Never infer correctness from a two-to-one vote.
8. Produce a repair checklist, but do not change code.

## Decision

Set `comparison_status` to:

- `PASS`: every release-critical case and metric passes its frozen rule.
- `FAIL`: at least one release-critical comparison fails.
- `BLOCKED_ENV`: evidence is missing, incompatible, skipped, or from different inputs/specifications.

Informational metrics may warn without failing only when the manifest classified them before execution.

## Prohibited Actions

- Do not adjust tolerances after observing values.
- Do not omit NaN, Inf, non-convergence, missing output, or skipped cases.
- Do not overwrite raw engine artifacts.
- Do not repair source code or issue the final attestation.
- Do not describe Stata, R, or Python as ground truth solely because two agree.

## Invocation Order

Run after Python, Stata, and R normalized artifacts exist. On `FAIL`, send findings to a separate analyzer/repair agent. After repair, discard no history: create a new run ID and repeat both audits and all three engines before comparing again.
