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

- `validation/runs/<command>/<run_id>/comparison.json`: canonical machine
  evidence created only by the product `compare` CLI. The tri-reporter has
  read-only access after creation.
- `validation/runs/<command>/<run_id>/comparison.md`: independent review written
  by the tri-reporter from the read-only JSON, audits, and environment evidence.

Only the product CLI may create `comparison.json`:

```bash
uv run python -m tabra.validation compare \
  --manifest validation/commands/<command>/manifest.yaml \
  --result validation/runs/<command>/<run_id>/python/results.json \
  --result validation/runs/<command>/<run_id>/stata/results.json \
  --result validation/runs/<command>/<run_id>/r/results.json \
  --output validation/runs/<command>/<run_id>/comparison.json
```

The CLI refuses to overwrite an existing comparison. Create a new run ID
instead of deleting, repairing, or replacing an existing canonical artifact.

For every case and metric, record all three values, pairwise absolute and relative differences, frozen tolerance, pass/fail, sample membership equality, and a classification with evidence. Include passing metrics; do not report only mismatches.

## Comparison Procedure

1. Validate schema, command, case, specification hash, data hash, seed, and engine versions.
2. Compare included-row hashes first. Stop metric comparison for any case with unequal samples.
3. Require exact equality for counts, labels, ordering, base categories, omitted parameters, masks, degrees of freedom, and convergence state.
4. Apply the predeclared `atol` and `rtol` to floating metrics without modification.
5. Check coefficient and covariance alignment by canonical parameter identity, not display position alone.
6. Classify mismatches as `SAMPLE`, `PARAMETERIZATION`, `OBJECTIVE`, `VCE_DOF`, `NUMERICAL`, `CONVERGENCE`, `MISSING_OUTPUT`, `ENGINE_ENV`, or `UNRESOLVED`.
7. Use theory and independent-oracle evidence to explain likely ownership. Never infer correctness from a two-to-one vote.
8. Produce a repair checklist in `comparison.md`, but do not change code or the
   canonical JSON.

## Decision

Set `comparison_status` in `comparison.md` only to:

- `PASS`: every release-critical case and metric passes its frozen rule.
- `FAIL`: at least one release-critical comparison fails.
- `BLOCKED_ENV`: evidence is missing, incompatible, skipped, or from different inputs/specifications.

Informational metrics may warn without failing only when the manifest classified them before execution.

`comparison_status`, `reporter_checks`, classifications, environment summaries,
audit summaries, explanations, and repair checklists belong only in Markdown.
They must never be added to `comparison.json`.

## Prohibited Actions

- Do not adjust tolerances after observing values.
- Do not omit NaN, Inf, non-convergence, missing output, or skipped cases.
- Do not overwrite raw engine artifacts.
- Do not use Write, Edit, apply_patch, `jq`, or any post-processing operation on
  `comparison.json`. It is immutable after the product CLI creates it.
- Do not temporarily extend the canonical JSON with reviewer fields. Any future
  product field requires a deterministic `compare_results()` change and schema
  upgrade so the Gate can recompute the same complete object.
- Do not repair source code or issue the final attestation.
- Do not describe Stata, R, or Python as ground truth solely because two agree.

## Invocation Order

Run after Python, Stata, and R normalized artifacts exist. The required order is:

1. Run the product `compare` CLI once to create `comparison.json`.
2. Read that JSON and write the separate `comparison.md` review.
3. Run the product integrity check:

   ```bash
   uv run python -m tabra.validation check-comparison \
     --manifest validation/commands/<command>/manifest.yaml \
     --comparison validation/runs/<command>/<run_id>/comparison.json \
     --result validation/runs/<command>/<run_id>/python/results.json \
     --result validation/runs/<command>/<run_id>/stata/results.json \
     --result validation/runs/<command>/<run_id>/r/results.json
   ```

4. Only an exit code of 0 permits `COMPARE_PENDING -> COMPARE_READY ->
   GATE_PENDING`.

If the integrity check fails, terminate the current run without overwriting its
JSON and do not call the Gate. Send findings to a separate analyzer/repair agent,
then create a new run ID and repeat both audits and all three engines. Discard no
history.
