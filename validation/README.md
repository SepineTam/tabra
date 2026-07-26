# Scientific validation infrastructure

This directory is the evidence layer for Tabra's econometric commands. It is
deliberately separate from ordinary unit tests: unit tests protect code behavior,
while this pipeline asks whether that behavior is mathematically supported and
reproducible across Python, Stata, and R.

## Acceptance layers

1. **Frozen specification** — cases, datasets, outputs, and tolerances are fixed
   before engine results are observed.
2. **Independent code audit** — implementation formulas, samples, degrees of
   freedom, covariance paths, and failure behavior are checked against an oracle
   that does not call the estimator internals.
3. **Independent theory audit** — primary literature is mapped to formulas, code,
   tests, assumptions, parameterization, and limitations.
4. **Three-engine execution** — Python, Stata, and R run the same frozen bytes.
5. **Comparison** — exact fields and predeclared floating tolerances are checked
   pairwise; there is no two-against-one majority rule.
6. **Independent gate** — evidence is rehashed and recomputed. Missing engines,
   skipped cases, missing audits, stale reports, or dirty release inputs cannot
   produce a passing attestation.

## Command-line entry point

From the repository root, either form is supported:

```bash
uv run tabra-validate preflight --registry validation/registry.yaml
uv run python -m tabra.validation preflight \
  --manifest validation/commands/reg/manifest.yaml
```

The pilot runners are documented in `commands/reg/README.md`. A typical result
comparison is:

```bash
uv run tabra-validate compare \
  --manifest validation/commands/reg/manifest.yaml \
  --result validation/runs/reg/<run_id>/python/results.json \
  --result validation/runs/reg/<run_id>/stata/results.json \
  --result validation/runs/reg/<run_id>/r/results.json \
  --output validation/runs/reg/<run_id>/comparison.json
```

The gate additionally requires `--run-id`, `--code-audit`, and
`--theory-audit`. Its output path is immutable: an existing attestation is never
overwritten, even by an error report.

## Autonomous agent workflow

Use the `verify-all-commands` skill as the coordinator. It delegates to separate
code-auditor, theory-auditor, R-runner, comparison, analysis/repair, and
gatekeeper roles. The corresponding skill contracts are under `.agents/skills/`
and role definitions under `.codex/agents/`.

Every autonomous run must use a new
`validation/runs/<command>/<run_id>/` directory and retain raw results, normalized
results, environments, findings, comparison, and terminal decision. Never call a
missing engine a skip that passed.

## Current pilot boundary

`reg` covers ordinary least squares with and without a constant using classical
homoskedastic covariance. Robust covariance is explicitly blocked. Successful
Python-Stata-R numerical agreement is only the result layer; it is not a full
scientific attestation until both independent audits and the final gate pass.
