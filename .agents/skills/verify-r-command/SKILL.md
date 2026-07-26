---
name: verify-r-command
description: Execute the frozen validation cases in an independent R implementation and emit normalized, reproducible results. Use as the R leg of Python-Stata-R validation after command specification, or when R package versions, mappings, defaults, or output extraction need verification.
---

# Verify R Command

Run R as an independent result engine. Execution and normalization are the only responsibilities of this skill.

## Inputs

- Frozen `manifest.yaml` and `cases.yaml`.
- `validation/commands/<command>/r/run.R` or permission to create it.
- Identical input data and expected SHA256 values.
- Project `renv.lock` and required R package/function mappings.
- Run ID and current Git commit.

## Outputs

Write only inside the run directory:

- `validation/runs/<command>/<run_id>/r/results.json`
- `validation/runs/<command>/<run_id>/r/environment.json`
- `validation/runs/<command>/<run_id>/r/raw/` for logs or package-native artifacts

The normalized JSON must include schema version, command, case ID, engine, sample hash, included-row hash, convergence state, warnings, coefficient names/order, every required metric, missing metrics, elapsed time, and error details. Preserve full numeric precision.

## Workflow

1. Check `Rscript`, `renv.lock`, required package versions, and package availability without changing the environment.
2. Verify input bytes and row IDs against the frozen hash before fitting.
3. Invoke explicit R functions and options from the manifest; never accept package defaults unless the manifest records them.
4. Capture warnings, convergence codes, rank, omitted parameters, sample membership, and all required statistics.
5. Normalize results to JSON using stable metric names and explicit `null` for unavailable values.
6. Validate JSON shape and record the R session and package versions.

## Failure Conditions

- `BLOCKED_ENV`: R/Rscript, `renv`, a locked package, required data, or the declared R reference implementation is unavailable.
- `FAILED`: a case crashes, hashes differ, output is malformed, a required metric is absent, or expected convergence is not achieved.
- `COMPLETE`: every case executed and normalized; this is not a scientific pass decision.

Any skipped R case must be represented explicitly and prevents `PASSED` downstream.

## Prohibited Actions

- Do not install or upgrade R or packages without explicit authorization.
- Do not compare engines, explain discrepancies, modify Python code, or issue an attestation.
- Do not replace missing values with zeros or parse rounded table text when structured values are accessible.
- Do not alter data, seeds, options, or tolerances.

## Invocation Order

Run only after the specification is frozen and both code and theory audits complete. It may run alongside Python and Stata runners. Pass normalized artifacts to `compare-three-engines` only when all three engines completed.
