# `reg` three-engine validation pilot

This directory is a reproducible pilot, not yet an attestation. A passing
attestation requires Tabra, Stata, and R to complete on the exact same CSV and a
separate comparison gate to accept every frozen metric.

## Generate the fixture

From the repository root:

```bash
uv run python validation/commands/reg/data/generate.py
```

The generator writes `reg_fixture.csv`, `reg_fixture.csv.sha256`, and
`reg_fixture.csv.metadata.json`. Every runner computes SHA-256 before fitting;
editing only the CSV or only its sidecar causes a hard failure.

## Run each engine

```bash
uv run python validation/commands/reg/python/run.py \
  --data validation/commands/reg/data/reg_fixture.csv \
  --output validation/runs/reg/<run_id>/python/results.json

STATA_CMD=/path/to/stata-mp uv run python validation/commands/reg/stata/run.py \
  --data validation/commands/reg/data/reg_fixture.csv \
  --output validation/runs/reg/<run_id>/stata/results.json

RSCRIPT_CMD=/path/to/Rscript uv run python validation/commands/reg/r/run.py \
  --data validation/commands/reg/data/reg_fixture.csv \
  --output validation/runs/reg/<run_id>/r/results.json
```

`STATA_CMD` may contain executable arguments. `RSCRIPT_CMD` must name an
executable. If either engine is absent, its runner exits nonzero; the validation
must remain incomplete. R uses only base `stats`, but `renv.lock` still freezes
the expected R version. The orchestration layer must record actual versions and
must not silently substitute another version for a release attestation.

On the current development machine, the tested executables are Stata 19.5 at
`/usr/local/bin/stata-mp`, R 4.4.1 at `/usr/local/bin/Rscript`, and
`stata-mcp` 1.21.1 through `/opt/homebrew/bin/uvx`. These paths are local
observations, not portable defaults; every run must capture its actual versions.

## Supported scope

- `basic`: constant plus two regressors, classical homoskedastic covariance.
- `no_constant`: two regressors, uncentered no-constant fit statistics.
- `robust`: explicitly blocked because Tabra `reg` does not currently expose or
  implement a robust covariance option.

The files named `.yaml` contain strict JSON. This is intentional: pipeline code
can parse them with Python's standard `json` module while they remain valid
YAML 1.2 documents.
