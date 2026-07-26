---
name: prepare-command-spec
description: Freeze a reproducible validation contract for one econometric command before any comparison or repair. Use when adding a command to the scientific validation pipeline, expanding supported options, changing estimator behavior, or preparing Python, Stata, and R cross-validation cases.
---

# Prepare Command Spec

Create the immutable contract that every later audit and engine run must follow. Do not inspect observed differences before setting metrics and tolerances.

## Inputs

- Command name and public Python API.
- Relevant implementation, documentation, and existing tests.
- Intended Stata command and R package/function mappings.
- Supported model variants, options, VCE modes, and returned statistics.
- Candidate datasets or deterministic data generators.
- Primary methodological sources, if already known.

If the command, Stata/R mapping, required primary source, or usable data cannot be established, record `BLOCKED_ENV`; do not produce a reduced specification.

## Outputs

Write only command-level specifications:

- `validation/commands/<command>/manifest.yaml`
- `validation/commands/<command>/cases.yaml`
- `validation/commands/<command>/data/README.md` when data provenance needs explanation

`manifest.yaml` must contain:

- specification version and command identifier;
- Python, Stata, and R invocation mappings;
- supported variants and deliberately unsupported behavior;
- required outputs, with type, units, names, and ordering;
- sample-selection, missing-value, constant, weights, base-category, and convergence rules;
- VCE, small-sample correction, and degrees-of-freedom definitions;
- per-metric comparison rules fixed in advance;
- required environment and primary-source identifiers;
- release-critical versus informational metrics.

`cases.yaml` must give every case a stable ID, deterministic seed, data reference and SHA256 expectation, engine-specific arguments, expected convergence status, and tags such as `basic`, `vce`, `edge`, `identification`, or `numerical-stress`.

## Workflow

1. Inventory the actual public API and implementation branches.
2. Enumerate the Cartesian dimensions that alter estimates, samples, inference, or output.
3. Select a minimal covering set of normal, simulated, edge, and failure cases.
4. Define engine mappings explicitly; never rely on defaults being equivalent.
5. Define metrics before execution. Counts, labels, masks, and degrees of freedom require exact equality. Floating metrics require declared `atol` and `rtol` justified by numerical behavior.
6. Record data origin, license, generator version, seed, and hash.
7. Validate the YAML structure and freeze its content hash for the run.

## Failure Conditions

- `BLOCKED_ENV`: R, Stata, a required package, a primary paper, licensed data, or deterministic test data is unavailable.
- `BLOCKED_SPEC`: the public behavior or engine mapping is ambiguous and cannot be resolved from authoritative material.
- `FAILED`: cases omit a material option, tolerances are selected after seeing results, or hashes/seeds are missing.

A skipped case is never a passing case.

## Prohibited Actions

- Do not modify source code or tests.
- Do not run comparisons or tune tolerance to observed output.
- Do not assume Stata, R, or Python is ground truth.
- Do not silently remove a case because an engine cannot run it.
- Do not overwrite a previously used specification; increment its version.

## Invocation Order

Run first. On `SPEC_READY`, run `audit-command-code` and `audit-command-theory`. Only after both audits are complete may engine runners start.
