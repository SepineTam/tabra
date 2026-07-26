# Scientific support policy

`SCIENTIFIC_SUPPORT.csv` is the release ledger for Tabra's research-facing features. A passing unit-test suite proves that the package runs; it does not by itself prove that an estimator is scientifically safe to use.

## Stability levels

| Level | Meaning | User guidance |
| --- | --- | --- |
| Stable | Public API is expected to remain compatible and the advertised numerical results have strong independent validation. | Suitable for research use within the documented assumptions. |
| Beta | The method is implemented and tested, but validation coverage or parity is incomplete. | Use only after checking results against a second package or a published replication. |
| Experimental | Known discrepancies, convergence risks, missing diagnostics, or incomplete validation can affect inference. | Do not use as the sole basis for published results. |

## Evidence grades

| Grade | Minimum evidence |
| --- | --- |
| A | Canonical methodological support, an independent numerical oracle, and reproducible Stata cross-validation. |
| B | Canonical methodological support plus either an independent oracle or substantial cross-validation. |
| C | Deterministic engineering semantics with targeted unit and invariant tests; a research paper is not required. |
| D | Known failed, stale, or incomplete validation that can affect scientific conclusions. |

## Scientific-safety rules

1. Every estimator must have a stable `support_id`, a primary methodological reference, explicit assumptions, and a test file.
2. Every statistic shown to users must have an independent numerical oracle. Formatting tests alone are insufficient.
3. `Stable` estimators must have reproducible Stata comparison inputs and reports. Skipped tests that depend on an unavailable external dataset do not satisfy this gate.
4. A known coefficient, covariance, degrees-of-freedom, convergence, or diagnostic discrepancy immediately prevents `Stable` status.
5. Optimizers must expose convergence state. Failed convergence, non-finite covariance, or invalid identification must raise or produce a prominent warning; they must never look like an ordinary successful result.
6. Data-changing operations must make lossy behavior explicit. Joins, duplicate removal, collapse, winsorization, and reshape require row-count or key-integrity tests.
7. Each pull request that changes a research-facing formula must update the matching CSV row, tests, and verification report in the same change.

## Required supporting comment

Core estimator modules contain a short English comment such as:

```python
# Scientific support: EST-IVPROBIT in SCIENTIFIC_SUPPORT.csv.
```

The comment is intentionally short. The CSV is the single source of truth for the reference, validation evidence, known risks, and release gate, so those details cannot silently drift across multiple code files.

## Current release blockers

- `EST-OLS`: the new Python-Stata-R result layer passes, including extreme-tail
  p-value preservation, but its exact theory mapping and independent code/theory
  audits have not yet produced a clean-run attestation. It therefore remains
  `Beta`, not `Stable`.
- `EST-TOBIT` and `EST-IVTOBIT`: the checked-in Stata report records inference or convergence discrepancies and must be re-run against the current implementation.
- `EST-IVHDFE`: the checked-in report records a clustered multi-absorb crash and incorrect or missing fit and identification statistics.
- `EST-IV-DIAG`: Kleibergen-Paap diagnostics need formula-level oracles and strict parity; very loose tolerances are not acceptable for a research release.

These rows remain `Experimental` even when the ordinary test suite passes.
