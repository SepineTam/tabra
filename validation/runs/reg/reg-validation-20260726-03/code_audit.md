# `reg` code audit — `reg-validation-20260726-03`

## Decision

`code_audit_status: PASS`

This is run-scoped code-audit evidence, not a final validation attestation. At
commit `64c92fe6b136b5b3adc12895d408e5c20af9214c`, both frozen cases and all 17
release-critical metrics pass the independent executable oracle. The oracle was
run through its formal entry point with its hash guard intact. The six prior
code findings remain closed under fresh tests and probes; the newly mapped
confidence-interval and no-constant-summary behavior also passes independent
formula checks.

## Frozen identity and environment

- Run ID: `reg-validation-20260726-03`
- Audited commit: `64c92fe6b136b5b3adc12895d408e5c20af9214c`
- Manifest SHA-256: `902392d562a50aa9919cfc58bc95b0018e63044452fa8b6203608142814ecf79`
- Cases SHA-256: `1b107cfe0d79a0997a04a5a251e4a1eb909404943d7e68f337aae8ea14612c07`
- Data SHA-256: `53f6fea92a7978cbce09212c224eb6e29e9c581ab920e0285ef541c4b09d37c4`
- Oracle SHA-256: `41a47dcda1e3f83a4b21cada2098e001ff861b00ed9a6eb100dfce87b09e438c`
- Runtime: Python 3.11.11, NumPy 2.4.3, SciPy 1.17.1, pandas 3.0.1,
  pytest 9.0.2.
- Git index and tracked files were clean before audit evidence was written.
  Pre-existing untracked documentation and other validation-run directories
  were not modified.
- Regenerating the fixture with seed `20260722` produced byte-identical CSV
  data with the frozen SHA-256.

Relevant source SHA-256 values were:

| File | SHA-256 |
|---|---|
| `src/tabra/models/estimate/ols.py` | `37712b20fb308b0582a84b9f5b725a69828e0f474531c13f879b5520aa94ea71` |
| `src/tabra/models/estimate/base.py` | `e43d7f01b0ec8f268a1a81077e18a50427a9ad787380c85304e64682da36f28f` |
| `src/tabra/models/estimate/stats.py` | `bab3ffd3156772b1b6cd9c3556d081883c3cba47198ca26682d483f0b006e0c4` |
| `src/tabra/ops/stats.py` | `2d9b774d3d55fb9a5b7ba972b5b0da0cada8dfe54346c79f2f8ed749af9f635c` |
| `src/tabra/results/ols_result.py` | `dbc39dc09b472fb91a1b59a1e429a315bfd1c7e89595c057c56275f0f61992d0` |
| `src/tabra/api/estimation/linear.py` | `598bdf96982adb5bd4f08e9085ab3c8072c880c40ff8fb7ba14d1328fc9950b5` |
| `validation/commands/reg/python/run.py` | `3ea141ddc60e504956207e97f49a4557fd2ee9c8d32bba4d1ca510afb4f7e521` |
| `validation/commands/reg/runner_common.py` | `38a17ac7d42a466449436d74ad8d51241a8b334b6841ab69d8790aec98939ee1` |

## Formula and implementation trace

### Public path, sample, and design

`TabraData.est.reg` calls `OLS.fit` through the estimation accessor. The public
signature exposes only `y`, ordered `x`, and `is_con`; it does not expose
weights, robust/cluster/bootstrap covariance, constraints, absorbed effects, or
categorical encoding. Those paths are unsupported and were not counted as
passed.

`BaseModel._prepare_df` selects exactly `[y] + x` and applies one row-wise
`dropna`, preserving input row order and ignoring missingness in unused columns.
A fresh partially-missing probe retained source rows `[0, 2, 4, 6, 7, 8]`,
including row 6 despite a missing unused value. It returned `n_obs=6` and
parameter order `x2, x1, _cons`. Against a separately constructed NumPy design,
maximum errors were `3.77e-15` for coefficients, `8.88e-16` for standard errors,
and `7.11e-15` for residuals.

The design preserves requested regressor order and appends one all-ones column
named `_cons` only when `is_con=True`. Non-numeric categories and non-finite
numeric values fail explicitly; exact collinearity is rejected after the
rank-revealing solve. There is no sorting, base-category selection, singleton
handling, or observation weighting.

### Objective, derivatives, identification, and convergence

For `S(beta)=(y-X beta)'(y-X beta)`, independent differentiation gives
`gradient S(beta)=-2X'(y-X beta)` and Hessian `2X'X`. A full-column-rank design
therefore has the unique least-squares minimizer satisfying
`X'(y-X beta)=0`.

Production forms `X_s=X D^-1`, where `D` contains maximum absolute column
scales, solves `min ||y-X_s gamma||` using rank-revealing
`numpy.linalg.lstsq`, requires `rank=k`, and maps back with
`beta=D^-1 gamma`. This is a coordinate transformation of the same objective.
The frozen oracle reports ranks `3/3` and `2/2`, condition numbers `1.3528` and
`1.3295`, and maximum normal-equation residuals `4.13e-14` and `9.77e-15` for
`basic` and `no_constant`.

The estimator is closed form. Iteration limits, iterative non-convergence,
separation, censoring limits, and empty-group convergence have no applicable
execution path. A uniform `1e-8` regressor scale change altered fitted values by
at most `2.22e-16` and reciprocally mapped coefficients by `1.11e-16`; reversing
rows altered coefficients by at most `2.22e-16`. A full-rank design with
condition number `2.014e7` matched independent `lstsq` coefficients exactly in
the fresh probe and fitted values within `4.44e-16`.

### Covariance, degrees of freedom, and inference

The sole supported covariance is conventional homoskedastic OLS:

`s2=RSS/(N-k)` and `V(beta)=s2(X'X)^-1`.

Production obtains a covariance factor from the SVD of the scaled design and
maps it back through `D^-1`; squared row norms equal the diagonal of
`(X'X)^-1`. The independent oracle constructs the unscaled design and computes
the covariance without calling Tabra. The frozen maximum standard-error error
was `2.78e-17`. Robust, clustered, bootstrap, and other model-specific
covariances are not implemented by `reg` and remain outside the passed surface.

The code uses `df_model=k-c` and `df_resid=N-k`, with `c=1` only when a constant
is fitted. TSS is centered for `c=1` and equals `y'y` for `c=0`; adjusted
R-squared consequently uses numerator df `N-1` or `N`. The model statistic is
`F=[SS_model/(k-c)]/[RSS/(N-k)]`. Coefficient probabilities use
`2*scipy.stats.t.sf(abs(t),df_resid)` and the model probability uses
`scipy.stats.f.sf`. Fresh extreme probes retained nonzero tails of
`9.994267861337003e-37` for `t=20,df=100` and
`4.846696230808534e-60` for `F=500,(3,100)`.

The public 95% confidence interval is
`beta +/- t_(0.975,N-k)*SE`. Independent checks for both constant conventions
matched the displayed three-decimal endpoints within `3.90e-4` and `2.11e-4`,
respectively. The no-constant ANOVA Total row uses df `N` and MS `TSS/N`; its
fresh five-observation line was:

```text
       Total | 57.9100     5 11.582000    Root MSE      =   0.1363
```

## Formal independent oracle and guard

The oracle at `validation/commands/reg/oracles/code_audit_oracle.py` imports no
Tabra estimator code. It reads the frozen CSV without pandas, constructs both
designs, derives least-squares estimates and classical covariance with NumPy
SVD/`lstsq`, and obtains t/F tails directly from SciPy.

The formal sequence was:

```bash
.venv/bin/python validation/commands/reg/python/run.py \
  --data validation/commands/reg/data/reg_fixture.csv \
  --output /tmp/tabra-reg-audit-03.syaOVn/python-results.json
.venv/bin/python validation/commands/reg/oracles/code_audit_oracle.py \
  --manifest validation/commands/reg/manifest.yaml \
  --cases validation/commands/reg/cases.yaml \
  --data validation/commands/reg/data/reg_fixture.csv \
  --actual /tmp/tabra-reg-audit-03.syaOVn/python-results.json
```

It exited `0` with `oracle_status=PASS`. All three frozen-hash checks, all five
status/sample checks, and all 17 metric comparisons passed. There were no
failed or skipped required metrics. Maximum absolute errors were
`1.33e-15` for coefficients, `2.78e-17` for standard errors, `1.42e-14` for t
statistics, `5.29e-19` for coefficient p-values, and zero for F statistics and
F probabilities. Both zero-mismatch safety checks were false.

The guard was then exercised without modifying repository files by passing a
process-substitution manifest with one extra blank line. It exited `1` before
comparison with `RuntimeError: Frozen input hash mismatch`; the changed digest
was `298f8b6bc7fd47cd66f4d31a571a6dad22c3671354b018440602e0f8fef748e6`.
The passing formal run did not bypass the guard.

## Tests and failure-mode probes

- Required targeted command covering oracle, guard hashes, partial missingness,
  both confidence-interval conventions, and the no-constant Total row:
  `6 passed in 2.08s`.
- Relevant regression command over OLS, summaries, statistics, API, oracle, and
  validation-gate tests: `128 passed in 2.97s`.
- Full suite: `787 passed, 52 skipped, 6 warnings in 134.75s`, exit code 0. The
  skips concern other modules and were not used as passing `reg` evidence.
- Exact collinearity raises `ValueError: OLS design is rank deficient`.
  `Inf`, an all-missing sample, `N=k`, and categorical data all fail explicitly.
- A `1e200` regressor probe returned finite core estimates and inference.
  Zero response and intercept-only inputs report undefined t/F inference as
  NaN, never as false zero probabilities.
- An empty predictor list with `is_con=False` fails before returning any result
  (NumPy zero-size reduction error). This zero-parameter configuration is not a
  frozen supported case; the failure is explicit and does not violate a frozen
  release invariant, though its message is less specific than the guarded
  supported-input errors.
- Empty groups, absorbed effects, singleton removal, separation, censoring,
  weights, and robust/cluster/bootstrap VCE have no path in this numeric,
  unweighted, closed-form API. None was relabeled as a passed feature.

## Prior findings and disposition

| Finding | Current disposition | Fresh evidence |
|---|---|---|
| `REG-CODE-001` tail cancellation | CLOSED | survival-function probes and frozen zero-mismatch checks pass |
| `REG-CODE-002` scale/rank instability | CLOSED | scaled `lstsq`/SVD, scale, ill-conditioning, row-order, and rank checks pass |
| `REG-CODE-003` non-finite/overflow output | CLOSED | finite guards and the `1e200` probe pass |
| `REG-CODE-004` degenerate df/inference | CLOSED | `N-k` rejection and NaN undefined-inference probes pass |
| `REG-CODE-005` no-constant Total df/MS | CLOSED | Total row reports df `N` and MS `TSS/N`; mapped regression passes |
| `REG-CODE-006` stale oracle/input mismatch | CLOSED | current guarded oracle passes and a changed manifest is rejected |

No confirmed implementation or safety defect remains open in the frozen `reg`
surface. Structured evidence and reproduction steps are in
`code_findings.json`.
