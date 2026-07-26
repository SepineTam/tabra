# `reg` code audit — `reg-validation-20260726-02`

## Decision

`code_audit_status: PASS`

This is run-scoped code-audit evidence, not a final validation attestation. At
commit `4caa15acacb770c5ee95b0fe6ffdea08fd1d20bd`, all release-critical
invariants and both frozen cases pass the independent executable oracle. The
oracle was executed through its formal entry point without bypassing its hash
guard. The previous stale-oracle blocker is closed, the previous five product
defects remain closed under fresh tests and probes, and the new partial-missing
test independently verifies the joint complete-case sample.

## Frozen identity and environment

- Run ID: `reg-validation-20260726-02`
- Audited commit: `4caa15acacb770c5ee95b0fe6ffdea08fd1d20bd`
- Manifest SHA-256: `3ace49ec0668c8deb699db3e6db90ade5f7e7a8105c61ff9163dd438c628068e`
- Cases SHA-256: `1b107cfe0d79a0997a04a5a251e4a1eb909404943d7e68f337aae8ea14612c07`
- Data SHA-256: `53f6fea92a7978cbce09212c224eb6e29e9c581ab920e0285ef541c4b09d37c4`
- Oracle SHA-256: `007099d58e6a26ba4e0eb3236537d004e453b14595b3ec78a4012dfc954ea3af`
- Runtime: Python 3.11.11, NumPy 2.4.3, SciPy 1.17.1, pandas 3.0.1,
  pytest 9.0.2.
- Tracked files and the Git index were clean at the final pre-write check.
  Pre-existing untracked documentation and validation-run evidence were not
  modified.

Relevant source SHA-256 values were:

| File | SHA-256 |
|---|---|
| `src/tabra/models/estimate/ols.py` | `37712b20fb308b0582a84b9f5b725a69828e0f474531c13f879b5520aa94ea71` |
| `src/tabra/models/estimate/base.py` | `e43d7f01b0ec8f268a1a81077e18a50427a9ad787380c85304e64682da36f28f` |
| `src/tabra/models/estimate/stats.py` | `bab3ffd3156772b1b6cd9c3556d081883c3cba47198ca26682d483f0b006e0c4` |
| `src/tabra/ops/stats.py` | `2d9b774d3d55fb9a5b7ba972b5b0da0cada8dfe54346c79f2f8ed749af9f635c` |
| `src/tabra/results/ols_result.py` | `dbc39dc09b472fb91a1b59a1e429a315bfd1c7e89595c057c56275f0f61992d0` |

## Formula and implementation trace

### Public path, sample, and design

`TabraData.est.reg` constructs the command label and calls `OLS.fit` through
`EstimationCommandBase._run_model`. `BaseModel._prepare_df` selects `[y] + x`
in caller order and performs one row-wise `dropna`, so the estimation sample is
the common complete-case sample across the dependent variable and all requested
regressors. It neither sorts nor examines unused columns.

The fresh partial-missing probe retained source rows `[0, 2, 4, 6, 7, 8]`,
including row 6 despite a missing value in the unused column. It returned
`n_obs=6` and parameter order `x2, x1, _cons`. Against an independently
constructed NumPy design, the maximum absolute errors were `3.77e-15` for
coefficients, `8.88e-16` for standard errors, and `7.11e-15` for residuals.
The dedicated test also checks R-squared. This closes the prior absence of an
independent partial-missing check.

The design preserves requested regressor order and appends a ones column named
`_cons` only when `is_con=True`. There are no weights, constraints, absorbed
effects, singleton processing, or categorical expansion in the `reg` API.
Non-numeric categories fail conversion explicitly; `Inf` is rejected after
conversion. Robust, clustered, bootstrap, and model-specific covariance options
are not exposed and are not counted as passed.

### Objective, derivatives, identification, and convergence

For

`S(beta) = (y - X beta)'(y - X beta)`,

independent differentiation gives

`gradient S(beta) = -2 X'(y - X beta)` and
`Hessian S(beta) = 2 X'X`.

For full column rank, the unique minimizer satisfies `X'(y-X beta)=0`.
Production forms `X_s = X D^-1`, where `D` contains maximum absolute column
scales, solves `min ||y-X_s gamma||` with rank-revealing
`numpy.linalg.lstsq`, requires `rank=k`, and maps back with
`beta=D^-1 gamma`. Thus the scaling changes coordinates, not the least-squares
objective or fitted values.

Both frozen designs are full rank. The independent oracle reports ranks `3/3`
and `2/2`, condition numbers `1.3528` and `1.3295`, and maximum normal-equation
residuals `4.13e-14` and `9.77e-15` for `basic` and `no_constant`. The estimator
is closed form: iteration limits, iterative non-convergence, separation, and
censoring limits have no applicable execution path.

### Covariance, degrees of freedom, and inference

The sole supported covariance is classical homoskedastic OLS:

`s2 = RSS/(N-k)` and `V(beta) = s2 (X'X)^-1`.

Production obtains the covariance factor from the SVD of the scaled design and
maps its rows back through `D^-1`; row squared norms therefore equal the
diagonal of `(X'X)^-1`. The formal independent oracle builds the unscaled design
and derives the same covariance directly. Maximum absolute standard-error error
was `2.78e-17`.

The implementation uses `df_model=k-c` and `df_resid=N-k`, with `c=1` only
when a constant is fitted. TSS is centered for `c=1` and is `y'y` for `c=0`.
Therefore adjusted R-squared uses numerator df `N-1` or `N`, and the model F
statistic is

`F = [SS_model/(k-c)] / [RSS/(N-k)]`.

Coefficient probabilities use `2*scipy.stats.t.sf(abs(t), df_resid)` and the F
probability uses `scipy.stats.f.sf`. At `t=20, df=100`, production preserves
`9.994267861337003e-37`; at `F=500, (3,100)`, it preserves
`4.846696230808534e-60`, while the corresponding `1-CDF` calculations round to
zero. Confidence intervals in the summary use
`beta +/- scipy.stats.t.ppf(0.975, df_resid)*SE`. Exact fits/zero response return
NaN t/F inference, and an intercept-only model returns `df_model=0` with NaN
F inference, rather than false zero probabilities.

The no-constant ANOVA summary uses Total df `N` and Total MS `TSS/N`. The fresh
five-observation probe produced:

```text
Total | 57.9100     5 11.582000    Root MSE      =   0.1363
```

## Formal independent oracle

The required oracle is
`validation/commands/reg/oracles/code_audit_oracle.py`. It imports no Tabra
module and derives samples, coefficients, sums of squares, classical covariance,
t/F statistics, probabilities, and invariants from the frozen CSV with
NumPy/SciPy and direct algebra.

The exact formal command was:

```bash
.venv/bin/python validation/commands/reg/oracles/code_audit_oracle.py \
  --manifest validation/commands/reg/manifest.yaml \
  --cases validation/commands/reg/cases.yaml \
  --data validation/commands/reg/data/reg_fixture.csv \
  --actual /tmp/tabra-reg-audit-02.bR6AkL/python-results.json
```

It exited `0` with `oracle_status=PASS`. All three guard checks, all five result
status/sample checks, and all 17 frozen metric comparisons passed. There were no
failed metrics. The largest absolute numeric differences were:

| Metric | Maximum absolute error |
|---|---:|
| coefficients | `1.3322676295501878e-15` |
| standard errors | `2.7755575615628914e-17` |
| t statistics | `1.4210854715202004e-14` |
| coefficient p-values | `5.285485590866834e-19` |
| F statistic / F p-value | `0` / `0` |

The stale-input guard was tested separately without bypassing it: a manifest
copy differing by one trailing newline produced exit code `1` and
`RuntimeError: Frozen input hash mismatch`. Conversely, the current manifest,
cases, and data hashes equal the oracle's embedded hashes. The stale-oracle
blocker from the preceding run is therefore closed while the guard remains
effective.

## Tests and numerical failure modes

- Focused command:
  `.venv/bin/pytest --no-cov -q tests/test_ols.py tests/test_ols_result.py tests/test_estimate_base_stats.py tests/ops/test_stats.py tests/test_api.py tests/test_reg_validation_oracle.py tests/test_validation_core.py`
  -> `126 passed in 3.65s`.
- Full suite: `.venv/bin/pytest --no-cov -q` -> `785 passed, 52 skipped,
  6 warnings in 144.63s`, exit code `0`. The skips concern other modules and
  were not used as passing `reg` evidence.
- The formal-oracle regression tests verify current embedded hashes and execute
  the runner plus guarded oracle entry point.
- Uniform `1e-8` regressor scaling preserved fitted values to `2.22e-16` and
  reciprocally mapped coefficients to `1.11e-16`. Reversing rows changed
  coefficients by at most `2.22e-16` and standard errors by `1.08e-18`.
- Near-collinear and correlated-design NumPy/SVD tests passed; exact
  collinearity raises `ValueError: OLS design is rank deficient: rank=2, k=3`.
- `Inf`, an all-missing sample, `N=k`, and categorical input each fail
  explicitly. The respective numerical guards reject non-finite values, empty
  complete samples, and nonpositive residual degrees of freedom.
- A `1e200` regressor probe returned finite coefficients, standard errors,
  statistics, probabilities, residuals, fitted values, R-squared, and F output.
- Zero response and intercept-only input return NaN for mathematically undefined
  inference. They do not serialize as successful frozen cases; the frozen cases
  are nondegenerate and finite.
- Empty groups, absorbed effects, singleton removal, separation, censoring,
  robust/cluster/bootstrap VCE, and iterative convergence are outside this
  numeric unweighted closed-form `reg` implementation. None was relabeled as a
  passed supported feature.

## Prior findings and disposition

| Finding | Current disposition | Fresh evidence |
|---|---|---|
| `REG-CODE-001` tail cancellation | CLOSED | survival-function tests and probes preserve representable nonzero tails |
| `REG-CODE-002` scale/rank instability | CLOSED | scaled `lstsq`/SVD, scale, near-collinearity, row-order, and rank-deficiency checks pass |
| `REG-CODE-003` non-finite/overflow output | CLOSED | finite-input/result guards and `1e200` probe pass |
| `REG-CODE-004` degenerate df/inference | CLOSED | `N-k` rejection and NaN undefined-inference probes pass |
| `REG-CODE-005` no-constant Total df/MS | CLOSED | Total row reports df `N` and MS `TSS/N` |
| `REG-CODE-006` stale oracle/input mismatch | CLOSED | guarded formal oracle passes current hashes and rejects a changed manifest |

No confirmed implementation or safety defect remains open in the frozen `reg`
surface. Structured evidence and reproduction steps are in
`code_findings.json`.
