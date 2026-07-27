# `reg` code audit — `reg-validation-20260726-01`

## Decision

`code_audit_status: BLOCKED_ENV`

This is a run-scoped code-audit decision, not a final validation attestation.
The current estimator closes the five defects reported by the previous audit,
and an independent algebra diagnostic agrees with every frozen metric. However,
the frozen executable oracle cannot accept this run's frozen manifest and cases:
its own immutable hash guard still names the earlier manifest and cases. Because
the required oracle exits before producing comparisons, the audit cannot record
`PASS`; bypassing that guard is diagnostic evidence only.

## Frozen identity and environment

- Run ID: `reg-validation-20260726-01`
- Audited commit: `4500f443f8a021cd9d9ae9cfee028ae2961dda96`
- Manifest SHA-256: `414ed36be847a31db40f46ed00c5fb9793fb0773d04072bc656f8d03e38c56b5`
- Cases SHA-256: `1b107cfe0d79a0997a04a5a251e4a1eb909404943d7e68f337aae8ea14612c07`
- Data SHA-256: `53f6fea92a7978cbce09212c224eb6e29e9c581ab920e0285ef541c4b09d37c4`
- Oracle SHA-256: `527b609e22a95163b99149c271e448774de76bd19702de7bbb79fcb4118a39ce`
- Runtime: Python 3.11.11, NumPy 2.4.3, SciPy 1.17.1, pandas 3.0.1,
  pytest 9.0.2.
- The tracked worktree was clean during the audit. Untracked run evidence and
  unrelated documentation were present and were not modified.

Relevant source SHA-256 values were:

| File | SHA-256 |
|---|---|
| `src/tabra/models/estimate/ols.py` | `37712b20fb308b0582a84b9f5b725a69828e0f474531c13f879b5520aa94ea71` |
| `src/tabra/models/estimate/base.py` | `e43d7f01b0ec8f268a1a81077e18a50427a9ad787380c85304e64682da36f28f` |
| `src/tabra/models/estimate/stats.py` | `bab3ffd3156772b1b6cd9c3556d081883c3cba47198ca26682d483f0b006e0c4` |
| `src/tabra/ops/stats.py` | `2d9b774d3d55fb9a5b7ba972b5b0da0cada8dfe54346c79f2f8ed749af9f635c` |
| `src/tabra/results/ols_result.py` | `dbc39dc09b472fb91a1b59a1e429a315bfd1c7e89595c057c56275f0f61992d0` |

## Formula and implementation trace

### Sample, design, and identification

`BaseModel._prepare_df` selects `[y] + x` in caller order and applies row-wise
complete-case deletion. It does not sort. A two-row missing-value probe retained
118 of 120 observations. The frozen fixture itself has 120 finite complete rows.
Weights are not accepted, and `robust`/cluster/bootstrap covariance is not
exposed. String categories fail numeric conversion; there is no silent category
expansion. The supported matrix is therefore numeric, unweighted, and has no
constraints or absorbed effects.

The design preserves requested regressor order and appends a ones column named
`_cons` only for `is_con=True`. The implementation scales every design column by
its maximum absolute value, solves with `numpy.linalg.lstsq`, rejects
`rank < k`, and maps coefficients back to original units. Empty complete samples,
non-finite inputs, and `N-k <= 0` are rejected before solving or inference.

### Objective, derivatives, covariance, and convergence

For `S(b)=(y-Xb)'(y-Xb)`, independent differentiation gives
`grad S(b)=-2X'(y-Xb)` and Hessian `2X'X`; with full column rank the minimizer is
the least-squares solution. The estimator is closed form, so iterative
convergence, iteration limits, separation, censoring limits, and empty-group
convergence paths do not apply.

The only covariance path is conventional homoskedastic OLS:
`s2=RSS/(N-k)` and `V(b)=s2(X'X)^-1`. Production derives a stable SVD covariance
factor from the scaled design and computes each standard error as a scaled row
norm. An independent SVD covariance on the unscaled frozen design agrees to a
maximum absolute standard-error error of `2.78e-17`. Robust, clustered, and
bootstrap covariances remain explicitly unsupported and were not counted as
passed.

### Degrees of freedom, summaries, and tails

The implementation uses `df_model=k-c`, `df_resid=N-k`, centered TSS when
`c=1`, and uncentered `y'y` when `c=0`. Consequently adjusted R-squared uses
`N-1` with a constant and `N` without one, while the F test covers the
nonconstant coefficients for `c=1` and all coefficients for `c=0`. The public
no-constant ANOVA summary now uses Total df `N` and Total MS `TSS/N`; the probe
reported `Total | 57.9100     5 11.582000` for five observations.

Coefficient probabilities use `2*scipy.stats.t.sf(abs(t), df_resid)` and the F
probability uses `scipy.stats.f.sf`. At `t=20, df=100`, the implementation keeps
`9.994267861337003e-37` while `2*(1-CDF)` is zero. At `F=500, (3,100)`, it keeps
`4.846696230808534e-60` while `1-CDF` is zero. Exact fits/zero response produce
NaN t/F inference rather than false zero probabilities; intercept-only models
likewise report the undefined F statistic and probability as NaN.

## Independent numerical evidence

The required executable oracle is
`validation/commands/reg/oracles/code_audit_oracle.py`. It imports no Tabra
module and independently derives coefficients, sums of squares, conventional
covariance, t/F statistics, and survival-function probabilities using NumPy and
SciPy. Its file digest matches the run's frozen oracle digest.

Its formal entry point nevertheless failed before reading results. Lines 23--26
freeze manifest `533e96...` and cases `d6dc76...`, while this run freezes
manifest `414ed3...` and cases `1b107c...`; lines 205--213 reject the mismatch.
The exact command was:

```bash
.venv/bin/python validation/commands/reg/oracles/code_audit_oracle.py \
  --manifest validation/commands/reg/manifest.yaml \
  --cases validation/commands/reg/cases.yaml \
  --data validation/commands/reg/data/reg_fixture.csv \
  --actual /tmp/tabra-reg-audit.SsdMsg/python-results.json
```

It raised `RuntimeError: Frozen input hash mismatch`. This is the sole open
finding and the reason for `BLOCKED_ENV`.

For diagnostic isolation only, invoking the same oracle's `derive_case` and
`compare_values` functions after independently verifying the task-supplied
hashes yielded no failed metrics. Maximum absolute errors were
`1.33e-15` coefficients, `2.78e-17` standard errors, `1.42e-14` t statistics,
`0` F statistic, and `5.29e-19` p-values. Both designs were full rank;
`cond(X)` was `1.3528` (`basic`) and `1.3295` (`no_constant`), and the largest
normal-equation residual was `4.13e-14`. All required p-values were nonzero,
including F tails `5.790493786860803e-92` and `6.324940034154269e-25`.

## Tests and failure-mode probes

- Relevant tests: `.venv/bin/pytest --no-cov -q tests/test_ols.py
  tests/test_ols_result.py tests/test_estimate_base_stats.py
  tests/ops/test_stats.py tests/test_api.py` -> `86 passed in 1.03s`.
- Full suite: `.venv/bin/pytest --no-cov -q` -> `781 passed, 52 skipped,
  6 warnings in 149.18s`; the skips are outside the required `reg` checks and
  were not used as passing evidence.
- The same focused command without `--no-cov` had all 86 tests pass but exited
  nonzero solely because repository-wide coverage was 20%, below the configured
  80% threshold; this is not treated as a failed estimator assertion.
- Uniform `1e-8` regressor scaling preserved rank 2 and `cond(X)=1.1761`;
  maximum fitted-value change was `2.22e-16`, and reciprocal coefficient
  rescaling differed by `1.11e-16`.
- The near-collinear and correlated-design SVD/`(X'X)^-1` tests passed; exact
  collinearity raises an explicit rank-deficiency error.
- `Inf`, all-missing input, and `N=k` each raise explicit `ValueError`s. An
  extreme `1e200` regressor returns only finite core results in its unit probe.
- Zero response and intercept-only input return NaN for undefined inference,
  never a false p-value of zero.
- Reversing rows is algebraically immaterial because the fit does not sort or
  otherwise depend on row order. The result does not expose the retained source
  index; that is outside the frozen normalized output contract.
- Separation, censoring limits, and empty groups have no path in unweighted
  closed-form OLS. The explicitly excluded robust case was not relabeled as a
  pass.

## Prior findings and disposition

| Finding | Current disposition | Key evidence |
|---|---|---|
| `REG-CODE-001` tail cancellation | CLOSED | t/F survival functions; extreme-tail tests; frozen diagnostic tails nonzero |
| `REG-CODE-002` scale/rank instability | CLOSED | scaled `lstsq`/SVD; scale and near-collinearity tests pass |
| `REG-CODE-003` non-finite/overflow output | CLOSED | finite-input and finite-result guards; invalid probes fail explicitly |
| `REG-CODE-004` degenerate degrees of freedom/inference | CLOSED | `N-k` hard check and NaN undefined inference |
| `REG-CODE-005` no-constant Total df/MS | CLOSED | summary stores `is_con` and uses `N-c` |
| `REG-CODE-006` frozen oracle/input mismatch | OPEN, blocking | formal oracle exits before comparison |

The structured finding details are in `code_findings.json`. A new frozen oracle
whose embedded manifest and cases hashes match this run is required, followed by
a rerun of this audit from the beginning.
