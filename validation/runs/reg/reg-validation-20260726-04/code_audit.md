# `reg` code audit — `reg-validation-20260726-04`

## Findings

No open P0, P1, or in-scope P2 code findings were reproduced. This decision is
limited to the two frozen, unweighted, homoskedastic OLS cases. It is not a
final validation attestation.

## Decision

`code_audit_status: PASS`

At commit `47aebd1b2681607027a70275eb443060c5df0457`, the independent
executable oracle passed both frozen cases and all 17 release-critical metrics.
Its frozen-input guard rejected a mutated manifest before comparison. The
relevant regression tests and the full repository suite completed without a
failure.

## Frozen identity

- Run ID: `reg-validation-20260726-04`
- Audited commit: `47aebd1b2681607027a70275eb443060c5df0457`
- Manifest SHA-256: `bfee2c597697569b212aeb67edbbf34d9e42421272d4430253d6724a45f121b7`
- Cases SHA-256: `1b107cfe0d79a0997a04a5a251e4a1eb909404943d7e68f337aae8ea14612c07`
- Theory SHA-256: `fd71801b0d33503cf2acd224edb7662904c8de7de89cc2bd15c82dfcc047fb8c`
- Formula-map SHA-256: `5b3c7233a6522311ceed5936a72433d6c52f46596a92353b96aa8d90dad07eb0`
- Data SHA-256: `53f6fea92a7978cbce09212c224eb6e29e9c581ab920e0285ef541c4b09d37c4`
- Oracle SHA-256: `0d9ea04248ca057d09ccaa0feb1d7c2510f6435ff6fe2e4ced85d03509cb6d68`
- Normalized-result schema SHA-256: `05823ed8e65c55dab2db5d308305e1293fc433c3492511f32d85142d83a60a89`
- Runtime: macOS/Darwin 25.3.0 arm64, Python 3.11.11, NumPy 2.4.3,
  SciPy 1.17.1, pytest 9.0.2, uv 0.11.28.
- The Git index and tracked worktree were clean before these run-scoped audit
  files were written. Pre-existing untracked documentation and earlier run
  directories were not modified.

Regenerating the fixture with seed `20260722` produced a byte-identical CSV
with SHA-256
`53f6fea92a7978cbce09212c224eb6e29e9c581ab920e0285ef541c4b09d37c4`.

## Formula and implementation trace

### Sample and design

`TabraData.est.reg` delegates to `OLS.fit`. `BaseModel._prepare_df` selects
exactly `[y] + x` and applies joint complete-case deletion. Requested regressor
order is preserved; one all-ones column named `_cons` is appended last only
when `is_con=True`. The runner independently reconstructs the complete-case
mask and requires its count to match `result.n_obs`.

The normalized result binds each case to the frozen cases SHA-256, seed, an
ordered hash of included `row_id` values, and an omitted-parameter list. The
external-engine normalizer rejects duplicate, out-of-order, or invalid sample
row numbers and requires their count to equal reported `n_obs`.

Weights, categorical base encoding, constraints, absorbed effects, singleton
handling, robust/cluster/bootstrap covariance, and sorting are not exposed by
the current public `reg` API. They were not counted as passed functionality.

### Objective, identification, and numerical path

For

`S(beta)=(y-X beta)'(y-X beta)`, the independently derived gradient is
`-2X'(y-X beta)` and the Hessian is `2X'X`. A full-column-rank design therefore
has a unique minimizer satisfying `X'(y-X beta)=0`.

Production rescales each design column, solves the same least-squares objective
with rank-revealing `numpy.linalg.lstsq`, requires `rank=k`, and maps the
coefficient back to original units. The frozen designs have ranks `3/3` and
`2/2`, condition numbers `1.3528` and `1.3295`, and maximum normal-equation
residuals `4.13e-14` and `9.77e-15`.

This estimator is closed form, so iterative convergence, separation,
censoring limits, and empty-group convergence are not applicable. Existing
tests exercise uniform scaling, condition number around `2.0e7`, exact rank
deficiency, `1e200` scale, non-finite input, all-missing input, and `N=k`.

### Covariance, degrees of freedom, and inference

The sole supported covariance is conventional homoskedastic OLS:

`s2=RSS/(N-k)` and `V(beta)=s2(X'X)^-1`.

Production derives an SVD covariance factor for the scaled design and maps it
back to original units. Squared row norms equal the diagonal of
`(X'X)^-1`. The independent oracle constructs the unscaled design without
calling Tabra; its maximum standard-error difference was `2.78e-17`.

The code uses `df_model=k-c` and `df_resid=N-k`, with `c=1` only for a fitted
constant. TSS is centered for `c=1` and is `y'y` for `c=0`. Adjusted R-squared,
F statistics, no-constant Total df, and Total mean square use the same
constant convention. Coefficient probabilities use
`2*scipy.stats.t.sf(abs(t),df_resid)` and the model probability uses
`scipy.stats.f.sf`; the oracle's zero-mismatch checks for both probability
families passed. Public 95% confidence intervals use the t critical value with
`N-k` degrees of freedom.

Empty complete samples, non-finite input, nonpositive residual degrees of
freedom, exact rank deficiency, and non-finite numerical results fail
explicitly. Exact-fit or zero-variance inference is represented as undefined,
not as a false zero p-value.

## Independent executable oracle

The oracle reads the frozen CSV with the standard library, builds both design
matrices itself, and derives coefficients, covariance, fit statistics, t/F
statistics, and tail probabilities with NumPy/SciPy. It does not import or call
Tabra estimator internals.

The formal sequence was:

```bash
.venv/bin/python validation/commands/reg/python/run.py \
  --data validation/commands/reg/data/reg_fixture.csv \
  --output /tmp/tabra-reg-audit-04.C2DAV4/python-results.json
.venv/bin/python validation/commands/reg/oracles/code_audit_oracle.py \
  --manifest validation/commands/reg/manifest.yaml \
  --cases validation/commands/reg/cases.yaml \
  --data validation/commands/reg/data/reg_fixture.csv \
  --actual /tmp/tabra-reg-audit-04.C2DAV4/python-results.json
```

The oracle exited `0` with `oracle_status=PASS`. All three hash checks, all five
status/sample checks, and all 17 metric comparisons passed. Maximum absolute
differences were `1.33e-15` for coefficients, `2.78e-17` for standard errors,
`1.42e-14` for t statistics, and `5.29e-19` for coefficient p-values. F
statistics, F probabilities, R-squared values, sums of squares, and Root MSE
matched exactly in the emitted floats.

For the guard test, one trailing line feed was added to a temporary manifest
copy. The oracle exited `1` before comparison with
`RuntimeError: Frozen input hash mismatch`; the changed digest was
`29f13e43f918347172a9d5ac6950b1bfce6c9efc5dbbc58a05bb84b3e0f7b419`.

## Test evidence

Preflight:

```bash
uv run python -m tabra.validation preflight \
  --manifest validation/commands/reg/manifest.yaml
```

Result: `PASS`; Python, Stata, and R executables were all available, tolerances
were frozen, and all 17 required metrics were present.

Relevant regression command:

```bash
uv run pytest -q -o addopts='' \
  tests/test_ols.py \
  tests/test_estimate_base_stats.py \
  tests/test_ols_result.py \
  tests/test_api.py \
  tests/ops/test_stats.py \
  tests/test_reg_validation_oracle.py \
  tests/test_reg_runner_common.py \
  tests/test_validation_core.py
```

Result: `148 passed in 4.17s`, with no failure or skip.

Full repository command:

```bash
uv run pytest tests/ -v
```

Result: `807 passed, 52 skipped, 6 warnings in 220.00s`; total coverage was
`82.17%`. The skips belong to other estimator modules and were not used as
passing `reg` evidence. The warnings are plotting-backend or Matplotlib
deprecation warnings outside the `reg` estimation and validation paths.

## Unverified areas and residual risk

- Robust, clustered, bootstrap, weighted, categorical, absorbed-effect, and
  constrained regression remain unsupported and are not passed by this audit.
- Cross-engine Python/Stata/R agreement belongs to the separate comparison
  stage. This audit neither performs that adjudication nor issues the final
  attestation.
- This fresh execution used one macOS arm64 environment. The frozen fixture is
  well-conditioned, but broader BLAS/LAPACK and platform variation remains a
  residual reproducibility risk.

## Skips and warnings

No required `reg` check was skipped, xfailed, or hidden by an environment
condition. Full-suite skips and warnings were outside the current command scope
and are recorded above rather than silently treated as passes.

## Cross-engine differences

Not adjudicated in this code-audit role. The next pipeline stage must compare
the separately normalized Python, Stata, and R results without majority voting.

## Final ruling

`PASS`
