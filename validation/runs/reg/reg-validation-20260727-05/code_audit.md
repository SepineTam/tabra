# Code audit: `reg`

## Findings

No open P0, P1, or in-scope P2 finding was found. No product code was modified
and no final attestation was issued.

## Frozen identity

- Run ID: `reg-validation-20260727-05`
- Audited commit: `9ce5b597d9e3a2d7b3168262519b2be3019116e9`
- Manifest SHA-256: `bfee2c597697569b212aeb67edbbf34d9e42421272d4430253d6724a45f121b7`
- Cases SHA-256: `1b107cfe0d79a0997a04a5a251e4a1eb909404943d7e68f337aae8ea14612c07`
- Theory SHA-256: `fd71801b0d33503cf2acd224edb7662904c8de7de89cc2bd15c82dfcc047fb8c`
- Formula map SHA-256: `5b3c7233a6522311ceed5936a72433d6c52f46596a92353b96aa8d90dad07eb0`
- Data SHA-256: `53f6fea92a7978cbce09212c224eb6e29e9c581ab920e0285ef541c4b09d37c4`
- Oracle SHA-256: `0d9ea04248ca057d09ccaa0feb1d7c2510f6435ff6fe2e4ced85d03509cb6d68`
- Result schema SHA-256: `05823ed8e65c55dab2db5d308305e1293fc433c3492511f32d85142d83a60a89`
- Runtime: Darwin 25.3.0 arm64, Python 3.11.11, NumPy 2.4.3, SciPy
  1.17.1, pytest 9.0.2, uv 0.11.28.

The tracked worktree was clean before these run-scoped files were written.
Regenerating the fixture with seed `20260722` produced byte-identical data.

## Formula and implementation trace

`TabraData.est.reg` delegates to `OLS.fit`. Sample construction selects exactly
`[y] + x`, applies joint complete-case deletion, preserves requested regressor
order, and appends `_cons` last only when requested. Each normalized case binds
the cases hash, seed, ordered included-row hash, and omitted-parameter list.

For `S(beta)=(y-X beta)'(y-X beta)`, the independent gradient is
`-2X'(y-X beta)` and Hessian is `2X'X`. Production column-scales the design,
uses rank-revealing least squares, requires full rank, and maps coefficients
back to original units. Frozen ranks were `3/3` and `2/2`; condition numbers
were `1.3528` and `1.3295`; maximum normal-equation residuals were `4.13e-14`
and `9.77e-15`.

The supported covariance is conventional homoskedastic OLS,
`s2=RSS/(N-k)` and `V(beta)=s2(X'X)^-1`. The independent oracle's maximum
standard-error difference was `2.78e-17`. Degrees of freedom are
`df_model=k-c` and `df_resid=N-k`; centered and uncentered sums of squares,
adjusted R-squared, F tests, and Total df follow the constant convention.
Coefficient and model probabilities use t and F survival functions.

Tests cover scale changes, near collinearity, rank deficiency, extreme values,
non-finite input, empty and partially missing samples, `N=k`, exact fits, and
intercept-only inference. Unsupported weights, robust/cluster/bootstrap
covariance, categorical encoding, constraints, and absorbed effects were not
counted as passed.

## Independent oracle

The fresh oracle does not import Tabra estimator internals. It exited `0` with
`oracle_status=PASS`: three hash checks, five sample/status checks, and all 17
metrics passed. Maximum absolute differences were `1.33e-15` for coefficients,
`2.78e-17` for standard errors, `1.42e-14` for t statistics, and `5.29e-19`
for coefficient p-values. Other emitted fit metrics matched exactly.

A negative probe supplied `cases.yaml` in the manifest slot without altering
any frozen file. The oracle exited `1` before comparison with
`RuntimeError: Frozen input hash mismatch`.

## Canonical comparison repair

`check_comparison_integrity` recomputes the complete deterministic
`compare_results()` object and requires full Python object equality. Added,
removed, or changed nested fields fail; the check is not limited to a status or
selected hashes. `evaluate_gate` calls this same function and consumes its
canonical match, scientific status, violations, and object hashes.

Executable probes confirmed:

- A product-generated report passed `check-comparison` with exit `0`; supplied
  and recomputed canonical object hashes both equalled `2f71427d...f343f`.
- The real polluted run04 report exited `1`: scientific comparison was true,
  canonical match was false, and the violation was
  `comparison_report_mismatch` (`125a33ff...181` versus `2f71427d...f343f`).
- Tests insert `comparison_status` and `reporter_checks`; both the integrity
  CLI and Gate reject them with matching hashes.
- A second `compare` invocation against an existing output exited `2`. Its
  physical SHA-256 remained `2b01f0e...51890` before and after.
- Product comparison emits no reporter fields. The tri-reporter contract keeps
  narrative status and checks in `comparison.md`; later JSON mutation is still
  detected by `check-comparison` and Gate.

## Test evidence

- Preflight: PASS; Python, Stata, and R were available.
- Relevant regression: `152 passed in 7.04s`; no failure, skip, or warning.
- Full suite: `811 passed, 52 skipped, 6 warnings in 309.66s`; coverage 82.17%.
- `git diff --check`: PASS.
- Ruff reported only I001 import ordering in the unchanged fixture generator;
  this is style-only and does not affect frozen bytes or scientific behavior.

All 52 skips concern other estimator modules and were not used as `reg`
evidence. The six warnings are plotting backend or Matplotlib deprecations
outside the audited paths.

## Unverified areas and residual risk

- Unsupported robust, clustered, bootstrap, weighted, categorical,
  absorbed-effect, and constrained variants remain outside this audit.
- Fresh run05 Python/Stata/R adjudication belongs to the separate comparison
  stage. This audit does not issue a final attestation.
- This execution used one macOS arm64 environment; broader BLAS/platform
  variation remains residual risk.
- Filesystem permissions do not physically lock `comparison.json`. CLI refusal
  prevents comparator overwrite, and recomputation detects external mutation.

## Skips and warnings

No required `reg` check was skipped, xfailed, or hidden by an environment
condition. The unrelated full-suite skips and warnings are disclosed above.

## Cross-engine differences

Not adjudicated in this role. The next stage must compare fresh run05 Python,
Stata, and R results without majority voting.

## Final ruling

`PASS`
