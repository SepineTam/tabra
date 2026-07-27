# `reg` theory audit

- Run ID: `reg-validation-20260726-01`
- Audit date: 2026-07-26
- Audited commit: `4500f443f8a021cd9d9ae9cfee028ae2961dda96`
- `theory_audit_status`: **FAIL**
- Auditor role: independent theory and literature audit; no gate decision

## Decision

The supported estimator and reporting formulas are mathematically consistent
with the authoritative Stata 19 `regress` formulas and with the inspected
production implementation.  The frozen theory package nevertheless cannot
pass because its independent evidence chain is incomplete:

1. every release-critical formula is mapped to
   `validation/commands/reg/oracles/code_audit_oracle.py`, but that oracle is
   locked to earlier manifest and cases hashes and aborts before deriving any
   metric for this run;
2. the release-critical complete-case sample rule is not independently tested
   by either frozen case, the oracle, or the mapped API test; and
3. the formula map uses the Stata software manual as the sole exact source for
   every implemented mathematical formula, while the cited original papers are
   explicitly only historical context and have no exact formula locations in
   the frozen map.  This does not satisfy the required separation between
   original mathematical support and software-compatibility documentation.

These are frozen-artifact deficiencies, not inaccessible-source deficiencies,
so `BLOCKED_ENV` is not appropriate.

## Frozen inputs

| Input | Frozen value | Observed value | Result |
|---|---|---|---|
| Git commit | `4500f443f8a021cd9d9ae9cfee028ae2961dda96` | same | match |
| `manifest.yaml` SHA-256 | `414ed36be847a31db40f46ed00c5fb9793fb0773d04072bc656f8d03e38c56b5` | same | match |
| `cases.yaml` SHA-256 | `1b107cfe0d79a0997a04a5a251e4a1eb909404943d7e68f337aae8ea14612c07` | same | match |
| fixture SHA-256 | `53f6fea92a7978cbce09212c224eb6e29e9c581ab920e0285ef541c4b09d37c4` | same | match |
| `theory.md` SHA-256 | `a55caa8832d1b3c6f92e3b5c49b66997c7313dc2ef004bd9004605a7d1443050` | same | match |
| `formula_map.csv` SHA-256 | `1dc518e2edf22c5ec0a6c756b24be45477aa939477031571579e8f86e10be3fe` | same | match |

The tracked worktree was clean during the audit.  Untracked files belonging to
this validation run and unrelated documentation work were present.  This audit
did not modify product code, tests, command specifications, registry, state, or
events.

## Sources and evidence roles

### Original methodological sources

- Fisher, R. A. (1922), “The Goodness of Fit of Regression Formulae, and
  the Distribution of Regression Coefficients,” *Journal of the Royal
  Statistical Society* 85(4), 597–612,
  <https://doi.org/10.2307/2341124>.  The article metadata and page span were
  checked through the Rothamsted Research repository.  The frozen theory calls
  this primary context for regression inference but asserts no exact equation
  or page location from it.
- Aitken, A. C. (1936), “On Least Squares and Linear Combination of
  Observations,” *Proceedings of the Royal Society of Edinburgh* 55, 42–48,
  <https://doi.org/10.1017/S0370164600014346>.  The DOI metadata, article
  description, journal, volume, and page span were checked.  The frozen theory
  calls it historical primary context but maps no release-critical formula to
  an exact location in it.
- White, H. (1980), “A Heteroskedasticity-Consistent Covariance Matrix
  Estimator and a Direct Test for Heteroskedasticity,” *Econometrica* 48(4),
  817–838, <https://doi.org/10.2307/1912934>.  It applies only to the explicitly
  excluded robust extension.  No implemented robust VCE is inferred from it.

The original papers were not treated as if they supplied unverified modern
matrix formulas or equation numbers.  Their role in the frozen artifacts is
context, not exact formula support.

### Authoritative software and formula documentation

- StataCorp LLC (2025), *Stata 19 Base Reference Manual*, `[R] regress`,
  <https://www.stata.com/manuals/rregress.pdf>.  Printed p. 15 documents that an
  observation is dropped when a model variable is missing.  Printed p. 22 gives
  the OLS minimizer and coefficient formula (equation (2)), centered versus
  uncentered TSS, RSS, MSS, their degrees of freedom, residual variance, and
  Root MSE.  Printed p. 23 gives F, R-squared, adjusted R-squared, and the
  conventional homoskedastic covariance (equation (3)).  Printed p. 26 gives
  `N-k` degrees of freedom for homoskedastic coefficient p-values and confidence
  intervals.  Printed pp. 7 and 9–10 establish the two-sided coefficient test,
  right-tail overall F interpretation, and `noconstant` convention.

This manual is authoritative for Stata compatibility and accurately states the
needed formulas.  Stata or R numerical output was not treated as mathematical
proof.

## Mathematical contract checked

For the retained complete cases, let `X` be an `N x k` full-column-rank design,
let `c=1` when a constant is fitted and `c=0` otherwise, and let
`e = y - X beta_hat`.  The frozen contract requires

\[
\hat\beta=(X'X)^{-1}X'y,
\quad RSS=e'e,
\quad TSS=\begin{cases}\sum_i(y_i-\bar y)^2,&c=1,\\y'y,&c=0,\end{cases}
\quad MSS=TSS-RSS,
\]

\[
df_m=k-c,
\quad df_r=N-k,
\quad s^2=RSS/(N-k),
\quad \widehat V(\hat\beta)=s^2(X'X)^{-1},
\]

\[
R^2=1-RSS/TSS,
\quad \bar R^2=1-(1-R^2)(N-c)/(N-k),
\quad F=\frac{MSS/(k-c)}{RSS/(N-k)},
\quad Root\ MSE=\sqrt{RSS/(N-k)}.
\]

For coefficient `j`, `t_j=beta_hat_j/se_j` and
`p_j=2 Pr(T_(N-k) >= |t_j|)`.  The overall p-value is
`Pr(F_(k-c,N-k) >= F)`.  With a constant the overall null excludes the
constant; without a constant it covers all `k` fitted coefficients.

Static source tracing confirmed these formulas at:

- sample selection: `src/tabra/models/estimate/base.py:19-23`;
- constant construction, scaled least-squares coefficients, residuals, and
  centered/uncentered sums of squares: `src/tabra/models/estimate/ols.py:36-90`;
- `RSS/(N-k)` and SVD-based diagonal covariance calculation:
  `src/tabra/models/estimate/ols.py:91-106`;
- coefficient t ratios and two-sided p-values:
  `src/tabra/models/estimate/ols.py:127-142` and
  `src/tabra/ops/stats.py:40-50`;
- model/residual degrees of freedom, adjusted R-squared, R-squared, F, MSE, and
  Root MSE: `src/tabra/models/estimate/stats.py:13-44,69-76`;
- no-constant total ANOVA degrees of freedom and mean square in the public
  summary: `src/tabra/results/ols_result.py:107-115`.

The source correctly maps normalized `ss_residual` to Tabra `SSR`,
`ss_model` to Tabra `SSE`, and `ss_total` to Tabra `SST`.  The unusual names are
explicitly documented in the frozen theory and map.

## Assumptions and interpretation

- Full column rank is required for identification of the coefficient vector.
- `E[u|X]=0` is required for conditional unbiasedness; fitting the regression
  does not establish it.
- The implemented covariance requires `Var(u|X)=sigma^2 I`: homoskedastic and
  conditionally uncorrelated errors.
- Exact finite-sample t and F reference distributions additionally require
  conditional Gaussian errors.  Without this assumption, software-computed
  tails are not exact finite-sample probabilities.
- Complete-case fitting changes the estimation sample.  Population
  interpretation needs a defensible missingness/selection argument.
- No OLS formula by itself supports a causal or structural interpretation.

The frozen cases satisfy the stated nondegenerate scope (`N>k`, positive TSS,
positive RSS, and positive model degrees of freedom).  Robust covariance,
weights, rank-deficient designs, and degenerate inference are outside the
release cases.

## Release-critical traceability

The frozen map contains 20 rows: 19 marked `implemented` and one excluded robust
row marked `blocked`.  The mathematical expressions, assumptions,
constant/no-constant parameterization, production code locations, and named
unit-test locations are accurate for the implemented scope.  Relevant mapped
unit tests were executed with the captured virtual environment:

```text
.venv/bin/python -m pytest -q -o addopts='' \
  tests/test_ols.py tests/test_estimate_base_stats.py \
  tests/test_api.py::TestRegAPI tests/ops/test_stats.py
55 passed in 1.64s
```

This test result is software evidence only.  It is not a substitute for the
mathematical sources or an executable independent frozen-case oracle.

| Area | Frozen formula IDs | Source/code result | Independent evidence result |
|---|---|---|---|
| case IDs and sample | `REG-CASE-001`, `REG-SAMPLE-001` | code and Stata behavior align | **fail**: oracle rejects current hashes; no partial-missing case/test |
| parameterization and df | `REG-PARAM-001`, `REG-DF-001`, `REG-NOCONSTANT-001` | constant conventions and df formulas align | named unit tests pass, but frozen oracle is unusable |
| objective and coefficients | `REG-OBJ-001`, `REG-COEF-001` | scaled solve maps back to OLS estimand | named unit tests pass, but frozen-case oracle is unusable |
| RSS/TSS/MSS and fit statistics | `REG-RSS-001`, `REG-TSS-001`, `REG-MSS-001`, `REG-R2-001`, `REG-AR2-001`, `REG-F-001`, `REG-RMSE-001` | formulas and `c` branches align | named unit tests pass, but frozen-case oracle is unusable |
| VCE, SE, t and p values | `REG-VCE-001`, `REG-SE-001`, `REG-T-001`, `REG-TP-001`, `REG-FP-001` | homoskedastic VCE and tail definitions align | some unit tests pass; full frozen-case mapping is invalidated by oracle hashes |
| robust extension | `REG-VCE-ROBUST-001` | correctly not implemented | correctly excluded, not counted as passing support |

## Findings

### TA-REG-20260726-001 — independent oracle is bound to stale frozen inputs (`critical`)

`code_audit_oracle.py:23-27` expects manifest hash
`533e96bde7c0fd15dcc532f127789ab300c21f378c7b344ac99be6a4cafa26d7`
and cases hash
`d6dc76648bcd0433795b60c2f65d46cc89167565384c23e24d864525d42ea7e7`.
This run freezes manifest hash
`414ed36be847a31db40f46ed00c5fb9793fb0773d04072bc656f8d03e38c56b5`
and cases hash
`1b107cfe0d79a0997a04a5a251e4a1eb909404943d7e68f337aae8ea14612c07`.
The guard at `code_audit_oracle.py:200-213` therefore aborts before loading the
specification or calculating expected metrics.

Direct reproduction with the captured environment returned exit code 1 and
`RuntimeError: Frozen input hash mismatch`.  Because every implemented formula
row uses this oracle as frozen-case independent evidence, no release-critical
metric has a working end-to-end independent oracle for this run.  An unmapped
or non-executable critical formula cannot pass.

### TA-REG-20260726-002 — complete-case selection lacks an independent test (`critical`)

The manifest makes complete-case selection release-critical through `nobs` and
the estimation contract.  Production code at `base.py:19-23` correctly selects
`[y]+x` and applies rowwise `dropna()`, consistent with Stata `[R] regress`,
printed p. 15.  However:

- both frozen cases use the same 120-row fixture with no missing values;
- `code_audit_oracle.py::read_numeric_csv` converts every CSV cell directly to
  `float` and performs no complete-case selection;
- `tests/test_api.py::TestRegAPI::test_reg_with_constant` uses a fully observed
  sample and only asserts `n_obs == 100`; and
- the only OLS missing-value unit test checks that an entirely missing sample is
  rejected, not that partially missing rows across `y`, `x1`, and `x2` are
  jointly excluded and that `nobs` and estimates use the resulting common
  sample.

Thus `REG-SAMPLE-001` explicitly admits a limitation that leaves a
release-critical sampling rule without the independent test/case required by
the audit procedure.

### TA-REG-20260726-003 — exact mathematical support is not separated from compatibility documentation (`high`)

All implemented mathematical rows in `formula_map.csv:3-20` use the Stata 19
`regress` manual as `primary_reference`.  The manual is authoritative and its
locations are accurate, but it simultaneously defines Stata behavior.  The
frozen theory says Aitken (1936) and Fisher (1922) are only historical primary
context and deliberately supplies no exact equation or narrower page location
from either.  Consequently the formula map does not separately identify exact
original-method support for the coefficient, covariance, and finite-sample
inference claims.

The remediation is not to invent historical equation numbers.  A newly frozen
map must either add verified exact original methodological locations for the
claims they genuinely support or explicitly justify an authoritative primary
mathematical source distinct from the Stata compatibility manual.

## Required remediation before re-audit

1. Re-freeze an independent oracle whose manifest and cases hashes exactly
   match this specification, and execute it against both supported cases.
2. Add an independent partially missing OLS case or unit test that verifies the
   joint complete-case sample, `nobs`, parameter ordering, and estimates.
3. Separate exact mathematical-source support from Stata compatibility support
   in the frozen formula map, using only source locations that have actually
   been verified.
4. Re-freeze the affected artifacts and rerun the independent theory audit.
