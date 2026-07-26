# `reg` theory audit

- Run ID: `reg-audit-20260723-01`
- Audit date: 2026-07-23
- `theory_audit_status`: **FAIL**
- Auditor role: independent theory audit; no gate decision

## Frozen inputs and repository state

| Input | Frozen value | Observed value | Result |
|---|---|---|---|
| Git commit | `0242978e7add4a7d44ab2056c4cbc9657f24a5f9` | same | match |
| `manifest.yaml` SHA-256 | `533e96bde7c0fd15dcc532f127789ab300c21f378c7b344ac99be6a4cafa26d7` | same | match |
| `cases.yaml` SHA-256 | `d6dc76648bcd0433795b60c2f65d46cc89167565384c23e24d864525d42ea7e7` | same | match |
| fixture SHA-256 | `53f6fea92a7978cbce09212c224eb6e29e9c581ab920e0285ef541c4b09d37c4` | same | match |
| `theory.md` SHA-256 | not supplied | `37c8f490413a42bc559073b4336208eb2e90e7fa6a29ed858b0c332aaab6fe84` | recorded |
| `formula_map.csv` SHA-256 | not supplied | `5cda46c1f134e52af159b3249d8a570dbf808bda3c6f90129cee98c61bac8223` | recorded |

The tracked worktree was dirty at audit start. Tracked modifications were present in `.github/workflows/ci.yml`, `.github/workflows/publish.yml`, `.gitignore`, `pyproject.toml`, multiple estimator modules including `src/tabra/models/estimate/ols.py`, `src/tabra/ops/stats.py`, and `tests/ops/test_stats.py`. The audit used the observed dirty source tree together with the frozen validation-file hashes. No pre-existing file was modified by this audit.

## Sources and their roles

### Mathematical and inferential support

- Fisher, R. A. (1922), “The Goodness of Fit of Regression Formulae, and the Distribution of Regression Coefficients,” *Journal of the Royal Statistical Society* 85(4), 597–612, DOI: <https://doi.org/10.2307/2341124>. This is original methodological context for finite-sample regression-coefficient and goodness-of-fit distributions. The frozen formula map does not give an exact page or formula within this paper, so it does not by itself close any mapped formula.
- Aitken, A. C. (1936), “On Least Squares and Linear Combination of Observations,” *Proceedings of the Royal Society of Edinburgh* 55, 42–48, DOI: <https://doi.org/10.1017/S0370164600014346>. This is the original source cited by the frozen map for least squares and variance. Each mapped row cites the entire article and explicitly says that no exact equation is asserted; that is insufficient for this audit's exact-location requirement.
- StataCorp LLC (2025), *Stata 19 Base Reference Manual*, `[R] regress`, pp. 22–23, <https://www.stata.com/manuals/rregress.pdf>. Page 22 gives the OLS minimization and coefficient formula (equation (2)), centered/uncentered TSS, RSS, MSS, degrees of freedom, residual variance, and Root MSE. Page 23 gives the F statistic, R-squared, adjusted R-squared, and homoskedastic covariance (equation (3)). These exact official formulas are sufficient authoritative support for the mathematical objects, but they are not recorded in the frozen formula map.

### Software-compatibility documentation

- StataCorp LLC (2025), `[R] regress`, pp. 9–10 documents the `noconstant` output and the change to uncentered ANOVA statistics; pp. 22–23 define the convention using `c=1` with a constant and `c=0` otherwise; p. 26 states that homoskedastic-regression p-values and confidence intervals use a t distribution with `N-k` degrees of freedom. These are compatibility rules, not empirical proof of Tabra's correctness.
- Cross-engine Stata or R output was not treated as mathematical evidence.

## Required mathematical contract

Condition on a design matrix `X` with `N` complete observations and full column rank `k`. Let `c=1` when an intercept is fitted and `c=0` otherwise. The supported model is

\[
y=X\beta+u,\qquad \hat\beta=(X'X)^{-1}X'y.
\]

With residuals `e=y-X beta_hat`, the required reporting parameterization is

\[
RSS=e'e,\quad
TSS=\begin{cases}\sum_i(y_i-\bar y)^2,&c=1,\\y'y,&c=0,\end{cases}\quad
MSS=TSS-RSS,
\]

\[
df_r=N-k,\quad df_m=k-c,\quad s^2=RSS/(N-k),\quad
\widehat{V}(\hat\beta)=s^2(X'X)^{-1},
\]

\[
R^2=1-RSS/TSS,\quad
\bar R^2=1-(1-R^2)(N-c)/(N-k),\quad
F=\frac{MSS}{(k-c)s^2},\quad
Root\ MSE=s.
\]

For coefficient `j`, `t_j=beta_hat_j/se_j` and the two-sided p-value is `2 Pr(T_(N-k) >= |t_j|)`. The overall p-value is `Pr(F_(k-c,N-k) >= F)`. Exact finite-sample t and F inference requires `u|X` to be Gaussian with zero conditional mean and covariance `sigma^2 I`; without normality, these are not exact finite-sample reference distributions. Conditional unbiasedness requires `E[u|X]=0`; homoskedastic standard errors require the spherical covariance condition. None of these assumptions establishes causal interpretation without a defensible design and absence of relevant confounding/endogeneity.

## Coverage and traceability

The frozen `formula_map.csv` has seven rows: three `implemented`, three `provisional`, and one unsupported robust row marked `blocked`.

| Reported object | Exact authoritative location available | Frozen map | Implementation | Independent test/case mapping | Audit result |
|---|---|---|---|---|---|
| coefficients/objective | `[R] regress`, p. 22, eq. (2) | cites all of Aitken pp. 42–48; no exact equation | `ols.py:46-50` | basic test listed; no-constant test exists but is not mapped | incomplete |
| homoskedastic covariance/SE | `[R] regress`, pp. 22–23, eq. (3) | cites all of Aitken pp. 42–48; no exact equation | `ols.py:54-56` | basic test listed; no-constant test exists but is not mapped | incomplete |
| centered R-squared | `[R] regress`, pp. 22–23 | `REG-R2-001` provisional, source pending | `ols.py:61-70`; `stats.py:34-37` | `test_r_squared`; case `basic` | fail |
| uncentered R-squared | `[R] regress`, pp. 9–10 and 22–23 | `REG-R2-002` provisional, source pending | `ols.py:64-70`; `stats.py:34-37` | `test_no_constant_regression`; case `no_constant` | fail |
| adjusted R-squared | `[R] regress`, p. 23 | no row | `stats.py:22-27` | helper tests and both OLS cases exist, but no map | fail |
| F statistic | `[R] regress`, p. 23 | `REG-F-001` provisional; indirect code location | `stats.py:40-44` via `base.py:43-45` | mapped only to basic test despite claiming both cases | fail |
| coefficient t/p and F p-value | `[R] regress`, pp. 7, 23, 26 | no rows | `ols.py:57-58`, `ops/stats.py:40-64`, `stats.py:40-44` | OLS p-value test checks only range; distribution helper tests exist but are not mapped | fail |
| `N`, model/residual df, names | `[R] regress`, pp. 22 and 26 | no rows | `base.py:19-23`, `ols.py:37-44`, `stats.py:14-19` | partial helper/name tests; no map | fail |
| RSS, MSS, centered/uncentered TSS | `[R] regress`, p. 22 | no dedicated rows | `ols.py:60-67` | decomposition test covers only constant case; no map | fail |
| Root MSE | `[R] regress`, pp. 22–23 | no row | `stats.py:69-76` | generic helper test only; no map | fail |

The validation cases `basic` and `no_constant` exercise the two values of `c`, but an engine-comparison case is not a substitute for a complete formula-map row with an exact source, assumptions, parameterization, implementation location, and independent test/case ID.

## Findings

### TA-REG-001 — release-critical source mappings remain provisional (`critical`)

`REG-R2-001`, `REG-R2-002`, and `REG-F-001` say that primary/authoritative support is pending and record `page_or_equation=not_asserted`. The official formulas are available at `[R] regress`, pp. 22–23, so this is not an environmental block; it is an incomplete frozen specification. Under the audit rule that a missing citation or skipped formula cannot pass, this finding alone requires `FAIL`.

### TA-REG-002 — many reported quantities have no formula-map row (`critical`)

The manifest requires `nobs`, `df_model`, `df_resid`, parameter names, coefficient t statistics and p-values, adjusted R-squared, F p-value, residual/model/total sums of squares, and Root MSE. These objects have no row recording an exact citation, assumptions, parameterization, code location, independent test/case, status, and limitations. Confidence intervals and the ANOVA total degrees of freedom/mean square are also publicly displayed by `OLSResult.summary()` but unmapped. Complete coverage is a release-critical requirement.

### TA-REG-003 — no-constant public ANOVA summary conflicts with the authoritative convention (`high`)

For `c=0`, `[R] regress`, p. 22 defines total degrees of freedom as `N`, and total mean square is consequently `TSS/N`. `src/tabra/results/ols_result.py:113` always computes total mean square as `SST/(N-1)`, and line 150 always displays total degrees of freedom as `N-1`. `OLSResult` does not retain `is_con`/`c`, so the `no_constant` public summary cannot select the correct convention. No test covers this display path. This is an implementation conflict even though the normalized validation schema does not export total df or total MS.

### TA-REG-004 — code and test locations are not complete or exact (`high`)

The R-squared rows point only to `OLS.fit`, while the formula is implemented in `EstimationStats.r_squared`. The F row points to `BaseModel._f_statistics`, which only delegates; the formula is in `EstimationStats.f_statistics`. Rows claiming `basic|no_constant` list a test that exercises only `basic`; the existing `test_no_constant_regression` is omitted from the map. Coefficient p-values are checked only for the range `[0,1]` at the OLS level, and no independent fixed-value OLS oracle is mapped for t/p, F p-value, sums of squares, or Root MSE.

### TA-REG-005 — assumptions and notation limitations are under-mapped (`medium`)

The theory text states exogeneity and homoskedasticity, but no row maps the Gaussian assumption needed for exact t/F inference or the complete-case sampling rule. The implementation names residual sum of squares `SSR` and model sum of squares `SSE`, the reverse of a common convention; normalization translates these correctly, but each affected row must state this parameterization to prevent future mapping errors. Degenerate guards returning `(0,0)` for invalid F inputs and `0` for nonpositive TSS/df are also undocumented limitations outside the frozen nondegenerate cases.

## Decision

`theory_audit_status = FAIL`.

All required source documents and code locations were accessible, so `BLOCKED_ENV` is not appropriate. A future audit can pass only after the frozen theory/formula artifacts precisely map every reported quantity to authoritative locations and independent tests/cases, and after the no-constant summary conflict is repaired and tested. The unsupported `robust` case remains excluded and was not counted as implemented support.
