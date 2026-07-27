# `reg` theory audit

- Run ID: `reg-validation-20260726-02`
- Audit date: 2026-07-26
- Audited commit: `4caa15acacb770c5ee95b0fe6ffdea08fd1d20bd`
- `theory_audit_status`: **FAIL**
- Auditor role: independent theory audit; no gate decision

## Decision summary

The OLS formulas implemented for the frozen `basic` and `no_constant` cases agree with the independently checked primary sources, the official Stata 19 compatibility conventions, the production code, and the independent executable oracle. The joint complete-case rule is also correctly mapped to an exact implementation and a meaningful partial-missing unit test.

The audit nevertheless fails because the frozen formula map is not complete for all publicly reported quantities. `OLSResult.summary()` reports coefficient confidence intervals, but no frozen formula row records their formula, assumptions, source location, implementation location, and independent test or case ID. In addition, `REG-NOCONSTANT-001` maps the public summary implementation but omits the existing test that actually verifies its total-degrees-of-freedom and total-mean-square convention. Under the audit rule, an unmapped release-critical formula cannot pass.

## Frozen inputs

| Input | Frozen SHA-256 / value | Observed | Result |
|---|---|---|---|
| Git commit | `4caa15acacb770c5ee95b0fe6ffdea08fd1d20bd` | same | match |
| `manifest.yaml` | `3ace49ec0668c8deb699db3e6db90ade5f7e7a8105c61ff9163dd438c628068e` | same | match |
| `cases.yaml` | `1b107cfe0d79a0997a04a5a251e4a1eb909404943d7e68f337aae8ea14612c07` | same | match |
| fixture | `53f6fea92a7978cbce09212c224eb6e29e9c581ab920e0285ef541c4b09d37c4` | same | match |
| `theory.md` | `fc99f885e2308657ba0ddb231d0496bd54aed7d7a07d038d5db08747a4e4d4b8` | same | match |
| `formula_map.csv` | `51420b1104a1db0e110fb74760cc9b992ba89c9595afd605b1d43b69d705dcf7` | same | match |
| independent oracle | `007099d58e6a26ba4e0eb3236537d004e453b14595b3ec78a4012dfc954ea3af` | same | match |

Only untracked files were present in the worktree at audit time; no tracked source modification was observed. This audit wrote only the two run-specific theory-audit outputs.

## Independently verified sources and roles

### Primary mathematical source

Lai, T. L. and Xing, H. (2008), *Statistical Models and Methods for Financial Markets*, Springer Texts in Statistics, DOI: <https://doi.org/10.1007/978-0-387-77827-3>; chapter DOI: <https://doi.org/10.1007/978-0-387-77827-3_1>.

The printed locations were checked directly:

- p. 4, equation (1.2): least-squares RSS objective;
- p. 5, equations (1.4)–(1.5): coefficient solution and orthogonal decomposition, followed by the residual/RSS definition;
- p. 7, equations (1.9)–(1.10): homoskedastic covariance and `RSS/(n-p)` residual-variance estimator under full rank, zero-mean errors, common variance, and no cross-observation covariance;
- pp. 8–10, equations (1.11)–(1.14) and (1.22): exact normal-error coefficient distribution, Student t statistic, and two-sided coefficient test;
- pp. 10–12, equations (1.23)–(1.25) and Table 1.1: projection/SS and degrees-of-freedom decomposition and right-tail F test;
- pp. 13–14, equations (1.27)–(1.28): centered intercept-model R-squared and adjusted R-squared.

These locations support the modern mathematical contract. Equations (1.27)–(1.28) do not directly establish Stata's uncentered no-intercept convention; that parameterization is separately documented by the official software manual.

### Original methodological source

Fisher, R. A. (1922), “The Goodness of Fit of Regression Formulae, and the Distribution of Regression Coefficients,” *Journal of the Royal Statistical Society* 85(4), 597–612, DOI: <https://doi.org/10.2307/2341124>; institutional scan: <http://hdl.handle.net/2440/15174>.

Journal pp. 607–610, section 6 were checked from the scan. Page 608 derives the simple-regression residual-variance estimate using the `n-2` divisor and independence from a fitted coefficient; p. 609 obtains Student's Type VII form and extends the argument to orthogonal multiple-regression coefficients; p. 610 gives the `n-p-1` divisor after fitting `p+1` constants and the corresponding coefficient distribution. This is valid original support for finite-sample regression-coefficient t inference. It is not treated as a source for modern matrix notation or the current software's numerical algorithm.

### Software compatibility source

StataCorp LLC (2025), *Stata 19 Base Reference Manual*, `[R] regress`, <https://www.stata.com/manuals/rregress.pdf>.

The official manual was used only to resolve software behavior and reporting parameterization:

- p. 7: two-sided coefficient significance and right-tail overall F interpretation;
- pp. 8–10: default intercept, `noconstant`, and the uncentered ANOVA/output convention;
- p. 15: an observation is dropped when one or more model variables are missing;
- p. 22, equation (2): Stata's OLS coefficient formula, objective, centered/uncentered TSS, RSS/MSS, and degrees of freedom;
- p. 23, equation (3): F, R-squared, adjusted R-squared, and conventional homoskedastic covariance;
- p. 26: t reference distribution with `N-k` degrees of freedom for homoskedastic p-values and confidence intervals.

The Stata manual is authoritative compatibility documentation, not independent mathematical proof and not evidence that Tabra's implementation is correct. Stata and R output were not used as theory proof.

## Mathematical contract checked

For the retained complete-case sample, let `N` be the observation count, `X` an `N x k` full-column-rank design, and `c=1` with a fitted constant and `c=0` without one. The supported estimator and conventional covariance are

\[
\hat\beta=(X'X)^{-1}X'y,\qquad
s^2=RSS/(N-k),\qquad
\widehat V(\hat\beta)=s^2(X'X)^{-1}.
\]

With `e=y-X beta_hat`, the reporting conventions are

\[
RSS=e'e,\quad
TSS=\begin{cases}\sum_i(y_i-\bar y)^2,&c=1,\\y'y,&c=0,\end{cases}
\quad MSS=TSS-RSS,
\]

\[
df_{model}=k-c,\quad df_{resid}=N-k,\quad df_{total}=N-c,
\]

\[
R^2=1-RSS/TSS,\quad
\bar R^2=1-(1-R^2)(N-c)/(N-k),
\]

\[
F=\frac{MSS/(k-c)}{RSS/(N-k)},\qquad Root\ MSE=\sqrt{RSS/(N-k)}.
\]

For coefficient `j`, `t_j=beta_hat_j/se_j` and `p_j=2 Pr(T_(N-k) >= |t_j|)`; the overall p-value is `Pr(F_(k-c,N-k) >= F)`. Exact finite-sample t/F calibration requires conditionally spherical normal errors. Conditional unbiasedness requires `E[u|X]=0`; the conventional covariance requires `Var(u|X)=sigma^2 I`. Cross-engine agreement establishes none of these assumptions, and no causal interpretation follows without a separate identification design.

## Formula, code, and test trace

| Objects | Frozen formula IDs | Source result | Exact implementation | Independent evidence | Audit result |
|---|---|---|---|---|---|
| case scope, sample, names, df | `REG-CASE-001`, `REG-SAMPLE-001`, `REG-PARAM-001`, `REG-DF-001` | exact/authoritative locations verified | `runner_common.py`, `base.py:19-23`, `ols.py:36-43`, `stats.py:14-19` | frozen cases, oracle, named unit/API tests | complete |
| objective, coefficients | `REG-OBJ-001`, `REG-COEF-001` | Lai–Xing (1.2), (1.4)–(1.5) verified | `ols.py:54-72` | oracle and mapped NumPy/lstsq tests | complete |
| RSS, TSS, MSS | `REG-RSS-001`, `REG-TSS-001`, `REG-MSS-001` | Lai–Xing and Stata parameterization verified | `ols.py:73-88` | oracle and mapped decomposition/no-constant tests | complete |
| covariance, SE | `REG-VCE-001`, `REG-SE-001` | Lai–Xing (1.9)–(1.10) verified | `ols.py:91-106` | SVD oracle and mapped SE tests | complete |
| t and coefficient p-value | `REG-T-001`, `REG-TP-001` | Lai–Xing and Fisher exact locations verified | `ols.py:127-142`, `ops/stats.py:40-50` | oracle and mapped distribution tests | complete |
| R-squared, adjusted R-squared | `REG-R2-001`, `REG-AR2-001` | intercept formulas and no-constant compatibility convention verified | `ols.py:144-145`, `stats.py:22-37` | oracle and mapped constant/no-constant tests | complete |
| F, F p-value, Root MSE | `REG-F-001`, `REG-FP-001`, `REG-RMSE-001` | exact source locations verified | `ols.py:146-153`, `stats.py:40-44,69-76`, `ops/stats.py:53-64` | oracle and mapped tests | complete |
| no-constant public ANOVA summary | `REG-NOCONSTANT-001` | Stata convention verified | `ols_result.py:107-152` | exact test exists but is absent from frozen row | **incomplete map** |
| coefficient 95% confidence interval | no formula ID | Lai–Xing (1.15) and Stata p. 26 are available | `ols_result.py:158-166` | no numerical confidence-interval test found | **unmapped** |

`REG-VCE-ROBUST-001` remains explicitly `blocked` and outside the frozen supported cases. It was not counted as implemented support and was not used to justify the homoskedastic VCE.

## Required partial-missing check

The partial-missing mapping is valid:

- compatibility rule: Stata 19 `[R] regress`, p. 15;
- implementation: `src/tabra/models/estimate/base.py:19-23` first restricts to `[y] + x` and then applies rowwise `dropna()`;
- caller: `src/tabra/models/estimate/ols.py:36` applies that sample before constructing `y` and `X`;
- independent test: `tests/test_ols.py::test_partial_missing_rows_use_joint_complete_case_sample` places missing values separately in `y`, `x1`, and `x2`, puts a missing value in an unused column, reverses requested regressor order, and independently checks retained `N`, names, coefficients, standard errors, residuals, and R-squared.

The test passed in the frozen `.venv` environment. The frozen three-engine fixture itself has no missing observations, a limitation correctly recorded by `REG-SAMPLE-001`. Neither the unit test nor complete-case computation establishes ignorability of the missingness mechanism.

## Findings

### TA-REG-02-001 — publicly reported confidence intervals are unmapped (`critical`)

`src/tabra/results/ols_result.py:155-166` labels and reports 95% confidence intervals, using `stats.t.ppf(0.975, df_resid)` and `coef ± t_crit * se`. No row in `formula_map.csv` defines this reported quantity or records its assumptions, exact source location, parameterization, implementation, and test/case ID. No numerical confidence-interval test was found. Mentioning confidence intervals in `theory.md` or in the Stata compatibility discussion does not constitute the required end-to-end formula map.

Required action: add a frozen confidence-interval formula row, using an exact mathematical source such as Lai and Xing p. 8, equation (1.15), record the `N-k`/two-sided 95% parameterization and normal-error condition, map `OLSResult._summary_style_stata`, and add/map an independent numerical test for both supported constant conventions.

### TA-REG-02-002 — no-constant public summary test is not recorded in the frozen map (`high`)

`REG-NOCONSTANT-001` maps `OLSResult._summary_style_stata` and claims that the public ANOVA summary uses total df `N` and total MS `TSS/N`, but its `independent_test_location` does not list the test that exercises this code path. The listed oracle and tests validate normalized no-constant fit statistics, not the rendered summary's total row. The exact existing test is `tests/test_ols_result.py::TestAnovaAlignment::test_total_df_and_ms_follow_constant_convention`, and it passed, but a test that is absent from the frozen formula row is not a complete recorded mapping.

Required action: add that exact test ID to `REG-NOCONSTANT-001` (or to a dedicated public-summary row) in a newly frozen formula map.

## Execution evidence

- Eight targeted mapped tests, including the required partial-missing test and the no-constant total-row test, passed under `.venv/bin/python`: `8 passed`.
- The independent oracle ran without importing Tabra, verified the frozen manifest/cases/data hashes, and returned `oracle_status = PASS` for every frozen normalized metric in `basic` and `no_constant`.
- Oracle and test success are implementation evidence only. They cannot cure a missing formula-map row or omitted test ID.

## Decision

`theory_audit_status = FAIL`.

The required primary sources, official manual, source locations, fixture, and tests were accessible, so `BLOCKED_ENV` is not appropriate. This audit does not issue the validation gate decision. A later theory audit may pass only after the confidence-interval formula has a complete frozen source/code/test mapping and the no-constant public-summary test is explicitly recorded.
