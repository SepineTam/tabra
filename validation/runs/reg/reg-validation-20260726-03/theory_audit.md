# `reg` theory audit

- Run ID: `reg-validation-20260726-03`
- Audit date: 2026-07-26
- Audited commit: `64c92fe6b136b5b3adc12895d408e5c20af9214c`
- `theory_audit_status`: **PASS**
- Auditor role: independent theory audit; no gate decision issued

## Decision summary

The frozen `reg` contract passes this independent theory audit. Every normalized
release-critical metric, together with the two public-summary quantities outside
the normalized schema, has authoritative support and a complete formula-to-code-
to-test trace. In particular, `REG-CI95-001` correctly fixes a two-sided 95%
Student-t interval with residual degrees of freedom `N-k` and has a parameterized
numerical test for both constant conventions. `REG-NOCONSTANT-001` correctly maps
the public `Total` row to uncentered `TSS=y'y`, total degrees of freedom `N`, and
total mean square `TSS/N`, and names the exact test that checks the rendered
summary.

No Stata or R result was used as mathematical proof. Lai and Xing (2008) and
Fisher (1922) provide mathematical and original-method evidence; the Stata 19
manual is used only for software compatibility and reporting parameterization.

## Frozen inputs

| Input | Frozen value | Independently observed | Result |
|---|---|---|---|
| Git commit | `64c92fe6b136b5b3adc12895d408e5c20af9214c` | same | match |
| `manifest.yaml` | `902392d562a50aa9919cfc58bc95b0018e63044452fa8b6203608142814ecf79` | same | match |
| `cases.yaml` | `1b107cfe0d79a0997a04a5a251e4a1eb909404943d7e68f337aae8ea14612c07` | same | match |
| fixture | `53f6fea92a7978cbce09212c224eb6e29e9c581ab920e0285ef541c4b09d37c4` | same | match |
| `theory.md` | `fd71801b0d33503cf2acd224edb7662904c8de7de89cc2bd15c82dfcc047fb8c` | same | match |
| `formula_map.csv` | `5b3c7233a6522311ceed5936a72433d6c52f46596a92353b96aa8d90dad07eb0` | same | match |
| independent oracle | `41a47dcda1e3f83a4b21cada2098e001ff861b00ed9a6eb100dfce87b09e438c` | same | match |

The tracked worktree was clean at audit time. Untracked files belonging to other
work were present and were not modified. This audit writes only the two
run-specific theory-audit outputs.

## Sources independently checked and evidence roles

### Primary mathematical source

Tze Leung Lai and Haipeng Xing (2008), *Statistical Models and Methods for
Financial Markets*, Springer Texts in Statistics, book DOI
<https://doi.org/10.1007/978-0-387-77827-3>, chapter 1 “Linear Regression Models”,
chapter DOI <https://doi.org/10.1007/978-0-387-77827-3_1>, pp. 3–35.

The cited printed locations were checked against the source:

- p. 4, equation (1.2): least-squares residual-sum-of-squares objective;
- p. 5, equations (1.4)–(1.5): coefficient solution and projection/RSS
  decomposition;
- p. 7, equations (1.9)–(1.10): homoskedastic covariance and the
  `RSS/(n-p)` variance estimator;
- pp. 8–10, equations (1.11)–(1.15) and (1.22): exact normal-error t
  distribution, two-sided coefficient test, and confidence interval;
- pp. 10–12, equations (1.23)–(1.25) and Table 1.1: sums-of-squares and
  degrees-of-freedom decomposition and right-tail F test;
- pp. 13–14, equations (1.27)–(1.28): centered intercept-model R-squared and
  adjusted R-squared.

The source's standard treatment assumes nonrandom regressors. The contract's
conditional-on-`X` formulation preserves the same inferential conditions. Its
intercept formulas are not presented as support for Stata's uncentered
`noconstant` reporting convention.

### Original methodological source

R. A. Fisher (1922), “The Goodness of Fit of Regression Formulae, and the
Distribution of Regression Coefficients,” *Journal of the Royal Statistical
Society* 85(4), 597–612, DOI <https://doi.org/10.2307/2341124>. The University of
Adelaide R. A. Fisher Digital Archive record is
<http://hdl.handle.net/2440/15174>; the institutional scan was checked through
its preserved repository copy.

Journal pp. 607–610, section 6, were checked directly. Page 608 uses the `n-2`
residual-variance divisor after fitting the two simple-regression constants and
establishes independence used for Student's distribution. Pages 609–610 extend
the result to orthogonal multiple-regression coefficients and to `p+1` fitted
constants. Fisher is used only as original finite-sample regression-coefficient
distribution support, not as a source for modern matrix notation, the present
SVD implementation, R-squared, or Stata reporting conventions.

### Authoritative software-compatibility source

StataCorp LLC (2025), *Stata 19 Base Reference Manual*, `[R] regress`,
<https://www.stata.com/manuals/rregress.pdf>.

The manual was checked at p. 7 (two-sided coefficient significance and
right-tail F interpretation), pp. 8–10 (default constant, `noconstant`, and
uncentered ANOVA reporting), p. 15 (model-variable complete-case rule), p. 22
equation (2) and the ANOVA definitions, p. 23 equation (3), and p. 26 (`N-k`
t-reference degrees of freedom for homoskedastic p-values and confidence
intervals). These are compatibility definitions and cross-checks, not
mathematical proof or evidence that Tabra code is correct.

## Population model, identification, and inferential target

For the complete-case sample, let `N` be its size, `X` an `N x k`
full-column-rank design, `y` the response, and `c=1` when a constant is fitted
and `c=0` otherwise. The supported model and estimator are

\[
y=X\beta+u,\qquad
\hat\beta=\arg\min_b (y-Xb)'(y-Xb)=(X'X)^{-1}X'y.
\]

Conditional unbiasedness requires `E[u|X]=0`. The conventional covariance
requires `Var(u|X)=sigma^2 I`. Exact finite-sample t, F, p-value, and 95%
coverage statements additionally require conditional spherical normal errors.
Full column rank, `N>k`, positive residual variance, and the relevant positive
degrees of freedom are required for the nondegenerate frozen formulas.
Complete-case fitting changes the target sample; neither software agreement nor
the missing-row test establishes an ignorable missingness mechanism. No causal
or structural interpretation follows without a separate identification design.

## Complete release-critical formula trace

The manifest contains 18 normalized release-critical metrics. The public summary
also reports coefficient 95% intervals and total ANOVA degrees of freedom/mean
square. Model and residual mean squares are the numerator and denominator
components already fixed by `REG-MSS-001`, `REG-RSS-001`, `REG-DF-001`,
`REG-F-001`, and `REG-RMSE-001`; the total-row constant convention is fixed by
`REG-DF-001`, `REG-TSS-001`, and the direct rendered-row mapping
`REG-NOCONSTANT-001`.

| Public or normalized quantity | Formula ID(s) | Production location | Independent evidence | Result |
|---|---|---|---|---|
| case IDs | `REG-CASE-001` | `runner_common.py:16-19`, `python/run.py:65-93` | oracle case/hash checks | complete |
| `nobs` and complete-case sample | `REG-SAMPLE-001` | `base.py:19-23`, `ols.py:36-43` | partial-missing test and oracle | complete |
| parameter names/order | `REG-PARAM-001` | `ols.py:38-42` | constant/no-constant name tests and oracle | complete |
| `df_model`, `df_resid`, total df | `REG-DF-001`, `REG-NOCONSTANT-001` | `stats.py:13-19`, `ols.py:89-90`, `ols_result.py:113-114` | df helper, oracle, and rendered Total-row test | complete |
| OLS objective and coefficients | `REG-OBJ-001`, `REG-COEF-001` | `ols.py:54-72` | independent `lstsq` tests and oracle | complete |
| `ss_residual` / public residual SS | `REG-RSS-001` | `ols.py:71-73` | decomposition test and oracle | complete |
| `ss_total` / centered or uncentered total SS | `REG-TSS-001`, `REG-NOCONSTANT-001` | `ols.py:74-80` | no-constant test and oracle | complete |
| `ss_model` / public model SS | `REG-MSS-001` | `ols.py:80-88` | decomposition test and oracle | complete |
| homoskedastic covariance and standard errors | `REG-VCE-001`, `REG-SE-001` | `ols.py:91-106` | independent inverse/SVD tests and oracle | complete |
| coefficient t statistics and two-sided p-values | `REG-T-001`, `REG-TP-001` | `ols.py:127-142`, `ops/stats.py:40-50` | tail-probability tests and oracle | complete |
| coefficient 95% intervals | `REG-CI95-001` | `ols_result.py:155-166` | parameterized constant/no-constant numerical test | complete |
| R-squared and adjusted R-squared | `REG-R2-001`, `REG-AR2-001` | `ols.py:144-145`, `stats.py:22-37` | constant/no-constant tests and oracle | complete |
| F statistic and right-tail p-value | `REG-F-001`, `REG-FP-001` | `ols.py:146-151`, `stats.py:39-44`, `ops/stats.py:53-64` | manual-formula/tail tests and oracle | complete |
| Root MSE / residual MS | `REG-RMSE-001`, `REG-VCE-001` | `ols.py:91,152-153`, `stats.py:68-76`, `ols_result.py:111-114` | MSE/Root-MSE helper test and oracle | complete |
| no-constant Total df and Total MS | `REG-NOCONSTANT-001` | `ols_result.py:107-152` | `TestAnovaAlignment::test_total_df_and_ms_follow_constant_convention` | complete |

The formula map has 20 data rows: 19 implemented rows and one explicitly blocked
future robust-VCE row. `REG-VCE-ROBUST-001` is not counted as implemented support;
the public `reg` API exposes no robust option and the frozen `robust` case remains
excluded.

## Focused checks required for this run

### `REG-CI95-001`

The public summary computes

\[
[\hat\beta_j-t_{0.975,N-k}se_j,\ \hat\beta_j+t_{0.975,N-k}se_j].
\]

`OLSResult._summary_style_stata` uses
`stats.t.ppf(0.975, self._df_resid)` at line 158 and applies the critical value at
lines 164–165. `self._df_resid=N-k` originates in `OLS.fit`. The mapped test
`tests/test_ols_result.py::TestCoefficientConfidenceInterval::test_95_percent_confidence_interval_uses_t_distribution`
is parameterized over `is_con=True` and `False`, independently reconstructs
`X'X`, `RSS/(N-k)`, standard errors, and the 0.975 t quantile, then checks every
displayed endpoint at the summary's three-decimal precision. Both parameter sets
passed. This is an exact formula/parameterization test, not a label-only test.

### `REG-NOCONSTANT-001`

For `is_con=False`, production uses uncentered `TSS=y'y`, `df_model=k`,
`df_total=N`, adjusted-R-squared numerator `N`, and `Total MS=TSS/N`.
`OLSResult._summary_style_stata` sets
`total_df = n_obs - (1 if is_con else 0)` and `ms_total=SST/total_df` at lines
113–114. The formula row explicitly names
`tests/test_ols_result.py::TestAnovaAlignment::test_total_df_and_ms_follow_constant_convention`,
which parses the public `Total` row for both constant conventions and checks
`N-1`, `TSS/(N-1)`, `N`, and `TSS/N`. The test passed.

### Source-role separation

Every implemented mathematical row has a non-Stata mathematical source and
location. The software-only case, sample, parameter-label, and no-constant
convention rows mark mathematical support `not_applicable` and separately cite
the frozen scope or official Stata behavior. Fisher is restricted to
`REG-T-001` and `REG-TP-001`. The automated source-role structural test passed.

## Execution evidence

- `76 passed` for the independently rerun OLS, OLS-result, estimation-statistics,
  tail-probability, oracle, and formula-source-role tests with coverage disabled
  for this targeted audit invocation.
- An earlier invocation of the same selected tests produced `75 passed` but a
  nonzero process status solely because the repository's global 80% coverage
  threshold is not meaningful for a targeted subset; rerunning with `--no-cov`
  separated assertion results from that unrelated threshold.
- The independent executable oracle imported no Tabra code, verified all frozen
  manifest/cases/data hashes, reported completed/non-skipped/converged status,
  and returned `oracle_status=PASS` with every normalized metric comparison
  passing for `basic` and `no_constant`.
- Oracle agreement is implementation evidence only and did not substitute for
  the source audit.

## Findings and decision

No open theory finding remains. Required papers, the authoritative manual,
source locations, and independent tests were accessible, so `BLOCKED_ENV` does
not apply.

`theory_audit_status = PASS`.

This is the theory-audit decision only. It does not issue the final validation
gate decision.
