# `reg` theory audit

- Run ID: `reg-validation-20260727-05`
- Audit date: 2026-07-27
- Audited commit: `9ce5b597d9e3a2d7b3168262519b2be3019116e9`
- `theory_audit_status`: **PASS**
- Auditor role: independent theory audit; no gate decision issued

## Findings

No findings. I found no mathematical contradiction, unsupported implemented
formula, assumption misstatement, or formula-to-code-to-test gap in the frozen
`reg` scope.

## Frozen evidence

The current `HEAD` and every hash in run05 `inputs.json` were independently
recomputed and matched: manifest `bfee2c59...21b7`, cases `1b107cfe...2c07`,
theory `fd71801b...8c`, formula map `5b3c7233...eb0`, fixture `53f6fea9...37c4`,
oracle `0d9ea042...6d68`, and normalized-result schema `05823ed8...0a89`.

The manifest contains 17 release-critical normalized metrics. The formula map
contains 21 data rows: 20 implemented mappings and one explicitly blocked
future robust-VCE mapping. The 17 metrics, public 95% coefficient intervals,
and constant/no-constant Total-row degrees of freedom and mean square all have
source, parameterization, production-code, test, and case mappings.

## Source audit

Lai and Xing (2008), *Statistical Models and Methods for Financial Markets*,
DOI <https://doi.org/10.1007/978-0-387-77827-3>, supplies the modern mathematical
support. I independently checked its chapter preview and searchable page
evidence at p. 4 equation (1.2), p. 5 equations (1.4)-(1.5), p. 7 equations
(1.9)-(1.10), pp. 8-10 equations (1.11)-(1.15) and (1.22), pp. 10-12 equations
(1.23)-(1.25) and Table 1.1, and pp. 13-14 equations (1.27)-(1.28).

Fisher (1922), “The Goodness of Fit of Regression Formulae, and the
Distribution of Regression Coefficients,” *JRSS* 85(4):597-612, DOI
<https://doi.org/10.2307/2341124>, supplies original finite-sample provenance.
The institutional scan (969966 bytes; SHA-256
`96fe8c1d8c57e62457c17ba46c18ddd12103ae6ee59abb7466ac96d67f96077f`)
was independently OCR-checked at journal pp. 607-610. Page 608 uses the `n-2`
residual divisor after two fitted constants and establishes the independence
used for Student's distribution; pp. 609-610 extend the argument to orthogonal
multiple-regression coefficients and `p+1` fitted constants. Fisher is not used
as a source for modern matrix notation or software behavior.

StataCorp (2025), Stata 19 `[R] regress`,
<https://www.stata.com/manuals/rregress.pdf>, was independently checked at p. 7,
pp. 8-10, p. 15, pp. 22-23, and p. 26. It confirms complete-case selection,
constant/no-constant ANOVA conventions, `N-k` residual degrees of freedom,
conventional homoskedastic covariance, two-sided coefficient inference, and
right-tailed F reporting. It is compatibility documentation, not mathematical
proof. Stata and R outputs were not used as theory evidence.

## Formula and implementation conclusions

- Complete cases are selected jointly across `y` and all requested `x` in
  `base.py:19-23`; `ols.py:36-43` then preserves requested regressor order and
  appends `_cons` only when requested.
- The scaled `lstsq` path in `ols.py:54-72` computes the full-rank OLS solution
  and maps coefficients back to original units.
- With `c=1`, TSS is centered, total df is `N-1`, and model df is `k-1`; with
  `c=0`, TSS is `y'y`, total df is `N`, and model df is `k`. Residual df is
  `N-k` in both cases (`ols.py:74-90`, `stats.py:14-27`).
- The SVD covariance path in `ols.py:91-106` implements the diagonal of
  `s^2(X'X)^-1`, where `s^2=RSS/(N-k)`, under the frozen full-rank,
  homoskedastic model.
- Coefficient statistics use `t=beta_hat/se` and two-sided Student-t survival
  probabilities with `N-k` df (`ols.py:127-142`, `ops/stats.py:40-50`). Public
  95% intervals use `t_(0.975,N-k)` for both constant conventions
  (`ols_result.py:155-166`).
- The overall statistic is `[MSS/(k-c)]/[RSS/(N-k)]` and its p-value is the
  right F tail (`ols.py:146-151`, `stats.py:39-44`, `ops/stats.py:53-64`).
- R-squared, adjusted R-squared, sums of squares, and Root MSE follow the same
  centered/uncentered convention (`ols.py:71-88,144-153`, `stats.py:22-37,69-76`).

## Assumptions and interpretation boundary

Conditional unbiasedness requires `E[u|X]=0`. The reported covariance requires
`Var(u|X)=sigma^2 I`. Exact finite-sample t, F, p-value, and 95% coverage claims
additionally require conditionally spherical normal errors. Full column rank,
`N>k`, positive relevant degrees of freedom, and nondegenerate sums of squares
are required for the frozen formulas. Complete-case estimation does not prove
ignorable missingness. Cross-engine agreement does not establish exogeneity,
and no coefficient is causal without a separate identification design.

## Test evidence and boundaries

`uv run pytest -q -o addopts='' tests/test_ols.py tests/test_ols_result.py tests/test_estimate_base_stats.py tests/ops/test_stats.py tests/test_reg_validation_oracle.py`
completed with **76 passed in 3.91s**, zero skips, and zero warnings. These tests
cover both constant conventions, complete-case selection, OLS coefficients,
VCE/SE, df, sums of squares, R-squared, adjusted R-squared, t/F tails, 95%
intervals, Root MSE, and frozen input provenance.

An earlier invocation ran the same 76 tests successfully but exited because the
focused subset alone produced 20.12% total-project coverage, below the
repository-wide 80% threshold. The rerun above disabled repository addopts to
isolate theory tests; full-project coverage remains a code-audit/gate concern.

Robust, weighted, clustered, rank-deficient, and other excluded designs are not
inside this PASS. The frozen fixture contains no missing values; complete-case
behavior is established by its dedicated unit test rather than the fixture.

## Final status

`PASS`: all release-critical mathematical claims and the additional public
report quantities have accessible support and complete implementation/test
traces. No finding remains. This is not the final release gate decision.
