# `reg` theory audit

- Run ID: `reg-validation-20260726-04`
- Audit date: 2026-07-26
- Audited commit: `47aebd1b2681607027a70275eb443060c5df0457`
- `theory_audit_status`: **PASS**
- Auditor role: independent theory audit; no gate decision issued

## Decision

The frozen `reg` contract passes this independent theory audit. I found no
mathematical contradiction, unsupported implemented formula, or mismatch between
the supported formulas and their production implementation. The manifest has 17
normalized release-critical metrics, all of which have complete formula-to-code-
to-test mappings. The public 95% coefficient intervals and ANOVA Total-row
degrees of freedom/mean square are additional mapped report quantities.

No Stata or R result was used as mathematical proof. Lai and Xing (2008) and
Fisher (1922) provide mathematical and original-method evidence. The Stata 19
manual is used only for software compatibility and reporting conventions.

## Frozen inputs

| Input | SHA-256 or commit | Result |
|---|---|---|
| Git commit | `47aebd1b2681607027a70275eb443060c5df0457` | match |
| `manifest.yaml` | `bfee2c597697569b212aeb67edbbf34d9e42421272d4430253d6724a45f121b7` | match |
| `cases.yaml` | `1b107cfe0d79a0997a04a5a251e4a1eb909404943d7e68f337aae8ea14612c07` | match |
| fixture | `53f6fea92a7978cbce09212c224eb6e29e9c581ab920e0285ef541c4b09d37c4` | match |
| `theory.md` | `fd71801b0d33503cf2acd224edb7662904c8de7de89cc2bd15c82dfcc047fb8c` | match |
| `formula_map.csv` | `5b3c7233a6522311ceed5936a72433d6c52f46596a92353b96aa8d90dad07eb0` | match |
| independent oracle | `0d9ea04248ca057d09ccaa0feb1d7c2510f6435ff6fe2e4ced85d03509cb6d68` | match |
| normalized-result schema | `05823ed8e65c55dab2db5d308305e1293fc433c3492511f32d85142d83a60a89` | match |

The tracked worktree was clean when the run inputs were captured. Untracked
files owned by other work were preserved. This audit writes only its two
run-specific outputs.

## Sources and evidence roles

### Mathematical source

Tze Leung Lai and Haipeng Xing (2008), *Statistical Models and Methods for
Financial Markets*, DOI <https://doi.org/10.1007/978-0-387-77827-3>, chapter 1,
chapter DOI <https://doi.org/10.1007/978-0-387-77827-3_1>.

The source was checked at these exact printed locations:

- p. 4, equation (1.2): least-squares RSS objective;
- p. 5, equations (1.4)-(1.5): coefficient solution and projection/RSS;
- p. 7, equations (1.9)-(1.10): homoskedastic covariance and `RSS/(N-k)`;
- pp. 8-10, equations (1.11)-(1.15) and (1.22): exact t inference,
  two-sided tests, and confidence intervals;
- pp. 10-12, equations (1.23)-(1.25) and Table 1.1: ANOVA and right-tail F;
- pp. 13-14, equations (1.27)-(1.28): intercept-model R-squared and adjusted
  R-squared.

### Original methodological source

R. A. Fisher (1922), “The Goodness of Fit of Regression Formulae, and the
Distribution of Regression Coefficients,” *JRSS* 85(4), 597-612, DOI
<https://doi.org/10.2307/2341124>; institutional record
<http://hdl.handle.net/2440/15174>.

The preserved scan was checked at journal pp. 607-610, section 6. Page 608 uses
the `n-2` residual divisor after fitting two simple-regression constants and the
independence needed for Student's distribution; pp. 609-610 extend the result to
multiple orthogonal regression coefficients and `p+1` fitted constants. This
source is not used to justify modern matrix notation, SVD implementation, or
Stata-specific reporting.

### Software-compatibility source

StataCorp LLC (2025), *Stata 19 Base Reference Manual*, `[R] regress`,
<https://www.stata.com/manuals/rregress.pdf>.

The manual was checked at p. 7 (two-sided coefficient tests and right-tail F),
pp. 8-10 (`noconstant` and uncentered ANOVA), p. 15 (complete cases), p. 22
(OLS and ANOVA definitions), p. 23 (homoskedastic VCE, F, R-squared, adjusted
R-squared), and p. 26 (`N-k` t-reference convention). These are compatibility
definitions, not mathematical or implementation proof.

## Model, assumptions, and interpretation

For the complete-case sample, let `N` be its size, `X` an `N x k` full-rank
design, `y` the response, and `c=1` when a constant is fitted and `c=0`
otherwise. The supported estimator is

\[
\hat\beta=\arg\min_b (y-Xb)'(y-Xb)=(X'X)^{-1}X'y.
\]

Conditional unbiasedness requires `E[u|X]=0`; the conventional covariance
requires `Var(u|X)=sigma^2 I`. Exact finite-sample t, F, p-value, and 95%
coverage statements additionally require conditional spherical normal errors.
The nondegenerate formulas require full column rank, `N>k`, positive residual
variance, and positive relevant degrees of freedom. Complete-case fitting does
not establish ignorable missingness. No causal interpretation follows without a
separate identification design.

## Formula-to-code-to-test trace

| Quantity | Formula ID(s) | Production location | Test evidence |
|---|---|---|---|
| case IDs | `REG-CASE-001` | `validation/runner_common.py:16-19`, `validation/python/run.py:65-93` | runner/hash tests in the validation suite |
| `nobs`, complete-case sample | `REG-SAMPLE-001` | `base.py:19-23`, `ols.py:36-43` | `test_partial_missing_rows_use_joint_complete_case_sample` |
| parameter names/order | `REG-PARAM-001` | `ols.py:38-42` | constant/no-constant parameter tests |
| model/residual/total df | `REG-DF-001`, `REG-NOCONSTANT-001` | `stats.py:14-19`, `ols.py:89-90`, `ols_result.py:107-152` | df helper and rendered Total-row tests |
| OLS objective/coefficients | `REG-OBJ-001`, `REG-COEF-001` | `ols.py:54-72` | independent least-squares tests |
| RSS, TSS, MSS | `REG-RSS-001`, `REG-TSS-001`, `REG-MSS-001` | `ols.py:71-88` | decomposition and no-constant tests |
| homoskedastic VCE/SE | `REG-VCE-001`, `REG-SE-001` | `ols.py:91-106` | inverse/SVD numerical tests |
| t statistics/two-sided p | `REG-T-001`, `REG-TP-001` | `ols.py:127-142`, `ops/stats.py:40-50` | coefficient and tail-probability tests |
| 95% coefficient interval | `REG-CI95-001` | `ols_result.py:155-166` | `test_95_percent_confidence_interval_uses_t_distribution` for both constant conventions |
| R-squared/adjusted R-squared | `REG-R2-001`, `REG-AR2-001` | `ols.py:144-145`, `stats.py:22-37` | constant/no-constant tests |
| F/right-tail p | `REG-F-001`, `REG-FP-001` | `ols.py:146-151`, `stats.py:39-44`, `ops/stats.py:53-64` | manual-formula and tail tests |
| residual MSE/Root MSE | `REG-RMSE-001`, `REG-VCE-001` | `ols.py:91,152-160`, `stats.py:69-76`, `ols_result.py:107-114` | MSE/Root-MSE helper tests |
| no-constant Total df/MS | `REG-NOCONSTANT-001` | `ols_result.py:107-152` | `test_total_df_and_ms_follow_constant_convention` |

The formula map contains 21 data rows: 20 implemented rows and one explicitly
blocked robust-VCE row. `REG-VCE-ROBUST-001` is not counted as implemented or
release-supported.

## Focused conclusions

- With a constant, `TSS=sum((y-ybar)^2)`, total df is `N-1`, and model df is
  `k-1`. Without a constant, `TSS=y'y`, total df is `N`, and model df is `k`.
- Residual df is `N-k` in both conventions; residual variance is `RSS/(N-k)`.
- The coefficient interval is
  `[beta_hat_j - t_(0.975,N-k) se_j, beta_hat_j + t_(0.975,N-k) se_j]`.
- Coefficient p-values are two-sided; the joint F p-value is right-tailed.
- The SVD implementation computes the same diagonal of
  `s^2 (X'X)^(-1)` under the supported full-rank assumptions.

## Execution evidence and boundaries

The focused command

`uv run pytest -q -o addopts='' tests/test_ols.py tests/test_estimate_base_stats.py tests/test_ols_result.py tests/ops/test_stats.py`

completed with **73 passed in 1.16s**, zero skips, and zero warnings.

This theory audit did not execute the independent oracle and did not use any
Stata/R engine comparison as proof. Robust, weighted, clustered, rank-deficient,
and other excluded designs remain outside this PASS. The frozen fixture contains
no missing observations; complete-case implementation is supported by a focused
unit test, not by the fixture itself.

## Final status

`PASS`: no findings. Required papers, definitions, source locations, production
locations, and tests were available. This is a theory-audit result only; the
gatekeeper must issue the final validation decision.
