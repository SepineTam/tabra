# `reg` mathematical contract

This specification fixes the mathematical meaning of every quantity exported by
the frozen `reg` manifest. Agreement among Tabra, Stata, and R is validation
evidence, not a substitute for this contract or for an independent formula
oracle.

## Supported scope and notation

The supported cases are unweighted ordinary least squares with the conventional
homoskedastic covariance:

- `basic`: regressors `x1`, `x2`, and a fitted constant named `_cons`;
- `no_constant`: regressors `x1` and `x2`, with no fitted constant;
- `robust`: excluded because the public API does not expose a robust covariance
  option.

Tabra first retains rows that are complete for the dependent variable and every
requested regressor. Stata's `[R] regress`, p. 15, likewise documents dropping an
observation when any model variable is missing. Let $N$ (`nobs`) be the number
of retained observations, $X$ the resulting $N\times k$ full-column-rank
design matrix, $y$ the retained response, and

\[
c=\begin{cases}
1,&\text{a constant is fitted},\\
0,&\text{no constant is fitted}.
\end{cases}
\]

Here $k$ counts every fitted coefficient, including the constant when
$c=1$. Thus `df_model` is $k-c$, `df_resid` is $N-k$, and total ANOVA
degrees of freedom are $N-c$. Lai and Xing (2008), p. 7, equations
(1.9)--(1.10), support the residual degrees of freedom and residual-variance
formula for a full-rank model. The role of $c$ and the no-constant reporting
parameterization are Stata compatibility conventions: `[R] regress`, pp. 8--10
and 22, explains that `noconstant`, unlike `hascons`, uses the uncentered ANOVA
convention.

## Model, objective, and coefficients

Conditionally on $X$, the model is

\[
y=X\beta+u.
\]

The estimator minimizes the residual sum of squares,

\[
\hat\beta=\arg\min_b (y-Xb)'(y-Xb)
            =(X'X)^{-1}X'y.
\]

Lai and Xing (2008), p. 4, equation (1.2), gives the least-squares objective;
p. 5, equations (1.4)--(1.5), gives the coefficient solution and associated
orthogonal decomposition. The inverse representation requires full column
rank. `[R] regress`, p. 22, equation (2), is a software-compatibility
cross-check. Tabra orders the slope coefficients as supplied by `x` and appends
`_cons` only for `is_con=True`; the no-constant case has no `_cons` entry.

## ANOVA decomposition and fit statistics

Let $e=y-X\hat\beta$. Lai and Xing (2008), p. 5, equation (1.4) and the
following RSS definition, supports $RSS=e'e$. Its pp. 10--12, equations
(1.23)--(1.25) and Table 1.1, supports the centered sums-of-squares and
degrees-of-freedom decomposition. Using the $c$ convention above,

\[
RSS=e'e,
\qquad
TSS=\begin{cases}
\sum_i(y_i-\bar y)^2,&c=1,\\
y'y,&c=0,
\end{cases}
\qquad
MSS=TSS-RSS.
\]

The normalized schema maps these quantities as follows:

| Normalized name | Mathematical name | Tabra result attribute |
|---|---|---|
| `ss_residual` | $RSS$ | `SSR` |
| `ss_model` | $MSS$ | `SSE` |
| `ss_total` | $TSS$ | `SST` |

The Tabra attribute names `SSR` and `SSE` are therefore the reverse of a common
sum-of-squares naming convention; normalization must preserve the mapping in
this table.

With $s^2=RSS/(N-k)$, the remaining reported fit quantities are

\[
R^2=1-\frac{RSS}{TSS},
\]

\[
\bar R^2=1-(1-R^2)\frac{N-c}{N-k},
\]

\[
F=\frac{MSS/(k-c)}{RSS/(N-k)}
  =\frac{MSS}{(k-c)s^2},
\qquad
Root\ MSE=s=\sqrt{\frac{RSS}{N-k}}.
\]

Lai and Xing (2008), pp. 13--14, equations (1.27)--(1.28), supports the
intercept-model $R^2$ and adjusted $R^2$ formulas. Its p. 7, equations
(1.9)--(1.10), supports $s^2=RSS/(N-k)$ and Root MSE. `[R] regress`, pp. 9--10
and 22--23, supplies the software-specific no-constant parameterization and
cross-checks the reported quantities. In the `basic` case $c=1$; in
`no_constant` $c=0$, so TSS is uncentered, model and total degrees of freedom
increase by one relative to an intercept fit with the same $k$, and the
adjusted-$R^2$ numerator is $N$, not $N-1$. Lai and Xing equation (1.28) does
not directly support this no-constant version.

## Classical covariance and inference

Lai and Xing (2008), p. 7, equations (1.9)--(1.10), supports the conventional
homoskedastic covariance estimator and the residual-variance estimate:

\[
\widehat{V}(\hat\beta)=s^2(X'X)^{-1}.
\]

For coefficient $j$, the normalized standard error, test statistic, and
two-sided p-value are

\[
se_j=\sqrt{\widehat{V}(\hat\beta)_{jj}},
\qquad
t_j=\frac{\hat\beta_j}{se_j},
\qquad
p_j=2\Pr\{T_{N-k}\ge |t_j|\}.
\]

The public two-sided 95% coefficient confidence interval is

\[
CI_{0.95}(\beta_j)=
[\hat\beta_j-t_{0.975,N-k}se_j,\
 \hat\beta_j+t_{0.975,N-k}se_j].
\]

Lai and Xing (2008), p. 8, equation (1.15), is the mathematical source for
this interval. The parameterization uses $\alpha=0.05$ with 0.025 in each
tail. The fitted constant, when present, is included in $k$, so both constant
conventions use $N-k$ residual degrees of freedom. `[R] regress`, p. 26,
supplies this software-compatibility parameterization. The interval requires
full column rank, $N>k$, $se_j>0$, and the conventional homoskedastic spherical
covariance assumptions above; exact finite-sample 95% coverage additionally
requires $u\mid X\sim N(0,\sigma^2 I_N)$. Robust and cluster intervals are
outside the supported scope. The public summary's three decimal places are
display precision, not internal estimation precision.

The overall test is $H_0:\beta_j=0$ for every nonconstant coefficient when
$c=1$, and for all $k$ coefficients when $c=0$. Its normalized p-value is

\[
p_F=\Pr\{F_{k-c,N-k}\ge F\}.
\]

Lai and Xing (2008), pp. 8--10, equations (1.11)--(1.14) and (1.22), supports
the exact t distribution under normal errors, coefficient tests, and the
two-sided absolute-value statistic. Its pp. 10--12, equations (1.23)--(1.25)
and Table 1.1, supports the F ratio and right-tail rejection rule. Fisher
(1922), journal pp. 607--610, section 6, is original-paper support for
finite-sample t inference for regression coefficients: p. 608 uses the
$n-2$ residual-variance divisor for simple regression, and pp. 609--610 extends
the argument to multiple orthogonal regression coefficients and the Student
Type VII distribution. Fisher is not used as if it stated the modern matrix
formulas.

For software compatibility, `[R] regress`, p. 7, describes the coefficient t
tests as two-sided and the overall F p-value as the probability of an F
statistic at least as large as the reported value. Page 23 supplies the Stata F
formula and its two degrees of freedom. Page 26 specifies that
homoskedastic-regression coefficient p-values and confidence intervals use a t
distribution with $N-k$ degrees of freedom. Its p. 22, equation (2), and p. 23,
equation (3), cross-check the Stata coefficient and covariance parameterization.

## Assumptions and interpretation boundary

The numerical formulas and their scientific interpretation require distinct
conditions:

1. $X$ must have full column rank. The implementation rescales each design
   column by its maximum absolute value before the `lstsq` solve, maps the
   coefficients and SVD covariance factor back to the original units, and
   rejects a rank deficiency detected on the scaled design. This preserves the
   estimand while reducing false rank failures caused only by measurement units.
2. $E[u\mid X]=0$ is required for conditional unbiasedness of OLS. It is not
   established by fitting the model or by cross-engine agreement.
3. The reported conventional covariance requires
   \(\operatorname{Var}(u\mid X)=\sigma^2 I_N\): common conditional variance and
   no conditional correlation. Heteroskedastic or dependent errors require a
   different covariance estimator, which is outside the supported scope.
4. Exact finite-sample t and F reference distributions additionally require
   $u\mid X\sim N(0,\sigma^2 I_N)$. Without conditional normality, the reported
   tails are not exact finite-sample probabilities merely because software can
   compute them.
5. Complete-case fitting changes the target sample. Population interpretation
   requires a defensible missing-data/selection argument. The frozen
   three-engine fixture contains no missing values, while
   `tests/test_ols.py::test_partial_missing_rows_use_joint_complete_case_sample`
   independently verifies the joint complete-case implementation. That unit
   test does not establish that the missingness mechanism is ignorable.
6. None of these conditions makes a coefficient causal. A causal or structural
   interpretation needs a separate identification design and a defense against
   confounding, simultaneity, measurement error, and other endogeneity.

The formulas above assume nondegenerate supported cases: $N>k$, $TSS>0$,
$RSS>0$, and $k-c>0$. Tabra returns zero for some undefined fit helpers and
NaN test statistics/p-values for degenerate inference. Those guards are
implementation behavior, not extensions of the statistical definitions, and
the frozen `basic` and `no_constant` cases do not exercise degenerate designs.

## Sources and evidence roles

### Mathematical source

- Lai, T. L. and Xing, H. (2008), *Statistical Models and Methods for Financial
  Markets*, Springer Texts in Statistics,
  <https://doi.org/10.1007/978-0-387-77827-3>. The formula map uses the printed
  book pages and equations: p. 4, equation (1.2); p. 5, equations (1.4)--(1.5)
  and the following RSS definition; p. 7, equations (1.9)--(1.10); pp. 8--10,
  equations (1.11)--(1.15) and (1.22); pp. 10--12, equations (1.23)--(1.25)
  and Table 1.1; and pp. 13--14, equations (1.27)--(1.28).

### Original historical sources

- Fisher, R. A. (1922), “The Goodness of Fit of Regression Formulae, and the
  Distribution of Regression Coefficients,” *Journal of the Royal Statistical
  Society* 85(4), 597--612, <https://doi.org/10.2307/2341124>, scanned-record
  copy at <http://hdl.handle.net/2440/15174>. Journal pp. 607--610, section 6,
  supports the finite-sample regression-coefficient t inference described
  above. It is mapped only to `REG-T-001` and `REG-TP-001` and is not treated as
  a source of modern matrix notation.
- Aitken, A. C. (1936), “On Least Squares and Linear Combination of
  Observations,” *Proceedings of the Royal Society of Edinburgh* 55, 42--48,
  <https://doi.org/10.1017/S0370164600014346>. This is historical background
  only. No formula-level location was verified for the current contract, so it
  is excluded from release-critical formula mapping and from the manifest's
  primary sources.
- White, H. (1980), “A Heteroskedasticity-Consistent Covariance Matrix
  Estimator and a Direct Test for Heteroskedasticity,” *Econometrica* 48(4),
  817--838, <https://doi.org/10.2307/1912934>. This identifies the excluded
  robust extension only; it is not support for an implemented `reg` option.

### Stata compatibility source

- StataCorp LLC (2025), *Stata 19 Base Reference Manual*, `[R] regress`,
  <https://www.stata.com/manuals/rregress.pdf>. This source defines the target
  software behavior: p. 15 for complete-case selection; pp. 8--10 and 22--23
  for constant/no-constant, centered/uncentered TSS, degrees of freedom, and
  reporting conventions; and pp. 7 and 26 for two-sided coefficient t
  probabilities, right-tailed F interpretation, and the $N-k$ homoskedastic
  reference degrees of freedom. Page 22, equation (2), and p. 23, equation (3),
  are compatibility cross-checks, not the sole mathematical source.

The independent executable oracle is
`validation/commands/reg/oracles/code_audit_oracle.py::derive_case`. It imports
no Tabra code and derives every frozen manifest metric from the fixture using an
SVD/lstsq path and distribution survival functions. The formula map links each
metric both to its production implementation and to this oracle plus the
applicable frozen case. Stata and R runners remain comparison evidence only.

## Reproducibility boundary

`data/generate.py` owns the deterministic fixture. Its seed, row count,
generation algorithm, CSV bytes, and SHA-256 are recorded. Each runner verifies
the CSV against the sidecar before executing. Missing R or Stata executables
produce a nonzero exit and are never converted to a passing skip.
