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
degrees of freedom are $N-c$. These definitions and the role of $c$ are
given in `[R] regress`, p. 22. Pages 9--10 explain that `noconstant`, unlike
`hascons`, uses the uncentered ANOVA convention.

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

This is `[R] regress`, p. 22, equation (2). The inverse representation requires
full column rank. Tabra orders the slope coefficients as supplied by `x` and
appends `_cons` only for `is_con=True`; the no-constant case has no `_cons`
entry.

## ANOVA decomposition and fit statistics

Let $e=y-X\hat\beta$. Using the $c$ convention above, `[R] regress`,
pp. 22--23, defines

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

These formulas appear in `[R] regress`, p. 22 for the sums of squares, their
degrees of freedom, $s^2$, and Root MSE, and p. 23 for $F$, $R^2$, and
adjusted $R^2$. In the `basic` case $c=1$; in `no_constant` $c=0$, so
TSS is uncentered, model and total degrees of freedom increase by one relative
to an intercept fit with the same $k$, and the adjusted-$R^2$ numerator is
$N$, not $N-1$.

## Classical covariance and inference

The conventional homoskedastic covariance estimator is `[R] regress`, p. 23,
equation (3):

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

The overall test is $H_0:\beta_j=0$ for every nonconstant coefficient when
$c=1$, and for all $k$ coefficients when $c=0$. Its normalized p-value is

\[
p_F=\Pr\{F_{k-c,N-k}\ge F\}.
\]

`[R] regress`, p. 7, describes the coefficient t tests as two-sided and the
overall F p-value as the probability of an F statistic at least as large as the
reported value. Page 23 supplies the F formula and its two degrees of freedom.
Page 26 specifies that homoskedastic-regression coefficient p-values and
confidence intervals use a t distribution with $N-k$ degrees of freedom.

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
   requires a defensible missing-data/selection argument; the frozen fixture
   contains no missing values and does not validate such an argument.
6. None of these conditions makes a coefficient causal. A causal or structural
   interpretation needs a separate identification design and a defense against
   confounding, simultaneity, measurement error, and other endogeneity.

The formulas above assume nondegenerate supported cases: $N>k$, $TSS>0$,
$RSS>0$, and $k-c>0$. Tabra returns zero for some undefined fit helpers and
NaN test statistics/p-values for degenerate inference. Those guards are
implementation behavior, not extensions of the statistical definitions, and
the frozen `basic` and `no_constant` cases do not exercise degenerate designs.

## Sources and evidence roles

- StataCorp LLC (2025), *Stata 19 Base Reference Manual*, `[R] regress`,
  <https://www.stata.com/manuals/rregress.pdf>. Exact locations used here are
  pp. 7, 9--10, 15, 22--23, and 26. This official manual is the authoritative
  formula and Stata-compatibility source for the supported contract.
- Aitken, A. C. (1936), “On Least Squares and Linear Combination of
  Observations,” *Proceedings of the Royal Society of Edinburgh* 55, 42--48,
  <https://doi.org/10.1017/S0370164600014346>. This is historical primary
  context; no equation or narrower page claim from it is used in the formula
  map.
- Fisher, R. A. (1922), “The Goodness of Fit of Regression Formulae, and the
  Distribution of Regression Coefficients,” *Journal of the Royal Statistical
  Society* 85(4), 597--612, <https://doi.org/10.2307/2341124>. This is primary
  context for regression inference; no unverified equation location from it is
  asserted here.
- White, H. (1980), “A Heteroskedasticity-Consistent Covariance Matrix
  Estimator and a Direct Test for Heteroskedasticity,” *Econometrica* 48(4),
  817--838, <https://doi.org/10.2307/1912934>. This identifies the excluded
  robust extension only; it is not support for an implemented `reg` option.

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
