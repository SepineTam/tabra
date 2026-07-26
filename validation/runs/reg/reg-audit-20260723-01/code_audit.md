# `reg` code audit — `reg-audit-20260723-01`

## Decision

`code_audit_status: FAIL`

This is a code-audit decision, not a final validation attestation. The frozen
commit fails the frozen `reject_zero_mismatch` safety rule for coefficient and F
p-values. Additional scale, non-finite-input, degenerate-inference, and displayed
degrees-of-freedom defects are independently reproducible.

## Frozen identity and source state

- Run ID: `reg-audit-20260723-01`
- Audited commit: `0242978e7add4a7d44ab2056c4cbc9657f24a5f9`
- Manifest SHA-256: `533e96bde7c0fd15dcc532f127789ab300c21f378c7b344ac99be6a4cafa26d7`
- Cases SHA-256: `d6dc76648bcd0433795b60c2f65d46cc89167565384c23e24d864525d42ea7e7`
- Data SHA-256: `53f6fea92a7978cbce09212c224eb6e29e9c581ab920e0285ef541c4b09d37c4`
- Oracle SHA-256: `527b609e22a95163b99149c271e448774de76bd19702de7bbb79fcb4118a39ce`
- Audit time: `2026-07-22T16:56:41Z`
- Environment: Python 3.11.11, NumPy 2.4.3, SciPy 1.17.1, pandas 3.0.1,
  pytest 9.0.2.

The tracked worktree was dirty and was not altered or reset. Tracked changes
included 18 files, among them the relevant files
`src/tabra/models/estimate/ols.py`, `src/tabra/ops/stats.py`, and
`tests/ops/test_stats.py`. The OLS diff is only a scientific-support comment.
The current dirty `ops/stats.py` replaces `1-CDF` with survival functions and
therefore is materially different from the frozen commit. All conclusions about
the frozen target were obtained by exporting the commit's `src/tabra` tree from
Git objects to a temporary directory. Current-worktree results are reported
separately and are not treated as evidence that the frozen commit passes.

Frozen source SHA-256 values were:

| File | SHA-256 |
|---|---|
| `models/estimate/ols.py` | `42d6d45764d34f854c7306e8583068608db1ff1afa41c6ab6e208fc339bae5bc` |
| `models/estimate/base.py` | `e43d7f01b0ec8f268a1a81077e18a50427a9ad787380c85304e64682da36f28f` |
| `models/estimate/stats.py` | `4a44426cdab333bbf83b91cb28e3ceaaa4be61966a9d55053031c0529bc77d35` |
| `ops/linalg.py` | `233bf1cb0343f2b0656ea8734016ce38da1ab4c82cef5f7b309949559fac1170` |
| `ops/stats.py` | `fc1b875579050660d2919f9a54d4a16d2f292c1df45ac702e75132d24980f6a4` |
| `results/ols_result.py` | `2fc42ce58903116cf438a78cedf7a1e793388fb33930e4885a6052896408bfcc` |
| `api/estimation/linear.py` | `598bdf96982adb5bd4f08e9085ab3c8072c880c40ff8fb7ba14d1328fc9950b5` |

## Formula and implementation trace

### Sample and design

`BaseModel._prepare_df` selects `[y] + x` and performs complete-case deletion
with `dropna()`. It does not sort. A probe with two incomplete fixture rows
returned `n_obs=118` and 118 residuals, as expected. Reversing row order changed
coefficients by at most `6.66e-16`. The original retained row index/sample mask is
not exposed on `OLSResult`.

There are no weight, categorical-encoding, robust, cluster, or bootstrap
arguments in either public `LinearCommands.reg(y, x, is_con=True)` or
`OLS.fit(df, y, x, is_con=True)`. Passing `weights` or `robust` raises
`TypeError`. String categories fail numeric conversion. Thus there is no silent
categorical expansion or weighting. The frozen robust case remains explicitly
blocked and was not counted as passed.

The parameter order is the caller's `x` order followed by `_cons` when
`is_con=True`. No constant is added otherwise. There are no constraints or
absorbed effects. `X'X` is inverted through an unpivoted custom LU routine with
an absolute `1e-14` pivot cutoff. There is no explicit matrix-rank or condition
number check.

### Objective, derivatives, and solution

For `S(b)=(y-Xb)'(y-Xb)`, independent differentiation gives
`grad S(b)=-2X'(y-Xb)` and Hessian `2X'X`. Full rank makes the stationary point
`b=(X'X)^(-1)X'y`. The implementation follows that equation directly. It is not
iterative, so there is no convergence tolerance or iteration metadata. The
frozen fixture is well conditioned: `cond(X)=1.3528` (`basic`) and `1.3295`
(`no_constant`); oracle normal-equation residuals were at most `4.14e-14`.

The direct normal-equations inverse squares conditioning and the LU cutoff is
absolute rather than scale-relative. A full-rank no-constant design with both
columns uniformly scaled from 1 to `1e-8` retained rank 2 and
`cond(X)=1.176099915`, but changed from a completed fit to `ValueError: Matrix is
singular`. At `cond(X)=8.224e6`, a separate near-collinear probe silently differed
from the SVD solution by 3.214 in coefficient space and 0.00609 in fitted values.

### Covariance, degrees of freedom, and inference

The sole covariance path is classical homoskedastic OLS:
`s2=RSS/(n-k)` and `Var(b)=s2(X'X)^(-1)`. Standard errors are its diagonal square
roots. Coefficient statistics use `t=b/se`, `df_resid=n-k`, and a two-sided
Student-t tail. `df_model=k-1` with a constant and `k` without one. The model F
statistic is `[(TSS-RSS)/df_model]/[RSS/df_resid]` with a right-tail F p-value.
Centered TSS and numerator adjustment `n-1` are used with a constant; uncentered
TSS and `n` are used without one. These formulas match the independent oracle on
ordinary full-rank data.

The frozen commit computes tails as `1-CDF`. On the frozen fixture this suffers
catastrophic cancellation: all three `basic` coefficient p-values, the `x1`
p-value in `no_constant`, and both F p-values become exactly zero. Independent
SciPy survival-function values are respectively
`[5.0893e-93, 3.4212e-38, 1.1032e-78]`, `6.9550e-26`, and
`[5.7905e-92, 6.3249e-25]`. This violates the manifest's explicit
`reject_zero_mismatch` rule. The dirty worktree's uncommitted survival-function
change corrects this particular defect but is outside the frozen commit.

The 95% confidence intervals printed by `OLSResult.summary()` use
`t.ppf(0.975, df_resid)` and `b +/- critical*se`, which is correct for this VCE;
confidence intervals are not exposed as structured result metadata. The
no-constant summary nevertheless labels Total df as `n-1` and computes Total MS
with `n-1`; uncentered total df must be `n`.

## Independent oracle evidence

The executable oracle is
`validation/commands/reg/oracles/code_audit_oracle.py`. It imports no Tabra code,
verifies all three frozen hashes, builds each matrix from the CSV and case file,
uses `numpy.linalg.lstsq`/SVD for the solution and covariance, and uses SciPy
survival functions for t and F tails. It then compares the independently derived
values to runner output under the frozen manifest tolerances.

On the **current dirty worktree**, all frozen metrics passed; maximum errors
included `4.44e-16` for coefficients, `2.78e-17` for standard errors,
`2.84e-14` for t statistics, and `4.55e-13` for F statistics. This only confirms
the ordinary fixture under the dirty source state.

On the **frozen commit**, the oracle exited 1 with `oracle_status=FAIL`.
`p_values` and `f_p_value` failed, and both had `zero_mismatch=true`. All other
frozen metrics passed their tolerances.

Reproduction:

```bash
audit_tmp_dir=$(mktemp -d /tmp/tabra-reg-head-audit.XXXXXX)
git archive 0242978e7add4a7d44ab2056c4cbc9657f24a5f9 src/tabra | tar -x -C "$audit_tmp_dir"
PYTHONPATH="$audit_tmp_dir/src" .venv/bin/python validation/commands/reg/python/run.py \
  --data validation/commands/reg/data/reg_fixture.csv \
  --output "$audit_tmp_dir/results.json"
PYTHONPATH="$audit_tmp_dir/src" .venv/bin/python \
  validation/commands/reg/oracles/code_audit_oracle.py \
  --manifest validation/commands/reg/manifest.yaml \
  --cases validation/commands/reg/cases.yaml \
  --data validation/commands/reg/data/reg_fixture.csv \
  --actual "$audit_tmp_dir/results.json"
```

## Tests and failure matrix

- Frozen commit's relevant original tests: `67 passed in 1.51s`. They encode
  `1-CDF` as their own expected value and therefore do not detect tail
  cancellation.
- Current dirty relevant tests: `69 passed in 1.06s`. The two added extreme-tail
  tests validate the uncommitted survival-function change.
- Oracle, dirty worktree: exit 0, `PASS` for the two ordinary fixture cases.
- Oracle, frozen commit: exit 1, `FAIL` for required p-value metrics.

| Probe | Raw outcome | Audit interpretation |
|---|---|---|
| Uniform `1e-8` scaling, no constant | rank 2, `cond(X)=1.1761`, `ValueError` singular | False identification failure; scale invariance broken |
| Exact collinearity | `ValueError` singular | Failure occurs, but through LU cutoff rather than an explicit rank diagnostic |
| Near collinearity | `cond(X)=8.224e6`; coefficient delta 3.214 vs SVD | Silent numerical inaccuracy with no warning |
| Missing values | 120 input, 118 complete cases, `n_obs=118` | Complete-case construction correct; sample index not returned |
| All missing | generic `ValueError` singular | No explicit empty-sample validation |
| `Inf` in regressor | returns NaN coefficients/p-values and warnings | Non-finite output is not rejected |
| `1e200` regressor | returns partly finite misleading output plus NaN p-value | Overflow is not rejected |
| Zero response | zero SE, NaN t/p; `F=0`, `Prob>F=0` | Invalid inference and false zero p-value sentinel |
| `n=k` full-rank design | `ZeroDivisionError` | Residual df is not validated before variance division |
| Intercept-only | `df_model=0`, `F=0`, `Prob>F=0` | Undefined F test is reported with a false p-value of zero |
| String category | numeric-conversion `ValueError` | No categorical encoding, consistent with the narrow API |
| Separation/censoring/empty groups | no applicable OLS path | Not silently marked passed; these model-specific paths are absent |

## Findings and disposition

Detailed structured findings are in `code_findings.json`.

1. `REG-CODE-001` (critical): frozen required p-values collapse to zero and
   directly violate the frozen safety rule. Open; a dirty worktree change is not
   a disposition for the frozen commit.
2. `REG-CODE-002` (high): the inverse/LU path is scale-dependent and silently
   inaccurate near collinearity. Open; replace with a rank-revealing stable solve
   and explicit condition/rank policy, then rerun from the beginning.
3. `REG-CODE-003` (high): non-finite and overflow inputs can return non-finite or
   misleading estimates instead of a hard validation error. Open.
4. `REG-CODE-004` (high): invalid residual/model degrees of freedom and zero
   variance yield exceptions, NaNs, or a false p-value of zero. Open.
5. `REG-CODE-005` (medium): no-constant summary reports centered Total df/MS for
   an uncentered decomposition. Open.
