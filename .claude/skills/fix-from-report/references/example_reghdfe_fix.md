# reghdfe 修复建议

## Bug 1: 常数项未恢复 (b_cons 错误 + se_cons 错误 + ll 错误)

**根因**: `fit()` 第 74-75 行在 MAP 去均值数据上添加了常数列 `np.ones(n)`。MAP 去均值后常数列方差为零（或接近零），导致 XtX 矩阵接近奇异，常数项系数为 0。

**修复方案**:
1. OLS 在去均值数据上**不加常数列**: `X_tilde_full = X_tilde` (只包含斜率变量)
2. 斜率系数: `beta_slope = (X_tilde' X_tilde)^{-1} X_tilde' y_tilde`
3. 常数项恢复: `_cons = mean(y) - mean(X) @ beta_slope`
4. 残差用**原始数据**计算: `resid = y_vec - X @ beta_slope - _cons`
5. XtX_inv 只基于斜率变量（无常数列），SE 计算也不含常数列
6. 常数项的 SE 需要单独计算（基于 delta method 或恢复公式）

**详细实现**:
```python
# Step 1: OLS on transformed data (NO constant)
XtX = mat_mul(mat_transpose(X_tilde), X_tilde)
Xty = mat_mul(mat_transpose(X_tilde), y_tilde.reshape(-1, 1))
XtX_inv = mat_inv(XtX)
beta_slope = mat_mul(XtX_inv, Xty).flatten()

# Step 2: Recover constant
x_means = np.mean(X, axis=0)
y_mean = np.mean(y_vec)
b_cons = y_mean - x_means @ beta_slope

# Step 3: Residuals on original data
resid = y_vec - X @ beta_slope - b_cons
SSR = float(resid @ resid)

# Step 4: Full beta vector
beta = np.append(beta_slope, b_cons)
var_names = x_cols + ["_cons"]

# Step 5: df and sigma2
k_full = len(var_names)  # k + 1
df_resid = n - k_full - df_a
sigma2 = SSR / df_resid

# Step 6: VCE for slope coefficients
if vce == "unadjusted":
    var_beta_slope = sigma2 * XtX_inv
elif vce == "robust":
    var_beta_slope = _robust_vce(X_tilde, resid, n, k, XtX_inv)
elif vce == "cluster":
    var_beta_slope = _cluster_vce(...)

# Step 7: Constant SE (delta method)
se_cons = np.sqrt(sigma2 * (1.0/n + x_means @ XtX_inv @ x_means))

std_err_slope = np.sqrt(np.diag(np.abs(var_beta_slope)))
std_err = np.append(std_err_slope, se_cons)
```

## Bug 2: df_a 少算 1

**根因**: `_compute_df_a()` 中单 FE 用 `K[0] - 1`，但 reghdfe 的行为是：
- 单 FE: `df_a = K` (不减 1，因为 intercept 已经在单独处理)
- 双 FE: `df_a = K1 + (K2 - M)` (不是 `(K1-1) + (K2-M)`)

**修复**: 将第一维从 `K[0] - 1` 改为 `K[0]`，双 FE 从 `(K[0]-1) + (K1-M)` 改为 `K[0] + (K1-M)`。

## Bug 3: 无共线性检测 (nlswork Case 4-6 崩溃)

**根因**: `grade` 是个体不变量（每个人 grade 相同），被 idcode FE 完全吸收。MAP 去均值后 `X_tilde[:, grade_idx]` 全为 0，导致 XtX 奇异，`mat_inv` 崩溃。

**修复方案**: 在 MAP 去均值后、OLS 前添加共线性检测：
1. 检查每列的方差: `col_var = np.var(X_tilde, axis=0)`
2. 若 `col_var < tolerance`，标记为 omitted
3. 从 `X_tilde` 中移除 omitted 列
4. 记录 omitted 列名，系数设为 0，SE 设为 0

## Bug 4: Robust SE 偏差 (衍生自 Bug 1)

常数列的加入改变了 XtX_inv 和 HC1 adjustment。修复 Bug 1 后（不加常数列做 OLS），robust SE 应该自动修正。

## 修复优先级

1. **Bug 1** (常数项恢复) - 影响 Case 1-3 的 b_cons, se_cons, ll
2. **Bug 2** (df_a 修复) - 影响 Case 1, 3 的 df_a
3. **Bug 3** (共线性检测) - 阻塞 Case 4-6
4. **Bug 4** (robust SE) - 依赖 Bug 1 修复，影响 Case 2 的 se_weight, se_mpg
