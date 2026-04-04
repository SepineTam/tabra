# reghdfe -- Linear regression with multiple fixed effects

## Syntax

```
reghdfe depvar [indepvars] [if] [in] [weight], absorb(absvars) [options]
```

核心只有一个必须项：`absorb()` 指定要吸收的固定效应。

## Sub-commands / Variants

| 变体 | 语法 | 说明 |
|------|------|------|
| OLS (无FE) | `reghdfe y x` | 不加 `absorb()`，等价于 `regress` |
| 单维FE | `reghdfe y x, absorb(id)` | 等价于 `areg` / `xtreg, fe` |
| 多维FE | `reghdfe y x, absorb(id year)` | 核心场景，多个固定效应 |
| 交互FE | `reghdfe y x, absorb(id#year)` | 交互固定效应 |
| 异质斜率 | `reghdfe y x, absorb(id##c.time)` | 吸收个体截距+个体时间趋势 |
| Group FE | `reghdfe y x, absorb(inv) group(patent) individual(inv)` | 个体FE + 组级结果变量 |

## Key Options

| 选项 | 说明 |
|------|------|
| `vce(robust)` | 异方差稳健标准误 |
| `vce(cluster var1 var2)` | 多维聚类标准误 |
| `vce(dkraay #)` | Driscoll-Kraay HAC 标准误 |
| `residuals(newvar)` | 保存残差 |
| `tolerance(#)` | 收敛容限，默认 1e-8 |
| `technique(map\|lsmr\|lsqr)` | 投影算法选择 |
| `verbose(#)` | 调试信息级别 |

## Algorithm / Estimation Method

reghdfe 的核心流程：

### 1. 数据预处理
- 解析 `absorb()` 中的固定效应变量
- **迭代删除 singleton 观测值**：如果某观测值在任何一个固定效应维度上是唯一值，则删除。这会持续进行直到没有新的 singleton 出现。
- 计算 degrees-of-freedom 调整量 `e(df_a)`

### 2. 固定效应吸收（Partialling-out）
核心使用 **Method of Alternating Projections (MAP)**：

```
对于每个固定效应维度 j = 1, 2, ..., J:
    X_residual = X - P_j(X)
其中 P_j 是向第 j 个固定效应子空间的投影
```

迭代收敛条件：`max|X_new - X_old| < tolerance`

可选算法：
- **MAP** (默认): 交替投影法，配合共轭梯度加速
- **LSMR**: 稀疏最小二乘，适用于 group+individual FE
- **LSQR**: 类似 LSMR 但精度略低

### 3. OLS 回归
对去固定效应后的变量执行 OLS：

```
y_tilde = y - D*alpha  (D 是固定效应矩阵)
X_tilde = X - D*Gamma
beta = (X_tilde' X_tilde)^(-1) X_tilde' y_tilde
```

### 4. 标准误计算
根据 `vce()` 选项计算：
- Conventional: sigma^2 * (X'X)^(-1)
- Robust: HC1 sandwich estimator
- Cluster: Cameron-Gelbach-Miller 多维聚类
- Driscoll-Kraay: 截面均值矩条件的 Newey-West HAC

### 5. DoF 调整
- `e(df_a)`: 因固定效应损失的自由度
- 通过 pairwise connected subgraphs 算法识别冗余 FE
- 公式: `df_a = sum(K_j - M_j)` 其中 K_j 是第 j 个 FE 的类别数，M_j 是连通子图数

## Input/Output

**Input:**
- 因变量 `depvar`: 连续变量
- 自变量 `indepvars`: 连续/因子变量
- `absorb()`: 分类变量（固定效应维度）
- 数据要求：矩形面板或截面数据

**Output (stored in `e()`):**

| 结果 | 含义 |
|------|------|
| `e(b)` | 系数向量 |
| `e(V)` | 方差-协方差矩阵 |
| `e(N)` | 有效观测数（去除 singleton 后） |
| `e(r2)` / `e(r2_within)` | R² / 组内 R² |
| `e(df_a)` | FE 损失的自由度 |
| `e(rss)` | 残差平方和 |
| `e(N_hdfe)` | 吸收的 FE 维度数 |
| `e(N_clust#)` | 第 # 个聚类变量的类别数 |

## Implementation Notes (Python 复刻要点)

1. **Singleton 删除**是 reghdfe 的关键创新，不做这步会导致 SE 有偏。实现逻辑：对每个 FE 维度，统计各类别的频次，删除频次=1 的观测，循环直到稳定。

2. **MAP 投影**的本质是对每个 FE 维度做组内去均值（within transformation），然后迭代直到收敛。Python 中可以用 `pandas.groupby().transform('mean')` 实现单步投影。

3. **DoF 调整**需要图论中的连通分量算法（`scipy.sparse.csgraph.connected_components`）。

4. **多维聚类标准误**：先对每个聚类维度分别计算 meat 矩阵 S_j，然后用 Cameron et al. (2011) 的近似公式合并。

5. 推荐依赖：`numpy`, `scipy.sparse`, `pandas`, `linearmodels`（面板工具）。
