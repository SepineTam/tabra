# `reg` 理论审计修复分析

- 分析对象：`reg-validation-20260726-02`
- 分析边界：只处理 `TA-REG-02-001` 与 `TA-REG-02-002`；不修改
  `src/tabra/`，不扩展 normalized schema、三引擎指标、案例、数据或容差。
- 结论：2 个开放项都需要修复验证规格/测试映射；0 个需要产品代码修复；
  0 个可按浮点误差接受。

## 摘要

| 报告项 | 影响案例 | 根因 | 最小修复 |
|---|---|---|---|
| `TA-REG-02-001` | `basic`、`no_constant` | 公开 95% 系数置信区间已有正确实现，但缺少完整公式行与独立数值测试 | 新增 `REG-CI95-001`，补充 `theory.md`，新增一个同时覆盖两种常数约定的 summary 数值测试 |
| `TA-REG-02-002` | `no_constant` | 已存在的 Total df/MS 精确测试未写入冻结公式行 | 把完整测试 ID 追加到 `REG-NOCONSTANT-001.independent_test_location` |

现有实现没有算法、自由度、VCE 或数据预处理错误：

- `src/tabra/results/ols_result.py:158-166` 已按
  `t_(0.975,N-k)` 计算 `beta_hat_j ± t_crit * se_j`；没有使用 `1.96`
  近似。
- `src/tabra/results/ols_result.py:113-114` 已按有常数 `N-1`、无常数 `N`
  计算 Total df 与 Total MS。
- 两项 finding 都是证据链缺口，不是数值差异，因此没有“可接受的末位浮点误差”项。

## Fix 1：为公开 95% coefficient CI 建立完整映射

- **报告项**：`TA-REG-02-001`，`basic` 与 `no_constant`
- **根因**：`src/tabra/results/ols_result.py:155-166` 公开标示并计算
  `[95% Conf. Interval]`，但 `validation/commands/reg/formula_map.csv:1-21`
  没有对应公式 ID，`tests/test_ols_result.py:1-250` 也没有数值测试。
- **产品代码结论**：公式和自由度已经正确；不得修改
  `src/tabra/results/ols_result.py`。
- **修复位置**：
  - `validation/commands/reg/formula_map.csv:14` 附近，在 t/p-value 行后新增
    `REG-CI95-001`；
  - `validation/commands/reg/theory.md:128-163`，加入 CI 公式、来源、参数化和
    假设；
  - `validation/commands/reg/theory.md:205-211`，把 Lai–Xing 来源位置补入
    p. 8 eq. (1.15)；
  - `tests/test_ols_result.py:162` 附近新增独立数值测试；
  - `validation/commands/reg/manifest.yaml:3,141-149`，规格版本升为
    `2.0.2`，并把 eq. (1.15) 加入 Lai–Xing 的 `exact_locations`。

### `REG-CI95-001` 应记录的准确内容

| 字段 | 内容 |
|---|---|
| `formula_id` | `REG-CI95-001` |
| `mathematical_reference` | Lai, T. L. and Xing, H. (2008), *Statistical Models and Methods for Financial Markets*, DOI `10.1007/978-0-387-77827-3` |
| `mathematical_location` | p. 8, eq. (1.15) |
| `compatibility_reference` | StataCorp LLC (2025), Stata 19 Base Reference Manual, `[R] regress` |
| `compatibility_location` | p. 26，homo-skedastic regression 的 coefficient CI 使用 `N-k` 自由度的 t reference distribution |
| `assumptions` | `X` 满列秩、`N>k`、`se_j>0`；沿用 `REG-VCE-001` 的同方差球形协方差假设；精确有限样本覆盖还要求 `u|X ~ N(0, sigma^2 I_N)` |
| `mathematical_object` | `CI_0.95(beta_j) = [beta_hat_j - t_(0.975,N-k) se_j, beta_hat_j + t_(0.975,N-k) se_j]` |
| `parameterization` | 双侧 `alpha=0.05`，每侧 `0.025`；`k` 包含拟合常数，因此两种常数约定均使用 `df=N-k`；公开 summary 最终显示三位小数 |
| `code_location` | `src/tabra/results/ols_result.py::OLSResult._summary_style_stata`（当前 `:155-166`） |
| `independent_test_location` | `tests/test_ols_result.py::TestCoefficientConfidenceInterval::test_95_percent_confidence_interval_uses_t_distribution` |
| `validation_cases` | `basic|no_constant` |
| `status` | `implemented` |
| `limitations` | 只支持当前 conventional homoskedastic VCE；不支持 robust/cluster CI；正态条件不成立时不能声称精确有限样本 95% 覆盖；summary 的三位小数是显示精度，不是内部估计精度 |

### 独立测试的最小设计

在 `tests/test_ols_result.py` 新增上述单个参数化测试，参数为
`is_con=True/False`，从而让完整 pytest node ID 同时执行 `basic` 和
`no_constant` 两种自由度约定。测试应：

1. 使用固定的小样本 `y` 与单个 `x`，让 `N-k` 较小，使 t 临界值与 `1.96`
   明显不同。
2. 测试代码直接从原始数组构造 `X`，独立计算
   `(X'X)^(-1)X'y`、残差、`RSS/(N-k)`、标准误和
   `scipy.stats.t.ppf(0.975, N-k)`；不得调用产品私有 helper 生成期望值。
3. 调用 `OLS().fit(...).summary()`，按参数名解析每个公开系数行最后两个字段，
   与独立得到的上下界比较。
4. 因公开输出固定为三位小数，按显示精度比较（建议绝对容差 `5e-4`），同时
   断言行数和参数顺序；不要拿未格式化期望值做 `1e-10` 比较。
5. 测试内新增的变量名与注释遵守仓库要求：变量名使用简单英文，注释使用英文。

这项测试独立验证公开输出的数值路径；只检查标题存在、区间顺序或
`lo < coef < hi` 均不足以关闭 finding。

### `theory.md` 的最小补充

在 “Classical covariance and inference” 中明确加入：

```text
CI_0.95(beta_j) = [beta_hat_j - t_(0.975,N-k) se_j,
                   beta_hat_j + t_(0.975,N-k) se_j].
```

并明确：Lai and Xing (2008), p. 8, eq. (1.15) 是数学来源；Stata 19
`[R] regress`, p. 26 只负责 `N-k` 的软件兼容参数化；精确有限样本 95%
覆盖要求条件球形正态误差。来源清单中的 Lai–Xing 精确位置也必须加入
`p. 8, eq. (1.15)`，避免正文、formula map 和 manifest 来源清单互相矛盾。

- **预期效果**：公开 CI 从“有实现、无证据链”变成
  公式—精确来源—假设—参数化—代码—两种常数约定独立测试的完整映射。

## Fix 2：把现有 Total df/MS 测试写入 `REG-NOCONSTANT-001`

- **报告项**：`TA-REG-02-002`，`no_constant`
- **根因**：`validation/commands/reg/formula_map.csv:20` 已映射
  `OLSResult._summary_style_stata`，但其 `independent_test_location` 没有列出
  直接解析公开 Total 行的现有测试。
- **修复位置**：`validation/commands/reg/formula_map.csv:20`。
- **修复方式**：在 `REG-NOCONSTANT-001.independent_test_location` 末尾追加：

```text
tests/test_ols_result.py::TestAnovaAlignment::test_total_df_and_ms_follow_constant_convention
```

保留该行已有 oracle 和 fit-statistics 测试，不新增 dedicated formula row，
不复制或改写现有测试。现有测试位于 `tests/test_ols_result.py:130-160`，同时断言
有常数 Total df/MS 为 `N-1`、`TSS/(N-1)`，无常数为 `N`、`TSS/N`，已经足以
验证该公开报告约定。

- **预期效果**：`REG-NOCONSTANT-001` 声称映射的 public summary 代码有对应的
  精确测试 ID，形成完整的实现—测试追踪。

## 严格最小修改范围

| 文件 | 最小修改 |
|---|---|
| `tests/test_ols_result.py` | 新增一个参数化的公开 95% CI 独立数值测试 |
| `validation/commands/reg/formula_map.csv` | 新增 `REG-CI95-001`；给 `REG-NOCONSTANT-001` 追加现有精确测试 ID |
| `validation/commands/reg/theory.md` | 加入 95% CI 公式、Lai–Xing eq. (1.15)、Stata p. 26 参数化、假设与限制 |
| `validation/commands/reg/manifest.yaml` | `2.0.1 -> 2.0.2`；在 Lai–Xing `exact_locations` 加入 eq. (1.15) |
| `validation/commands/reg/oracles/code_audit_oracle.py` | 所有规格修改定稿后，只更新 `FROZEN_HASHES["manifest"]` |
| `validation/runs/reg/<new_run_id>/...` | 生成新的不可变审计、三引擎比较和 gate 证据 |

明确不应修改或扩展：

- `src/tabra/results/ols_result.py` 与其他 `src/tabra/` 文件；
- `cases.yaml`、fixture、sidecar、数据生成器；
- normalized result schema、runner 输出字段、manifest metrics/tolerances；
- Stata/R/Python runner 的 CI 输出。公开 summary CI 由独立单元测试覆盖即可；
  把它加入 normalized 三引擎协议会把一次证据链修复扩大成新功能。

## 规格版本、oracle 哈希与新运行判断

| 项目 | 判断 | 原因与准确操作 |
|---|---|---|
| `specification_version` | **需要递增到 `2.0.2`** | 新增一项 release-critical 公开报告公式，并需把 eq. (1.15) 写入 manifest 的精确来源清单；这是补充证据合同的 patch 级变化，不改变公开计算语义或 normalized schema |
| oracle 哈希重绑 | **需要，但只重绑 manifest** | manifest 字节会变化，所以在所有规格编辑定稿后重算 SHA-256，并替换 `code_audit_oracle.py:24`；cases/data 未变，`:25-26` 必须保持现值。formula map/theory 不属于该 oracle 的三项内嵌 guard，不应借机扩大 guard |
| oracle 文件自身哈希 | **需要在新 run 重新冻结** | 修改 manifest 常量后 oracle 文件 SHA-256 会变化；它不写回自身，而是由新 run 的 `inputs.json`/审计工件记录 |
| `run_id` | **必须新建** | `verify-all-commands` 要求任何 repair 后建立新运行并重跑两项审计、三引擎、比较和 gate；不得覆盖 `reg-validation-20260726-02`。若当日实施且目录仍不存在，可用 `reg-validation-20260726-03` |

顺序必须是：先修改测试、formula map、theory 和 manifest；再重算最终 manifest
哈希并更新 oracle；最后创建新 run。若先重绑 oracle 再编辑 manifest，会立即制造
新的过期哈希。

## 应执行的测试与验证命令

以下命令只应在修复实施后执行。`<new_run_id>` 必须替换为唯一的新 ID。

### 1. 静态检查和定向测试

```bash
git diff --check

.venv/bin/python -m tabra.validation preflight \
  --manifest validation/commands/reg/manifest.yaml

.venv/bin/python -m pytest -q -o addopts='' \
  tests/test_ols_result.py::TestCoefficientConfidenceInterval::test_95_percent_confidence_interval_uses_t_distribution \
  tests/test_ols_result.py::TestAnovaAlignment::test_total_df_and_ms_follow_constant_convention \
  tests/test_validation_core.py::test_reg_formula_map_separates_mathematical_and_compatibility_sources
```

### 2. OLS 与冻结 oracle 回归测试

```bash
.venv/bin/python -m pytest -q -o addopts='' \
  tests/test_ols.py \
  tests/test_ols_result.py \
  tests/test_estimate_base_stats.py \
  tests/test_api.py::TestRegAPI \
  tests/ops/test_stats.py \
  tests/test_validation_core.py \
  tests/test_reg_validation_oracle.py

shasum -a 256 \
  validation/commands/reg/manifest.yaml \
  validation/commands/reg/cases.yaml \
  validation/commands/reg/data/reg_fixture.csv \
  validation/commands/reg/theory.md \
  validation/commands/reg/formula_map.csv \
  validation/commands/reg/oracles/code_audit_oracle.py
```

验收时，`tests/test_reg_validation_oracle.py::test_frozen_hashes_match_current_reg_inputs`
必须证明 oracle 只更新后的 manifest 哈希与当前文件一致；formal-entry 测试必须继续
返回 `oracle_status == PASS`。

### 3. 新运行的三引擎证据

```bash
.venv/bin/python validation/commands/reg/python/run.py \
  --data validation/commands/reg/data/reg_fixture.csv \
  --output validation/runs/reg/<new_run_id>/python/results.json

.venv/bin/python validation/commands/reg/oracles/code_audit_oracle.py \
  --manifest validation/commands/reg/manifest.yaml \
  --cases validation/commands/reg/cases.yaml \
  --data validation/commands/reg/data/reg_fixture.csv \
  --actual validation/runs/reg/<new_run_id>/python/results.json

STATA_CMD=/usr/local/bin/stata-mp \
  .venv/bin/python validation/commands/reg/stata/run.py \
  --data validation/commands/reg/data/reg_fixture.csv \
  --output validation/runs/reg/<new_run_id>/stata/results.json

RSCRIPT_CMD=/usr/local/bin/Rscript \
  .venv/bin/python validation/commands/reg/r/run.py \
  --data validation/commands/reg/data/reg_fixture.csv \
  --output validation/runs/reg/<new_run_id>/r/results.json

.venv/bin/python -m tabra.validation compare \
  --manifest validation/commands/reg/manifest.yaml \
  --result validation/runs/reg/<new_run_id>/python/results.json \
  --result validation/runs/reg/<new_run_id>/stata/results.json \
  --result validation/runs/reg/<new_run_id>/r/results.json \
  --output validation/runs/reg/<new_run_id>/comparison.json
```

在 engine comparison 前，必须由相互独立的 code-auditor 与 theory-auditor 对新
commit 和新冻结哈希生成新 run 的审计工件。理论审计至少确认：

- `REG-CI95-001` 的公式、Lai–Xing p. 8 eq. (1.15)、Stata p. 26、假设、代码和
  参数化测试 ID 全部一致；
- `REG-NOCONSTANT-001` 明确列出既有 Total df/MS 测试；
- 不再有公开报告量未映射。

### 4. 独立 gate

两项审计与 comparison 均通过后才能执行：

```bash
.venv/bin/python -m tabra.validation gate \
  --manifest validation/commands/reg/manifest.yaml \
  --comparison validation/runs/reg/<new_run_id>/comparison.json \
  --run-id <new_run_id> \
  --code-audit validation/runs/reg/<new_run_id>/code_audit.md \
  --theory-audit validation/runs/reg/<new_run_id>/theory_audit.md \
  --result validation/runs/reg/<new_run_id>/python/results.json \
  --result validation/runs/reg/<new_run_id>/stata/results.json \
  --result validation/runs/reg/<new_run_id>/r/results.json \
  --workdir . \
  --output validation/runs/reg/<new_run_id>/attestation.json
```

最终验收条件是：新增 CI 测试两种常数参数均通过；既有 Total df/MS 测试通过；
oracle 的最终 manifest 哈希 guard 与 formal entry 均通过；新 code/theory audit 无
开放项；三引擎 comparison 通过；独立 gate 在新 run 写出不可变的 PASS attestation。
