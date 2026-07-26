# `reg` 修复分析

- 分析对象：`reg-validation-20260726-01`
- 分析边界：只分析验证工件与证据链，不修改产品代码、命令规格、测试或既有运行证据。
- 结论：3 个开放项都需要修复验证工件；0 个是产品 OLS 公式缺陷；0 个可仅按浮点误差接受。

## 摘要

| 项目 | 根因类别 | 是否需要产品代码修复 | 最小修复 |
|---|---|---:|---|
| `REG-CODE-006` / `TA-REG-20260726-001` | 冻结顺序与哈希绑定错误 | 否 | 所有规格/文献修订完成后，最后重算并写入 oracle 的 manifest、cases、data 哈希 |
| `TA-REG-20260726-002` | 数据预处理规则缺少独立测试 | 否 | 增加一个手工构造、部分缺失、独立 NumPy 期望值的单元测试，并映射到 `REG-SAMPLE-001` |
| `TA-REG-20260726-003` | 数学来源与 Stata 兼容性来源角色混用 | 否 | 用已核实页码/公式的数学来源支持公式，用 Stata 手册单独支持软件行为和参数化 |

哈希保护被绕过后的诊断结果已经表明，现有 `basic` 和 `no_constant`
数值计算与独立 oracle 一致：系数最大绝对误差
`1.33e-15`、标准误 `2.78e-17`、t 统计量 `1.42e-14`。这些是正常的末位浮点差异，
但不能把正式 oracle 在比较前退出的问题降级为“可接受数值误差”。

## 修复顺序

必须按以下顺序实施，不能先把报告里的当前哈希直接抄回 oracle：

1. 先完成完整案例测试和文献分层修订。
2. 决定是否只增加单元测试，还是新增一个冻结的 `partial_missing` 三引擎案例。
3. 定稿 `manifest.yaml`、`cases.yaml` 和数据文件后，重算它们的 SHA-256。
4. 最后更新 `code_audit_oracle.py::FROZEN_HASHES`，再计算 oracle 自身哈希。
5. 创建新 `run_id`，重新执行代码审计、理论审计、Python/Stata/R、三引擎比较和独立 gate。旧运行目录保持不变。

否则，若先把本次报告中的 manifest 哈希
`414ed36b...` 写入 oracle，随后又因文献来源修订 `manifest.yaml`，oracle 会立即再次过期。

## Fix 1：oracle 绑定了旧 manifest/cases 哈希

- **报告项**：`REG-CODE-006`、`TA-REG-20260726-001`
- **根因**：`validation/commands/reg/oracles/code_audit_oracle.py:23-27`
  手工内嵌的是较早版本的 manifest
  `533e96bd...` 和 cases `d6dc7664...`；本次运行冻结的是
  manifest `414ed36b...` 和 cases `1b107cfe...`。
  `code_audit_oracle.py:200-213` 因而在读取规格和计算任何指标前退出。
  这是冻结工件的版本/顺序错误，不是 OLS 算法、自由度、VCE 或数据处理错误。
- **修复位置**：
  - `validation/commands/reg/oracles/code_audit_oracle.py:23-27`
  - `validation/commands/reg/oracles/code_audit_oracle.py:200-213`
  - 建议新增 `tests/test_reg_validation_oracle.py`
- **具体修改**：
  1. 在其他修复全部定稿后，用最终文件字节重算 manifest、cases、data 的
     SHA-256，再替换 `FROZEN_HASHES`。不得删除哈希保护，也不得在运行时把传入文件
     的哈希自动当作期望值，否则 oracle 将失去独立冻结作用。
  2. 保留 `main()` 的强制拒绝逻辑；错误信息应继续同时输出 expected 与 actual，
     便于定位过期工件。
  3. 增加端到端回归测试：先用当前 Python runner 在 `tmp_path` 生成 normalized
     result，再以当前 manifest/cases/data 调用 oracle；断言退出码为 0、
     `oracle_status == "PASS"`、三个 `hash_checks` 都为真，而且全部 metric
     comparison 通过。
  4. 增加一个轻量静态断言，直接计算三个文件的 SHA-256 并与
     `FROZEN_HASHES` 比较，使将来只修改 manifest/cases 而忘记重冻 oracle
     时能在测试阶段失败。
- **预期效果**：正式入口不再出现 `RuntimeError: Frozen input hash mismatch`，
  两个支持案例的所有 release-critical metrics 由正式、未绕过保护的 oracle
  产生比较结果。

## Fix 2：complete-case 规则没有部分缺失值的独立证据

- **报告项**：`TA-REG-20260726-002`，公式 `REG-SAMPLE-001`
- **根因**：产品代码
  `src/tabra/models/estimate/base.py:19-23` 已经对 `[y] + x` 做联合
  `dropna()`，并没有发现实现错误；证据缺口来自：
  - `validation/commands/reg/cases.yaml:4-48` 的两个案例都使用无缺失的同一数据；
  - `code_audit_oracle.py:40-50` 直接把每个 CSV 单元转为 `float`，没有可验证的
    complete-case 分支；
  - `tests/test_api.py:50-55` 只测试完整的 100 行；
  - `tests/test_ols.py:316-322` 只测试“全部缺失时拒绝”，没有测试 y、x1、x2
    分别缺失时是否取共同样本。
- **推荐的最小修复位置**：
  - `tests/test_ols.py:316` 附近
  - `validation/commands/reg/formula_map.csv:3`（`REG-SAMPLE-001`）
  - `validation/commands/reg/theory.md:5-16,162-164,196-201`
- **具体修改**：
  1. 新增 `test_partial_missing_rows_use_joint_complete_case_sample`。
     数据至少包含：一行 y 缺失、一行 x1 缺失、一行 x2 缺失、一行仅未请求列
     缺失，以及不少于 4 行完整且满秩的数据。
  2. 测试中硬编码应保留的行号，不调用产品的 `_prepare_df`，也不复用
     `DataFrame.dropna()` 生成期望样本。用这些明确行和
     `numpy.linalg.lstsq` 独立计算期望系数。
  3. 以反向请求顺序（例如 `x=["x2", "x1"]`）拟合，并同时断言：
     `n_obs` 等于共同完整行数；`var_names == ["x2", "x1", "_cons"]`；
     系数与独立 NumPy 结果一致；标准误、残差长度和至少一个拟合统计量使用同一
     保留样本；仅未请求列缺失的行不会被删除。
  4. 把该测试的完整名称加入 `REG-SAMPLE-001.independent_test_location`，
     删除“仅验证 N、未验证缺失选择后果”的旧限制，改为准确说明：
     三引擎冻结案例仍无缺失，但独立单元测试覆盖联合 complete-case 行为；
     单元测试不证明缺失机制可忽略。
- **为何这是最小修复**：审计要求是“独立的部分缺失案例或单元测试”。
  该方案不改变三引擎案例、数据集、输出形状或容差，能够直接关闭所报告的测试缺口。
- **预期效果**：`REG-SAMPLE-001` 有独立、可执行、会因错误样本掩码而失败的证据；
  `nobs`、参数顺序和估计值均由同一个明确的共同样本验证。

### 更强但非最小的方案

| 方案 | 修改范围 | 优点 | 风险/成本 | 规格版本 |
|---|---|---|---|---|
| A：独立单元测试（推荐） | `tests/test_ols.py`、formula map、theory | 最小，直接覆盖报告要求；不改三引擎输出结构 | complete-case 不进入 frozen engine comparison | 仅因此项无需增加 |
| B：新增 `partial_missing` 冻结案例 | manifest、cases、数据/sidecar、Python/Stata/R runners、oracle、formula map、theory、README | 三引擎与 oracle 都直接验证缺失样本 | 所有 case 数组、哈希、runner 与数据均改变，修复面显著扩大 | 建议 `2.1.0` |

若采用 B，oracle 的 `read_numeric_csv` 不能再直接对空字符串调用 `float`；
它必须独立解析缺失标记，并由 `derive_case` 对 dependent 与所有 requested
independent 列建立共同有限值掩码。Stata 和 R runner 也必须从 `cases.yaml`
驱动或显式加入第三案例，不能继续只硬编码 `basic/no_constant`。

## Fix 3：数学来源与 Stata 软件兼容性来源未精确分层

- **报告项**：`TA-REG-20260726-003`
- **根因**：`validation/commands/reg/formula_map.csv:3-20` 把 Stata 19
  `regress` 手册同时作为所有数学公式的唯一精确来源和 Stata 行为来源；
  `validation/commands/reg/theory.md:175-190` 中 Aitken (1936) 与
  Fisher (1922) 只有历史背景角色，没有公式级位置。因此公式的数学支持与
  软件参数化无法独立追踪。
- **修复位置**：
  - `validation/commands/reg/formula_map.csv:1-21`
  - `validation/commands/reg/theory.md:24-140,175-194`
  - `validation/commands/reg/manifest.yaml:3,141-151`
  - 建议在 `tests/test_validation_core.py:777` 后增加 formula-map
    来源角色结构测试
- **最小结构修改**：
  1. 将 formula map 的来源字段拆成至少四列：
     `mathematical_reference`、`mathematical_location`、
     `compatibility_reference`、`compatibility_location`。
     也可以保留现有列并新增明确的 `reference_role`，但每行必须能无歧义地指出
     哪个来源支持数学公式、哪个来源只规定 Stata 行为。
  2. `theory.md` 分成“数学来源”“原始历史来源”“Stata 兼容性来源”三层。
     不再写“Stata 手册是唯一 authoritative formula source”。
  3. 更新 manifest 的 `primary_sources`：加入下面已核实的 Lai & Xing
     和 Fisher 来源及角色；Stata 标记为 `software_compatibility`。
     Aitken 不应继续作为 release-critical 公式的无位置来源。
  4. 增加结构测试：所有 `implemented` 的数学行必须有数学来源和精确位置；
     Stata 软件专属的 sample、名称顺序、`noconstant` 报告约定可以明确标为
     `mathematical_reference=not_applicable`，但必须有精确 compatibility
     位置；任何数学行不得只剩 Stata 手册这一列。

### 已核实、可以直接使用的数学来源

**Lai, T. L. and Xing, H. (2008), _Statistical Models and Methods for
Financial Markets_, Springer Texts in Statistics.**

- DOI：<https://doi.org/10.1007/978-0-387-77827-3>
- Springer 书目页：<https://link.springer.com/book/10.1007/978-0-387-77827-3>
- 公式级位置（均为书内印刷页码，不是 PDF 查看器页码）：

| formula-map 组 | 已核实位置 | 可支持的主张 |
|---|---|---|
| `REG-OBJ-001` | p. 4, eq. (1.2) | 最小化残差平方和 |
| `REG-COEF-001`, `REG-RSS-001` | p. 5, eqs. (1.4)-(1.5) 及 eq. (1.4) 后的 RSS 定义 | \(\hat\beta=(X'X)^{-1}X'Y\)、RSS 与正交分解 |
| `REG-VCE-001`, `REG-SE-001`, `REG-DF-001`, `REG-RMSE-001` | p. 7, eqs. (1.9)-(1.10) | \(\operatorname{Cov}(\hat\beta)=\sigma^2(X'X)^{-1}\)、\(s^2=RSS/(n-p)\)、残差自由度 |
| `REG-T-001`, `REG-TP-001` | pp. 8-10, eqs. (1.11)-(1.14) 与 (1.22) | 正态误差下的精确 t 分布、系数检验和双侧绝对值统计量 |
| `REG-MSS-001`, `REG-F-001`, `REG-FP-001` | pp. 10-12, eqs. (1.23)-(1.25), Table 1.1 | 平方和/自由度分解、一般线性假设的 F 比率及右尾拒绝规则 |
| `REG-R2-001`, `REG-AR2-001` | pp. 13-14, eqs. (1.27)-(1.28) | 有常数模型的 \(R^2\) 与 adjusted \(R^2\) |

该书的 `p` 表示拟合参数个数；映射到当前合同的 `k` 即可。其 eq. (1.28)
是有常数模型。Tabra/Stata 的无常数 uncentered \(R^2\) 和 adjusted
\(R^2\) 是软件兼容性参数化，应继续由 Stata 手册精确支持，不应声称
Lai & Xing 的 eq. (1.28) 已直接给出无常数版本。

**Fisher, R. A. (1922), “The Goodness of Fit of Regression Formulae,
and the Distribution of Regression Coefficients,” JRSS 85(4):597-612.**

- DOI：<https://doi.org/10.2307/2341124>
- 可核实扫描件记录：<http://hdl.handle.net/2440/15174>
- 已核实位置：journal pp. 607-610, §6 “The distribution of regression
  coefficients”。p. 608 对简单线性回归使用以 `n-2` 为除数的残差方差估计；
  pp. 609-610 推广到多个正交回归系数，并把回归系数显著性归入 Student
  Type VII（现代 t）分布。
- 角色：这是回归系数有限样本 t 推断的原始论文支持，可加入
  `REG-T-001` 与 `REG-TP-001`；现代矩阵 OLS/VCE/F 公式仍以 Lai & Xing
  的精确公式为主，不要把 Fisher 的历史符号强行改写成未出现的现代矩阵公式。

### Stata 来源只承担兼容性角色

StataCorp LLC (2025), _Stata 19 Base Reference Manual_, `[R] regress`,
<https://www.stata.com/manuals/rregress.pdf>：

- p. 15：模型变量任一缺失时删除观测；
- pp. 8-10、22-23：常数、`noconstant`、centered/uncentered TSS、
  模型与总自由度和相应报告约定；
- p. 7、p. 26：Stata 系数表的双侧 t 概率、总体 F 的右尾解释及
  homoskedastic 情形使用 `N-k`；
- p. 22 eq. (2)、p. 23 eq. (3)：可以作为 Stata 实现兼容性的交叉佐证，
  但不再是数学公式的唯一 primary reference。

### 明确的 BLOCKED 边界

- Aitken (1936) 的文章元数据、页段 pp. 42-48 可以核实，但本次没有核实到
  能支持当前现代矩阵 OLS、VCE、t/F 和拟合统计量的具体公式位置。
  因此，不得把 Aitken 写入这些行的公式级位置。
- 最小修复是把 Aitken 降为不参与 release-critical 映射的历史背景，或从
  manifest 的 `primary_sources` 中移除。若维护者坚持必须用 Aitken
  作为公式证明，则该部分为 **BLOCKED：需要取得并逐页核实原文后才能映射**。
- White (1980) 仍只对应未实现的 robust 扩展；不得用它支持当前
  homoskedastic VCE，也无需为关闭本次已实现范围的缺陷虚构公式编号。

- **预期效果**：每项数学主张和每项 Stata 特有行为都有独立、准确、可核实的
  来源角色；审计不再把软件手册当作原始数学证明，也不再把仅有元数据的历史论文
  当作公式级证据。

## 应修改的文件清单

### 推荐最小方案

| 文件 | 具体修改 |
|---|---|
| `validation/commands/reg/oracles/code_audit_oracle.py` | 在其他工件定稿后更新最终 hashes；保留严格 guard |
| `tests/test_reg_validation_oracle.py`（新增） | 正式入口端到端 PASS 测试和静态 hash 同步测试 |
| `tests/test_ols.py` | 增加 y/x1/x2 分别部分缺失的独立共同样本测试 |
| `validation/commands/reg/formula_map.csv` | 映射新 complete-case 测试；拆分数学与兼容性来源及精确位置 |
| `validation/commands/reg/theory.md` | 加入已核实来源、页码/公式和角色边界；保留缺失机制与因果解释限制 |
| `validation/commands/reg/manifest.yaml` | `specification_version` 由 `2.0.0` 增至建议的 `2.0.1`；更新 primary source 及其角色 |
| `tests/test_validation_core.py` | 增加 formula-map 来源角色完整性测试 |
| `validation/runs/reg/<new_run_id>/...` | 生成全新、不可变的完整审计与三引擎证据；不得覆盖当前运行 |

不应修改 `src/tabra/`：报告没有发现需修复的算法、自由度、VCE 或
complete-case 产品实现错误。

## `specification_version` 与 `run_id`

| 变化 | `specification_version` | 新 `run_id` |
|---|---|---|
| 仅重绑 oracle 最终哈希 | 不需要 | 必须 |
| 仅增加独立单元测试 | 不需要 | 修复后重审必须 |
| 仅修订 `theory.md`/`formula_map.csv` 的来源和映射，manifest 不变 | 不需要 | 必须，因为冻结 theory/map 哈希改变 |
| 按推荐方案更新 manifest 的 `primary_sources`/角色 | 必须；建议 `2.0.1` | 必须 |
| 新增 `partial_missing` 冻结案例或改变 fixture/case 数组 | 必须；建议 `2.1.0` | 必须 |
| 改变输出、容差、估计样本语义或公开行为 | 必须；按兼容性影响决定 minor/major | 必须 |

`prepare-command-spec` 规定不得覆盖已经使用过的 specification；
`verify-all-commands` 规定任何 repair 后必须创建新 run，并完整重跑。
因此，即使只修 oracle 哈希，也不能在
`reg-validation-20260726-01` 中覆盖原审计文件或补发 attestation。

## 应新增的测试

1. `tests/test_ols.py::test_partial_missing_rows_use_joint_complete_case_sample`
   - y、每个 requested x 分别缺失；
   - 未请求列缺失不影响样本；
   - 手工固定共同样本；
   - 独立 NumPy 系数/标准误；
   - `n_obs`、参数顺序、残差长度和拟合统计量断言。
2. `tests/test_reg_validation_oracle.py::test_frozen_hashes_match_current_reg_inputs`
   - 当前 manifest/cases/data 字节哈希必须等于 oracle 常量。
3. `tests/test_reg_validation_oracle.py::test_reg_oracle_formal_entry_passes`
   - 不绕过 guard 的正式端到端 oracle 测试。
4. `tests/test_validation_core.py::test_reg_formula_map_separates_mathematical_and_compatibility_sources`
   - 数学行有可核实数学来源和位置；
   - 软件特有行为有 Stata compatibility 位置；
   - 禁止所有 implemented 数学行继续只引用 Stata。

## 验证命令

以下命令应在新 run 中执行；`<new_run_id>` 替换为唯一新 ID。

```bash
.venv/bin/python -m tabra.validation preflight \
  --manifest validation/commands/reg/manifest.yaml

.venv/bin/python -m pytest -q -o addopts='' \
  tests/test_ols.py tests/test_estimate_base_stats.py \
  tests/test_api.py::TestRegAPI tests/ops/test_stats.py \
  tests/test_validation_core.py tests/test_reg_validation_oracle.py

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

在运行三个 engine 前，必须由独立角色重新生成新 run 的
`code_audit.md`/`code_findings.json` 与
`theory_audit.md`/`theory_findings.json`。只有两项审计和 comparison
都通过后，才执行 gate：

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

验收条件是：正式 oracle `PASS`；部分缺失测试通过；formula map
来源角色测试通过；代码审计和理论审计均无开放项；三引擎所有要求案例完成且
comparison 通过；最终 gate 生成新 run 的不可变通过证明。
