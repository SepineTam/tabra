# `reg` 第三次修复分析

- 分析对象：`reg-validation-20260726-03`
- 分析边界：只修复 `cases_sha256`、`seed`、逐案例
  `included_row_sha256`、逐案例 `omitted_parameters` 四项 provenance 证据及其机器校验；
  不改变案例、fixture、估计样本语义、估计值、17 项指标、容差或
  `src/tabra/` 产品 estimator。
- 结论：报告有 3 组阻断项，落到 4 个缺失/未统一字段；全部需要验证基础设施修复，
  0 项需要产品代码修复，0 项可仅按数值误差关闭。

## 报告阻断项

| 范围 | 缺失证据 | 当前状态 | 根因类别 |
|---|---|---|---|
| 全局，Python/Stata | `cases_sha256` | normalized result 未记录；各自 environment 也未记录 | `ENGINE_ENV` / provenance 协议缺口 |
| `basic`、`no_constant`，Python/Stata | `seed` | R 的 `metadata.case_execution[]` 有值，Python/Stata 没有 | `ENGINE_ENV` / runner 归一化缺口 |
| `basic`、`no_constant`，Python/Stata | `included_row_sha256` | R 两案例均为 `11ebba9a...`；Python/Stata 无字段 | `MISSING_OUTPUT` / 样本身份证据缺口 |
| `basic`、`no_constant`，Python/Stata | `omitted_parameters` | R 显式为空数组；Python/Stata 无字段 | `MISSING_OUTPUT` / 参数化证据缺口 |
| 全局，机器 compare | 上述四项均未强制校验 | 原始机器结果错误地得到 `passed: true` | fail-open 比较逻辑 |

R 的本 run 工件虽已有后三项，但仓库当前 R runner 不能生成这些增强字段；因此修复必须
让三端都从同一个 `runner_common` 输出同名、同层、同类型字段，不能只给 Python/Stata
补一份手工 environment JSON。

现有 17 项指标均通过冻结容差。最大差异属于正常末位浮点漂移，例如系数
`4.88e-15`、t 统计量 `5.68e-14`、平方和约 `3.41e-13`；它们不需要修复，
也不能抵消 provenance 阻断。算法、`N-k` 自由度、同方差 VCE 和当前完整数据上的
complete-case 结果均无 mismatch。

## 唯一 normalized contract

四项字段固定在 normalized result 的 `metadata` 层，不加入 `metrics`，因此不新增
release-critical metric，也不触碰容差：

```json
{
  "schema_version": 2,
  "metadata": {
    "cases_sha256": "<64 lowercase hex>",
    "case_execution": [
      {
        "case_id": "basic",
        "seed": 20260722,
        "included_row_sha256": "<64 lowercase hex>",
        "omitted_parameters": []
      },
      {
        "case_id": "no_constant",
        "seed": 20260722,
        "included_row_sha256": "<64 lowercase hex>",
        "omitted_parameters": []
      }
    ]
  }
}
```

约束如下：

1. `metadata.cases_sha256` 是 runner 实际读取的 `cases.yaml` 文件字节 SHA-256。
2. `metadata.case_execution` 必须与 `metrics.case_ids` 及
   `manifest.supported_cases` 数量、顺序和 `case_id` 完全一致，不允许缺项、重复项或
   额外案例。
3. `seed` 必须是 JSON integer，且逐案例等于冻结 manifest 的 `data.seed`；当前两案例
   都是 `20260722`。
4. `included_row_sha256` 必须按下述唯一规范计算，并与该案例的实际估计样本绑定。
5. `omitted_parameters` 必须是按 canonical parameter identity 排列、元素唯一的字符串
   数组；没有省略参数时必须显式写 `[]`，不能省略或写 `null`。
6. R 现有额外诊断字段可以保留，但上述四项的字段名、层级和类型必须与 Python/Stata
   完全相同；机器结论只依赖统一字段，不解析 R 专用 environment 结构。

## included-row hash 唯一规范

采用当前 R 工件已使用并已由 fixture 验证为
`11ebba9a3453b6af0b448a00ad5c27aa9f5508a1cfdfacfe130c6752545dcf76`
的规范：

1. 以原始 CSV 数据顺序排列该案例实际包含的行。
2. 每行取原始 CSV 中 `row_id` 的 UTF-8 文本值；不排序、不 trim、不做数值重新格式化、
   不做 Unicode normalization。
3. 每个 `row_id` 后追加一个字节 `LF` (`0x0a`)，包括最后一个；空样本的 preimage
   是空字节串，但 completed 案例仍由既有 `nobs > 0` 规则拒绝空样本。
4. 对拼接后的字节串计算 lowercase SHA-256。
5. runner 在计算前必须拒绝缺失、空字符串或重复的 `row_id`；外部引擎回传的是
   1-based 原始数据行号，由 Python wrapper 映射回未经重新格式化的 CSV `row_id`。

用形式化记号表示，若实际包含的原始行依次为 `i1,...,im`，则：

```text
SHA256(UTF8(row_id[i1] + "\n" + ... + row_id[im] + "\n"))
```

选择这一规范而不是对浮点数据行或 engine 内部 row name 哈希，是因为 `row_id` 已是
fixture 的稳定唯一身份；原始词法值由共享 wrapper 读取，可避开 Stata/R 的数值显示和
locale 差异。不能让 wrapper 仅根据 `nobs` 假定“前 N 行”或“全体行”。

## 最小修复项

### Fix 1：升级 normalized evidence contract，并把 cases/seed 冻结值交给比较器

- **报告项**：全局 `cases_sha256`、两个案例的 `seed`，以及旧结果可通过 schema。
- **根因**：
  - `validation/schema/normalized-result.schema.json:19-49` 只要求旧 metadata；
  - `src/tabra/validation/io.py:18-21,119-125,346-432` 用同一个
    `SCHEMA_VERSION=1` 校验 manifest、registry 和 result，且不解析四项证据；
  - `src/tabra/validation/models.py:169-184` 的 `NormalizedResult` 无对应字段；
  - manifest 有 `data.seed`，但没有冻结的 cases 文件哈希供 compare 对照。
- **修复位置**：
  - `validation/schema/normalized-result.schema.json:19-49`
  - `src/tabra/validation/io.py:18-21,119-125,162-280,346-432`
  - `src/tabra/validation/models.py:101-126,169-184`
  - `validation/commands/reg/manifest.yaml:2-35`
- **修复方式**：
  1. normalized result 的 `schema_version` 升为 `2`；schema 将
     `metadata.cases_sha256` 和上述严格结构的 `metadata.case_execution` 设为 required。
  2. 在 `io.py` 拆分版本常量：manifest/registry 仍校验版本 1，normalized result
     校验版本 2；不要把全局 `SCHEMA_VERSION` 直接改成 2，否则会无关地迫使 registry
     和 manifest 结构升级。
  3. `load_normalized_result()` 严格校验 hash、integer seed、case object、唯一字符串
     omitted 数组，并把 `cases_sha256`、冻结后的 `case_execution` 暴露给
     `NormalizedResult`。
  4. manifest 新增 `cases: {"path": "cases.yaml", "sha256": "1b107cfe..."}`；
     `load_manifest()` 校验并保存 `cases_sha256`，同时把现有 `data.seed` 校验为 integer
     并保存为 `CommandManifest.seed`。cases 文件和 seed 值本身不变。
  5. manifest 的 `required_outputs.metadata/case` 补记四项证据合同。
- **预期效果**：旧 v1 或缺字段工件在加载阶段即以非零退出，无法再次产生
  `passed: true`；compare 有冻结的 cases hash 和 seed 作为独立对照。

### Fix 2：三端 runner 生成同名四字段

- **报告项**：Python/Stata 缺少全部四项证据；当前仓库 R runner 不能重现本 run 的
  enriched result。
- **根因**：`runner_common.make_document()` 只生成旧 metadata；Python 未保留案例样本
  身份；Stata/R 的 TSV 只输出参数与统计量，wrapper 没有实际样本行号或 omitted flag。
- **修复位置**：
  - `validation/commands/reg/runner_common.py:12-19,52-121,124-179`
  - `validation/commands/reg/python/run.py:27-62,71-93`
  - `validation/commands/reg/stata/run.do:5,16-52`
  - `validation/commands/reg/stata/run.py:66-125`
  - `validation/commands/reg/r/run.R:1-73`
  - `validation/commands/reg/r/run.py:47-81`
- **修复方式**：
  1. `runner_common` 新增唯一的 row-id 读取/验证/hash helper；`make_document()` 从每个
     normalized case 的 evidence 生成统一 `metadata.case_execution`，并输出
     `schema_version: 2` 和实际 `cases_sha256`。
  2. Python 对每个 case 先以 `[dependent, *independent]` 构造 complete-case mask，
     用该 mask 的 dataframe 调用 `OLS.fit()`，以原始 dataframe 顺序对应的 raw
     `row_id` 计算 hash，并断言 mask 行数等于 `result.n_obs`。这不改变当前样本；只是
     让传给 estimator 的 dataframe 与记录的身份显式一致。`omitted_parameters` 由
     请求的 canonical 参数集合与 `result.var_names` 比较得到；当前满秩案例为 `[]`。
  3. Stata `run.do` 增加第三个临时输出路径。每次 `regress` 后立即按 `e(sample)` 写出
     `case_id` 与 1-based 原始数据行号到 sample sidecar；参数 TSV 同时输出每个 term
     的 omitted flag（用 `_ms_omit_info e(b)`/`r(omit)`，不能凭 `_b==0` 推断）。
     `stata/run.py` 在 direct 和 MCP driver 两条路径都传入 sidecar 路径，再由共享
     wrapper 将实际行号映射到 raw CSV `row_id` 并计算 hash，将 flagged term 按
     canonical 顺序组成 `omitted_parameters`。
  4. R `run.R` 同样输出每个 `lm` 对象实际 model frame 对应的 1-based 原始数据行号
     和 term omitted 信息；`r/run.py` 调用同一共享 normalizer/hash helper。不得从
     Python 预先计算的完整案例 mask 直接填充 R/Stata 字段，那不能证明外部引擎的
     实际 `e(sample)`/model frame。
  5. 三个 runner 都直接读取并哈希实际 `cases.yaml`，seed 从各 case specification
     写入；wrapper 校验每个 raw case 恰好出现一次且顺序与 cases 文件一致。
- **预期效果**：三个 normalized result 可重复地产生完全同名、同层的四项证据；
  当前两案例的 included-row hash 均为上面的 `11ebba9a...`，omitted 均为 `[]`。

### Fix 3：机器 compare 先校验证据，再比较指标

- **报告项**：当前 `compare_results()` 只比较总行数和输入数据 hash，因此原始机器
  comparison 在必需证据缺失时仍 `passed: true`。
- **根因**：`src/tabra/validation/compare.py:139-417` 没有 cases hash、seed、逐案例
  sample identity 或 omitted parameter 检查。
- **修复位置**：
  - `src/tabra/validation/compare.py:139-417`
  - `tests/test_validation_core.py:38-114,252-474`
- **修复方式**：在构造任何 metric pair 前，按 manifest case 顺序执行：
  1. 每个 engine 的 `cases_sha256 == manifest.cases_sha256`；否则记录
     `case_specification_hash_mismatch`。
  2. `case_execution.case_id` 与冻结 case 顺序 exact equal，且每个 `seed ==
     manifest.seed`；否则记录 `case_execution_mismatch`/`case_seed_mismatch`。
  3. 每案例三端 `included_row_sha256` exact equal；否则记录
     `included_row_hash_mismatch`。
  4. 每案例三端 `omitted_parameters` exact equal；否则记录
     `omitted_parameters_mismatch`。
  5. 只有上述所有检查通过才进入现有 17 项 metric 循环。任一 provenance/sample/
     parameterization 检查失败时，`passed=false`、CLI 退出 1，并将
     `metric_comparison_performed=false`；结构缺失或 schema v1 则 loader 直接失败、
     CLI 退出 2。两条路径都不得写出或返回成功比较。
  6. comparison JSON 增加确定性的 `case_evidence` 和
     `metric_comparison_performed`，列出三端原值和逐案例 pass/fail；不得只给一个总布尔值。

该全局停表策略比“只跳过坏案例”更保守，但对当前仅两个案例的扁平 metrics 数组是严格
最小且 fail-closed 的实现；它满足先比较 included-row hashes、样本不一致时不比较估计值
的要求。`gate.evaluate_gate()` 已重新调用 `compare_results()`，无需修改 gate；修复后它会
自动拒绝缺字段、篡改或旧 comparison。

## 明确不修改

- 不修改 `src/tabra/models/estimate/ols.py`、`BaseModel._prepare_df`、结果类或公开 API。
- 不修改 `cases.yaml`、fixture CSV、sidecar、generator、17 项 metrics 或任何 tolerance。
- 不把 provenance 四字段加入浮点 tolerance 体系；它们全部 exact/fail-closed。
- 不回写 `reg-validation-20260726-03` 的 Python/Stata/R result、environment、comparison
  或审计工件；本文件是该 run 唯一新增内容。

## 版本、oracle hash 与新 run 判断

| 项目 | 判断 | 操作 |
|---|---|---|
| normalized result `schema_version` | 必须 `1 -> 2` | 必需字段改变，旧工件必须被明确拒绝；只升级 result schema，不升级 manifest/registry schema |
| manifest `schema_version` | 保持 `1` | 不重做通用 manifest/registry artifact schema；由 command 的 specification version 标记本次合同补丁 |
| manifest `specification_version` | 必须 `2.0.2 -> 2.0.3` | required evidence contract 和 cases hash binding 改变，但案例、计算语义、指标、容差均未变，属于 patch |
| oracle 内嵌 hash | 只更新 `FROZEN_HASHES["manifest"]` | manifest 字节改变；cases/data 字节未变，其两个常量必须保持现值 |
| oracle 文件自身 hash | 新 run 重新记录 | 修改 manifest 常量后 oracle 文件 hash 会变；写入新 run `inputs.json`/审计，不能自我回填 |
| `run_id` | 必须新建 | `reg-validation-20260726-03` 已是不可变失败证据；实施后使用尚不存在的唯一 ID，例如 `reg-validation-20260726-04` |

顺序必须是：先完成 schema、manifest、runner、compare 和测试；再计算最终 manifest hash
并仅更新 oracle 的 manifest 常量；提交修复；最后以该干净 commit 创建新 run，重做两项
审计、三引擎、comparison 和 gate。不能用当前 run 的 reporter 注释 JSON 作为 gate 输入。

## TDD 清单

先写以下失败测试，再实施对应修复：

1. `tests/test_validation_core.py`
   - v1 normalized result 或缺少任一四字段时 `load_normalized_result()` 失败；
   - 非法 cases hash、bool seed、重复/非字符串 omitted、非法 included hash 失败；
   - 三端完整且相同证据通过；
   - 三端共享错误 cases hash 也失败，证明不是多数表决；
   - 任一 case seed 不等于 manifest seed 失败；
   - 任一 case included hash 不同，`passed=false` 且不执行 metrics；
   - 任一 case omitted 列表不同，exact failure；
   - case_execution 缺项、重复或顺序错误失败。
2. `tests/test_reg_validation_oracle.py`
   - Python runner 的两个案例产生四字段；hash 为 `11ebba9a...`、seed 为
     `20260722`、omitted 为 `[]`；
   - oracle 的 manifest/cases/data frozen hashes 与最终文件一致，正式 oracle 继续 PASS。
3. 新增 `tests/test_reg_runner_common.py`
   - canonical hash helper 对 `1\n2\n...\n120\n` 得到固定 digest；改顺序、去末尾 LF、
     重复/空 row_id 均被区分或拒绝；
   - 模拟 Stata/R raw result + sample sidecar，验证实际行号到 raw row_id 的映射、逐案例
     hash、omitted flag 和 case 顺序；无需在单元测试中依赖已安装的 Stata/R。
4. 现有 compare/gate 测试 fixture 全部升级到 normalized schema v2，确认 gate 重算后
   同样执行四项检查。

## 完整复验命令

以下命令只在实施修复后运行；`reg-validation-20260726-04` 若届时已存在，必须换成新的
唯一 ID。

```bash
git diff --check

.venv/bin/python -m pytest -q -o addopts='' \
  tests/test_validation_core.py \
  tests/test_reg_runner_common.py \
  tests/test_reg_validation_oracle.py

.venv/bin/python -m pytest -q -o addopts='' \
  tests/test_ols.py \
  tests/test_ols_result.py \
  tests/test_estimate_base_stats.py \
  tests/test_api.py::TestRegAPI \
  tests/ops/test_stats.py \
  tests/test_validation_core.py \
  tests/test_reg_runner_common.py \
  tests/test_reg_validation_oracle.py

.venv/bin/python -m tabra.validation preflight \
  --manifest validation/commands/reg/manifest.yaml

shasum -a 256 \
  validation/commands/reg/manifest.yaml \
  validation/commands/reg/cases.yaml \
  validation/commands/reg/data/reg_fixture.csv \
  validation/schema/normalized-result.schema.json \
  validation/commands/reg/oracles/code_audit_oracle.py

.venv/bin/python validation/commands/reg/python/run.py \
  --data validation/commands/reg/data/reg_fixture.csv \
  --output validation/runs/reg/reg-validation-20260726-04/python/results.json

STATA_CMD=/usr/local/bin/stata-mp \
  .venv/bin/python validation/commands/reg/stata/run.py \
  --data validation/commands/reg/data/reg_fixture.csv \
  --output validation/runs/reg/reg-validation-20260726-04/stata/results.json

RSCRIPT_CMD=/usr/local/bin/Rscript \
  .venv/bin/python validation/commands/reg/r/run.py \
  --data validation/commands/reg/data/reg_fixture.csv \
  --output validation/runs/reg/reg-validation-20260726-04/r/results.json

.venv/bin/python validation/commands/reg/oracles/code_audit_oracle.py \
  --manifest validation/commands/reg/manifest.yaml \
  --cases validation/commands/reg/cases.yaml \
  --data validation/commands/reg/data/reg_fixture.csv \
  --actual validation/runs/reg/reg-validation-20260726-04/python/results.json

.venv/bin/python -m tabra.validation compare \
  --manifest validation/commands/reg/manifest.yaml \
  --result validation/runs/reg/reg-validation-20260726-04/python/results.json \
  --result validation/runs/reg/reg-validation-20260726-04/stata/results.json \
  --result validation/runs/reg/reg-validation-20260726-04/r/results.json \
  --output validation/runs/reg/reg-validation-20260726-04/comparison.json
```

在 compare 之前，由独立 code-auditor 与 theory-auditor 对同一新 commit、manifest hash
和 run ID 重新生成审计工件。确认三端 `case_evidence` 全部 exact PASS、17 项指标仍按
原容差 PASS、两项审计均 PASS 后，执行：

```bash
.venv/bin/python -m tabra.validation gate \
  --manifest validation/commands/reg/manifest.yaml \
  --comparison validation/runs/reg/reg-validation-20260726-04/comparison.json \
  --run-id reg-validation-20260726-04 \
  --code-audit validation/runs/reg/reg-validation-20260726-04/code_audit.md \
  --theory-audit validation/runs/reg/reg-validation-20260726-04/theory_audit.md \
  --result validation/runs/reg/reg-validation-20260726-04/python/results.json \
  --result validation/runs/reg/reg-validation-20260726-04/stata/results.json \
  --result validation/runs/reg/reg-validation-20260726-04/r/results.json \
  --workdir . \
  --output validation/runs/reg/reg-validation-20260726-04/attestation.json
```

最终验收条件：旧 v1/缺字段 result 无法通过加载；三端四字段逐案例 exact；样本 hash
不一致时机器不比较指标且退出非零；当前 17 项数值与原冻结容差不变；正式 oracle、两项
独立审计、machine comparison 和 gate 均在新 run、同一干净 commit 上通过。
