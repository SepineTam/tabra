# reg 独立验证 Gate 报告

## 决定

`PASSED`

产品 Gate CLI 已且仅已执行一次，退出码为 0，并创建不可覆盖的
`attestation.json`。两项独立审计均为 PASS；Python、Stata、R 均执行
2/2 个冻结案例；全部 17 个 release-critical 指标通过；必需范围内没有
失败、警告、跳过、NaN、Inf、缺失输出或非收敛。

## 冻结身份

- Run ID：`reg-validation-20260727-05`
- 审计提交：`9ce5b597d9e3a2d7b3168262519b2be3019116e9`
- Gate 时 tracked worktree：clean；未授权 dirty worktree
- Manifest SHA-256：`bfee2c597697569b212aeb67edbbf34d9e42421272d4430253d6724a45f121b7`
- Canonical comparison 对象 SHA-256：`2f71427d97e4fa94ecd449b6ebba5e4a749093d32bec200647a3d5a05f6f343f`
- Comparison 文件 SHA-256：`2b01f0e1762b31a578c12f954fa4f454202007de4403729b916560102ad51890`
- Attestation 身份 SHA-256：`0ff3c9915a751e93f0b9f7ef3fc11974fb34ee2f63dc616d145d1ede758194f9`
- Attestation 文件 SHA-256：`49f41b6380bda4cc02dd794470bc2e12c4e1cef56d5f6173af46af42dfb3efb3`

结果文件 SHA-256：Python
`33488e61673c56eaede753dd3588d6fdb21504a79366a02a4d983e80d937945a`，
Stata `4ba938a20b66e19adcb385024d19b60489801e86d2b43b321a8d8d9f67029b3a`，
R `93dd1dba7643ae9dd765ac2913b748ec64fe365e324d58740aa9f2caaff343c2`。

## 证据核验

- `state.json` 为 `GATE_PENDING`，状态日志完整到 Gate 提交阶段。
- run05 属于新的 validation cycle，明确承接但不修改历史失败 run04。
- run04 的 attestation 仍为 FAILED，原因仍是
  `comparison_report_mismatch`；其 tracked 历史证据未改变。
- 当前 HEAD、inputs、两项审计、三端环境和 Gate attestation 的提交身份一致。
- 七项冻结输入哈希均与当前文件实算值一致。
- 三端案例顺序均为 `basic`、`no_constant`；seed、cases、dataset、样本成员和
  omitted parameters 完全一致。
- 三端结果均为 `completed`、`converged=true`、`skipped=false`。
- `comparison-check.json` 为 `canonical_match=true`、
  `scientific_passed=true`，supplied 与 recomputed 对象哈希一致。

## 限制

- 本次通过只覆盖冻结范围内的经典同方差 OLS，以及含常数和不含常数两个案例。
  robust、cluster、bootstrap、weights、分类变量编码、约束和吸收效应未获通过。
- 只在一套 macOS arm64 环境中验证，仍存在跨平台和 BLAS 差异风险。
- 冻结三端数据没有缺失值；complete-case 行为由独立单元测试覆盖，而不是由本次
  三端 fixture 覆盖。
- 全量测试中的 52 个 skip 和 6 个 warning 属于其他估计器或绘图路径，未作为
  `reg` 通过证据；必需 `reg` 范围没有 skip 或 warning。
- 文件系统没有物理锁定 canonical comparison；CLI 拒绝覆盖，Gate 会通过重算
  检测事后修改。

## 下一步

允许状态转换为 `GATE_PENDING -> PASSED`。下游只有依据本次 PASSED attestation
才可更新 registry 或稳定性标签；Gatekeeper 本身不执行这些更新。
