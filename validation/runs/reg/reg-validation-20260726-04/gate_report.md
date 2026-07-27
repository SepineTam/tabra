# `reg` 最终验证门报告

## 决定

`FAILED`

产品 gate CLI 在唯一一次最终执行中拒绝放行。失败原因为
`comparison_report_mismatch`：存储的 `comparison.json` 与 gate 从冻结
manifest 和三份 normalized results 重新计算的比较对象不完全相同。

- 存储比较对象的规范化 SHA-256：
  `125a33ffaa9261fb23e5ea3b4758af7848f40c7a77d76b513b74c90510aae181`
- 重新计算比较对象的规范化 SHA-256：
  `2f71427d97e4fa94ecd449b6ebba5e4a749093d32bec200647a3d5a05f6f343f`

这不是环境缺失，因此不能判为 `BLOCKED_ENV`。根据 fail-closed 规则，
本 run 也不能判为 `PASSED`。

## 冻结身份

- Run ID：`reg-validation-20260726-04`
- Audited commit：`47aebd1b2681607027a70275eb443060c5df0457`
- Manifest SHA-256：
  `bfee2c597697569b212aeb67edbbf34d9e42421272d4430253d6724a45f121b7`
- Cases SHA-256：
  `1b107cfe0d79a0997a04a5a251e4a1eb909404943d7e68f337aae8ea14612c07`
- Data SHA-256：
  `53f6fea92a7978cbce09212c224eb6e29e9c581ab920e0285ef541c4b09d37c4`
- Repair attempts used：`3`
- Gate 执行时受版本控制的工作区：clean

## 已核对但不足以覆盖 gate 失败的证据

- 独立代码审计：`PASS`，零开放发现。
- 独立理论审计：`PASS`，零开放发现，必需文献可用。
- Python、Stata、R：均完成 `2/2` 必需案例；三端均为
  `completed / skipped=false / converged=true`。
- 必需指标：存储报告声称 `17/17` 通过；无必需案例 skip、NaN、Inf、
  非收敛或遗漏参数。
- 三端案例、seed `20260722`、样本成员哈希和数据哈希一致。

上述科学结果不能替代 gate 对比较工件不可篡改性的重新计算检查。

## 主要证据文件 SHA-256

- `attestation.json`：
  `42586b1e4d20441eb3fc631d274831bb709bf68f5533c719d00cb6832890bf33`
- `code_findings.json`：
  `3c10982e46dcb179b4c2c999b435e9caaaa65ba7f896d78548263c54d64e0cbf`
- `theory_findings.json`：
  `c363f8a87a1b1d187e0cb26e25233051f2bfc47c55bec0baebfaaf3c5f2a9ab8`
- `comparison.json`：
  `463243ac5bbcf6a2b70a0e7dc3d24b4191adaf21ee34a6dad9d9dc934be83542`
- Python result：
  `33488e61673c56eaede753dd3588d6fdb21504a79366a02a4d983e80d937945a`
- Stata result：
  `4ba938a20b66e19adcb385024d19b60489801e86d2b43b321a8d8d9f67029b3a`
- R result：
  `93dd1dba7643ae9dd765ac2913b748ec64fe365e324d58740aa9f2caaff343c2`

## 限制与下一状态

本决定只覆盖冻结的 `basic` 与 `no_constant`、无权重、同方差 OLS；不放行
robust、cluster、bootstrap、weighted、分类变量、吸收效应、约束回归或因果
解释。完整测试套件中的 52 个 skip 属于其他估计模块，没有作为 `reg` 的通过
证据；本 run 的必需工作没有 skip。

本 run 的唯一允许后续状态是 `GATE_PENDING -> FAILED`。不得覆盖
`attestation.json`，不得在本 run 内修补 comparison 或再次 gate。若要继续，
必须由非 gatekeeper 角色诊断比较工件的结构差异，修复后建立新 run ID，并从
规格绑定、两项独立审计、三引擎执行和比较重新走完整流程。
