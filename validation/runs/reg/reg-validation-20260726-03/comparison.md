# reg 三引擎比较报告

## 结论

`comparison_status: BLOCKED_ENV`

仓库现有命令 `.venv/bin/python -m tabra.validation compare` 已以退出码 0 生成机器比较；其未注释原始 JSON 的 SHA-256 为 `d1d07472bddba2db9125122fd158227fe3470ed8115d839434ef3f11bafb77a8`。机器比较确认三引擎的命令、数据哈希、必需案例、版本及 manifest 中全部 17 个指标通过冻结规则，且无数值 mismatch。

这不能形成总体 PASS。Python 与 Stata 的 normalized result 和 environment 均未记录 `cases_sha256`、seed、逐案例 `included_row_sha256` 或等价样本 mask；只有 R 工件记录了这些字段。因此无法证明三引擎使用同一冻结案例定义、seed 和完全相同的估计样本。依照 `compare-three-engines` 规则，缺少必需引擎证据必须记为 `BLOCKED_ENV`，不能把相同的 `nobs=120`、数据哈希或三方数值接近当作样本身份哈希的替代。

顶层 `comparison.json` 保留机器命令生成的 `passed: true` 作为该命令自身范围内的原始结果，并另以 `comparison_status: BLOCKED_ENV` 和 `reporter_checks` 记录严格报告结论。该 JSON 已经报告者注释，后续 gate 不应把它当作未经扩展的机器 JSON；本报告没有运行 gate，也没有签发 attestation。

## 冻结身份与原始证据

| 项目 | 冻结值 | 当前证据 | 检查 |
|---|---|---|---|
| run ID | `reg-validation-20260726-03` | 三份结果及环境目录相同 | PASS |
| Git commit | `64c92fe6b136b5b3adc12895d408e5c20af9214c` | 当前 HEAD、inputs、审计、Python/Stata/R 环境一致 | PASS |
| manifest | `902392d562a50aa9919cfc58bc95b0018e63044452fa8b6203608142814ecf79` | 文件实算、inputs、三引擎可得环境证据一致 | PASS |
| cases | `1b107cfe0d79a0997a04a5a251e4a1eb909404943d7e68f337aae8ea14612c07` | 文件实算与 inputs/R 一致；Python、Stata 环境未记录 | BLOCKED_ENV / `ENGINE_ENV` |
| data | `53f6fea92a7978cbce09212c224eb6e29e9c581ab920e0285ef541c4b09d37c4` | 文件实算、manifest、三份结果及环境一致 | PASS |
| seed | `20260722` | cases、代码审计、R 案例环境一致；Python、Stata 工件未记录 | BLOCKED_ENV / `ENGINE_ENV` |
| formula map | `5b3c7233a6522311ceed5936a72433d6c52f46596a92353b96aa8d90dad07eb0` | 文件实算、inputs、理论审计一致 | PASS |

原始工件哈希：Python result `78c957b4de118cb40da9e3bb5bbfa150d71ee86f4dd5a12ee08ba53fcb94820c`，Stata result `3f51adfbd99b639bd6aad78b74cb1469aa3f9e6db5a731d69a7c95c3618509e1`，R result `f4bea6b077320b4757980a8b46013b80e7d57f57a9b8848ce159f34f6301ce60`。Python/Stata environment 中记录的 result hash 与实算相同；R environment 未记录 result hash，但本报告保留了实算值。

环境工件 SHA-256：Python `f2dbf8fb540fdfccf02325d52ad590e1e54eb289306df12cf8696a98b2bc9e50`，Stata `0c244e0c6ea48f53208547defafbc1d1859de68e3c63768691d2988b141fef29`，R `d541cc4e381f0335b023e72cc0ffbd35b8567a590617c9cf14cc9e6c5a673347`。

## 样本身份检查

| 引擎 | basic nobs | no_constant nobs | included-row SHA-256 | 判定 |
|---|---:|---:|---|---|
| Python | 120 | 120 | 未记录 | BLOCKED_ENV / `MISSING_OUTPUT` |
| Stata | 120 | 120 | 未记录 | BLOCKED_ENV / `MISSING_OUTPUT` |
| R | 120 | 120 | `11ebba9a3453b6af0b448a00ad5c27aa9f5508a1cfdfacfe130c6752545dcf76`（两案例） | 仅 R 自身可验证 |

R 采用的规范化规则为：按数据顺序将 UTF-8 `row_id` 每行一个并保留末尾换行后计算哈希。Python 和 Stata 没有提供同口径证据，故 `sample_membership_equality = UNVERIFIABLE`。下面的指标比较作为完整的条件性数值证据保留，但不能解除环境阻断。

## 版本、状态和参数化

| 检查 | Python | Stata | R | 结果 |
|---|---|---|---|---|
| 引擎版本 | 3.11.11，要求 ≥3.11 | 19.5，要求 ≥16 | 4.4.1，要求 =4.4.1 | PASS |
| command | `reg` | `reg` | `reg` | PASS |
| status/skipped/converged | completed / false / true | completed / false / true | completed / false / true | PASS |
| case status | basic、no_constant 均 completed | 同左 | 同左 | PASS |
| 参数顺序 basic | `x1,x2,_cons` | 同左 | 同左 | PASS |
| 参数顺序 no_constant | `x1,x2` | 同左 | 同左 | PASS |
| df basic | model 2 / resid 117 | 同左 | 同左 | PASS |
| df no_constant | model 2 / resid 118 | 同左 | 同左 | PASS |
| base category | 不适用 | 不适用 | 不适用 | PASS |
| omitted parameters | 未显式记录 | 未显式记录 | 空数组 | BLOCKED_ENV / `MISSING_OUTPUT` |
| 样本 mask/hash | 未记录 | 未记录 | 两案例均有 included-row hash | BLOCKED_ENV / `MISSING_OUTPUT` |

参数标签、顺序、常数参数化、自由度和收敛状态完全相等；但 Python/Stata 未显式记录 omitted parameters 和样本 mask/hash，不能把 R 的空数组或三方相同的参数名多数表决为完整证据。

## 全部 17 个指标

相对差异定义为 `abs(left-right) / max(abs(left), abs(right))`，两值都为零时记 0。向量指标按案例列出全部三引擎值，pairwise 列为该案例所有分量的最大绝对差和最大相对差。`P-S`、`P-R`、`S-R` 分别表示 Python–Stata、Python–R、Stata–R。所有容差均直接来自冻结 manifest，未作调整。`p_values` 和 `f_p_value` 还通过了冻结的 `reject_zero_mismatch=true` 检查。

| 指标 | 案例 | Python | Stata | R | P-S max abs / rel | P-R max abs / rel | S-R max abs / rel | 冻结容差 | 结果 | 分类 |
|---|---|---|---|---|---:|---:|---:|---|---|---|
| `case_ids` | 全局 | `[basic,no_constant]` | `[basic,no_constant]` | `[basic,no_constant]` | exact equal | exact equal | exact equal | exact | PASS | — |
| `nobs` | `basic` | `120` | `120` | `120` | exact equal | exact equal | exact equal | exact | PASS | — |
| `nobs` | `no_constant` | `120` | `120` | `120` | exact equal | exact equal | exact equal | exact | PASS | — |
| `df_model` | `basic` | `2` | `2` | `2` | exact equal | exact equal | exact equal | exact | PASS | — |
| `df_model` | `no_constant` | `2` | `2` | `2` | exact equal | exact equal | exact equal | exact | PASS | — |
| `df_resid` | `basic` | `117` | `117` | `117` | exact equal | exact equal | exact equal | exact | PASS | — |
| `df_resid` | `no_constant` | `118` | `118` | `118` | exact equal | exact equal | exact equal | exact | PASS | — |
| `parameter_names` | `basic` | `[x1,x2,_cons]` | `[x1,x2,_cons]` | `[x1,x2,_cons]` | exact equal | exact equal | exact equal | exact | PASS | — |
| `parameter_names` | `no_constant` | `[x1,x2]` | `[x1,x2]` | `[x1,x2]` | exact equal | exact equal | exact equal | exact | PASS | — |
| `coefficients` | `basic` | `[2.4069521063774233,-0.8200631259823029,1.6812901529393778]` | `[2.406952106377425,-0.8200631259823036,1.6812901529393773]` | `[2.40695210637742,-0.820063125982304,1.68129015293938]` | 1.7763568394002505e-15 / 8.1229577781243711e-16 | 3.1086244689504383e-15 / 1.3538262963540612e-15 | 4.8849813083506888e-15 / 2.0295299168635361e-15 | atol=1e-10, rtol=1e-9 | PASS | — |
| `coefficients` | `no_constant` | `[2.298318605567287,-0.8206119775920795]` | `[2.298318605567289,-0.82061197759208]` | `[2.29831860556729,-0.82061197759208]` | 1.7763568394002505e-15 / 7.7289407791301254e-16 | 3.1086244689504383e-15 / 1.3525646363477712e-15 | 1.3322676295501878e-15 / 5.7967055843475906e-16 | atol=1e-10, rtol=1e-9 | PASS | — |
| `standard_errors` | `basic` | `[0.03752003308109958,0.0424577073591154,0.03512838040010806]` | `[0.03752003308109959,0.04245770735911541,0.035128380400108045]` | `[0.0375200330810996,0.0424577073591154,0.035128380400108]` | 1.3877787807814457e-17 / 3.9505914163272288e-16 | 5.5511151231257827e-17 / 1.5802365665308915e-15 | 4.163336342344337e-17 / 1.1851774248981692e-15 | atol=1e-10, rtol=1e-8 | PASS | — |
| `standard_errors` | `no_constant` | `[0.16917168766937923,0.19178609635590663]` | `[0.1691716876693792,0.19178609635590663]` | `[0.169171687669379,0.191786096355907]` | 2.7755575615628914e-17 / 1.6406749851590437e-16 | 3.6082248300317588e-16 / 1.8813797760061797e-15 | 3.6082248300317588e-16 / 1.8813797760061797e-15 | atol=1e-10, rtol=1e-8 | PASS | — |
| `test_statistics` | `basic` | `[64.15111898155298,-19.314823550081314,47.861305696126145]` | `[64.15111898155303,-19.31482355008133,47.86130569612615]` | `[64.151118981553,-19.3148235500813,47.8613056961262]` | 4.2632564145606011e-14 / 7.3574861703265033e-16 | 5.6843418860808015e-14 / 1.187669622339802e-15 | 4.9737991503207013e-14 / 1.4714972340653007e-15 | atol=1e-9, rtol=1e-8 | PASS | — |
| `test_statistics` | `no_constant` | `[13.585716600871223,-4.278787634684584]` | `[13.585716600871235,-4.278787634684587]` | `[13.5857166008712,-4.27878763468459]` | 1.2434497875801753e-14 / 9.1526256885149104e-16 | 2.3092638912203256e-14 / 1.6997733421527706e-15 | 3.5527136788005009e-14 / 2.6150359110042601e-15 | atol=1e-9, rtol=1e-8 | PASS | — |
| `p_values` | `basic` | `[5.089289619277885e-93,3.4212390558695086e-38,1.1031530193039544e-78]` | `[5.089289619277556e-93,3.421239055869285e-38,1.103153019303933e-78]` | `[5.08928961927799e-93,3.42123905586929e-38,1.1031530193039e-78]` | 2.2342642498426948e-51 / 6.5305703967384885e-14 | 2.1872820576731054e-51 / 6.393245318302399e-14 | 4.6982192169589377e-53 / 8.5338413839856592e-14 | atol=1e-10, rtol=1e-8 | PASS | — |
| `p_values` | `no_constant` | `[6.954966563121992e-26,3.8410727594931455e-5]` | `[6.954966563121544e-26,3.841072759493087e-5]` | `[6.95496656312169e-26,3.8410727594931e-5]` | 5.8275866771095863e-19 / 6.4370984347226172e-14 | 4.5400965972830498e-19 / 4.3409150982873037e-14 | 1.2874900798265365e-19 / 2.0961833364354049e-14 | atol=1e-10, rtol=1e-8 | PASS | — |
| `r_squared` | `basic` | `0.9724330765125854` | `0.9724330765125854` | `0.972433076512585` | 0 / 0 | 3.3306690738754696e-16 / 3.4250882187390953e-16 | 3.3306690738754696e-16 / 3.4250882187390953e-16 | atol=1e-10, rtol=1e-8 | PASS | — |
| `r_squared` | `no_constant` | `0.6110906237033638` | `0.6110906237033635` | `0.611090623703364` | 2.2204460492503131e-16 / 3.63357898668752e-16 | 2.2204460492503131e-16 / 3.6335789866875185e-16 | 4.4408920985006262e-16 / 7.267157973375037e-16 | atol=1e-10, rtol=1e-8 | PASS | — |
| `r_squared_adj` | `basic` | `0.9719618470512621` | `0.9719618470512621` | `0.971961847051262` | 0 / 0 | 1.1102230246251565e-16 / 1.1422495934314203e-16 | 1.1102230246251565e-16 / 1.1422495934314203e-16 | atol=1e-10, rtol=1e-8 | PASS | — |
| `r_squared_adj` | `no_constant` | `0.604498939359353` | `0.6044989393593527` | `0.604498939359353` | 2.2204460492503131e-16 / 3.6732009018965996e-16 | 0 / 0 | 2.2204460492503131e-16 / 3.6732009018965996e-16 | atol=1e-10, rtol=1e-8 | PASS | — |
| `f_stat` | `basic` | `2063.6084038161694` | `2063.608403816169` | `2063.60840381617` | 4.5474735088646412e-13 / 2.2036513809767078e-16 | 4.5474735088646412e-13 / 2.2036513809767073e-16 | 9.0949470177292824e-13 / 4.4073027619534147e-16 | atol=1e-9, rtol=1e-8 | PASS | — |
| `f_stat` | `no_constant` | `92.70629353764514` | `92.70629353764508` | `92.7062935376451` | 5.6843418860808015e-14 / 6.1315598641343238e-16 | 4.2632564145606011e-14 / 4.5986698981007431e-16 | 1.4210854715202004e-14 / 1.5328899660335817e-16 | atol=1e-9, rtol=1e-8 | PASS | — |
| `f_p_value` | `basic` | `5.790493786860803e-92` | `5.79049378686074e-92` | `5.79049378686107e-92` | 6.2677168844919979e-106 / 1.082414922664119e-14 | 2.6705924116531121e-105 / 4.6120288009164685e-14 | 3.2973641001023119e-105 / 5.6944437235805378e-14 | atol=1e-10, rtol=1e-8 | PASS | — |
| `f_p_value` | `no_constant` | `6.324940034154269e-25` | `6.324940034154432e-25` | `6.32494003415425e-25` | 1.6346718316122436e-38 / 2.5844858967596194e-14 | 1.8367099231598242e-39 / 2.9039167379322317e-15 | 1.818342823928226e-38 / 2.874877570552835e-14 | atol=1e-10, rtol=1e-8 | PASS | — |
| `ss_residual` | `basic` | `17.25765313422786` | `17.257653134227862` | `17.2576531342279` | 3.5527136788005009e-15 / 2.0586308295617831e-16 | 4.2632564145606011e-14 / 2.4703569954741341e-15 | 3.907985046680551e-14 / 2.2644939125179564e-15 | atol=1e-10, rtol=1e-8 | PASS | — |
| `ss_residual` | `no_constant` | `355.1395772823539` | `355.13957728235397` | `355.139577282354` | 5.6843418860808015e-14 / 1.6005937523435923e-16 | 1.1368683772161603e-13 / 3.2011875046871841e-16 | 5.6843418860808015e-14 / 1.6005937523435921e-16 | atol=1e-10, rtol=1e-8 | PASS | — |
| `ss_model` | `basic` | `608.7698809903771` | `608.7698809903771` | `608.769880990377` | 0 / 0 | 1.1368683772161603e-13 / 1.8674845992161213e-16 | 1.1368683772161603e-13 / 1.8674845992161213e-16 | atol=1e-10, rtol=1e-8 | PASS | — |
| `ss_model` | `no_constant` | `558.0283711588663` | `558.0283711588661` | `558.028371158866` | 2.2737367544323206e-13 / 4.0745898809955051e-16 | 3.4106051316484809e-13 / 6.1118848214932574e-16 | 1.1368683772161603e-13 / 2.0372949404977533e-16 | atol=1e-10, rtol=1e-8 | PASS | — |
| `ss_total` | `basic` | `626.027534124605` | `626.027534124605` | `626.027534124605` | 0 / 0 | 0 / 0 | 0 / 0 | atol=1e-10, rtol=1e-8 | PASS | — |
| `ss_total` | `no_constant` | `913.1679484412202` | `913.16794844122` | `913.16794844122` | 2.2737367544323206e-13 / 2.4899436717129574e-16 | 2.2737367544323206e-13 / 2.4899436717129574e-16 | 0 / 0 | atol=1e-10, rtol=1e-8 | PASS | — |
| `root_mse` | `basic` | `0.38405899135361266` | `0.3840589913536127` | `0.384058991353613` | 5.5511151231257827e-17 / 1.445380852446892e-16 | 3.3306690738754696e-16 / 8.672285114681346e-16 | 2.7755575615628914e-16 / 7.226904262234455e-16 | atol=1e-10, rtol=1e-8 | PASS | — |
| `root_mse` | `no_constant` | `1.7348364287725284` | `1.7348364287725284` | `1.73483642877253` | 0 / 0 | 1.5543122344752192e-15 / 8.9594166268168612e-16 | 1.5543122344752192e-15 / 8.9594166268168612e-16 | atol=1e-10, rtol=1e-8 | PASS | — |

## 审计解释与 mismatch 分类

代码审计状态为 PASS：冻结 oracle 未导入 Tabra，manifest/cases/data hash guard 全部通过，17 个 release-critical 指标全部通过，失败数为 0；尾概率的 zero mismatch 检查也通过。`code_audit.md`、`code_findings.json` 的 SHA-256 分别为 `a37f184b28a2d797bbb11234eef4838d0ff9ee10cb5e67fe4c474c6ed030fdc5`、`95c3d7f06da8c2f7e42afb6ddc157f8600756f6e83e606a2ffe085012f15b303`。

理论审计状态为 PASS：formula map 覆盖已支持的 normalized release-critical 指标和公开报告量，robust 路径继续明确列为 unsupported/blocked。`theory_audit.md`、`theory_findings.json` 的 SHA-256 分别为 `3ba43cbd475116dc772f2a3322fbc229fb7225d7c6573f0b653f466ccac39159`、`889252aab14d05f5a24caf5e81d627dd4dc5b72cbe1dabb95b46837237817327`。

没有 `SAMPLE`、`PARAMETERIZATION`、`OBJECTIVE`、`VCE_DOF`、`NUMERICAL`、`CONVERGENCE` 或 `UNRESOLVED` 的数值 mismatch。阻断项全部分类如下：

| 分类 | 引擎 | 证据 | 影响 |
|---|---|---|---|
| `ENGINE_ENV` | Python、Stata | environment 未记录 `cases_sha256` 和 seed | 无法验证二者与冻结案例文件及 seed 的引擎级绑定 |
| `MISSING_OUTPUT` | Python、Stata | result/environment 均未记录逐案例 included-row hash 或 mask | 无法进行三引擎样本成员精确相等检查 |
| `MISSING_OUTPUT` | Python、Stata | 未显式记录 omitted-parameter 列表 | 参数名虽相等，但缺少技能要求的显式省略参数证据 |

本结论依赖理论审计与独立代码 oracle 来解释数值证据，没有以 Python/Stata/R 中任何两者的一致性作为真值来源。

## 修复清单

1. 保留本 run 的全部原始工件；不得回写 Python、Stata 或 R 输出。
2. 在新的 run ID 中，让 Python 和 Stata runner 对每个案例记录 `cases_sha256`、seed、规范化方式以及 `included_row_sha256`，并显式记录 omitted parameters。
3. 新 run 必须重新生成相互独立的 code/theory audit，重新运行三引擎，再按原冻结容差比较。
4. 本报告不改源码、不改容差、不作 gate decision，也不签发 attestation。
