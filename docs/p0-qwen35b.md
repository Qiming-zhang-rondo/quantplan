# P0：Qwen3.5-35B-A3B Risk 可行性实验

> 状态说明：本文保留的是早期 W4A4 方法学预注册草案，不是当前 CLI 的可执行合同。当前实现范围以 README 为准：W4A8/W8A8/BF16，默认 8 Block 仅做 pilot 且不产出 YAML，完整覆盖后才允许编译。W4A4/C4 暂时阻断。

## 1. 决策问题

本实验只回答：

> 当前 Risk 信号是否足够稳定，并且能否在相同 A4 激活量化/计算覆盖范围下，比随机和简单经验规则产生更好的候选计划，从而值得进入下一阶段研发？

P0 不是量化排行榜，不负责证明全局最优，也不把 Risk 解释为 `ΔNLL` 或准确率损失预测。

## 2. 已知事实与前置风险

Qwen 官方模型卡将 Qwen3.5-35B-A3B 描述为 35B 总参数、每 token 约 3B 激活、40 层，采用 Gated DeltaNet、Gated Attention 与 256 experts MoE 的混合布局。模型还包含 Vision Encoder 和 MTP。[Qwen3.5-35B-A3B 模型卡](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)

截至审计提交 `376e2fad43bc999f9a49b48bdbc3eef06ffdd83a`，ModelSlim 对该型号提供官方 W8A8 权重生成实践；W4A8 与 W4A4 未在该型号支持矩阵中标记为已验证。该 W8A8 YAML 实际只覆盖部分 self-attention 和 routed experts，并排除部分投影、vision、linear-attention 路径与 MTP，因此不能简称为“全模型 W8A8 已验证部署”。[固定提交中的 Qwen3.5 W8A8 YAML](https://gitcode.com/Ascend/msmodelslim/blob/376e2fad43bc999f9a49b48bdbc3eef06ffdd83a/lab_practice/qwen3_5_moe/qwen3_5_moe_w8a8.yaml)

同一审计提交中的 ModelSlim Qwen3.5 多模态适配器还要求图文校准数据。P0-A 的 Risk 方法学实验仍以
text-only 工作负载为目标；P0-B 若复用官方适配器，则必须单独记录图文校准要求及其
分布差异。不能用“方法学 text-only”掩盖适配器约束，也不能把适配器阻塞记作算法失败。[固定提交中的适配器源码](https://gitcode.com/Ascend/msmodelslim/blob/376e2fad43bc999f9a49b48bdbc3eef06ffdd83a/msmodelslim/model/qwen3_5_moe/model_adapter.py)

因此：

- W4A8、W4A4 和 Block 级混配是 P0 的实验能力，不是现成产品能力；
- 后端无法执行时，结论是 `BACKEND_CAPABILITY_BLOCKED`；
- 参考 fake-quant 结果不能代替 ModelSlim 导出和目标硬件验证；
- W4A8/W4A4 混配只验证激活精度分配，不能外推为通用权重位宽分配。

## 3. 冻结范围

### 3.1 纳入范围

- 模型：固定 revision 的 Qwen3.5-35B-A3B；
- 模态：text-only；
- 输入上下文：不超过 4K tokens；
- 生成：非 thinking、固定解码参数；Risk 和单点真值优先采用 teacher forcing；
- 决策单元：40 个语言 Block；
- 候选：W4A8 与 W4A4；
- 目标：对 Block 进行 A8/A4 分配；
- 环境：固定 ModelSlim、CANN、驱动、推理引擎和硬件版本。

### 3.2 锁定高精度

- MoE router 权重；
- router logits；
- top-k expert 选择与归一化路径；
- 不在 40-Block 候选清单中的加载、归一化和输出边界。

一个 Block 内所有 routed experts 与 shared expert 作为同一决策组。不得按本批样本的路由频率只量化部分专家。

如果后端不能表达 router 高精度锁定，P0-B 不成立，不能静默放宽该条件。

### 3.3 排除范围

- Vision Encoder、视觉投影和任何多模态输入；
- MTP；
- C4；
- 超过 4K 的长上下文；
- expert 级或单 Linear 级分配；
- 自动写入 ModelSlim 最佳实践库；
- 跨模型通用性结论。

## 4. 预注册假设

- **H1，稳定性**：同分布独立样本组产生的 Risk 排序具有足够一致性。
- **H2，排序有效性**：Risk 与独立条件单点干预真值呈正相关。
- **H3，计划增益**：在相同 A4 成本覆盖下，Risk 计划优于随机和简单 heuristic。
- **H4，部署价值**：通过质量门禁的 Risk 计划在真实后端产生可测量性能收益。

H1/H2 通过不代表 H3/H4 自动通过。

## 5. Capability Gate

在计算 Risk 前生成 capability manifest，并完成：

1. BF16 和官方部分量化 W8A8 权重能生成、加载并完成固定 smoke prompts；
2. 统一 W4A8 能生成、加载并推理；
3. 统一 W4A4 能生成、加载并推理；
4. 单个 Block 可以在 W4A8/W4A4 间切换，其余配置不变；
5. router 高精度锁定可以被表达和验证；
6. 运行日志或 profiler 能确认没有静默 W8/FP 回退；
7. 参考 fake-quant 与真实后端在 pilot Block 上的影响方向一致。

W4A8/W4A4 只有全部通过后才能标记为 `SUPPORTED_EXPERIMENTAL`。失败后可以继续 P0-A 方法学分析，但不得给出产品 Go。

### 5.1 YAML Gate

当前原型已经执行的输入静态策略测试：

- 未知执行字段触发 fail-closed；非执行性顶层扩展只作为 opaque 证据保留；
- 未知 `qconfig` 子字段触发 fail-closed；
- 未注册 processor/saver、非法 qconfig 形态和值、未命中 pattern 被阻断；
- 原始文件不被修改并记录 SHA256；
- 静态预检通过只表示 `static_policy_passed`，执行能力仍为 `not_assessed`。

后续 patcher 的验收门槛：

- 已知且支持的字段可以形成最小补丁；
- 自定义 runner、processor 或 saver 保留并锁定；
- 输入出现 C4 时返回 `KNOWN_UNSUPPORTED`；
- anchor、alias、注释和未知字段不会因生成补丁而丢失；
- 原 YAML checksum 不一致时拒绝应用补丁。

## 6. 数据分组

使用 1 个校准组和 8 个互不重叠的评估组，共 768 条文本：

| 分组 | 数量 | 用途 |
| --- | ---: | --- |
| `C0` | 128 | 只用于量化参数校准 |
| `S1-S4` | 4 × 64 | 分别计算 Risk，评估排序稳定性 |
| `T1-T2` | 2 × 64 | 条件单点干预真值 |
| `H1-H2` | 2 × 128 | 只用于最终计划质量验证 |

每组采用相同的分层比例：

- 中文知识/指令；
- 英文知识/指令；
- 数学/代码；
- 1K-4K 较长上下文。

所有 bootstrap 以 prompt 为单位，不能把同一 prompt 的 token 当作独立样本。不得根据 `T` 或 `H` 调整 Risk 公式；不得根据 `H` 选择门槛或计划。

## 7. 基线

所有配置使用相同的校准组、量化方法、group size、对称性、runner、导出格式和运行环境。除 precision mask 外，不允许同时改变其他变量。

| 基线 | 目的 |
| --- | --- |
| BF16 | 质量和性能参考 |
| ModelSlim 官方部分 W8A8 | 当前权重生成锚点；部署与覆盖矩阵仍需 smoke |
| 全 W4A8 | 保守端点，实验能力 |
| 全 W4A4 | 激进端点，实验能力 |
| Random | 相同 A4 成本覆盖下的随机分配，至少 20 seeds |
| Depth heuristic | 首尾/深度保护经验规则 |
| Structure heuristic | Attention 与 DeltaNet/MoE 的结构保护规则 |
| Cheap metric | residual relative MSE、post-norm relative MSE、kurtosis 等便宜指标排序 |
| Risk | 当前待验证算法 |
| Marginal oracle | 根据独立单点真值排序的参考上界 |

Marginal oracle 不是组合空间的全局最优，因为它仍然假设单点效应近似可用于组合。

## 8. 真值和评估指标

### 8.1 条件单点真值

主真值以全 W4A8 为上下文，只将一个 Block 降为 W4A4：

\[
T_i = \operatorname{NLL}(P_{\mathrm{W4A8}, i\rightarrow \mathrm{A4}})
      - \operatorname{NLL}(P_{\mathrm{W4A8}})
\]

- 主指标：每 prompt 的 `ΔNLL/token`；
- 次指标：teacher logits KL/JS；
- 最终指标：留出任务正确率/指令遵循和真实硬件性能。

`T_i` 是验证 Risk 排序的外部真值，但 Risk 自身不得存储或展示为 `T_i` 的预测值。

### 8.2 成本口径

W4A8 和 W4A4 都使用 4-bit weights，因此不能按模型文件大小配平。候选计划使用：

- 工作负载下的 activation traffic；
- routing-weighted active FLOPs；
- 实测 kernel latency/throughput；

定义等成本的 A4 coverage。不得简单使用“多少个 Block 是 A4”。

## 9. 两阶段实验

### 9.1 Phase A：8-Block Pilot

运行前预注册 8 个 Block：四个深度区间各选择一个 Gated DeltaNet Block 和一个 Gated Attention Block，并使用 `C0` 的 BF16 路由覆盖作第三个分层条件。不得使用 Risk、`T` 或 `H` 结果选取 pilot Blocks。

执行：

1. 在 `S1-S4` 分别计算 8 个 Block 的 Risk；
2. 在 `T1-T2` 对 8 个 Block 分别执行 W4A8→W4A4 单点干预；
3. 比较 Risk 稳定性、真值测量噪声和 fake-quant/真实后端方向；
4. 检查 W4A4 是否出现全局崩溃、乱码或路由变化。

进入 40-Block 阶段的条件：

- Capability Gate 通过，或明确只进入 P0-A 方法学模式；
- scoring groups 的 Risk 排序中位 Spearman `ρ >= 0.50`；
- 单点真值的 block 间方差高于同一条件重复运行噪声；
- Risk 至少能区分 pilot 中的高、低风险组，方向不存在系统性反转；
- 预计完整实验未超过硬成本上限。

8 个样本点不足以宣称算法有效。Pilot 只用于停止无信号或不可执行的实验。

### 9.2 Phase B：40 Blocks

1. 在全部 40 Blocks 上计算四份独立 Risk 排序；
2. 对全部 40 Blocks 执行 W4A8→W4A4 条件单点干预；
3. 对预注册的 12 Blocks 执行 W4A4→W4A8 反向干预，检查上下文依赖；
4. 对 8 对 Block 做高/高、高/低、低/低和跨结构交互实验；
5. 在参考执行中比较 25%、50%、75% A4 coverage 的 Risk、oracle、heuristic 与 random；
6. 真实 ModelSlim 导出只在 50% 主比较点执行完整对照，减少 checkpoint 数量；
7. 在 `H1-H2` 上执行最终质量门禁，并在目标硬件测量性能。

持久运行配置最多为：BF16、W8A8、全 W4A8、全 W4A4、Risk、oracle、最强 heuristic 和预注册 random，共 8 个。

## 10. Go / No-Go 门槛

门槛必须在读取 `T` 和 `H` 结果前冻结。

### 10.1 搜索空间 Headroom

在 50% A4 coverage 下，marginal oracle 的 `ΔNLL` 损失必须比 random 中位数至少低 20%。

若未达到，结论为 `SEARCH_SPACE_NO_HEADROOM`：当前模型、粒度和 W4A8/W4A4 搜索空间不值得继续做位宽分配，不应通过修改 Risk 公式强行挽救。

### 10.2 Risk 排序

- 四个 scoring groups 之间的中位 Spearman `ρ >= 0.70`；
- Risk 与独立 `T_i` 的 Spearman `ρ >= 0.60`；
- prompt bootstrap 的 95% 置信下界 `ρ > 0.30`；
- 真值危险四分位，即 Top-10 高风险 Blocks 的 recall `>= 0.75`；
- `NDCG@10 >= 0.75`。

### 10.3 计划增益

在 50% A4 coverage 下：

- Risk 计划的 `ΔNLL` 损失比 random 中位数至少低 20%；
- 比预注册最强 heuristic 至少低 10%；
- 获得 marginal oracle 改善量的至少 70%；
- paired bootstrap 95% CI 不跨 0；
- 相对全 W4A8，最终任务总体下降不超过 1 个百分点，任一预注册切片不超过 2 个百分点。

### 10.4 硬件价值

至少一个通过质量门禁的 Risk 计划必须：

- 相对全 W4A8 获得至少 5% 的 TPOT 或吞吐改善；
- 没有未声明的 W8/FP 回退；
- 没有因混合精度边界导致收益被转换开销抵消。

若 Risk 排序和计划质量通过但实机无收益，结论为 `HARDWARE_VALUE_FAILED`，不能进入产品化。

## 11. 结论状态

| 状态 | 结论 |
| --- | --- |
| `GO` | 能力、排序、计划、质量、硬件价值和成本全部通过 |
| `METHOD_ONLY` | Risk 方法学通过，但 W4 后端能力尚未成立；不得产品化 |
| `BACKEND_CAPABILITY_BLOCKED` | W4A8/W4A4、router 锁定或 Block 混配无法执行 |
| `SEARCH_SPACE_NO_HEADROOM` | oracle 也不优于随机，当前分配问题无价值 |
| `RISK_VALIDATION_FAILED` | Risk 不稳定或与独立真值相关性不足 |
| `PLAN_VALIDATION_FAILED` | 排序有效但组合计划未通过留出质量门禁 |
| `HARDWARE_VALUE_FAILED` | 质量可接受但无实际性能收益 |

## 12. 成本上限

以下是待硬件 SKU、拓扑、推理引擎和版本冻结后复核的暂定预算假设。实验模板在这些
字段补齐前保持 `runnable: false`。暂按每节点 8 NPU 计：

- 计划预算：48 node-hours，即 384 NPU-hours；
- 硬上限：72 node-hours，即 576 NPU-hours；
- 持久 checkpoint：最多 8 个；
- 存储预算：不超过约 250GB；
- 目标墙钟：一周内形成 Go/No-Go 报告。

预算建议：

- capability smoke：10%；
- Risk 和单点真值：45%；
- 候选参考验证：20%；
- 最终 ModelSlim 导出与实机测试：20%；
- 重试储备：5%。

成本控制：模型常驻、Block 临时替换、不为 40 个单点干预保存 checkpoint、复用校准统计与量化权重、主要实验使用 teacher forcing，并在 8-Block pilot 无信号时提前停止。

## 13. P0 能支持的表述

若实验通过，只能表述为：

> 在指定 Qwen3.5-35B-A3B revision、text-only ≤4K 工作负载、固定软硬件环境和 Block 级 W4A8/W4A4 激活精度搜索空间内，当前 Risk 能对独立边际量化风险进行有效排序，并改善同成本候选计划。

通过所有门禁后，只能称为“在明确列出的已评估候选集合中，观测结果最佳且通过全部门禁的候选”，并同时记录 evaluated set、成本预算、停止条件和版本指纹；不能简称为“最优 YAML”。

不能表述为：

- Risk 能预测准确率或 `ΔNLL`；
- 已找到最优位宽分配；
- 结论适用于 Vision、MTP、C4、超长上下文或 expert 级量化；
- 结论适用于其他模型、硬件或推理引擎；
- 已验证通用权重位宽分配。

“框架可扩展”与“算法已通用”必须分开。只有在 Dense Transformer、普通 Attention-MoE、hybrid/recurrent-MoE 和非 Qwen 模型上完成独立复现后，才可以逐步提升算法通用性表述。
