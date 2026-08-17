# QuantPlan

QuantPlan 当前是一个面向模型量化工程师的**证据驱动量化规划 P0 原型**。目标是读取模型结构、工作负载、能力声明和现有 ModelSlim YAML，形成可审阅的风险证据、候选量化计划，最终在验证通过后生成最小 YAML 补丁。

QuantPlan 当前不是自动优化器，也不承诺找到全局最优配置。它首先回答：

- 哪些量化单元值得优先保护或进一步验证；
- 判断来自哪批数据、哪种量化候选和哪套软硬件环境；
- 哪些结论仍缺证据或受后端能力阻塞；
- 未来生成的 YAML 相对原文件具体改了什么。

## 当前能力边界

| 能力 | 状态 |
| --- | --- |
| 对角梯度加权 Risk 计算与 prompt 聚合 | 已实现 |
| 成本约束下的实验性 gain-per-cost 候选搜索 | 已实现；只产出待验证提案 |
| Risk 与留出单点真值的点估计排序评测 | 已实现 |
| ModelSlim YAML 静态字段策略、qconfig 形态与模块 pattern 预检 | 已实现 |
| 35B 模型前向、梯度采集与 fake-quant 注入 | 规划中 |
| 外部版本化 runtime capability manifest | 规划中 |
| 无损最小 YAML patcher | 规划中 |
| ModelSlim 执行、导出产物核验和目标硬件验证 | 规划中 |

当前代码不能生成“最优 YAML”，也不会把静态 YAML 预检通过解释为后端可执行。

## Risk 的含义

`Risk` 是一个用于相对排序和实验优先级分配的**证据信号**，不是预测的 `ΔNLL`、任务准确率下降或部署性能损失。

- 首版主 Risk 是对角梯度加权相对残差失真；它衡量局部残差扰动证据，不保留完整 Hessian 或跨坐标交互。
- Risk 高表示当前证据更支持“该单元在给定候选量化下需要优先验证或保护”。
- Risk 只在模型、数据、候选量化语义和分析版本相同的上下文中比较。
- Risk 与 `ΔNLL`、KL、任务指标的关系必须通过独立留出数据验证，不能在报告中直接换算。
- Risk 排序即使有效，也不等于相应组合在层间交互和真实后端上一定有效。

QuantPlan 输出“候选”“证据等级”和“适用条件”，不会把未经端到端验证的计划描述成“最优实践”。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

仓库目前提供两个诊断命令：

```bash
quantplan inspect-yaml examples/modelslim-practice.yaml \
  --modules examples/named_modules.txt
quantplan evaluate-ranking --scores examples/risk.json \
  --truth examples/truth.json --top-k 1
```

第一条命令只读审计 ModelSlim YAML，并用非零退出码阻断未知执行语义或未命中的
`include`/`exclude`。成功只代表内置静态策略通过，输出中的
`execution_readiness` 始终为 `not_assessed`。第二条命令比较 Risk 排序与留出单点干预
真值，只输出点估计指标，不替代完整 Go/No-Go 门禁。两个命令均不执行 35B 模型前向，
也不会自动生成或运行量化权重。

## 首个 P0 范围

首个可否证实验使用 Qwen3.5-35B-A3B，并严格限制为：

- text-only，输入上下文不超过 4K；
- 40 个语言模型 Block 为决策粒度；
- MoE router、路由 logits 和 top-k 路由路径保持高精度；
- 排除 Vision Encoder/视觉投影、多模态输入、MTP 和 C4；
- 截至审计提交 `376e2fad43bc999f9a49b48bdbc3eef06ffdd83a`，ModelSlim 存在该型号的官方 W8A8 权重生成实践，但它只覆盖部分
  self-attention/experts 路径，不能等同于“全模型 W8A8 已验证部署”；
- W4A8、W4A4 和 Block 级混合精度均视为实验能力，必须先通过能力探测。

P0 先执行 8-Block pilot，只有出现可测量信号且后端能力满足要求时，才扩展至全部 40 Blocks。完整实验设计见 [docs/p0-qwen35b.md](docs/p0-qwen35b.md)。

方法学评估保持 text-only；当前 ModelSlim 多模态适配器要求图文校准，因此后端
smoke test 与方法学实验分开记录，不能把适配器限制算作 Risk 失败。

## 能力优先

目标产品在生成计划前必须读取版本化 runtime capability manifest。能力状态只有：

- `SUPPORTED_VERIFIED`
- `SUPPORTED_EXPERIMENTAL`
- `KNOWN_UNSUPPORTED`
- `UNKNOWN`
- `OUT_OF_SCOPE`

当前实现只有一个内容寻址的内置保守 schema manifest，它不包含模型适配器、CANN、
推理引擎或硬件可执行性。未知能力不会被当作默认支持。原 YAML 中影响 `spec`、
processor、qconfig 或 saver 语义的未知字段默认 fail-closed。非执行性的顶层扩展和
metadata 可以作为 opaque 证据保留，但不会被解释或复制到新执行计划。

## YAML 修改原则（目标能力）

- 当前只读原 YAML、记录 checksum，不生成衍生 YAML，因此没有 round-trip 无损承诺。
- 未来 patcher 只修改 capability manifest 明确授权的字段路径。
- 执行语义未知字段阻断；非执行 unknown 仅作为 opaque evidence，不会猜测透传。
- 未来只输出最小 patch/diff，不重写整份 YAML。
- 注释、anchor、alias、排序和未知字段无损将作为 patcher 的验收门槛。
- 禁止静默删除字段、回退精度或改变 runner、dataset、save 等执行语义。

架构和故障语义见 [docs/architecture.md](docs/architecture.md)。

## 当前非目标

- 自动发布量化权重或写入最佳实践库；
- 对任务精度、延迟或吞吐作无条件保证；
- 将单模型实验外推为跨模型通用结论；
- 在 P0 中支持 Vision、MTP、C4、超长上下文或 expert 级位宽分配；
- 依赖 ModelSlim 的内部 `core`/`infra` API。
