# QuantPlan

QuantPlan 是面向模型量化工程师的**证据驱动混合精度规划 P0 工具**。它读取模型、text-only 工作负载和已有 ModelSlim YAML，对每个 Block 做候选精度干预，复测完整组合方案，只有方案通过冻结的质量门槛后才生成派生 YAML。

它当前不会承诺数学意义上的“全局最优”。准确表述是：

> 本次明确实测候选集合中，成本最低且通过 ΔNLL 门槛的方案。

## 当前能力

| 能力 | 状态 |
| --- | --- |
| `quantplan init / doctor / run` 一键工作流 | 已实现 |
| 整模型 teacher-forced 单组 W4A8/W8A8 fake-quant 前向 | 已实现；分析代理，不是 ModelSlim-exact |
| 普通 `nn.Linear`、Qwen3.5 和 GLM-5.3（三维 fused MoE） | 已实现结构发现与分析代理；fused 权重完整，A8 仅覆盖模块入口，router/DSA 索引路径默认排除 |
| 成本约束搜索与完整组合方案 ΔNLL 复测 | 已实现 |
| ModelSlim YAML 严格预检、派生 YAML、diff 和 provenance | 已实现 |
| 对角梯度加权 Risk 与排序评测研究接口 | 已实现；不是默认决策指标 |
| GLM-5.3 `glm_moe_dsa` 细粒度分组（attention/dense/shared/routed） | 已实现；A3/A5 kernel 兼容性仍需目标框架实测 |
| Qwen3.5-35B-A3B、GLM-5.3 A3/A5 内网 NPU 实测 | 待目标环境验证 |
| ModelSlim 权重生成、产物核验和目标硬件 benchmark | 规划中 |

静态 YAML 预检通过不代表后端可执行。所有派生结果都保留 `execution_readiness: not_assessed`，直到锁定版本的 ModelSlim、CANN、推理引擎和目标硬件完成验证。

P0 的保护精度语义是真实 BF16，不是“当前任意浮点模型”的别名。配置只接受 `bf16`/`bfloat16` 或 `auto`；使用 `auto` 时，运行时仍会逐个核验实际量化 target 的权重 dtype，发现 FP16/FP32 会阻断。

## 默认指标

默认运行不依赖梯度。它先计算一次 BF16 teacher-forced NLL；随后对每个决策组 `g` 和候选精度 `q`，只临时 fake-quant 该组并完成整个模型前向：

```text
observed_damage(g, q) = max(0, candidate_NLL(g, q) - BF16_NLL)
```

这是候选相关的整模型观测值，因此该 Block 的残差连接、RMSNorm 以及所有后续层传播都会影响结果。它仍然只是当前数据和 fake-quant 语义下的边际证据，不能直接代表任务准确率或真实 NPU kernel。

Qwen fused expert 的 P0 代理会对打包的 `gate_up_proj`/`down_proj` 权重全部做 W4/W8 QDQ，但普通 module hook 只能量化 expert 模块入口激活，暂时捕获不到内部 `down_proj` 的中间激活 A8。该限制会写入每个相关决策组的 warning；因此首轮结果用于排序与相关性验证，不应冒充 ModelSlim-exact W4A8/W8A8 结论。后续复用 `accuracy_bench` 流式 expert 路径时，会在每次内部矩阵乘之前补齐干预点。

规划器使用同一组 W4A8/W8A8/BF16 的实测差做 gain-per-cost 搜索，再将完整混合方案重新前向。组合方案超过 `search.max_delta_nll` 时，不生成派生 YAML。KL 可以通过 `analysis.compute_kl: true` 开启；默认关闭，以减少 35B 分析中的全词表计算。

仓库仍保留对角梯度加权 Risk，供研究和对照实验使用，但它不是一键工作流的默认判定依据。

## 内网快速开始

请在已经安装目标版本 `torch`、`torch_npu` 和 `transformers` 的内网环境中安装 QuantPlan；使用 `npu:auto` 时还需要与该组合兼容的 `accelerate>=0.28`。项目故意不自动安装这些包，避免破坏 ModelSlim 的严格版本组合。

```bash
git clone https://github.com/Qiming-zhang-rondo/quantplan.git
cd quantplan
python -m pip install -e .
```

开发与自测依赖：

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

### 1. 准备数据

输入为 JSONL，每行至少包含一个 `text`：

```json
{"id":"sample-001","text":"请解释量化误差如何传播。"}
{"id":"sample-002","text":"Write a short Python function for matrix multiplication."}
```

### 2. 一次性生成配置

只需要五项业务参数：模型目录、JSONL 数据、原始 ModelSlim YAML、输出目录和设备。

```bash
quantplan init quantplan-run.yaml \
  --model-path /models/Qwen3.5-35B-A3B \
  --data-path /data/calibration.jsonl \
  --original-yaml /configs/modelslim.yaml \
  --output-dir /results/quantplan-pilot \
  --device npu:auto
```

对 35B 模型，`npu:0` 表示整模单卡加载，只有模型与 fake-quant 临时空间都能放入该卡时才应使用。`npu:auto` 会请求 Hugging Face/Accelerate 的 `device_map="auto"` 并先确认 NPU runtime 可用；加载后 QuantPlan 还会核对实际参数与候选 target 的 placement。CPU/disk/meta offload和已检测到的多进程 distributed 执行会阻断。合同同时禁止 TP/FSDP/EP 和单进程参数切片，但任意自定义实现无法只靠通用反射完全识别，custom loader 必须如实声明。

不提供参数时，`quantplan init` 会生成带 `REPLACE_WITH_*` 占位符的模板；`doctor` 和 `run` 都会阻断未填写的模板。完整配置示例见 `examples/quantplan-run.yaml`。

针对 GLM-5.3 可在配置中显式设置 `runtime.hardware_profile: a3` 或 `a5`，分别保存两套
上下文绑定的分析结果；默认 `unknown`。该字段只绑定目标，不会把公开 recipe 或代理测量
自动升级为 PrologV3 kernel 支持。

### 3. 预检

```bash
quantplan doctor --config quantplan-run.yaml
```

`doctor` 不加载模型权重，只检查路径、依赖、设备、候选范围和 YAML 基础结构。具体 NPU 配置会实际导入 `torch_npu`，并调用 `torch.npu.is_available()` 和 `device_count()` 核对请求的卡号；`device: npu:auto` 还要求 Accelerate 可用，真实设备图在模型加载后校验。显存容量与 fake-quant 峰值会明确保持 `warning/not assessed`，不能由 doctor 返回码推断“模型一定放得下”。无法完成基础探测会给出明确的阻断原因。返回码 `0` 表示可以尝试开始分析，`2` 表示存在阻断项。

### 4. 运行

```bash
quantplan run --config quantplan-run.yaml
```

默认是精确模式。40 个 Block、2 个候选精度且 KL 关闭时，提案阶段需要
`1 + 40 × 2 = 81` 次整模型数据集前向，随后最多精确复测 8 个组合：

```bash
quantplan run --config quantplan-run.yaml --mode exact
```

快速模式使用一次完整数据集前向和一次反向获得各 Block 输出对
teacher-forced NLL 的梯度，再分别对 W4A8/W8A8 做 Block 局部无状态重放：

```bash
quantplan run --config quantplan-run.yaml --mode fast
```

对某个 Block 和候选精度，Fast V1 计算每个有效 token 的一阶影响
`z = sum(输出梯度 × 候选输出差)`，排序风险为
`abs(mean(z)) + rms(z)`。它同时考虑候选量化误差和该误差对最终 NLL 的影响，
但仍是局部一阶近似，不是整模型 `delta_nll`。Fast 最多精确复测成本最低的 3 个
完整组合；只有复测通过才会生成 YAML。

Fast 将模型参数冻结，因此不产生权重梯度，并且用无状态量化副本局部重放，
不改写原模型权重。它仍需保留一次反向所需的激活，显存可能明显高于纯前向；
不支持反向、Block 输出不是 `[batch, sequence, hidden]`、或一个决策组跨多个
Block root 时会明确阻断，不会悄悄退回低可信指标。对 GLM、DeepSeek、Kimi 等
新适配器，必须先在小规模 exact 对照上验证 Fast 排序相关性。

运行顺序：

1. 加载模型并发现 Block、普通 Linear 和 fused expert target；
2. 锁定 Vision、MTP、router 等默认排除路径；
3. 用完整 `named_modules` 审计原始 ModelSlim YAML；
4. 对每个 Block 的 W4A8/W8A8 候选执行整模型前向；
5. 在成本预算下生成候选组合；
6. 对完整组合重新测量 ΔNLL；
7. 仅在质量门槛通过后编译派生 YAML。

已有产物默认拒绝覆盖。确认要重跑时：

```bash
quantplan run --config quantplan-run.yaml --overwrite
```

成功返回码为 `0`；前置检查阻断为 `2`；没有组合方案通过质量门槛为 `3`，此时不会产生 `derived-modelslim.yaml`。

## 保守默认值

- 候选：W4A8、W8A8，BF16 作为保护精度；
- 8 个均匀覆盖深度的 Block；
- 16 条文本、最长 512 tokens、batch size 1；
- 25% 高精度预算；
- exact 最多复测 8 个组合候选，fast 最多复测 3 个；
- teacher-forced ΔNLL/token 上限 0.02；
- KL 关闭；
- Vision、Visual、MTP、router 和 `.mlp.gate` 默认排除；Qwen3.5 还锁定
  `conv1d`、`linear_attn` 和 `shared_expert_gate`；
- 未识别 expert target 默认 fail-closed。

这些参数只用于第一次内网 smoke。默认只分析 8 个 Block，因此只输出证据和候选诊断，**不会生成派生 YAML**。运行正常后，必须将 `analysis.max_groups` 调整到不少于实际发现的 Block 数（Qwen3.5-35B-A3B 预期为 40）才允许编译完整计划；同时可按预算提高 `max_samples` 和 `max_length`。`search.costs` 是归一化规划代理，不是硬件延迟或显存实测。

W4A4/C4 当前被阻断。先证明 W4A8/W8A8/BF16 链路和模型覆盖正确，再扩展候选空间。

## 模型插件

默认 loader 使用 Hugging Face `AutoModelForCausalLM`。如果模型需要特殊分片、ModelSlim adapter 或内部加载逻辑，把配置改为：

```yaml
runtime:
  loader: your_package.your_module:load_model
```

自定义函数接收模型路径、revision、device、dtype 和 loader 选项，可返回 `(model, tokenizer)`、`ModelBundle`，或包含 `model/tokenizer` 的 mapping。QuantPlan 核心不依赖 ModelSlim 的内部 `core`/`infra` Python API。

多卡自定义 loader 不能再只返回 `(model, tokenizer)`，还要声明 placement：

```python
from quantplan.runtime import ModelBundle, ModelPlacement

return ModelBundle(
    model=model,
    tokenizer=tokenizer,
    model_id=model_path,
    placement=ModelPlacement(
        strategy="pre_sharded",
        input_device="npu:0",
        device_map=tuple(model_device_map.items()),
    ),
)
```

首期合同只接受单进程、同一加速后端、完整参数常驻不同卡的模块分片。运行时会记录模型参数设备、目标权重设备和 placement 哈希，并检查 embedding 输入设备、BF16、meta/offload 与 target 稳定性；多卡 logits 的 labels/mask 会在计算 NLL/KL 时搬到 logits 所在末卡，不搬运全量 logits。自定义 placement 返回后以它为准，配置中的具体设备只作为传给 loader 的放置请求；`npu:auto` 这类带后端的请求仍会强制核对 NPU/CUDA/XPU 前缀。未返回 placement 的旧 loader 则必须让实际输入设备与配置一致。

显存仍不足时，不会在 QuantPlan 内重写一套层流式加载器。已选择复用 [`accuracy_bench`](https://github.com/Qiming-zhang-rondo/accuracy_bench) 的 `meta skeleton + indexed safetensors + layers_per_shard + streaming expert` 执行机制，当前审计基线为提交 `95dc32f`。它现有的 `ShardedBlockComparator` 是双 checkpoint 的精度对齐器，并非可直接返回 QuantPlan 单组 W4A8/W8A8 ΔNLL 的 loader；因此要通过独立 runner adapter 接入，不能把它伪装成 `ModelBundle`。详细边界见 [`docs/large-model-placement.md`](docs/large-model-placement.md)。

如果模型确实由 Transformers 的 `AutoModelForImageTextToText` 加载，但本次只验证 text-only 前向，可以显式开启受控 fallback：

```yaml
runtime:
  loader: huggingface
  allow_image_text_fallback: true
```

默认值为 `false`，`AutoModelForCausalLM` 失败时不会自动猜测架构。开启该选项只改变模型类 fallback，不会自动构造图片输入；如果 text-only 调用仍需要 processor 或额外参数，应使用自定义 loader。

自动 Block discovery 不适配新架构时，可填写 `modules.block_patterns`。每条 pattern 必须命中，重叠和空匹配都会阻断。`modules.include/exclude` 控制组内量化 target；fused expert 必须同时覆盖 `gate_up_proj` 和 `down_proj`，不能只选一半。

模型结构通过版本化 `ArchitectureAdapter` 插件隔离。Qwen3.5 MoE 与 GLM-5.3
`glm_moe_dsa` 已有独立适配器；GLM 会把一个 Block 拆成 attention、dense MLP、shared
expert、routed expert 四类决策域，使 W4A8/W8A8 混配可以表达。`indexer`、`weights_proj`、
`compressor`、`kv_b_proj`、`wk`、router 等路径仍保持保护精度。GLM 的分组只回答“可以测量
哪些域”，不等于 PrologV3 kernel 已命中：官方公开的 GLM-5 W4A8 recipe 目前标注 A2/A3，
A5 需要内网 capability manifest 或真实框架测试后才能生成可部署 YAML。DeepSeek、Kimi
仍需按代际提供独立插件。外部包可通过 `quantplan.architecture_adapters` entry point
接入，未知或歧义 custom/fused op 会阻断。完整合同见 [`docs/architecture-adapters.md`](docs/architecture-adapters.md)。

## 输出产物

- `model-inventory.json`：Block、量化 target、参数量和覆盖警告；
- `risk-evidence.jsonl`：exact 为每个 Block/精度的整模型 ΔNLL；fast 为明确标记的
  梯度加权局部风险，不含伪造的 `delta_nll` 字段；
- `candidate-plan.json`：成本受限候选及完整 assignments；
- `validation-report.json`：完整组合方案复测与冻结门槛；
- `derived-modelslim.yaml`：通过门槛后才出现；
- `plan.diff`：原 YAML 到派生 YAML 的统一 diff；
- `yaml-audit.json`：字段和 pattern 静态审计；
- `provenance.json`：数据、runtime、量化模板和 YAML 哈希；
- `run-manifest.json`：运行状态和失败原因。

派生 YAML 不会被 QuantPlan 自动执行或发布。请在内网使用锁定版本的 ModelSlim 完成权重生成，然后核对导出描述、逐模块实际精度、kernel fallback、任务质量和目标硬件性能。

## YAML 安全规则

- 原 YAML 永不原地修改，并记录 SHA256；
- 只替换唯一且无歧义的 `spec.process` 中 `linear_quant` 槽；
- W4A8/W8A8 模板从原 YAML 提取；缺失候选只允许在保持 scope、method、symmetric 和 ext 不变时派生，并记录来源；
- runner、dataset、save 和其他 processor 不由 QuantPlan 猜测修改；
- 执行语义 unknown 默认阻断；metadata 等非执行扩展保持原样并只作为 opaque evidence，不参与决策；
- 源 YAML 和派生 YAML 使用同一 guard；
- 无安全插入位置、多处 linear slot、危险 target 名、W4A4 或模板不完整时阻断；
- W4A8/W8A8 使用互斥 include；BF16 target 不进入任何量化 include。

## 独立诊断命令

```bash
quantplan inspect-yaml examples/modelslim-practice.yaml \
  --modules examples/named_modules.txt

quantplan evaluate-ranking \
  --scores examples/risk.json \
  --truth examples/truth.json \
  --top-k 1
```

第一条只读检查 YAML；第二条比较 proposal score 与留出单点干预真值。它们不会加载 35B 模型。

## P0 范围

首个实验针对 Qwen3.5-35B-A3B，范围固定为 text-only、最多 4K、Block 粒度，并排除 Vision、MTP、C4 和 expert 级独立分配。MoE router 权重、router logits、top-k 选择与归一化路径必须保持高精度。

截至审计的 ModelSlim 提交 `376e2fad43bc999f9a49b48bdbc3eef06ffdd83a`，该型号只有部分 W8A8 权重生成实践；W4A8、W4A4 和 Block 混配都不能预先视为已验证能力。完整实验设计见 `docs/p0-qwen35b.md`。

## 非目标

- 宣称找到全局最优 YAML；
- 自动发布量化权重或写入最佳实践库；
- 对任务精度、吞吐或延迟作无条件保证；
- 把单模型结论外推到 GLM、Kimi、DeepSeek 或其他架构；
- 在 P0 支持 Vision、MTP、C4、超长上下文或 expert 级位宽分配。
