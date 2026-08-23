# Architecture Adapter：模型家族边界与插件合同

> 本文描述模型架构插件边界，不代表某个模型已经完成量化或硬件验证。当前只有
> Qwen3.5 MoE 的一段 ModelSlim 后端映射经过源码审计；GLM、DeepSeek、Kimi 尚未接入，
> 也不应被表述为“已支持”。

## 为什么核心算法不绑定模型家族

QuantPlan 把“模型结构知识”和“精度决策”分开。Qwen、GLM、DeepSeek 或 Kimi 的差异
应停在 Architecture Adapter 边界，不进入指标公式、搜索器或 Canonical plan：

| 层 | 只依赖 | 不应依赖 |
| --- | --- | --- |
| 指标与验证 | 决策组、候选精度、固定 `context_id` 下的整模型前向结果 | 模型类名、Block 路径、expert 参数布局 |
| 搜索 | `GroupInventory`、`InterventionMeasurement`、成本和质量门槛 | Qwen/GLM 等家族分支、ModelSlim 模块名 |
| Canonical plan | `QuantPlan` 中 `group_id -> PrecisionFormat` 的映射及上下文、成本、告警 | 后端 YAML pattern、转换后的模块拓扑 |
| Architecture Adapter | 模型匹配、Block/target 语义、锁定区域、后端 target 映射 | 指标阈值、候选排序和计划选择 |

因此，同一套指标与搜索逻辑可以复用于不同家族，但“框架可复用”不等于“算法已在该
家族验证”。每个模型仍需独立的数据代表性、计划质量和目标硬件验证。

运行链路如下：

```text
loaded model
  -> resolve ArchitectureAdapter
  -> discover GroupInventory + locked/excluded targets + verified backend mappings
  -> measure candidate interventions
  -> build backend-neutral QuantPlan
  -> guarded YAML compiler consumes assignments and mappings separately
```

后端映射在 inventory discovery 时提前计算，以便未知 custom/fused target 尽早阻断；
它不会混入 `QuantPlan.assignments`。这样更换 ModelSlim 版本或换用其他后端时，不需要
重写指标和搜索，只需重新发现 inventory、核验 capability 并编译后端产物。

## `ArchitectureAdapter` 必须声明什么

当前协议位于 `quantplan.runtime.architecture`。一个插件至少承担以下合同：

| 合同 | 当前 API | 安全要求 |
| --- | --- | --- |
| 身份与版本 | 版本化 `adapter_id` | ID 必须绑定插件合同版本；若映射依赖特定 ModelSlim/模型 revision，也必须进入匹配条件或 provenance |
| 模型匹配 | `matches(model, config=None)`、`fallback` | 多个非 fallback 插件同时匹配时阻断；显式指定的插件不匹配时阻断 |
| Block discovery | `preferred_block_paths`、`allow_block_path_fallback` | 只声明已知 Block 根；禁止用“找到一个列表就算成功”掩盖架构不确定性 |
| 高精度锁定 | `nonremovable_exclude_patterns` | router、Vision、MTP 等安全排除不能被用户 include 抵消；锁定原因必须进入 inventory/provenance |
| runtime target 到后端映射 | `map_fused_target(...) -> FusedTargetMapping | None` | `runtime_parameter_names`、`backend_module_names`、`backend_pattern` 和 `activation_kwarg` 必须完整且可审计 |
| 能力状态 | runtime 输出的 `backend_adapter`、`backend_compilation_ready`、`backend_compilation_blockers` | `ready` 只表示后端 target 映射完整，不代表 ModelSlim 可执行、精度合格或硬件已验证 |

注册、枚举和解析使用：

- `register_architecture_adapter`；
- `unregister_architecture_adapter`；
- `registered_architecture_adapters`；
- `resolve_architecture_adapter`。

`ArchitectureAdapterRegistry` 对精确匹配优先，并拒绝重复 ID、歧义匹配和歧义 fallback。
内置 `GenericLinearAdapter` 只负责普通、可明确识别的 `nn.Linear` fallback；它没有权限
猜测 custom/fused op 的参数语义或 ModelSlim 转换后名称。

## 能力状态不是一个笼统的“支持”标签

架构插件只回答结构和映射问题。候选精度、ModelSlim recipe、运行环境和硬件是否可用，
仍应使用统一能力状态分别记录：

- `SUPPORTED_VERIFIED`：完全匹配的版本与场景已经验证；
- `SUPPORTED_EXPERIMENTAL`：链路可运行，但证据尚不完整；
- `KNOWN_UNSUPPORTED`：已确认无法表达或执行；
- `UNKNOWN`：没有足够声明或无法安全探测；
- `OUT_OF_SCOPE`：当前实验主动排除。

`backend_compilation_ready=True` 不能自动提升为 `SUPPORTED_VERIFIED`。它只允许进入下一道
YAML 编译检查；ModelSlim 加载、权重生成、kernel 命中、质量和硬件性能仍是独立门禁。

## 内置插件的能力边界

### Qwen3.5 MoE

`Qwen35MoeModelSlimAdapter` 是首个经过审计的架构插件。它的 `adapter_id` 固定到已审计的
ModelSlim Qwen3.5 MoE adapter 提交，并声明：

- 已知 Block 根；
- 不可移除的 router/Vision/MTP 等排除项；
- fused `gate_up_proj`/`down_proj` 到 ModelSlim 转换后 expert Linear 的映射；
- fused target 不完整、rank 不符或 expert 数不一致时阻断。

这只说明该版本映射有源码依据，不说明任意 Qwen3.5 revision、W4A8/W8A8 混配、NPU
权重生成或部署性能已经验证。相关结论仍以独立实验和 capability manifest 为准。

### 其他家族

GLM、DeepSeek、Kimi 必须按“家族 + 代际/变体 + 后端版本”分别实现和审计插件。不能用
一个按名称模糊匹配的通用 MoE 插件覆盖它们，因为不同代际可能改变：

- Block 根与 Attention/SSM/recurrent 混合方式；
- dense、routed、shared expert 的耦合边界；
- router 和归一化路径；
- fused 参数形状、激活参数名和 ModelSlim 转换后拓扑。

在对应插件、测试和版本证据落地前，这些家族的状态是 `UNKNOWN`，不是“已支持”，也不
应因为普通 Linear discovery 能运行就生成家族级支持结论。

## 未知 custom/fused op 必须 fail-closed

当模块暴露 custom/fused 参数，而已选插件无法给出完整 `FusedTargetMapping` 时，核心
不得把父模块名直接写入 YAML，也不得猜测 `*` pattern。安全行为是：

1. discovery 抛出带 adapter ID 和 target 路径的 `DiscoveryError`；
2. 不进入指标、搜索和 YAML 编译；
3. 如未来增加“只分析、不编译”的宽松路径，必须将
   `backend_compilation_ready` 置为 `false`、记录 blocker，并禁止生成派生 YAML。

同样，映射缺少部分参数、匹配为空或重叠、后端模块清单无法覆盖、插件版本不匹配时都
必须阻断。显式配置只能选择一个已注册且匹配的插件，不能绕过这些检查。

## 外部插件接入

外部插件有两种受控入口：

1. 安装包通过 `ARCHITECTURE_ADAPTER_ENTRY_POINT` 指定的 Python entry-point group
   `quantplan.architecture_adapters` 暴露插件；
2. 受信任的集成代码显式调用 `register_architecture_adapter`，并可用
   `resolve_architecture_adapter(..., adapter_id=...)` 固定选择。

entry point 名和 `adapter_id` 都应版本化；升级模型代际、后端映射合同或不兼容语义时应
发布新 ID，而不是静默改变旧插件。entry point 可以导出 adapter 实例或零参工厂；
`registered_architecture_adapters` 和 `resolve_architecture_adapter` 首次调用时延迟发现。
重复 ID、对象不满足 `ArchitectureAdapter`、导入失败或多个精确插件同时匹配均应
fail-closed。

架构插件是可执行 Python 代码，只应从受信任且版本锁定的包加载。原始 ModelSlim YAML
和模型仓库内容不得声明任意导入路径；所有插件身份、来源版本、匹配结果和 blocker 都应
进入 provenance。
