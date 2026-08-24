# 大模型放置与分片边界

QuantPlan 的算法需要对同一 BF16 参考模型反复执行候选干预。模型如何进入设备属于执行后端能力，不属于 Risk、搜索或 YAML 编译逻辑。

## 已实现、待内网验证的常驻路径

| 路径 | 配置 | 能力边界 |
| --- | --- | --- |
| 单设备常驻 | `device: npu:0` | 整模和临时 fake-quant workspace 都必须放入一张卡 |
| HF 模块常驻分片 | `device: npu:auto` | 使用 `device_map="auto"`；仅接受单进程、完整模块常驻多卡 |
| 内部自定义分片 | `runtime.loader: module:function` | 多设备返回值必须包含 `ModelPlacement(strategy="pre_sharded")` |

HF 自动放置还要求目标环境中的 `accelerate>=0.28`。`doctor` 只验证依赖与设备可见性，
显存容量和 fake-quant 峰值保持 `not assessed`，必须由实际加载与首轮 smoke 确认。

所有路径在分析前都核对 input embedding device、模型参数设备、候选 target 设备和 BF16 dtype。以下可识别状态默认阻断：

- `meta` 参数、CPU/disk offload；
- 已初始化且 `world_size > 1` 的 distributed 参数执行；
- 多设备模型没有显式 placement 合同；
- 配置输入设备与 embedding 实际设备不一致；
- selected target 不是完整、常驻的 BF16 权重。

这是因为 QuantPlan 的 fake-quant 会临时原地修改并恢复一个模块的权重。TP、FSDP、ZeRO、EP 与单进程参数切片同样不在合同内：它们需要全局 scale、跨 rank inventory 和同步恢复；offload 则需要与 materialize hook 建立严格顺序。通用反射无法识别任意自定义切片包装，因此 custom loader 必须如实声明，当前校验只对模型参数与 target 权重负责，不宣称覆盖全部 buffer 或私有运行时状态。

## 为什么不能只把流式 runner 当成 loader

[`accuracy_bench`](https://github.com/Qiming-zhang-rondo/accuracy_bench) 已有成熟的大模型低峰值机制：

1. 在 `meta` device 创建模型骨架；
2. 依据 safetensors index 只读取当前 shard；
3. 逐层或逐 expert 物化到 NPU；
4. 将 hidden state 传给下一个 shard；
5. 完成后把权重卸载回 `meta` 并回收显存。

审计基线固定为 `accuracy_bench@95dc32f`。可复用的底层能力主要位于：

- `accuracy_checker.model_loader.create_model_skeleton`；
- `build_weight_index` / `ShardWeightReader`；
- `load_layer_weights_indexed` / `unload_layers_to_meta`；
- `ShardedBlockComparator` 中 Qwen、GLM、Kimi 的逐层状态和 streaming expert 处理。

但当前 `ShardedBlockComparator` 的输入是 reference checkpoint 与 quantized checkpoint，输出是逐层对齐结果；QuantPlan 需要的是“同一 BF16 checkpoint + 指定决策组/精度干预 + 完整 teacher-forced NLL”。流式模型在任意时刻含有大量 `meta` 参数，因此不能返回一个普通 `ModelBundle` 交给 `model(**batch)`。

## 最薄的复用协议

后续的 `accuracy_bench` adapter 应实现 workflow 已有的 `AnalysisRuntimeLike`，而不是实现新的模型 loader。建议只增加以下稳定接口：

```text
discover() -> groups, module inventory, backend patterns
measure(group_id, precision, samples) -> baseline NLL, candidate NLL, token count
evaluate(assignments, samples) -> combined-plan NLL, token count
placement_provenance() -> backend commit, devices, shard policy, checkpoint hash
```

runner 内部继续复用 `accuracy_bench` 的 skeleton、indexed reader、模型结构探测和分片前向。候选干预必须发生在当前 Block 已物化之后、forward 之前，并在卸载之前恢复。最终 norm/lm_head 物化后流式累计 NLL，不保留数据集规模的 logits。

接入必须满足以下门槛：

- target 名称与 QuantPlan `ArchitectureAdapter` 的决策组严格对应；
- router、Vision、MTP 和跨层状态保持锁定；
- Qwen3.5、GLM、DeepSeek、Kimi 仍按代际分别声明 capability；
- grouped/streaming expert 路径必须提供真实的候选量化回调，不能依赖普通 `nn.Linear` hook；
- backend commit、模型 revision、数据 hash、shard 大小和设备列表进入 context ID；
- resident runtime 与 streaming runtime 在同一小模型上通过 NLL/恢复一致性测试。

在这些条件通过前，QuantPlan 只把 accuracy_bench 标记为“选定的复用后端”，不宣称流式分片已经可运行。
