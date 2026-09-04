# 推理框架插件与高速算子选择

QuantPlan 不用模型名称直接推断高速算子。最终算子选择由以下上下文共同决定：

```text
model architecture + inference framework + framework version/commit
+ hardware profile + quant config/capability snapshot
```

因此 GLM-5.3 只由 Architecture Adapter 识别为 `glm_moe_dsa`；它是否走 PrologV3，
必须由 vLLM-Ascend 对应的 Inference Framework Adapter 和目标环境 capability snapshot
确认。DeepSeek MLA、Qwen GDN 或 SGLang 的路径使用各自插件，不能复用“模型名到 kernel”
的静态映射。

模型侧只产生中性的 `mla_preprocess` quantization domain，不把它命名成 PrologV3。
框架解析结果再决定这个 domain 的 runtime owner 是 PrologV3、MLAPO、NATIVE 或其他算子。

输入 ModelSlim YAML 是已知可执行配置的安全锚点。框架插件约束的是“从锚点切换到新精度”
是否合法，而不是重新证明锚点本身。capability 未知时，QuantPlan 禁止改变对应融合域并保留
原 YAML 精度；capability 明确允许 W4A8/W8A8 时才开放相应候选。原 YAML 与 capability
声明冲突时仍保留原精度，同时记录冲突，不能据此推断其他精度同样可用。

## 插件合同

入口组为 `quantplan.inference_framework_adapters`。插件实现：

- `adapter_id`：版本化插件身份；
- `fallback`：是否只是 analysis-only fallback；
- `matches(FrameworkRequest)`：检查框架、模型结构、硬件和 capability 是否匹配；
- `resolve(FrameworkRequest)`：返回 `OperatorResolution`。

解析结果包含 framework/version/commit、operator、状态、量化域、fallback 链和原因。
多个非 fallback 插件同时匹配时 fail-closed；外部插件返回的 adapter ID 与注册对象不一致
时也会阻断。

内置 `ManifestInferenceFrameworkAdapter` 从版本化 capability snapshot 得到“候选算子”，
但返回 `selection_basis=capability_manifest`、`runtime_selected=false`，不会把静态声明冒充
真实 dispatch。内置 `AnalysisOnlyFrameworkAdapter` 只允许采集分析证据，不宣称任何
kernel。框架专用实现可以
在自己的包中调用 vLLM-Ascend、SGLang 或自研框架的真实 resolver，并通过相同 entry point
安装，不需要修改 QuantPlan 核心。

## 更新规则

当 PrologV3、MLA 或 GDN 增加新的量化组合时：

1. 框架内部 resolver 不变、只是 contract 扩展：重新生成 capability snapshot；
2. 私有 API 或 dispatch 逻辑变化：发布新版本 Framework Adapter；
3. 运行时重新绑定 framework version、commit 和 manifest hash；
4. 旧证据上下文不能直接编译新环境的 YAML，必须重新分析。

配置示例：

```yaml
runtime:
  inference_framework: vllm_ascend
  hardware_profile: a5
  backend_capability_manifest: /configs/vllm-ascend-a5-prologv3.yaml
```

`inference_framework: auto` 会优先使用 capability snapshot 声明的 backend；没有 snapshot
时退回 analysis-only。只有框架插件调用真实 resolver 后返回 `status=verified` 且
`runtime_selected=true`，才算完成高速算子选择验证；否则只能编译保持受约束融合域与原
YAML 一致的计划，任何改变该域的计划都会阻断。
