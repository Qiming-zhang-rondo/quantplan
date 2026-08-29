from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

from quantplan.domain.models import PrecisionFormat
from quantplan.runtime import (
    PROXY_SEMANTICS,
    QWEN35_MOE_ADAPTER_ID,
    ModelBundle,
    ModelPlacement,
    Qwen35MoeModelSlimAdapter,
    TorchAnalysisError,
    TorchAnalysisRuntime,
    TorchDependencyError,
    load_causal_lm,
    load_jsonl_texts,
    plan_evaluation_to_dict,
    torch_analysis,
)


def test_jsonl_text_loader_keeps_ids_and_metadata(tmp_path) -> None:
    source = tmp_path / "samples.jsonl"
    source.write_text(
        '\n{"id":"a","text":"hello","domain":"en"}\n{"text":"world","score":2}\n',
        encoding="utf-8",
    )

    samples = load_jsonl_texts(source)

    assert [sample.sample_id for sample in samples] == ["a", "3"]
    assert [sample.text for sample in samples] == ["hello", "world"]
    assert samples[0].metadata == {"domain": "en"}
    assert samples[1].metadata == {"score": 2}


def test_jsonl_text_loader_rejects_duplicate_ids(tmp_path) -> None:
    source = tmp_path / "duplicates.jsonl"
    source.write_text(
        '{"id":"same","text":"one"}\n{"id":"same","text":"two"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate sample id"):
        load_jsonl_texts(source)


def test_missing_torch_has_actionable_error(monkeypatch) -> None:
    real_import = importlib.import_module

    def import_without_torch(name: str, package=None):
        if name == "torch":
            error = ModuleNotFoundError("No module named 'torch'")
            error.name = "torch"
            raise error
        return real_import(name, package)

    monkeypatch.setattr(torch_analysis.importlib, "import_module", import_without_torch)

    with pytest.raises(TorchDependencyError, match="PyTorch is optional"):
        torch_analysis.discover_transformer_blocks(object())


def test_custom_loader_entrypoint_does_not_require_transformers(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        float32="fp32",
    )
    monkeypatch.setattr(torch_analysis, "_require_torch", lambda: fake_torch)
    seen = {}

    class DummyModel:
        def eval(self):
            seen["evaluated"] = True

    tokenizer = object()

    def loader(**kwargs):
        seen.update(kwargs)
        return DummyModel(), tokenizer

    bundle = load_causal_lm(
        "custom/model",
        revision="abc",
        loader_entrypoint=loader,
    )

    assert bundle.model_id == "custom/model"
    assert bundle.revision == "abc"
    assert bundle.tokenizer is tokenizer
    assert seen["torch_dtype"] == "bf16"
    assert seen["evaluated"] is True


def test_auto_torch_dtype_is_preserved_for_loader_semantics() -> None:
    assert torch_analysis._resolve_torch_dtype(SimpleNamespace(), "auto") == "auto"


def test_hf_auto_loader_records_device_map_placement(monkeypatch) -> None:
    torch = _torch_or_skip()
    model, tokenizer = _toy_objects(torch)
    model.hf_device_map = {"": "cpu"}
    seen = {}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(_path, **kwargs):
            seen["tokenizer_kwargs"] = kwargs
            return tokenizer

    class AutoModel:
        @staticmethod
        def from_pretrained(_path, **kwargs):
            seen["model_kwargs"] = kwargs
            return model

    monkeypatch.setattr(
        torch_analysis,
        "_require_transformers",
        lambda: SimpleNamespace(
            AutoTokenizer=AutoTokenizer,
            AutoModelForCausalLM=AutoModel,
        ),
    )

    bundle = load_causal_lm("toy/model", device="auto")

    assert seen["model_kwargs"]["device_map"] == "auto"
    assert bundle.placement == ModelPlacement(
        strategy="hf_device_map",
        input_device="cpu",
        device_map=(("", "cpu"),),
    )


def test_explicit_device_rejects_separate_hf_device_map(monkeypatch) -> None:
    _torch_or_skip()
    monkeypatch.setattr(
        torch_analysis,
        "_require_transformers",
        lambda: SimpleNamespace(),
    )

    with pytest.raises(TorchAnalysisError, match="cannot be combined"):
        load_causal_lm(
            "toy/model",
            device="cpu",
            model_kwargs={"device_map": "auto"},
        )


def test_qwen_target_filters_keep_gate_proj_and_lock_router_gate() -> None:
    include = ("*",)
    exclude = torch_analysis._effective_exclude_patterns(())

    assert not torch_analysis._target_selected(
        "model.language_model.layers.0.mlp.gate", include, exclude
    )
    assert not torch_analysis._target_selected(
        "model.language_model.layers.0.mlp.gate.weight", include, exclude
    )
    assert torch_analysis._target_selected(
        "model.language_model.layers.0.mlp.gate_proj", include, exclude
    )

    qwen_exclude = torch_analysis._effective_exclude_patterns(
        (), adapter=Qwen35MoeModelSlimAdapter()
    )
    for unsafe_name in (
        "model.language_model.layers.0.linear_attn.in_proj",
        "model.language_model.layers.0.linear_attn.conv1d",
        "model.language_model.layers.0.mlp.shared_expert_gate",
        "model.language_model.layers.0.mlp.shared_expert_gate.weight",
    ):
        assert not torch_analysis._target_selected(unsafe_name, include, qwen_exclude)


def test_uniform_group_selection_spans_full_depth() -> None:
    assert torch_analysis._uniform_indexes(40, 8) == {0, 6, 11, 17, 22, 28, 33, 39}


def _torch_or_skip():
    return pytest.importorskip("torch", reason="optional PyTorch dependency is not installed")


def _toy_objects(torch, *, cast_to_bf16: bool = True):
    nn = torch.nn

    class ToyTokenizer:
        pad_token_id = 0

        def __call__(self, texts, *, return_tensors, padding, truncation, max_length=None):
            del return_tensors, padding
            rows = []
            for text in texts:
                tokens = [1 + (ord(character) % 30) for character in text]
                if truncation and max_length is not None:
                    tokens = tokens[:max_length]
                rows.append(tokens)
            width = max(len(row) for row in rows)
            input_ids = torch.zeros((len(rows), width), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for index, row in enumerate(rows):
                input_ids[index, : len(row)] = torch.tensor(row)
                attention_mask[index, : len(row)] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class ToyFusedExperts(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(2, hidden_size * 2, hidden_size))
            self.down_proj = nn.Parameter(torch.randn(2, hidden_size, hidden_size * 2))

        def forward(self, hidden_states, top_k_index=None, top_k_weights=None):
            del top_k_index, top_k_weights
            up = torch.nn.functional.linear(hidden_states, self.gate_up_proj[0])
            return torch.nn.functional.linear(torch.tanh(up), self.down_proj[0])

    class ToyMlp(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.gate = nn.Linear(hidden_size, 2, bias=False)  # Router: excluded.
            self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.experts = ToyFusedExperts(hidden_size)

        def forward(self, hidden_states):
            projected = self.gate_proj(hidden_states)
            return projected + self.experts(hidden_states=hidden_states)

    class ToyBlock(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.mlp = ToyMlp(hidden_size)
            self.output = nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, hidden_states):
            return hidden_states + self.output(torch.tanh(self.mlp(hidden_states)))

    class ToyLanguageModel(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.layers = nn.ModuleList([ToyBlock(hidden_size), ToyBlock(hidden_size)])

    class ToyBackbone(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.language_model = ToyLanguageModel(hidden_size)

    class ToyCausalLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen3_5_moe")
            hidden_size = 6
            self.embed = nn.Embedding(32, hidden_size)
            self.model = ToyBackbone(hidden_size)
            self.lm_head = nn.Linear(hidden_size, 32, bias=False)
            self.fail_on_intervention = False

        def get_input_embeddings(self):
            return self.embed

        def forward(self, *, input_ids, attention_mask):
            del attention_mask
            has_active_hook = any(module._forward_pre_hooks for module in self.modules())
            if self.fail_on_intervention and has_active_hook:
                self.fail_on_intervention = False
                raise RuntimeError("intentional candidate failure")
            hidden_states = self.embed(input_ids)
            for block in self.model.language_model.layers:
                hidden_states = block(hidden_states)
            return SimpleNamespace(logits=self.lm_head(hidden_states))

    torch.manual_seed(7)
    model = ToyCausalLM()
    if cast_to_bf16:
        model.to(dtype=torch.bfloat16)
    return model, ToyTokenizer()


def _state_snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_state_equal(torch, model, expected) -> None:
    actual = model.state_dict()
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        assert torch.equal(value, expected[name]), name


def test_teacher_forced_metrics_exclude_left_padding_transition() -> None:
    torch = _torch_or_skip()
    input_ids = torch.tensor([[0, 1, 2, 3]])
    attention_mask = torch.tensor([[0, 1, 1, 1]])
    logits = torch.zeros((1, 4, 5))

    nll_sum, token_count = torch_analysis._nll_sum_and_count(
        torch, logits, input_ids, attention_mask
    )

    assert token_count == 2
    assert nll_sum == pytest.approx(2 * torch.log(torch.tensor(5.0)).item())
    assert torch_analysis._kl_sum(torch, logits, logits, attention_mask) == pytest.approx(0.0)


def test_toy_runtime_discovers_fused_experts_excludes_router_and_restores() -> None:
    torch = _torch_or_skip()
    model, tokenizer = _toy_objects(torch)
    before = _state_snapshot(model)

    runtime = TorchAnalysisRuntime(
        model,
        tokenizer,
        ["hello", "quant plan"],
        batch_size=1,
        linear_exclude=(),
        max_groups=2,
    )
    progress = []
    result = runtime.run(
        (PrecisionFormat.W4A8, PrecisionFormat.BF16),
        progress_callback=lambda *event: progress.append(event),
    )

    assert len(result.groups) == 2
    assert result.provenance.context_id == runtime.context_id
    assert result.provenance.context_schema == "quantplan.torch-analysis-context/v2"
    assert result.provenance.placement_schema == "quantplan.model-placement/v1"
    assert result.provenance.placement_strategy == "single_device"
    assert result.provenance.model_devices == ("cpu",)
    assert result.provenance.target_devices == ("cpu",)
    assert len(result.provenance.placement_sha256) == 64
    assert result.provenance.target_weight_dtypes == ("torch.bfloat16",)
    assert result.provenance.analysis_mode == "exact"
    assert result.provenance.full_model_dataset_passes == 3
    assert result.provenance.backward_dataset_passes == 0
    assert len(result.provenance.target_inventory_sha256) == 64
    assert runtime.backend_compilation_ready is True
    assert runtime.backend_adapter == QWEN35_MOE_ADAPTER_ID
    assert any(".mlp.experts.0.gate_proj" in name for name in runtime.backend_module_inventory)
    assert len(result.measurements) == 4
    for group in result.groups:
        assert group.parameter_count > 0
        assert any(name.endswith(".mlp.experts") for name in group.module_names)
        assert any(pattern.endswith(".mlp.experts.*") for pattern in group.backend_patterns)
        assert any(name.endswith(".mlp.gate_proj") for name in group.module_names)
        assert not any(name.endswith(".mlp.gate") for name in group.module_names)
        assert not any(name.endswith(".gate_up_proj") for name in group.module_names)
        assert any("module_input_a8_only" in warning for warning in group.warnings)
    for measurement in result.measurements:
        assert measurement.token_count > 0
        assert measurement.parameter_count > 0
        assert measurement.kl_divergence is None
        assert PROXY_SEMANTICS in measurement.warnings
    json.loads(result.to_json())
    assert progress == [
        (
            stage,
            current,
            4,
            result.groups[(current - 1) // 2].group_id,
            ("w4a8", "bf16")[(current - 1) % 2],
        )
        for current in range(1, 5)
        for stage in ("measurement_start", "measurement_complete")
    ]
    _assert_state_equal(torch, model, before)
    assert all(not module._forward_pre_hooks for module in model.modules())


def test_fast_runtime_produces_finite_approximate_risk_and_restores() -> None:
    torch = _torch_or_skip()
    model, tokenizer = _toy_objects(torch)
    before = _state_snapshot(model)
    requires_grad_before = tuple(parameter.requires_grad for parameter in model.parameters())
    runtime = TorchAnalysisRuntime(
        model,
        tokenizer,
        ["hello", "quant plan"],
        batch_size=1,
        linear_exclude=(),
        max_groups=2,
    )

    result = runtime.run_fast((PrecisionFormat.W4A8, PrecisionFormat.W8A8))

    assert len(result.measurements) == 4
    assert result.provenance.analysis_mode == "fast"
    assert result.provenance.full_model_dataset_passes == 1
    assert result.provenance.backward_dataset_passes == 1
    assert result.provenance.local_block_replays == 8
    assert all(item.estimated_risk >= 0 for item in result.measurements)
    assert all(item.metric_id.endswith("/v1") for item in result.measurements)
    assert all(not hasattr(item, "delta_nll") for item in result.measurements)
    assert tuple(parameter.requires_grad for parameter in model.parameters()) == requires_grad_before
    _assert_state_equal(torch, model, before)
    assert all(not module._forward_pre_hooks for module in model.modules())
    assert all(not module._forward_hooks for module in model.modules())


def test_fast_runtime_failure_restores_hooks_parameters_and_grad_flags(monkeypatch) -> None:
    torch = _torch_or_skip()
    model, tokenizer = _toy_objects(torch)
    before = _state_snapshot(model)
    requires_grad_before = tuple(parameter.requires_grad for parameter in model.parameters())
    runtime = TorchAnalysisRuntime(model, tokenizer, ["hello"], max_groups=1)

    def fail_replay(*_args, **_kwargs):
        raise RuntimeError("intentional fast replay failure")

    monkeypatch.setattr(runtime, "_fast_local_block_replay", fail_replay)
    with pytest.raises(RuntimeError, match="intentional fast replay failure"):
        runtime.run_fast((PrecisionFormat.W4A8,))

    assert tuple(parameter.requires_grad for parameter in model.parameters()) == requires_grad_before
    _assert_state_equal(torch, model, before)
    assert all(not module._forward_pre_hooks for module in model.modules())
    assert all(not module._forward_hooks for module in model.modules())


def test_runtime_context_fingerprint_binds_experiment_inputs(monkeypatch) -> None:
    torch = _torch_or_skip()

    def build(**overrides):
        model, tokenizer = _toy_objects(torch)
        options = {
            "batch_size": 1,
            "max_length": 8,
            "max_groups": 2,
            "model_id": "toy/qwen",
            "model_revision": "revision-a",
        }
        options.update(overrides)
        return TorchAnalysisRuntime(model, tokenizer, ["abcdef", "ghijkl"], **options)

    reference = build()
    assert build().context_id == reference.context_id
    assert build(model_id="toy/other").context_id != reference.context_id
    assert build(model_revision="revision-b").context_id != reference.context_id
    assert build(max_length=6).context_id != reference.context_id
    assert build(batch_size=2).context_id != reference.context_id
    assert build(max_groups=1).context_id != reference.context_id

    model, tokenizer = _toy_objects(torch)
    changed_data = TorchAnalysisRuntime(
        model,
        tokenizer,
        ["different"],
        batch_size=1,
        max_length=8,
        max_groups=2,
        model_id="toy/qwen",
        model_revision="revision-a",
    )
    assert changed_data.context_id != reference.context_id

    reference.max_length = 6
    with pytest.raises(TorchAnalysisError, match="analysis settings changed"):
        reference.run((PrecisionFormat.BF16,))

    monkeypatch.setattr(
        torch_analysis,
        "PROXY_SEMANTICS",
        f"{PROXY_SEMANTICS}; test-revision",
    )
    assert build().context_id != reference.context_id


def test_runtime_rejects_non_bf16_target_anchor_but_ignores_excluded_weights() -> None:
    torch = _torch_or_skip()
    fp32_model, tokenizer = _toy_objects(torch, cast_to_bf16=False)

    with pytest.raises(TorchAnalysisError, match="BF16 protection requires"):
        TorchAnalysisRuntime(fp32_model, tokenizer, ["abc"], max_groups=2)

    bf16_model, tokenizer = _toy_objects(torch)
    bf16_model.model.language_model.layers[0].mlp.gate.float()
    runtime = TorchAnalysisRuntime(bf16_model, tokenizer, ["abc"], max_groups=2)

    assert runtime.target_weight_dtypes == ("torch.bfloat16",)

    bf16_model.model.language_model.layers[0].mlp.gate_proj.float()
    with pytest.raises(TorchAnalysisError, match="BF16 protection requires"):
        runtime.run((PrecisionFormat.W4A8,))


def test_plan_evaluation_cannot_override_runtime_context() -> None:
    torch = _torch_or_skip()
    model, tokenizer = _toy_objects(torch)
    runtime = TorchAnalysisRuntime(model, tokenizer, ["abc"], max_groups=2)

    with pytest.raises(ValueError, match="must equal the runtime experiment context"):
        runtime.evaluate_plan(
            {runtime.groups[0].group_id: PrecisionFormat.W4A8},
            context_id="unrelated-context",
        )


def test_progress_callback_cannot_mutate_experiment_context() -> None:
    torch = _torch_or_skip()
    model, tokenizer = _toy_objects(torch)
    runtime = TorchAnalysisRuntime(model, tokenizer, ["abcdef"], max_length=6, max_groups=1)

    def mutate_runtime(stage, *_event) -> None:
        if stage == "measurement_start":
            runtime.max_length = 3

    with pytest.raises(TorchAnalysisError, match="analysis settings changed"):
        runtime.run((PrecisionFormat.BF16,), progress_callback=mutate_runtime)


def test_kl_is_streamed_and_plan_failure_restores_every_target() -> None:
    torch = _torch_or_skip()
    model, tokenizer = _toy_objects(torch)
    runtime = TorchAnalysisRuntime(
        model,
        tokenizer,
        ["abcdef", "ghijkl"],
        batch_size=1,
        compute_kl=True,
        max_groups=2,
    )
    group_ids = [group.group_id for group in runtime.groups]
    before = _state_snapshot(model)

    result = runtime.run((PrecisionFormat.W8A8,))
    assert all(item.kl_divergence is not None for item in result.measurements)
    assert all(item.kl_divergence >= 0 for item in result.measurements)

    plan = runtime.evaluate_plan(
        {group_id: PrecisionFormat.W4A8 for group_id in group_ids},
        plan_id="toy-plan",
    )
    assert plan.plan_id == "toy-plan"
    assert plan.token_count > 0
    assert plan.kl_divergence is not None
    json.dumps(plan_evaluation_to_dict(plan))
    _assert_state_equal(torch, model, before)

    model.fail_on_intervention = True
    with pytest.raises(RuntimeError, match="intentional candidate failure"):
        runtime.evaluate_plan({group_ids[0]: PrecisionFormat.W4A8}, plan_id="failure")
    _assert_state_equal(torch, model, before)
    assert all(not module._forward_pre_hooks for module in model.modules())


def test_fused_intervention_rejects_missing_activation_and_restores_weights() -> None:
    torch = _torch_or_skip()

    class MissingActivationExpert(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(
                torch.randn(2, 4, 4, dtype=torch.bfloat16)
            )
            self.down_proj = torch.nn.Parameter(
                torch.randn(2, 4, 4, dtype=torch.bfloat16)
            )

        def forward(self, *, token_ids):
            return token_ids

    expert = MissingActivationExpert()
    before = _state_snapshot(expert)

    with torch_analysis.temporary_fused_expert_fake_quant(
        (expert,), PrecisionFormat.W4A8, torch_module=torch
    ), pytest.raises(TorchAnalysisError, match="weight-only intervention"):
        expert(token_ids=1)

    _assert_state_equal(torch, expert, before)


def test_uniform_max_groups_selection_records_original_coverage() -> None:
    torch = _torch_or_skip()
    model, tokenizer = _toy_objects(torch)

    runtime = TorchAnalysisRuntime(model, tokenizer, ["abc"], max_groups=1)

    assert len(runtime.groups) == 1
    assert runtime.discovered_group_count == 2
    assert runtime.full_group_coverage is False
    assert runtime.groups[0].group_id.endswith(".1")
    assert any("1 of 2" in warning for warning in runtime.groups[0].warnings)


def test_custom_loader_can_declare_pre_sharded_placement(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        float32="fp32",
    )
    monkeypatch.setattr(torch_analysis, "_require_torch", lambda: fake_torch)

    class DummyModel:
        def eval(self):
            return self

    model = DummyModel()
    tokenizer = object()

    bundle = load_causal_lm(
        "custom/model",
        device="auto",
        loader_entrypoint=lambda **_kwargs: {
            "model": model,
            "tokenizer": tokenizer,
            "placement": {
                "strategy": "pre_sharded",
                "input_device": "npu:0",
                "device_map": {"model.layers.0": "npu:0", "model.layers.1": "npu:1"},
            },
        },
    )

    assert isinstance(bundle, ModelBundle)
    assert bundle.placement == ModelPlacement(
        strategy="pre_sharded",
        input_device="npu:0",
        device_map=(
            ("model.layers.0", "npu:0"),
            ("model.layers.1", "npu:1"),
        ),
    )


def test_pre_sharded_contract_rejects_cpu_offload() -> None:
    torch = _torch_or_skip()
    model, tokenizer = _toy_objects(torch)

    with pytest.raises(TorchAnalysisError, match="CPU/disk/meta offload"):
        TorchAnalysisRuntime(
            model,
            tokenizer,
            ["abc"],
            max_groups=2,
            placement=ModelPlacement(
                strategy="pre_sharded",
                input_device="cpu",
                device_map=(("", "cpu"),),
            ),
        )


def test_meta_targets_are_blocked_before_analysis() -> None:
    torch = _torch_or_skip()
    model, tokenizer = _toy_objects(torch)
    model.to("meta")

    with pytest.raises(TorchAnalysisError, match="meta/disk offload"):
        TorchAnalysisRuntime(model, tokenizer, ["abc"], max_groups=2)


def test_model_placement_normalizes_device_map_order() -> None:
    placement = ModelPlacement(
        strategy="pre_sharded",
        input_device="npu:0",
        device_map={"model.layers.1": "npu:1", "model.layers.0": "npu:0"},
    )

    assert placement.device_map == (
        ("model.layers.0", "npu:0"),
        ("model.layers.1", "npu:1"),
    )
    assert torch_analysis._requested_auto_backend("npu:auto") == "npu"
    assert torch_analysis._requested_auto_backend("auto") is None


def test_resident_device_map_rejects_incorrect_parameter_device() -> None:
    class Parameter:
        device = "npu:1"

    class Model:
        @staticmethod
        def named_modules():
            return (("", object()), ("model.layers.0", object()))

        @staticmethod
        def named_parameters():
            return (("model.layers.0.weight", Parameter()),)

    with pytest.raises(TorchAnalysisError, match="disagrees"):
        torch_analysis._validate_resident_device_map(
            Model(),
            (("model.layers.0", "npu:0"),),
            ("npu:1",),
        )


def test_typed_auto_device_rejects_a_different_loaded_backend() -> None:
    class Parameter:
        def __init__(self, device: str) -> None:
            self.device = device

    class Model:
        parameters_by_name = (
            ("model.embed_tokens.weight", Parameter("cuda:0")),
            ("model.layers.0.weight", Parameter("cuda:1")),
        )

        @classmethod
        def parameters(cls):
            return (parameter for _name, parameter in cls.parameters_by_name)

        @classmethod
        def named_parameters(cls):
            return iter(cls.parameters_by_name)

        @staticmethod
        def named_modules():
            return (
                ("", object()),
                ("model", object()),
                ("model.embed_tokens", object()),
                ("model.layers", object()),
                ("model.layers.0", object()),
            )

        @staticmethod
        def get_input_embeddings():
            return SimpleNamespace(weight=Parameter("cuda:0"))

    fake_torch = SimpleNamespace(
        distributed=SimpleNamespace(is_initialized=lambda: False),
    )
    target_inventory = (
        {
            "targets": (
                {
                    "name": "model.layers.0.weight",
                    "device": "cuda:1",
                    "contiguous": True,
                },
            )
        },
    )

    with pytest.raises(TorchAnalysisError, match="requires.*npu"):
        torch_analysis._resolve_runtime_placement(
            fake_torch,
            Model(),
            target_inventory,
            declared=ModelPlacement(
                strategy="pre_sharded",
                input_device="cuda:0",
                device_map={
                    "model.embed_tokens": "cuda:0",
                    "model.layers.0": "cuda:1",
                },
            ),
            requested_device="npu:auto",
        )
