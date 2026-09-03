from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from quantplan.runtime.architecture import (
    ARCHITECTURE_ADAPTER_ENTRY_POINT,
    GENERIC_ADAPTER_ID,
    GLM5_MOE_DSA_ADAPTER_ID,
    QWEN35_MOE_ADAPTER_ID,
    ArchitectureAdapterRegistry,
    ArchitectureAdapterResolutionError,
    ArchitectureTargetMappingError,
    FusedTargetMapping,
    GenericLinearAdapter,
    GlmMoeDsaModelSlimAdapter,
    Qwen35MoeModelSlimAdapter,
)


@dataclass(frozen=True)
class _AlwaysAdapter:
    adapter_id: str
    fallback: bool = False
    preferred_block_paths: tuple[str, ...] = ("layers",)
    nonremovable_exclude_patterns: tuple[str, ...] = ("*router*",)
    allow_block_path_fallback: bool = False

    def matches(self, model, config=None) -> bool:
        del model, config
        return True

    def map_fused_target(self, module_name, module, direct_parameters):
        del module_name, module, direct_parameters


class _EntryPoint:
    def __init__(self, name: str, value) -> None:
        self.name = name
        self.value = value

    def load(self):
        return self.value

    def __str__(self) -> str:
        return f"{ARCHITECTURE_ADAPTER_ENTRY_POINT}:{self.name}"


def _model(model_type: str = "unknown") -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(model_type=model_type))


@dataclass(frozen=True)
class _BrokenMatchAdapter(_AlwaysAdapter):
    def matches(self, model, config=None) -> bool:
        del model, config
        raise RuntimeError("broken matcher")


@dataclass(frozen=True)
class _NonBooleanMatchAdapter(_AlwaysAdapter):
    def matches(self, model, config=None):
        del model, config
        return "yes"


def test_registry_prefers_exact_qwen_adapter_and_keeps_generic_fallback() -> None:
    registry = ArchitectureAdapterRegistry(
        (GenericLinearAdapter(), Qwen35MoeModelSlimAdapter())
    )

    assert registry.resolve(_model()).adapter_id == GENERIC_ADAPTER_ID
    assert registry.resolve(_model("qwen3_5_moe")).adapter_id == QWEN35_MOE_ADAPTER_ID
    assert registry.resolve(_model("qwen3_5_moe_text")).adapter_id == QWEN35_MOE_ADAPTER_ID


def test_registry_resolves_glm_moe_dsa_adapter() -> None:
    registry = ArchitectureAdapterRegistry(
        (GlmMoeDsaModelSlimAdapter(), GenericLinearAdapter())
    )

    assert registry.resolve(_model("glm_moe_dsa")).adapter_id == GLM5_MOE_DSA_ADAPTER_ID


def test_glm_adapter_declares_fusion_domains_and_protected_dsa_paths() -> None:
    adapter = GlmMoeDsaModelSlimAdapter()

    assert adapter.decision_grouping == "glm5_fused_subgroups/v1"
    assert adapter.recipe_verified_hardware == ("a2", "a3")
    assert "*weights_proj*" in adapter.nonremovable_exclude_patterns
    assert "*kv_b_proj*" in adapter.nonremovable_exclude_patterns
    assert "routed_expert:atomic_gate_up_down" in adapter.fusion_contracts


def test_glm_adapter_maps_packed_experts_without_splitting_gate_up_down() -> None:
    adapter = GlmMoeDsaModelSlimAdapter()
    mapping = adapter.map_fused_target(
        "model.layers.0.mlp.experts",
        object(),
        {
            "gate_up_proj": SimpleNamespace(ndim=3, shape=(4, 16, 8)),
            "down_proj": SimpleNamespace(ndim=3, shape=(4, 8, 8)),
        },
    )

    assert mapping is not None
    assert mapping.runtime_parameter_names == ("gate_up_proj", "down_proj")
    assert mapping.backend_pattern == "model.layers.0.mlp.experts.*"
    assert len(mapping.backend_module_names) == 12


def test_qwen_adapter_locks_architecture_specific_unsafe_targets() -> None:
    excludes = Qwen35MoeModelSlimAdapter().nonremovable_exclude_patterns

    assert "*conv1d*" in excludes
    assert "*linear_attn*" in excludes
    assert "*.shared_expert_gate" in excludes
    assert "*.shared_expert_gate.weight" in excludes


def test_qwen_adapter_maps_fused_runtime_weights_to_audited_backend_names() -> None:
    adapter = Qwen35MoeModelSlimAdapter()
    parameters = {
        "gate_up_proj": SimpleNamespace(ndim=3, shape=(2, 12, 6)),
        "down_proj": SimpleNamespace(ndim=3, shape=(2, 6, 12)),
    }

    mapping = adapter.map_fused_target("model.layers.0.mlp.experts", object(), parameters)

    assert mapping is not None
    assert mapping.runtime_parameter_names == ("gate_up_proj", "down_proj")
    assert mapping.backend_pattern == "model.layers.0.mlp.experts.*"
    assert mapping.backend_module_names == (
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.0.up_proj",
        "model.layers.0.mlp.experts.0.down_proj",
        "model.layers.0.mlp.experts.1.gate_proj",
        "model.layers.0.mlp.experts.1.up_proj",
        "model.layers.0.mlp.experts.1.down_proj",
    )


def test_qwen_adapter_rejects_incomplete_or_wrong_rank_fused_layout() -> None:
    adapter = Qwen35MoeModelSlimAdapter()

    with pytest.raises(ArchitectureTargetMappingError, match="missing"):
        adapter.map_fused_target(
            "experts",
            object(),
            {"gate_up_proj": SimpleNamespace(ndim=3, shape=(2, 12, 6))},
        )
    with pytest.raises(ArchitectureTargetMappingError, match="non_3d"):
        adapter.map_fused_target(
            "experts",
            object(),
            {
                "gate_up_proj": SimpleNamespace(ndim=2, shape=(12, 6)),
                "down_proj": SimpleNamespace(ndim=3, shape=(2, 6, 12)),
            },
        )


def test_registry_fails_closed_for_unknown_and_ambiguous_matches() -> None:
    with pytest.raises(ArchitectureAdapterResolutionError, match="no registered"):
        ArchitectureAdapterRegistry().resolve(_model())

    ambiguous = ArchitectureAdapterRegistry(
        (_AlwaysAdapter("test.one/v1"), _AlwaysAdapter("test.two/v1"))
    )
    with pytest.raises(ArchitectureAdapterResolutionError, match="ambiguous"):
        ambiguous.resolve(_model())


def test_entry_point_factory_is_discovered_and_duplicate_id_fails_closed() -> None:
    adapter = _AlwaysAdapter("external.example/v1")
    registry = ArchitectureAdapterRegistry((GenericLinearAdapter(),))

    registry.discover_entry_points((_EntryPoint("external", lambda: adapter),))

    assert registry.resolve(_model()).adapter_id == adapter.adapter_id

    duplicate = ArchitectureAdapterRegistry((GenericLinearAdapter(),))
    with pytest.raises(ArchitectureAdapterResolutionError, match="duplicates"):
        duplicate.discover_entry_points(
            (_EntryPoint("duplicate", GenericLinearAdapter()),)
        )


def test_registry_wraps_broken_or_non_boolean_plugin_matchers() -> None:
    broken = ArchitectureAdapterRegistry((_BrokenMatchAdapter("broken/v1"),))
    with pytest.raises(ArchitectureAdapterResolutionError, match="broken matcher"):
        broken.resolve(_model())
    with pytest.raises(ArchitectureAdapterResolutionError, match="broken matcher"):
        broken.resolve(_model(), adapter_id="broken/v1")

    non_boolean = ArchitectureAdapterRegistry(
        (_NonBooleanMatchAdapter("bad-result/v1"),)
    )
    with pytest.raises(ArchitectureAdapterResolutionError, match="non-boolean"):
        non_boolean.resolve(_model())


def test_fused_mapping_requires_exact_names_covered_by_its_pattern() -> None:
    with pytest.raises(ValueError, match="cover every"):
        FusedTargetMapping(
            runtime_parameter_names=("weight",),
            backend_module_names=("model.layers.0.proj",),
            backend_pattern="model.layers.1.*",
        )
    with pytest.raises(ValueError, match="exact names"):
        FusedTargetMapping(
            runtime_parameter_names=("weight",),
            backend_module_names=("model.layers.*.proj",),
            backend_pattern="model.layers.*",
        )
