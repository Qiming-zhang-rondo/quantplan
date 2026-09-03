from dataclasses import dataclass
from pathlib import Path

import pytest

from quantplan.runtime.capability import load_capability_manifest
from quantplan.runtime.framework import (
    ANALYSIS_ONLY_FRAMEWORK_ADAPTER_ID,
    MANIFEST_FRAMEWORK_ADAPTER_ID,
    AnalysisOnlyFrameworkAdapter,
    FrameworkAdapterResolutionError,
    FrameworkRequest,
    InferenceFrameworkAdapterRegistry,
    ManifestInferenceFrameworkAdapter,
    OperatorResolution,
)


def _snapshot(tmp_path: Path, *, status: str = "verified"):
    path = tmp_path / "capability.yaml"
    path.write_text(
        f"""\
schema_version: quantplan.capability/v1
backend: {{name: vllm_ascend, version: 0.27.1, git_commit: abc, operator: prolog_v3}}
hardware: {{profile: a5}}
models: {{model_types: [glm_moe_dsa], architecture_adapter_ids: []}}
domains:
  - id: mla_preprocess
    status: {status}
    legal_modes: [native]
    allowed_precisions: [bf16]
    owners: ['*q_proj*']
    contract: {{}}
    fallback: [mlapo, native]
""",
        encoding="utf-8",
    )
    return load_capability_manifest(path, hardware_profile="a5")


def _request(snapshot=None, *, framework: str = "auto") -> FrameworkRequest:
    return FrameworkRequest(
        requested_framework=framework,
        architecture_adapter_id="modelslim.glm_moe_dsa/test",
        model_type="glm_moe_dsa",
        hardware_profile="a5",
        capability_snapshot=snapshot,
    )


def test_manifest_framework_adapter_resolves_operator_and_fallbacks(tmp_path: Path) -> None:
    registry = InferenceFrameworkAdapterRegistry(
        (ManifestInferenceFrameworkAdapter(), AnalysisOnlyFrameworkAdapter())
    )

    result = registry.resolve(_request(_snapshot(tmp_path)))

    assert result.framework_adapter_id == MANIFEST_FRAMEWORK_ADAPTER_ID
    assert result.framework_name == "vllm_ascend"
    assert result.operator == "prolog_v3"
    assert result.status == "capability_declared"
    assert result.runtime_selected is False
    assert result.verified is False
    assert result.fallback_modes == ("mlapo", "native")


def test_analysis_only_fallback_does_not_claim_operator() -> None:
    registry = InferenceFrameworkAdapterRegistry((AnalysisOnlyFrameworkAdapter(),))

    result = registry.resolve(_request(framework="sglang"))

    assert result.framework_adapter_id == ANALYSIS_ONLY_FRAMEWORK_ADAPTER_ID
    assert result.framework_name == "sglang"
    assert result.operator == "unknown"
    assert result.verified is False


def test_manifest_for_another_model_falls_back_to_analysis_only(tmp_path: Path) -> None:
    registry = InferenceFrameworkAdapterRegistry(
        (ManifestInferenceFrameworkAdapter(), AnalysisOnlyFrameworkAdapter())
    )
    request = FrameworkRequest(
        requested_framework="auto",
        architecture_adapter_id="modelslim.qwen/test",
        model_type="qwen3_5_moe",
        hardware_profile="a5",
        capability_snapshot=_snapshot(tmp_path),
    )

    result = registry.resolve(request)

    assert result.framework_adapter_id == ANALYSIS_ONLY_FRAMEWORK_ADAPTER_ID
    assert result.operator == "unknown"


@dataclass(frozen=True)
class _AlwaysFrameworkAdapter:
    adapter_id: str
    fallback: bool = False

    def matches(self, request: FrameworkRequest) -> bool:
        del request
        return True

    def resolve(self, request: FrameworkRequest) -> OperatorResolution:
        del request
        return OperatorResolution(
            framework_adapter_id=self.adapter_id,
            framework_name="test",
            framework_version="1",
            framework_git_commit=None,
            operator="test_kernel",
            status="verified",
            selection_basis="test_runtime_resolver",
            runtime_selected=True,
        )


def test_framework_registry_rejects_ambiguous_exact_plugins() -> None:
    registry = InferenceFrameworkAdapterRegistry(
        (_AlwaysFrameworkAdapter("one/v1"), _AlwaysFrameworkAdapter("two/v1"))
    )

    with pytest.raises(FrameworkAdapterResolutionError, match="ambiguous"):
        registry.resolve(_request())
