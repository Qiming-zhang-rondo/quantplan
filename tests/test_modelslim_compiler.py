from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from quantplan.adapters.modelslim import (
    CompilationBlockCode,
    ModelSlimPlanCompiler,
    PlanCompilationBlocked,
    compile_modelslim_plan,
)
from quantplan.domain.models import PrecisionFormat

MODULES = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.1.self_attn.q_proj",
    "model.layers.2.mlp.router",
)

W4A8_QCONFIG = {
    "act": {
        "dtype": "int8",
        "scope": "per_token",
        "symmetric": True,
        "method": "minmax",
    },
    "weight": {
        "dtype": "int4",
        "scope": "per_channel",
        "symmetric": True,
        "method": "minmax",
        "ext": {"group_size": 128},
    },
}

W8A8_QCONFIG = {
    "act": {
        "dtype": "int8",
        "scope": "per_token",
        "symmetric": True,
        "method": "minmax",
    },
    "weight": {
        "dtype": "int8",
        "scope": "per_channel",
        "symmetric": True,
        "method": "minmax",
    },
}

SOURCE = b"""\
# keep this source header byte-for-byte
apiversion: modelslim_v1
metadata:
  config_id: baseline # keep this non-process comment
spec:
  runner: auto
  process:
    - type: linear_quant
      qconfig:
        act: {dtype: int8, scope: per_token, symmetric: true, method: minmax}
        weight: {dtype: int8, scope: per_channel, symmetric: true, method: minmax}
      include:
        - 'model.layers.*'
  dataset: calibration.jsonl
"""


def _source(tmp_path: Path, content: bytes = SOURCE) -> Path:
    path = tmp_path / "modelslim.yaml"
    path.write_bytes(content)
    return path


def _groups() -> dict[str, list[str]]:
    return {
        "four_bit": [MODULES[0]],
        "eight_bit": [MODULES[1]],
        "bf16_fallback": [MODULES[2]],
    }


def _assignments() -> dict[str, PrecisionFormat]:
    return {
        "four_bit": PrecisionFormat.W4A8,
        "eight_bit": PrecisionFormat.W8A8,
        "bf16_fallback": PrecisionFormat.BF16,
    }


def _templates() -> dict[PrecisionFormat, dict]:
    return {
        PrecisionFormat.W4A8: W4A8_QCONFIG,
        PrecisionFormat.W8A8: W8A8_QCONFIG,
    }


def test_compiles_mixed_plan_without_writing_source_and_records_provenance(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    original_hash = hashlib.sha256(SOURCE).hexdigest()

    result = compile_modelslim_plan(
        source,
        _groups(),
        _assignments(),
        _templates(),
        MODULES,
    )

    assert source.read_bytes() == SOURCE
    assert list(tmp_path.iterdir()) == [source]
    assert "# keep this source header byte-for-byte" in result.derived_yaml
    assert "config_id: baseline # keep this non-process comment" in result.derived_yaml

    document = yaml.safe_load(result.derived_yaml)
    assert document["spec"]["runner"] == "auto"
    assert document["spec"]["dataset"] == "calibration.jsonl"
    assert len(document["spec"]["process"]) == 1
    group = document["spec"]["process"][0]
    assert group["type"] == "group"
    assert [config["type"] for config in group["configs"]] == [
        "linear_quant",
        "linear_quant",
    ]

    w4_config, w8_config = group["configs"]
    assert w4_config["qconfig"] == W4A8_QCONFIG
    assert w8_config["qconfig"] == W8A8_QCONFIG
    assert w4_config["include"] == [MODULES[0]]
    assert w8_config["include"] == [MODULES[1]]
    assert "exclude" not in w4_config
    assert "exclude" not in w8_config
    # BF16 has no invented qconfig and is absent from every disjoint include.
    assert all(MODULES[2] not in config["include"] for config in group["configs"])

    assert result.output_guard.static_policy_passed
    assert not result.output_guard.execution_ready
    assert result.output_guard.execution_readiness == "not_assessed"
    assert result.sidecar.source_sha256 == original_hash
    assert result.sidecar.derived_sha256 == hashlib.sha256(result.yaml_bytes).hexdigest()
    assert result.sidecar.execution_readiness == "not_assessed"
    assert result.sidecar.bf16_modules == (MODULES[2],)
    assert result.sidecar.quantized_modules == MODULES[:2]
    assert result.sidecar.quant_target_count == len(MODULES)
    assert result.sidecar.model_module_count == len(MODULES)
    assert result.sidecar.diff[0].path == "$.spec.process"
    assert result.sidecar.diff[0].operation == "replace"
    assert {item.precision for item in result.sidecar.templates} == {
        PrecisionFormat.W4A8,
        PrecisionFormat.W8A8,
    }
    assert result.sidecar_dict["source"]["sha256"] == original_hash
    assert result.sidecar_dict["inventories"]["quant_targets"]["count"] == len(MODULES)


def test_preserves_other_processors_in_an_existing_group(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        b"""\
apiversion: modelslim_v1
spec:
  process:
    - type: group
      configs:
        - type: quarot
          online: true
        - type: linear_quant
          qconfig:
            weight: {dtype: int8, scope: per_channel, symmetric: true, method: minmax}
          include: ['model.layers.*']
""",
    )
    assignments = {name: PrecisionFormat.W4A8 for name in _groups()}

    result = ModelSlimPlanCompiler().compile(
        source,
        _groups(),
        assignments,
        {PrecisionFormat.W4A8: W4A8_QCONFIG},
        MODULES,
    )

    configs = yaml.safe_load(result.derived_yaml)["spec"]["process"][0]["configs"]
    assert configs[0] == {"type": "quarot", "online": True}
    assert configs[1]["type"] == "linear_quant"
    assert configs[1]["qconfig"] == W4A8_QCONFIG
    assert "exclude" not in configs[1]


def test_full_model_inventory_is_only_used_for_pattern_guarding(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        b"""\
apiversion: modelslim_v1
spec:
  process:
    - type: flex_smooth_quant
      include: ['model.mtp.*']
    - type: linear_quant
      qconfig:
        weight: {dtype: int8, scope: per_channel, symmetric: true, method: minmax}
      include: ['model.layers.*']
""",
    )
    # Direct ``named_modules()`` output includes the model root as an empty name.
    model_inventory = ("", *MODULES, "model.mtp.0.proj")
    assignments = {name: PrecisionFormat.W4A8 for name in _groups()}

    result = compile_modelslim_plan(
        source,
        _groups(),
        assignments,
        {PrecisionFormat.W4A8: W4A8_QCONFIG},
        MODULES,
        model_module_names=model_inventory,
    )

    processes = yaml.safe_load(result.derived_yaml)["spec"]["process"]
    assert processes[0] == {
        "type": "flex_smooth_quant",
        "include": ["model.mtp.*"],
    }
    generated = processes[1]["configs"][0]
    assert generated["include"] == list(MODULES)
    assert "model.mtp.0.proj" not in generated["include"]
    assert "exclude" not in generated
    assert result.output_guard.static_policy_passed
    assert result.sidecar.quant_target_count == len(MODULES)
    assert result.sidecar.model_module_count == len(model_inventory) - 1
    assert (
        result.sidecar.quant_target_inventory_sha256 != result.sidecar.model_module_inventory_sha256
    )
    assert any(
        expansion.pattern == "model.mtp.*" and expansion.matched_modules == ("model.mtp.0.proj",)
        for expansion in result.output_guard.pattern_expansions
    )


def test_full_model_inventory_must_cover_all_quantization_targets(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)

    with pytest.raises(PlanCompilationBlocked) as caught:
        compile_modelslim_plan(
            source,
            _groups(),
            _assignments(),
            _templates(),
            MODULES,
            model_module_names=MODULES[:-1],
        )

    assert caught.value.code is CompilationBlockCode.PLAN_INVALID
    assert "missing_targets" in caught.value.details[0]


def test_can_add_group_to_missing_process_but_does_not_rewrite_other_fields(
    tmp_path: Path,
) -> None:
    original = b"""\
apiversion: modelslim_v1
metadata:
  deployment_ticket: opaque-and-preserved
spec:
  runner: layer_wise
  dataset: data.jsonl
"""
    source = _source(tmp_path, original)
    assignments = {name: PrecisionFormat.W4A8 for name in _groups()}

    result = compile_modelslim_plan(
        source,
        _groups(),
        assignments,
        {PrecisionFormat.W4A8: W4A8_QCONFIG},
        MODULES,
    )

    assert source.read_bytes() == original
    document = yaml.safe_load(result.derived_yaml)
    assert document["metadata"] == {"deployment_ticket": "opaque-and-preserved"}
    assert document["spec"]["runner"] == "layer_wise"
    assert document["spec"]["dataset"] == "data.jsonl"
    assert document["spec"]["process"][0]["type"] == "group"
    assert result.sidecar.diff[0].operation == "add"
    assert result.sidecar.diff[0].before is None


def test_all_bf16_removes_existing_linear_quant_without_inventing_a_template(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    assignments = {name: PrecisionFormat.BF16 for name in _groups()}

    result = compile_modelslim_plan(
        source,
        _groups(),
        assignments,
        {},
        MODULES,
    )

    assert yaml.safe_load(result.derived_yaml)["spec"]["process"] == []
    assert result.sidecar.templates == ()
    assert result.sidecar.bf16_modules == MODULES
    assert result.output_guard.static_policy_passed


def test_unknown_source_execution_field_is_blocked_and_source_is_unchanged(
    tmp_path: Path,
) -> None:
    original = SOURCE.replace(
        b"  runner: auto\n",
        b"  runner: auto\n  vendor_execution_mode: unsafe-to-guess\n",
    )
    source = _source(tmp_path, original)

    with pytest.raises(PlanCompilationBlocked) as caught:
        compile_modelslim_plan(
            source,
            _groups(),
            _assignments(),
            _templates(),
            MODULES,
        )

    assert caught.value.code is CompilationBlockCode.SOURCE_YAML_BLOCKED
    assert "unknown_field_blocked@$.spec.vendor_execution_mode" in caught.value.details
    assert source.read_bytes() == original


def test_invalid_qconfig_template_is_rejected_by_the_output_guard(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    invalid_template = yaml.safe_load(yaml.safe_dump(W4A8_QCONFIG))
    invalid_template["weight"]["method"] = "future_method"
    assignments = {name: PrecisionFormat.W4A8 for name in _groups()}

    with pytest.raises(PlanCompilationBlocked) as caught:
        compile_modelslim_plan(
            source,
            _groups(),
            assignments,
            {PrecisionFormat.W4A8: invalid_template},
            MODULES,
        )

    assert caught.value.code is CompilationBlockCode.DERIVED_YAML_BLOCKED
    assert any("quantizer_method_unsupported" in item for item in caught.value.details)
    assert source.read_bytes() == SOURCE


def test_precision_label_must_match_template_signature(tmp_path: Path) -> None:
    source = _source(tmp_path)
    assignments = {name: PrecisionFormat.W4A8 for name in _groups()}

    with pytest.raises(PlanCompilationBlocked) as caught:
        compile_modelslim_plan(
            source,
            _groups(),
            assignments,
            {PrecisionFormat.W4A8: W8A8_QCONFIG},
            MODULES,
        )

    assert caught.value.code is CompilationBlockCode.TEMPLATE_INVALID
    assert "weight.dtype='int8'" in str(caught.value)
    assert source.read_bytes() == SOURCE


def test_backend_pattern_can_target_post_adapter_expert_linears(tmp_path: Path) -> None:
    source = _source(tmp_path)
    parent = "model.layers.0.mlp.experts"
    backend_children = (
        f"{parent}.0.gate_proj",
        f"{parent}.0.up_proj",
        f"{parent}.0.down_proj",
    )

    result = compile_modelslim_plan(
        source,
        {"block_0": [parent]},
        {"block_0": PrecisionFormat.W4A8},
        {PrecisionFormat.W4A8: W4A8_QCONFIG},
        (parent,),
        model_module_names=(parent, *backend_children),
        module_patterns={parent: f"{parent}.*"},
    )

    generated = yaml.safe_load(result.derived_yaml)["spec"]["process"][0]["configs"][0]
    assert generated["include"] == [f"{parent}.*"]
    expansion = next(
        item
        for item in result.output_guard.pattern_expansions
        if item.pattern == f"{parent}.*"
    )
    assert expansion.matched_modules == backend_children


@pytest.mark.parametrize(
    ("assignments", "templates", "expected_code"),
    [
        (
            {name: PrecisionFormat.W4A4 for name in _groups()},
            {},
            CompilationBlockCode.PLAN_INVALID,
        ),
        (
            {name: PrecisionFormat.W4A8 for name in _groups()},
            {},
            CompilationBlockCode.TEMPLATE_MISSING,
        ),
    ],
)
def test_unrepresentable_precision_or_missing_template_blocks_explicitly(
    tmp_path: Path,
    assignments: dict[str, PrecisionFormat],
    templates: dict,
    expected_code: CompilationBlockCode,
) -> None:
    source = _source(tmp_path)

    with pytest.raises(PlanCompilationBlocked) as caught:
        compile_modelslim_plan(
            source,
            _groups(),
            assignments,
            templates,
            MODULES,
        )

    assert caught.value.code is expected_code
    assert source.read_bytes() == SOURCE


def test_unsafe_pattern_names_and_ambiguous_insertion_slots_are_blocked(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    unsafe_modules = ("model.layers.[0].q_proj", MODULES[1], MODULES[2])
    unsafe_groups = {
        "four_bit": [unsafe_modules[0]],
        "eight_bit": [unsafe_modules[1]],
        "bf16_fallback": [unsafe_modules[2]],
    }

    with pytest.raises(PlanCompilationBlocked) as caught:
        compile_modelslim_plan(
            source,
            unsafe_groups,
            _assignments(),
            _templates(),
            unsafe_modules,
        )
    assert caught.value.code is CompilationBlockCode.UNSAFE_PATTERN_NAME

    no_linear_source = _source(
        tmp_path,
        b"""\
apiversion: modelslim_v1
spec:
  process:
    - type: quarot
      online: true
""",
    )
    assignments = {name: PrecisionFormat.W4A8 for name in _groups()}
    with pytest.raises(PlanCompilationBlocked) as caught:
        compile_modelslim_plan(
            no_linear_source,
            _groups(),
            assignments,
            {PrecisionFormat.W4A8: W4A8_QCONFIG},
            MODULES,
        )
    assert caught.value.code is CompilationBlockCode.UNSAFE_SOURCE_STRUCTURE
    assert "no unambiguous linear_quant insertion slot" in str(caught.value)
