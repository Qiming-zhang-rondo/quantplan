from __future__ import annotations

from pathlib import Path

import pytest

from quantplan.domain.models import PrecisionFormat
from quantplan.run_config import (
    RunConfigError,
    init_run_config,
    load_run_config,
    loads_run_config,
    parse_run_config,
    render_example_run_config,
)

EXAMPLE = Path(__file__).parents[1] / "examples" / "quantplan-run.yaml"


def _minimal_document(**overrides):
    document = {
        "schema_version": "quantplan_v1",
        "model_path": "model",
        "data_path": "data.jsonl",
        "original_yaml": "modelslim.yaml",
        "output_dir": "artifacts",
        "device": "npu:0",
    }
    document.update(overrides)
    return document


def test_minimal_config_has_conservative_defaults_and_anchors_paths(tmp_path: Path) -> None:
    config = parse_run_config(_minimal_document(), base_dir=tmp_path)

    assert config.model_path == (tmp_path / "model").resolve()
    assert config.data_path == (tmp_path / "data.jsonl").resolve()
    assert config.original_yaml == (tmp_path / "modelslim.yaml").resolve()
    assert config.output_dir == (tmp_path / "artifacts").resolve()
    assert config.device == "npu:0"
    assert config.runtime.loader == "huggingface"
    assert config.runtime.analysis_only is True
    assert config.runtime.trust_remote_code is False
    assert config.runtime.allow_image_text_fallback is False
    assert config.runtime.strict_target_coverage is True
    assert config.runtime.hardware_profile == "unknown"
    assert config.modules.block_patterns == ()
    assert config.modules.include == ()
    assert config.modules.exclude == (
        "*vision*",
        "*visual*",
        "*mtp*",
        "*.mlp.gate",
        "*.mlp.gate.weight",
        "*router*",
    )
    assert config.candidates.precisions == (
        PrecisionFormat.W4A8,
        PrecisionFormat.W8A8,
    )
    assert config.analysis.max_samples == 16
    assert config.analysis.max_length == 512
    assert config.analysis.max_groups == 8
    assert config.analysis.compute_kl is False
    assert config.analysis.text_field == "text"
    assert config.analysis.id_field == "id"
    assert config.search.base_precision is PrecisionFormat.W4A8
    assert config.search.high_precision_fraction == 0.25
    assert config.search.max_candidates == 8
    assert config.search.max_delta_nll == 0.02
    assert config.search.costs[PrecisionFormat.BF16] == 2.5
    assert config.resolved_output_paths["risk_evidence"] == (
        tmp_path / "artifacts" / "risk-evidence.jsonl"
    ).resolve()


def test_full_config_accepts_custom_loader_selectors_candidates_and_costs(
    tmp_path: Path,
) -> None:
    document = _minimal_document(
        runtime={
            "loader": "my_project.loader:load_model",
            "revision": "commit-123",
            "torch_dtype": "auto",
            "analysis_only": False,
            "trust_remote_code": True,
            "allow_image_text_fallback": True,
            "strict_target_coverage": False,
            "hardware_profile": "A5",
        },
        modules={
            "block_patterns": ["model.layers.*"],
            "include": ["model.layers.*.mlp.*"],
            "exclude": ["model.layers.*.mlp.router"],
        },
        candidates={"precisions": ["w4a8", "w4a4"]},
        analysis={
            "max_samples": 8,
            "max_length": 1024,
            "batch_size": 2,
            "max_groups": 4,
            "compute_kl": True,
            "text_field": "prompt",
            "id_field": "sample_id",
            "seed": 7,
        },
        search={
            "base_precision": "w4a8",
            "high_precision_fraction": 0.5,
            "max_candidates": 4,
            "max_delta_nll": 0.01,
            "costs": {"bf16": 16, "w4a8": 6, "w4a4": 4},
        },
        output={
            "risk_evidence": "evidence/risk.jsonl",
            "candidate_plan": "plans/candidate.json",
            "model_inventory": "inventory/model.json",
            "validation_report": "reports/validation.json",
            "derived_yaml": "yaml/derived.yaml",
            "provenance": "reports/provenance.json",
            "diff": "reports/plan.diff",
            "yaml_audit": "audit/yaml.json",
            "run_manifest": "manifests/run.json",
        },
    )

    config = parse_run_config(document, base_dir=tmp_path)

    assert config.runtime.loader == "my_project.loader:load_model"
    assert config.runtime.revision == "commit-123"
    assert config.runtime.torch_dtype == "auto"
    assert config.runtime.allow_image_text_fallback is True
    assert config.runtime.strict_target_coverage is False
    assert config.runtime.hardware_profile == "a5"
    assert config.modules.block_patterns == ("model.layers.*",)
    assert config.candidates.precisions == (
        PrecisionFormat.W4A8,
        PrecisionFormat.W4A4,
    )
    assert config.search.costs[PrecisionFormat.BF16] == 16.0
    assert config.search.max_delta_nll == 0.01
    assert config.output.candidate_plan == Path("plans/candidate.json")


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            """\
schema_version: quantplan_v1
model_path: a
data_path: b
original_yaml: c
output_dir: d
device: npu:0
future_option: true
""",
            "unknown field",
        ),
        (
            """\
schema_version: quantplan_v1
model_path: a
data_path: b
original_yaml: c
output_dir: d
device: npu:0
runtime: {loader: package_without_function}
""",
            "module:function",
        ),
        (
            """\
schema_version: quantplan_v1
model_path: a
data_path: b
original_yaml: c
output_dir: d
device: npu:0
candidates: {precisions: [bf16]}
""",
            "cannot use bf16",
        ),
        (
            """\
schema_version: quantplan_v1
model_path: a
data_path: b
original_yaml: c
output_dir: d
device: npu:0
analysis: {max_samples: true}
""",
            "must be an integer",
        ),
        (
            """\
schema_version: quantplan_v1
model_path: a
data_path: b
original_yaml: c
output_dir: d
device: npu:0
runtime: {allow_image_text_fallback: enabled}
""",
            "must be a boolean",
        ),
        (
            """\
schema_version: quantplan_v1
model_path: a
data_path: b
original_yaml: c
output_dir: d
device: npu:0
runtime: {torch_dtype: float16}
""",
            "cannot be labeled BF16",
        ),
        (
            """\
schema_version: quantplan_v1
model_path: a
data_path: b
original_yaml: c
output_dir: d
device: npu:0
runtime: {hardware_profile: a2}
""",
            "one of unknown, a3 or a5",
        ),
        (
            """\
schema_version: quantplan_v1
model_path: a
data_path: b
original_yaml: c
output_dir: d
device: npu:0
analysis: {max_length: 4097}
""",
            "evidence boundary",
        ),
        (
            """\
schema_version: quantplan_v1
model_path: a
data_path: b
original_yaml: c
output_dir: d
device: npu:0
output: {risk_evidence: ../escape.jsonl}
""",
            "inside output_dir",
        ),
        (
            """\
schema_version: quantplan_v1
model_path: a
data_path: b
original_yaml: c
output_dir: d
device: npu:0
candidates: {precisions: [w4a8]}
search: {costs: {bf16: 16}}
""",
            "every candidate",
        ),
    ],
)
def test_strict_validation_rejects_unsafe_or_ambiguous_values(
    source: str, message: str
) -> None:
    with pytest.raises(RunConfigError, match=message):
        loads_run_config(source)


def test_unknown_nested_field_and_duplicate_key_are_rejected() -> None:
    with pytest.raises(RunConfigError, match="unknown field"):
        parse_run_config(
            _minimal_document(analysis={"max_samples": 4, "future_metric": "x"})
        )

    duplicate = """\
schema_version: quantplan_v1
model_path: a
model_path: b
data_path: c
original_yaml: d
output_dir: e
device: npu:0
"""
    with pytest.raises(RunConfigError, match="duplicate key"):
        loads_run_config(duplicate)


def test_example_is_generated_from_the_public_template_and_is_not_runnable(
    tmp_path: Path,
) -> None:
    assert EXAMPLE.read_text(encoding="utf-8") == render_example_run_config()

    config = load_run_config(EXAMPLE)
    assert config.blocking_placeholders == (
        "model_path",
        "data_path",
        "original_yaml",
        "output_dir",
        "device",
    )
    with pytest.raises(RunConfigError, match="blocking placeholders"):
        config.validate_for_run()

    destination = tmp_path / "configs" / "run.yaml"
    assert init_run_config(destination) == destination
    assert destination.read_text(encoding="utf-8") == render_example_run_config()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        init_run_config(destination)


def test_init_fills_five_required_values_without_manual_editing(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    data = tmp_path / "data with spaces.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    original_yaml = tmp_path / "source.yaml"
    original_yaml.write_text("apiversion: modelslim_v1\n", encoding="utf-8")
    output = tmp_path / "artifacts"
    destination = tmp_path / "quantplan-run.yaml"

    init_run_config(
        destination,
        model_path=model,
        data_path=data,
        original_yaml=original_yaml,
        output_dir=output,
        device="npu:0",
    )
    config = load_run_config(destination)

    assert config.blocking_placeholders == ()
    assert config.model_path == model
    assert config.data_path == data
    assert config.output_dir == output
    config.validate_for_run()


def test_validate_for_run_checks_inputs_without_creating_outputs(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    source_yaml = tmp_path / "modelslim.yaml"
    source_yaml.write_text("apiversion: modelslim_v1\n", encoding="utf-8")
    output = tmp_path / "not-created"
    config = parse_run_config(
        {
            "schema_version": "quantplan_v1",
            "model_path": str(model),
            "data_path": str(data),
            "original_yaml": str(source_yaml),
            "output_dir": str(output),
            "device": "npu:0",
        }
    )

    config.validate_for_run()

    assert not output.exists()


def test_validate_for_run_can_skip_mount_checks_but_not_placeholders() -> None:
    config = parse_run_config(_minimal_document())

    config.validate_for_run(check_paths=False)

    placeholder = parse_run_config(
        _minimal_document(model_path="REPLACE_WITH_MODEL_PATH")
    )
    with pytest.raises(RunConfigError, match="model_path"):
        placeholder.validate_for_run(check_paths=False)
