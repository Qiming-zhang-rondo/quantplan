from pathlib import Path
from types import SimpleNamespace

import pytest

from quantplan.artifacts import ArtifactError
from quantplan.domain import (
    GroupInventory,
    InterventionMeasurement,
    PlanEvaluation,
    PrecisionFormat,
)
from quantplan.run_config import loads_run_config
from quantplan.workflow import execute_workflow


class _FakeRuntime:
    def __init__(self) -> None:
        self.groups = (
            GroupInventory("block_0", ("model.layers.0.fc",), 100),
            GroupInventory("block_1", ("model.layers.1.fc",), 100),
        )
        self.module_inventory = tuple(
            name for group in self.groups for name in group.module_names
        )
        self.discovered_group_count = len(self.groups)
        self.full_group_coverage = True

    def run(self, precisions, *, progress_callback=None):
        del progress_callback
        measurements = []
        candidate_nlls = {
            ("block_0", PrecisionFormat.W4A8): 2.10,
            ("block_0", PrecisionFormat.W8A8): 2.01,
            ("block_1", PrecisionFormat.W4A8): 2.02,
            ("block_1", PrecisionFormat.W8A8): 2.01,
        }
        for group in self.groups:
            for precision in precisions:
                measurements.append(
                    InterventionMeasurement(
                        group_id=group.group_id,
                        precision=precision,
                        context_id="toy-context",
                        module_names=group.module_names,
                        parameter_count=group.parameter_count,
                        baseline_nll=2.0,
                        candidate_nll=candidate_nlls[(group.group_id, precision)],
                        token_count=64,
                        elapsed_seconds=0.1,
                        quantizer_id="toy-proxy",
                    )
                )
        return SimpleNamespace(
            groups=self.groups,
            module_inventory=self.module_inventory,
            measurements=tuple(measurements),
            provenance={"runtime": "toy"},
        )

    def evaluate_plan(self, assignments, *, plan_id):
        protected = assignments["block_0"] is not PrecisionFormat.W4A8
        return PlanEvaluation(
            plan_id=plan_id,
            context_id="toy-context",
            baseline_nll=2.0,
            candidate_nll=2.01 if protected else 2.10,
            token_count=64,
            elapsed_seconds=0.2,
        )


class _FailingPlanRuntime(_FakeRuntime):
    def evaluate_plan(self, assignments, *, plan_id):
        del assignments
        return PlanEvaluation(
            plan_id=plan_id,
            context_id="toy-context",
            baseline_nll=2.0,
            candidate_nll=2.10,
            token_count=64,
            elapsed_seconds=0.2,
        )


class _PartialRuntime(_FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.discovered_group_count = 40
        self.full_group_coverage = False


class _BackendBlockedRuntime(_FakeRuntime):
    backend_compilation_ready = False
    backend_compilation_blockers = ("unverified fused expert mapping",)


def _make_config(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    data = tmp_path / "data.jsonl"
    data.write_text('{"text":"hello world"}\n', encoding="utf-8")
    source_yaml = tmp_path / "source.yaml"
    source_yaml.write_text(
        """\
apiversion: modelslim_v1
spec:
  process:
    - type: linear_quant
      qconfig:
        act: {dtype: int8, scope: per_token, symmetric: true, method: minmax}
        weight: {dtype: int4, scope: per_channel, symmetric: true, method: minmax}
      include:
        - model.layers.*.fc
""",
        encoding="utf-8",
    )
    output = tmp_path / "artifacts"
    config = loads_run_config(
        f"""
schema_version: quantplan_v1
model_path: {model}
data_path: {data}
original_yaml: {source_yaml}
output_dir: {output}
device: cpu
search:
  base_precision: w4a8
  high_precision_fraction: 0.25
  max_candidates: 8
  max_delta_nll: 0.02
"""
    )

    return config, output


def test_workflow_emits_yaml_only_after_a_plan_passes(tmp_path: Path) -> None:
    config, output = _make_config(tmp_path)

    result = execute_workflow(config, _FakeRuntime())

    assert result.status == "complete_analysis_only"
    assert result.derived_yaml is not None
    assert result.derived_yaml.is_file()
    assert (output / "validation-report.json").is_file()
    assert (output / "run-manifest.json").read_text(encoding="utf-8").find(
        "complete_analysis_only"
    ) > 0


def test_workflow_does_not_emit_yaml_when_no_plan_passes(tmp_path: Path) -> None:
    config, output = _make_config(tmp_path)

    result = execute_workflow(config, _FailingPlanRuntime())

    assert result.status == "plan_validation_failed"
    assert result.derived_yaml is None
    assert not (output / "derived-modelslim.yaml").exists()
    assert "plan_validation_failed" in (output / "run-manifest.json").read_text(
        encoding="utf-8"
    )


def test_workflow_never_emits_yaml_for_a_pilot_subset(tmp_path: Path) -> None:
    config, output = _make_config(tmp_path)

    result = execute_workflow(config, _PartialRuntime())

    assert result.status == "complete_partial_analysis"
    assert result.derived_yaml is None
    assert not (output / "derived-modelslim.yaml").exists()
    report = (output / "validation-report.json").read_text(encoding="utf-8")
    assert '"scope": "pilot_subset"' in report
    assert "complete_partial_analysis" in (output / "run-manifest.json").read_text(
        encoding="utf-8"
    )


def test_workflow_overwrite_cannot_follow_artifact_parent_symlink(
    tmp_path: Path,
) -> None:
    config, output = _make_config(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "validation-report.json"
    protected.write_text("do not delete", encoding="utf-8")
    output.mkdir()
    (output / "reports").symlink_to(outside, target_is_directory=True)
    object.__setattr__(
        config.output,
        "validation_report",
        Path("reports/validation-report.json"),
    )

    with pytest.raises(ArtifactError, match="escapes output directory"):
        execute_workflow(config, _FakeRuntime(), overwrite=True)

    assert protected.read_text(encoding="utf-8") == "do not delete"


def test_workflow_never_emits_yaml_without_backend_target_mapping(
    tmp_path: Path,
) -> None:
    config, output = _make_config(tmp_path)

    result = execute_workflow(config, _BackendBlockedRuntime())

    assert result.status == "backend_capability_blocked"
    assert result.derived_yaml is None
    assert not (output / "derived-modelslim.yaml").exists()
    manifest = (output / "run-manifest.json").read_text(encoding="utf-8")
    assert "unverified fused expert mapping" in manifest
