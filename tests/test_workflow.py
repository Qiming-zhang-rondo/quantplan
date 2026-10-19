import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from quantplan.adapters.modelslim import inspect_modelslim_yaml
from quantplan.artifacts import ArtifactError
from quantplan.domain import (
    ApproximateMeasurement,
    DirectionalMeasurement,
    GroupInventory,
    InterventionMeasurement,
    PlanEvaluation,
    PrecisionFormat,
    StatisticalMeasurement,
)
from quantplan.run_config import CandidateConfig, loads_run_config
from quantplan.workflow import execute_workflow


class _FakeRuntime:
    def __init__(self) -> None:
        self.groups = (
            GroupInventory("block_0", ("model.layers.0.fc",), 100),
            GroupInventory("block_1", ("model.layers.1.fc",), 100),
        )
        self.module_inventory = tuple(name for group in self.groups for name in group.module_names)
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
            provenance={"runtime": "toy", "dataset_sha256": "a" * 64},
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


class _FastRuntime(_FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.exact_run_called = False
        self.fast_run_called = False
        self.evaluated_plan_ids = []

    def run(self, precisions, *, progress_callback=None):
        self.exact_run_called = True
        return super().run(precisions, progress_callback=progress_callback)

    def run_fast(self, precisions, *, progress_callback=None):
        del progress_callback
        self.fast_run_called = True
        risks = {
            ("block_0", PrecisionFormat.W4A8): 0.2,
            ("block_0", PrecisionFormat.W8A8): 0.01,
            ("block_1", PrecisionFormat.W4A8): 0.02,
            ("block_1", PrecisionFormat.W8A8): 0.01,
        }
        measurements = tuple(
            ApproximateMeasurement(
                group_id=group.group_id,
                precision=precision,
                context_id="toy-context",
                module_names=group.module_names,
                parameter_count=group.parameter_count,
                estimated_risk=risks[(group.group_id, precision)],
                token_count=64,
                elapsed_seconds=0.1,
                metric_id="fast-v1",
                mean_first_order_effect=0.0,
                rms_first_order_effect=risks[(group.group_id, precision)],
            )
            for group in self.groups
            for precision in precisions
        )
        return SimpleNamespace(
            groups=self.groups,
            module_inventory=self.module_inventory,
            measurements=measurements,
            provenance={"runtime": "toy-fast", "dataset_sha256": "a" * 64},
        )

    def evaluate_plan(self, assignments, *, plan_id):
        self.evaluated_plan_ids.append(plan_id)
        return super().evaluate_plan(assignments, plan_id=plan_id)


class _PartialRuntime(_FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.discovered_group_count = 40
        self.full_group_coverage = False


class _StatsRuntime(_FakeRuntime):
    def __init__(self, *, fail_all: bool = False) -> None:
        super().__init__()
        self.stats_run_called = False
        self.evaluated_plan_ids = []
        self.fail_all = fail_all

    def run(self, precisions, *, progress_callback=None):
        raise AssertionError("fast-stats must not use exact candidate interventions")

    def run_fast(self, precisions, *, progress_callback=None):
        raise AssertionError("fast-stats must not use local block replay")

    def run_fast_stats(self, precisions, *, progress_callback=None):
        self.stats_run_called = True
        measurements = []
        for group in self.groups:
            for precision in precisions:
                risk = {
                    ("block_0", PrecisionFormat.W4A8): 0.2,
                    ("block_0", PrecisionFormat.W8A8): 0.01,
                    ("block_1", PrecisionFormat.W4A8): 0.02,
                    ("block_1", PrecisionFormat.W8A8): 0.01,
                }[(group.group_id, precision)]
                measurements.append(
                    StatisticalMeasurement(
                        group_id=group.group_id,
                        precision=precision,
                        context_id="toy-context",
                        module_names=group.module_names,
                        parameter_count=group.parameter_count,
                        estimated_risk=risk,
                        token_count=64,
                        elapsed_seconds=0.1,
                        metric_id="stats-v1",
                        weight_risk=risk * 0.75,
                        activation_risk=risk * 0.25,
                    )
                )
        return SimpleNamespace(
            groups=self.groups,
            module_inventory=self.module_inventory,
            measurements=tuple(measurements),
            provenance={"runtime": "toy-stats", "dataset_sha256": "a" * 64},
        )

    def evaluate_plan(self, assignments, *, plan_id):
        self.evaluated_plan_ids.append(plan_id)
        evaluation = super().evaluate_plan(assignments, plan_id=plan_id)
        if self.fail_all:
            return replace(evaluation, candidate_nll=2.10)
        return evaluation


class _TaylorRuntime(_StatsRuntime):
    def __init__(self, *, fail_all: bool = False) -> None:
        super().__init__(fail_all=fail_all)
        self.taylor_run_called = False

    def run_fast_stats(self, precisions, *, progress_callback=None):
        raise AssertionError("fast-taylor must not use statistical candidate scores")

    def run_fast_taylor(self, precisions, *, progress_callback=None):
        self.taylor_run_called = True
        effects = {
            ("block_0", PrecisionFormat.W4A8): 0.2,
            ("block_0", PrecisionFormat.W8A8): -0.01,
            ("block_1", PrecisionFormat.W4A8): 0.02,
            ("block_1", PrecisionFormat.W8A8): 0.01,
        }
        offsets = {"block_0": 0.01, "block_1": 0.0}
        measurements = []
        for group in self.groups:
            for precision in (*precisions, PrecisionFormat.BF16):
                effect = effects.get((group.group_id, precision), 0.0)
                offset = offsets[group.group_id]
                measurements.append(
                    DirectionalMeasurement(
                        group_id=group.group_id,
                        precision=precision,
                        context_id="toy-context",
                        module_names=group.module_names,
                        parameter_count=group.parameter_count,
                        estimated_risk=effect + offset,
                        token_count=64,
                        elapsed_seconds=0.1,
                        metric_id="taylor-v1",
                        mean_first_order_effect=effect,
                        ranking_offset=offset,
                    )
                )
        return SimpleNamespace(
            groups=self.groups,
            module_inventory=self.module_inventory,
            measurements=tuple(measurements),
            provenance={"runtime": "toy-taylor", "dataset_sha256": "a" * 64},
        )


class _BackendBlockedRuntime(_FakeRuntime):
    backend_compilation_ready = False
    backend_compilation_blockers = ("unverified fused expert mapping",)


class _SourceAnchoredRuntime(_FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.groups = tuple(
            replace(
                group,
                allowed_formats=(PrecisionFormat.W8A8,),
                source_precision=PrecisionFormat.W8A8,
            )
            for group in self.groups
        )
        self.seen_precisions = ()

    def run(self, precisions, *, progress_callback=None):
        self.seen_precisions = tuple(precisions)
        return super().run(precisions, progress_callback=progress_callback)


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


def test_workflow_adds_source_yaml_anchor_when_candidate_omits_it(tmp_path: Path) -> None:
    config, output = _make_config(tmp_path)
    config.original_yaml.write_text(
        config.original_yaml.read_text(encoding="utf-8").replace(
            "weight: {dtype: int4", "weight: {dtype: int8"
        ),
        encoding="utf-8",
    )
    config = replace(
        config,
        candidates=CandidateConfig((PrecisionFormat.W4A8,)),
    )
    runtime = _SourceAnchoredRuntime()

    result = execute_workflow(config, runtime)

    assert result.status == "complete_analysis_only"
    assert runtime.seen_precisions == (PrecisionFormat.W4A8, PrecisionFormat.W8A8)
    candidate_set = json.loads(
        (output / "candidate-plan.json").read_text(encoding="utf-8")
    )
    assert candidate_set["source_yaml_safe_anchors"] == {
        "block_0": "w8a8",
        "block_1": "w8a8",
    }
    assert all(
        set(plan["assignments"].values()) == {"w8a8"}
        for plan in candidate_set["plans"]
    )


def test_fast_workflow_uses_approximate_ranking_then_exact_validation(
    tmp_path: Path,
) -> None:
    config, output = _make_config(tmp_path)
    runtime = _FastRuntime()

    result = execute_workflow(config, runtime, analysis_mode="fast")

    assert result.status == "complete_analysis_only"
    assert runtime.fast_run_called is True
    assert runtime.exact_run_called is False
    assert 1 <= len(runtime.evaluated_plan_ids) <= 3
    evidence = [
        json.loads(line)
        for line in (output / "risk-evidence.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(item["evidence_role"] == "gradient-weighted local approximation" for item in evidence)
    assert all("delta_nll" not in item for item in evidence)
    report = json.loads((output / "validation-report.json").read_text(encoding="utf-8"))
    assert report["analysis_mode"] == "fast"
    assert report["validation_cap"] == 3


@pytest.mark.parametrize("fail_all", [False, True])
@pytest.mark.parametrize("analysis_mode", ["fast-stats", "fast-taylor"])
def test_experimental_fast_workflow_validates_cheapest_first_and_stops_on_pass(
    tmp_path: Path, fail_all: bool, analysis_mode: str
) -> None:
    config, output = _make_config(tmp_path)
    config = replace(
        config,
        search=replace(config.search, high_precision_fraction=1.0),
    )
    runtime_type = _StatsRuntime if analysis_mode == "fast-stats" else _TaylorRuntime
    runtime = runtime_type(fail_all=fail_all)

    result = execute_workflow(config, runtime, analysis_mode=analysis_mode)

    if analysis_mode == "fast-stats":
        assert runtime.stats_run_called
    else:
        assert runtime.taylor_run_called
        assert not runtime.stats_run_called
    candidates = json.loads((output / "candidate-plan.json").read_text(encoding="utf-8"))
    ordered = sorted(
        candidates["plans"],
        key=lambda item: (item["estimated_cost"], item["proposal_score"]),
    )
    assert len(ordered) > 3
    expected_count = 3 if fail_all else 2
    assert runtime.evaluated_plan_ids == [item["plan_id"] for item in ordered[:expected_count]]
    evidence_role = (
        "factorized gradient statistics; no block replay"
        if analysis_mode == "fast-stats"
        else (
            "shared projection signed first-order; one extra local gradient projection per Linear/batch "
            "(possibly multiple chunk GEMMs), "
            "no candidate affine/block replay"
        )
    )
    assert evidence_role in candidates["claim"]
    evidence = [
        json.loads(line)
        for line in (output / "risk-evidence.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(item["evidence_role"] == evidence_role for item in evidence)
    assert all("delta_nll" not in item for item in evidence)
    if analysis_mode == "fast-stats":
        assert all("weight_risk" in item and "activation_risk" in item for item in evidence)
        assert all("mean_first_order_effect" not in item for item in evidence)
    else:
        assert all("mean_first_order_effect" in item and "ranking_offset" in item for item in evidence)
        assert any(item["mean_first_order_effect"] < 0 for item in evidence)
        assert any(item["precision"] == "bf16" for item in evidence)
    report = json.loads((output / "validation-report.json").read_text(encoding="utf-8"))
    assert report["analysis_mode"] == analysis_mode
    assert report["validation_cap"] == 3
    assert report["early_stop_on_pass"] is True
    assert len(report["evaluations"]) == expected_count
    assert report["passed"] is not fail_all
    if fail_all:
        assert result.status == "plan_validation_failed"
        assert result.derived_yaml is None
        assert not (output / "derived-modelslim.yaml").exists()
    else:
        assert result.status == "complete_analysis_only"
        assert result.selected_plan_id == ordered[1]["plan_id"]
        assert result.derived_yaml.is_file()


def test_fast_mode_keeps_validating_after_first_passing_plan(tmp_path: Path) -> None:
    config, output = _make_config(tmp_path)
    config = replace(config, search=replace(config.search, high_precision_fraction=1.0))
    runtime = _FastRuntime()

    result = execute_workflow(config, runtime, analysis_mode="fast")

    assert result.status == "complete_analysis_only"
    assert len(runtime.evaluated_plan_ids) == 3
    report = json.loads((output / "validation-report.json").read_text(encoding="utf-8"))
    assert report["evaluations"][1]["delta_nll"] <= config.search.max_delta_nll
    assert report["early_stop_on_pass"] is False


def test_workflow_emits_latency_report_and_uses_static_costs(tmp_path: Path) -> None:
    config, output = _make_config(tmp_path)
    config = replace(
        config,
        latency=replace(
            config.latency,
            mode="static",
            memory_bandwidth_gbps=1000.0,
            compute_tflops=100.0,
        ),
    )

    result = execute_workflow(config, _FakeRuntime(), analysis_mode="exact")

    assert result.status == "complete_analysis_only"
    latency_report = json.loads(
        (output / "latency-report.json").read_text(encoding="utf-8")
    )
    assert latency_report["provider"]["mode"] == "static"
    candidate_plan = json.loads(
        (output / "candidate-plan.json").read_text(encoding="utf-8")
    )
    assert candidate_plan["latency_model"]["mode"] == "static"
    assert all(item["latency_seconds"] is not None for item in candidate_plan["evidence"])


def test_provenance_binds_selection_validation_config_and_dataset(
    tmp_path: Path,
) -> None:
    config, output = _make_config(tmp_path)

    result = execute_workflow(config, _FakeRuntime())

    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    binding = provenance["evidence_binding"]
    assert binding["selected_plan_id"] == result.selected_plan_id
    assert binding["context_id"] == "toy-context"
    assert provenance["selected_evaluation"] == {
        "plan_id": result.selected_plan_id,
        "context_id": "toy-context",
    }

    validation_bytes = (output / "validation-report.json").read_bytes()
    assert binding["validation_report"] == {
        "path": "validation-report.json",
        "sha256": hashlib.sha256(validation_bytes).hexdigest(),
        "size_bytes": len(validation_bytes),
        "hash_scope": "exact_artifact_bytes",
    }
    assert binding["dataset"] == {
        "sha256": "a" * 64,
        "source": "runtime_provenance.dataset_sha256",
        "hash_scope": "runtime_selected_samples",
    }

    normalized_config = binding["run_config"]["normalized"]
    assert binding["run_config"]["sha256"] == _canonical_json_sha256(normalized_config)
    binding_without_digest = {
        key: value for key, value in binding.items() if key != "binding_sha256"
    }
    assert binding["binding_sha256"] == _canonical_json_sha256(binding_without_digest)
    assert provenance["derived_yaml_guard"]["guard_scope"] == "derived_yaml"
    assert provenance["derived_yaml_guard"]["findings"]


def test_yaml_audit_records_every_source_finding_and_pattern_expansion(
    tmp_path: Path,
) -> None:
    config, output = _make_config(tmp_path)
    runtime = _FakeRuntime()
    expected = inspect_modelslim_yaml(
        config.original_yaml,
        module_names=runtime.module_inventory,
    )

    execute_workflow(config, runtime)

    audit = json.loads((output / "yaml-audit.json").read_text(encoding="utf-8"))
    assert audit["guard_scope"] == "source_yaml"
    assert audit["source_size_bytes"] == config.original_yaml.stat().st_size
    assert audit["findings"] == [
        {"path": item.path, "status": item.status.value} for item in expected.findings
    ]
    assert audit["pattern_expansions"] == [
        {
            "field_path": item.field_path,
            "pattern": item.pattern,
            "matched_modules": list(item.matched_modules),
        }
        for item in expected.pattern_expansions
    ]
    assert audit["summary"]["finding_count"] == len(expected.findings)
    assert audit["summary"]["pattern_expansion_count"] == len(expected.pattern_expansions)
    assert sum(audit["summary"]["finding_counts"].values()) == len(expected.findings)


def test_workflow_does_not_emit_yaml_when_no_plan_passes(tmp_path: Path) -> None:
    config, output = _make_config(tmp_path)

    result = execute_workflow(config, _FailingPlanRuntime())

    assert result.status == "plan_validation_failed"
    assert result.derived_yaml is None
    assert not (output / "derived-modelslim.yaml").exists()
    assert "plan_validation_failed" in (output / "run-manifest.json").read_text(encoding="utf-8")


def test_workflow_never_emits_yaml_for_a_pilot_subset(tmp_path: Path) -> None:
    config, output = _make_config(tmp_path)

    result = execute_workflow(config, _PartialRuntime())

    assert result.status == "complete_partial_analysis"
    assert result.derived_yaml is None
    assert not (output / "derived-modelslim.yaml").exists()
    report = (output / "validation-report.json").read_text(encoding="utf-8")
    assert '"scope": "pilot_subset"' in report
    assert "complete_partial_analysis" in (output / "run-manifest.json").read_text(encoding="utf-8")


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


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
