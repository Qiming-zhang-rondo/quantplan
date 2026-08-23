from pathlib import Path

from quantplan.doctor import inspect_environment
from quantplan.run_config import loads_run_config


def test_doctor_reports_missing_runtime_without_claiming_execution_ready(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    data = tmp_path / "data.jsonl"
    data.write_text('{"text":"hello"}\n', encoding="utf-8")
    source_yaml = tmp_path / "source.yaml"
    source_yaml.write_text("apiversion: modelslim_v1\nspec: {}\n", encoding="utf-8")
    config = loads_run_config(
        f"""
schema_version: quantplan_v1
model_path: {model}
data_path: {data}
original_yaml: {source_yaml}
output_dir: {tmp_path / 'out'}
device: cpu
runtime:
  loader: huggingface
"""
    )

    report = inspect_environment(config)

    assert report.environment["execution_readiness"] == "not_assessed"
    assert any(item.code == "modelslim_yaml.syntax" for item in report.checks)
    assert isinstance(report.ready_for_analysis, bool)


def test_doctor_blocks_w4a4_proxy_request(tmp_path: Path) -> None:
    config = loads_run_config(
        f"""
schema_version: quantplan_v1
model_path: {tmp_path / 'missing-model'}
data_path: {tmp_path / 'missing-data'}
original_yaml: {tmp_path / 'missing-yaml'}
output_dir: {tmp_path / 'out'}
device: cpu
candidates:
  precisions: [w4a4]
search:
  base_precision: w4a4
"""
    )

    report = inspect_environment(config)

    check = next(item for item in report.checks if item.code == "candidate.proxy_support")
    assert check.status == "blocking"
