import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import quantplan.doctor as doctor_module
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


def test_doctor_blocks_non_analysis_runtime(tmp_path: Path) -> None:
    config = _valid_config(
        tmp_path,
        runtime="""
runtime:
  analysis_only: false
""",
    )

    report = inspect_environment(config)

    check = next(item for item in report.checks if item.code == "runtime.analysis_only")
    assert check.status == "blocking"
    assert report.ready_for_analysis is False


def test_doctor_reports_missing_dotted_custom_loader_without_raising(
    tmp_path: Path,
) -> None:
    config = _valid_config(
        tmp_path,
        runtime="""
runtime:
  loader: quantplan_parent_package_that_does_not_exist.child:load_model
""",
    )

    report = inspect_environment(config)

    check = next(item for item in report.checks if item.code == "runtime.custom_loader")
    assert check.status == "blocking"
    assert "cannot be found" in check.message
    assert "ModuleNotFoundError" in check.message


@pytest.mark.parametrize(
    ("available", "count", "device", "expected_status", "message"),
    [
        (True, 2, "npu:1", "pass", "npu:1 is available"),
        (True, 1, "npu:1", "blocking", "device_count() returned 1"),
        (False, 2, "npu:0", "blocking", "is_available() returned false"),
    ],
)
def test_doctor_actively_probes_requested_npu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    available: bool,
    count: int,
    device: str,
    expected_status: str,
    message: str,
) -> None:
    config = _valid_config(tmp_path, device=device)
    fake_torch = ModuleType("torch")
    fake_torch.npu = SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: count,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(
        doctor_module,
        "_safe_find_spec",
        lambda name: (
            (object(), None)
            if name in {"torch", "torch_npu", "transformers"}
            else (None, None)
        ),
    )

    report = inspect_environment(config)

    check = next(item for item in report.checks if item.code == "device.npu")
    assert check.status == expected_status
    assert message in check.message


def test_doctor_explains_when_npu_probe_cannot_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _valid_config(tmp_path, device="npu:0")
    monkeypatch.setattr(
        doctor_module,
        "_safe_find_spec",
        lambda name: (
            (object(), None)
            if name in {"torch_npu", "transformers"}
            else (None, None)
        ),
    )

    report = inspect_environment(config)

    check = next(item for item in report.checks if item.code == "device.npu")
    assert check.status == "blocking"
    assert "not probed because PyTorch cannot be resolved" in check.message


def test_doctor_blocks_torch_npu_binary_import_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _valid_config(tmp_path, device="npu:0")
    fake_torch = ModuleType("torch")
    fake_torch.npu = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
    )
    monkeypatch.setattr(
        doctor_module,
        "_safe_find_spec",
        lambda name: (
            (object(), None)
            if name in {"torch", "torch_npu", "transformers"}
            else (None, None)
        ),
    )

    def import_for_probe(name: str):
        if name == "torch":
            return fake_torch
        if name == "torch_npu":
            raise OSError("binary ABI mismatch")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(doctor_module.importlib, "import_module", import_for_probe)

    report = inspect_environment(config)

    dependency = next(
        item for item in report.checks if item.code == "dependency.torch_npu"
    )
    device = next(item for item in report.checks if item.code == "device.npu")
    assert dependency.status == "blocking"
    assert "OSError: binary ABI mismatch" in dependency.message
    assert device.status == "blocking"
    assert "import failed" in device.message


def test_doctor_rejects_invalid_npu_device_syntax(tmp_path: Path) -> None:
    config = _valid_config(tmp_path, device="npu:bad")

    report = inspect_environment(config)

    check = next(item for item in report.checks if item.code == "device.npu")
    assert check.status == "blocking"
    assert "npu:<non-negative-index>" in check.message


def _valid_config(
    tmp_path: Path,
    *,
    device: str = "cpu",
    runtime: str = "",
):
    model = tmp_path / "model"
    model.mkdir()
    data = tmp_path / "data.jsonl"
    data.write_text('{"text":"hello"}\n', encoding="utf-8")
    source_yaml = tmp_path / "source.yaml"
    source_yaml.write_text("apiversion: modelslim_v1\nspec: {}\n", encoding="utf-8")
    return loads_run_config(
        f"""
schema_version: quantplan_v1
model_path: {model}
data_path: {data}
original_yaml: {source_yaml}
output_dir: {tmp_path / 'out'}
device: {device}
{runtime}
"""
    )
