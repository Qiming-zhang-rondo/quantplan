"""Cheap preflight checks that do not load model weights."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Any

import yaml

from quantplan.domain import PrecisionFormat
from quantplan.run_config import RunConfig


@dataclass(frozen=True)
class DoctorCheck:
    code: str
    status: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "status": self.status, "message": self.message}


@dataclass(frozen=True)
class DoctorReport:
    ready_for_analysis: bool
    checks: tuple[DoctorCheck, ...]
    environment: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready_for_analysis": self.ready_for_analysis,
            "checks": [item.to_dict() for item in self.checks],
            "environment": self.environment,
        }


def inspect_environment(config: RunConfig) -> DoctorReport:
    """Check paths, optional dependencies and obvious unsupported requests.

    Passing this report does not mean ModelSlim or target-hardware execution is
    ready.  Full target coverage and YAML pattern checks require loading the
    model inventory and happen inside ``quantplan run``.
    """

    checks: list[DoctorCheck] = []

    checks.append(
        DoctorCheck(
            code="config.placeholders",
            status="blocking" if config.blocking_placeholders else "pass",
            message=(
                "configuration still contains: "
                + ", ".join(config.blocking_placeholders)
                if config.blocking_placeholders
                else "all required configuration values are filled"
            ),
        )
    )

    checks.append(
        DoctorCheck(
            code="runtime.analysis_only",
            status="pass" if config.runtime.analysis_only else "blocking",
            message=(
                "analysis-only execution is enabled"
                if config.runtime.analysis_only
                else (
                    "runtime.analysis_only=false is not implemented; "
                    "QuantPlan cannot execute ModelSlim"
                )
            ),
        )
    )

    for field_name in ("model_path", "data_path", "original_yaml"):
        path = getattr(config, field_name)
        exists = path.is_file() if field_name != "model_path" else path.exists()
        checks.append(
            DoctorCheck(
                code=f"path.{field_name}",
                status="pass" if exists else "blocking",
                message=f"{path} {'exists' if exists else 'does not exist'}",
            )
        )

    torch_spec, torch_spec_error = _safe_find_spec("torch")
    torch_available = torch_spec is not None
    checks.append(
        DoctorCheck(
            code="dependency.torch",
            status="pass" if torch_available else "blocking",
            message=(
                "PyTorch is importable"
                if torch_available
                else _unavailable_message(
                    "PyTorch is required; use the build paired with the target accelerator",
                    torch_spec_error,
                )
            ),
        )
    )
    if config.runtime.loader == "huggingface":
        transformers_spec, transformers_spec_error = _safe_find_spec("transformers")
        transformers_available = transformers_spec is not None
        checks.append(
            DoctorCheck(
                code="dependency.transformers",
                status="pass" if transformers_available else "blocking",
                message=(
                    "Transformers is importable"
                    if transformers_available
                    else _unavailable_message(
                        "Transformers is required by runtime.loader=huggingface",
                        transformers_spec_error,
                    )
                ),
            )
        )
    else:
        module_name = config.runtime.loader.partition(":")[0]
        loader_spec, loader_spec_error = _safe_find_spec(module_name)
        loader_available = loader_spec is not None
        checks.append(
            DoctorCheck(
                code="runtime.custom_loader",
                status="pass" if loader_available else "blocking",
                message=(
                    f"custom loader module {module_name!r} is importable"
                    if loader_available
                    else _unavailable_message(
                        f"custom loader module {module_name!r} cannot be found",
                        loader_spec_error,
                    )
                ),
            )
        )

    if config.device.lower().startswith("npu"):
        checks.extend(
            _inspect_npu_device(
                config.device,
                torch_available=torch_available,
            )
        )

    unsupported = [
        precision.value
        for precision in config.candidates.precisions
        if precision is PrecisionFormat.W4A4
    ]
    if unsupported:
        checks.append(
            DoctorCheck(
                code="candidate.proxy_support",
                status="blocking",
                message="W4A4 is not implemented by the first analysis runtime",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                code="candidate.proxy_support",
                status="pass",
                message="requested analysis proxies are implemented",
            )
        )

    yaml_sha256: str | None = None
    if config.original_yaml.is_file():
        try:
            source = config.original_yaml.read_bytes()
            yaml_sha256 = hashlib.sha256(source).hexdigest()
            document = yaml.safe_load(source)
            valid_root = isinstance(document, dict) and "spec" in document
            checks.append(
                DoctorCheck(
                    code="modelslim_yaml.syntax",
                    status="pass" if valid_root else "blocking",
                    message=(
                        "source YAML parses and contains spec; semantic pattern audit is deferred"
                        if valid_root
                        else "source YAML must be a mapping containing spec"
                    ),
                )
            )
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            checks.append(
                DoctorCheck(
                    code="modelslim_yaml.syntax",
                    status="blocking",
                    message=f"source YAML cannot be parsed safely: {exc}",
                )
            )

    modelslim_spec, modelslim_spec_error = _safe_find_spec("msmodelslim")
    modelslim_available = modelslim_spec is not None
    checks.append(
        DoctorCheck(
            code="dependency.msmodelslim",
            status="pass" if modelslim_available else "warning",
            message=(
                "msmodelslim is importable"
                if modelslim_available
                else _unavailable_message(
                    "msmodelslim is not required for analysis/YAML generation; "
                    "backend execution remains unassessed",
                    modelslim_spec_error,
                )
            ),
        )
    )

    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": config.device,
        "source_yaml_sha256": yaml_sha256,
        "packages": {
            "torch": _package_version("torch"),
            "torch_npu": _package_version("torch-npu", "torch_npu"),
            "transformers": _package_version("transformers"),
            "msmodelslim": _package_version("msmodelslim"),
        },
        "execution_readiness": "not_assessed",
    }
    return DoctorReport(
        ready_for_analysis=not any(item.status == "blocking" for item in checks),
        checks=tuple(checks),
        environment=environment,
    )


def _package_version(*distribution_names: str) -> str | None:
    for name in distribution_names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _safe_find_spec(module_name: str) -> tuple[Any | None, str | None]:
    """Resolve an import spec without leaking dotted-parent import failures."""

    try:
        return importlib.util.find_spec(module_name), None
    except (AttributeError, ImportError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _unavailable_message(message: str, detail: str | None) -> str:
    return f"{message} ({detail})" if detail else message


def _inspect_npu_device(
    device: str,
    *,
    torch_available: bool,
) -> tuple[DoctorCheck, ...]:
    torch_npu_spec, torch_npu_spec_error = _safe_find_spec("torch_npu")
    torch_npu_available = torch_npu_spec is not None
    requested_index, device_error = _parse_npu_device(device)
    if device_error is not None:
        dependency_status = "warning" if torch_npu_available else "blocking"
        dependency_message = (
            "torch_npu is resolvable but was not imported because the device syntax is invalid"
            if torch_npu_available
            else _unavailable_message(
                "an NPU device was requested but torch_npu cannot be resolved",
                torch_npu_spec_error,
            )
        )
        return (
            DoctorCheck("dependency.torch_npu", dependency_status, dependency_message),
            DoctorCheck("device.npu", "blocking", device_error),
        )
    if not torch_available:
        return (
            DoctorCheck(
                "dependency.torch_npu",
                "warning" if torch_npu_available else "blocking",
                (
                    "torch_npu is resolvable but was not imported because PyTorch "
                    "cannot be resolved"
                    if torch_npu_available
                    else _unavailable_message(
                        "an NPU device was requested but torch_npu cannot be resolved",
                        torch_npu_spec_error,
                    )
                ),
            ),
            DoctorCheck(
                "device.npu",
                "blocking",
                "NPU availability was not probed because PyTorch cannot be resolved",
            ),
        )
    if not torch_npu_available:
        return (
            DoctorCheck(
                "dependency.torch_npu",
                "blocking",
                _unavailable_message(
                    "an NPU device was requested but torch_npu cannot be resolved",
                    torch_npu_spec_error,
                ),
            ),
            DoctorCheck(
                "device.npu",
                "blocking",
                "NPU availability was not probed because torch_npu cannot be resolved",
            ),
        )

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # noqa: BLE001 - binary runtime import boundary
        return (
            DoctorCheck(
                "dependency.torch_npu",
                "warning",
                "torch_npu is resolvable but was not imported because PyTorch import failed",
            ),
            DoctorCheck(
                "device.npu",
                "blocking",
                "PyTorch import failed during the NPU probe: "
                f"{type(exc).__name__}: {exc}",
            ),
        )
    try:
        importlib.import_module("torch_npu")
    except Exception as exc:  # noqa: BLE001 - binary accelerator import boundary
        return (
            DoctorCheck(
                "dependency.torch_npu",
                "blocking",
                "torch_npu was resolved but failed to import: "
                f"{type(exc).__name__}: {exc}",
            ),
            DoctorCheck(
                "device.npu",
                "blocking",
                "NPU availability was not probed because torch_npu import failed",
            ),
        )

    checks = [
        DoctorCheck(
            "dependency.torch_npu",
            "pass",
            "torch_npu imported successfully",
        )
    ]

    npu = getattr(torch, "npu", None)
    is_available = getattr(npu, "is_available", None)
    device_count = getattr(npu, "device_count", None)
    if not callable(is_available) or not callable(device_count):
        checks.append(
            DoctorCheck(
                "device.npu",
                "blocking",
                "torch_npu imported, but torch.npu does not expose callable "
                "is_available() and device_count() probes",
            )
        )
        return tuple(checks)

    try:
        backend_available = bool(is_available())
        count = int(device_count())
    except Exception as exc:  # noqa: BLE001 - accelerator runtime probe boundary
        checks.append(
            DoctorCheck(
                "device.npu",
                "blocking",
                "torch.npu availability probe failed: "
                f"{type(exc).__name__}: {exc}",
            )
        )
        return tuple(checks)

    if not backend_available:
        message = "torch.npu.is_available() returned false"
        status = "blocking"
    elif count <= requested_index:
        message = (
            f"requested npu:{requested_index}, but torch.npu.device_count() returned {count}"
        )
        status = "blocking"
    else:
        message = (
            f"requested npu:{requested_index} is available; "
            f"torch.npu.device_count() returned {count}"
        )
        status = "pass"
    checks.append(DoctorCheck("device.npu", status, message))
    return tuple(checks)


def _parse_npu_device(device: str) -> tuple[int, str | None]:
    normalized = device.strip().lower()
    if normalized == "npu":
        return 0, None
    prefix, separator, raw_index = normalized.partition(":")
    if prefix != "npu" or separator != ":" or not raw_index.isdecimal():
        return 0, "NPU device must use 'npu' or 'npu:<non-negative-index>' syntax"
    return int(raw_index), None


__all__ = ["DoctorCheck", "DoctorReport", "inspect_environment"]
