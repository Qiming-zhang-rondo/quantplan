"""Cheap preflight checks that do not load model weights."""

from __future__ import annotations

import hashlib
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

    torch_available = importlib.util.find_spec("torch") is not None
    checks.append(
        DoctorCheck(
            code="dependency.torch",
            status="pass" if torch_available else "blocking",
            message=(
                "PyTorch is importable"
                if torch_available
                else "PyTorch is required; use the build paired with the target accelerator"
            ),
        )
    )
    if config.runtime.loader == "huggingface":
        transformers_available = importlib.util.find_spec("transformers") is not None
        checks.append(
            DoctorCheck(
                code="dependency.transformers",
                status="pass" if transformers_available else "blocking",
                message=(
                    "Transformers is importable"
                    if transformers_available
                    else "Transformers is required by runtime.loader=huggingface"
                ),
            )
        )
    else:
        module_name = config.runtime.loader.partition(":")[0]
        loader_available = importlib.util.find_spec(module_name) is not None
        checks.append(
            DoctorCheck(
                code="runtime.custom_loader",
                status="pass" if loader_available else "blocking",
                message=(
                    f"custom loader module {module_name!r} is importable"
                    if loader_available
                    else f"custom loader module {module_name!r} cannot be found"
                ),
            )
        )

    if config.device.lower().startswith("npu"):
        npu_available = importlib.util.find_spec("torch_npu") is not None
        checks.append(
            DoctorCheck(
                code="dependency.torch_npu",
                status="pass" if npu_available else "blocking",
                message=(
                    "torch_npu is importable"
                    if npu_available
                    else "an NPU device was requested but torch_npu is not importable"
                ),
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

    modelslim_available = importlib.util.find_spec("msmodelslim") is not None
    checks.append(
        DoctorCheck(
            code="dependency.msmodelslim",
            status="pass" if modelslim_available else "warning",
            message=(
                "msmodelslim is importable"
                if modelslim_available
                else "msmodelslim is not required for analysis/YAML generation; backend execution remains unassessed"
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


__all__ = ["DoctorCheck", "DoctorReport", "inspect_environment"]
