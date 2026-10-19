"""Fast, hardware-aware latency estimates for mixed-precision proposals.

The provider deliberately estimates *relative* deployment cost.  It is not a
replacement for a target-engine benchmark: kernel dispatch, fusion and
communication are represented by configurable overheads and optional
calibration entries.  This keeps the fast path cheap while making its
assumptions explicit in the artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite, prod
from pathlib import Path
from typing import Any, Protocol

from quantplan.domain.models import GroupInventory, PrecisionFormat

LATENCY_SCHEMA = "quantplan.latency/v1"
CALIBRATION_SCHEMA = "quantplan.latency-calibration/v1"


@dataclass(frozen=True)
class LatencyEstimate:
    """A candidate latency estimate and the assumptions behind it."""

    seconds: float
    lower_bound: float
    upper_bound: float
    confidence: float
    model: str
    components: Mapping[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("seconds", "lower_bound", "upper_bound"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound must not exceed upper_bound")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        if not self.model.strip():
            raise ValueError("model must not be empty")


class LatencyProvider(Protocol):
    """Plugin contract consumed by the architecture-neutral planner."""

    model_id: str

    def estimate(
        self,
        inventory: GroupInventory,
        precision: PrecisionFormat,
    ) -> LatencyEstimate: ...

    def summary(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class RooflineProfile:
    """Effective hardware parameters, intentionally separate per deployment."""

    name: str
    memory_bandwidth_gbps: float = 1000.0
    compute_tflops: float = 100.0
    memory_efficiency: float = 0.70
    compute_efficiency: float = 0.50
    launch_overhead_us: float = 2.0
    qdq_overhead_us: float = 0.0
    layout_overhead_us: float = 0.0
    fallback_overhead_us: float = 0.0
    uncertainty: float = 0.25

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("latency profile name must not be empty")
        for name in ("memory_bandwidth_gbps", "compute_tflops"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "launch_overhead_us",
            "qdq_overhead_us",
            "layout_overhead_us",
            "fallback_overhead_us",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("memory_efficiency", "compute_efficiency"):
            value = getattr(self, name)
            if not isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if not isfinite(self.uncertainty) or not 0 <= self.uncertainty < 1:
            raise ValueError("uncertainty must be finite and in [0, 1)")


def _bits(precision: PrecisionFormat) -> int:
    return {
        PrecisionFormat.BF16: 16,
        PrecisionFormat.W8A8: 8,
        PrecisionFormat.W4A8: 4,
        PrecisionFormat.W4A4: 4,
    }[precision]


def _activation_bits(precision: PrecisionFormat) -> int:
    return 16 if precision is PrecisionFormat.BF16 else (4 if precision is PrecisionFormat.W4A4 else 8)


@dataclass(frozen=True)
class RooflineLatencyProvider:
    """Static Roofline estimate with conservative runtime overheads.

    ``parameter_count`` is used as the shape-independent fallback.  Runtime
    plugins can provide richer shapes later without changing this interface.
    """

    profile: RooflineProfile
    deployment_mode: str = "decode_bs1"
    metadata_factor: float = 1.08
    activation_bytes_per_token: int = 2
    proxy_hardware_parameters: bool = False
    model_id: str = "quantplan.roofline/v1"

    def __post_init__(self) -> None:
        if not self.deployment_mode.strip():
            raise ValueError("deployment_mode must not be empty")
        if not isfinite(self.metadata_factor) or self.metadata_factor < 1:
            raise ValueError("metadata_factor must be finite and at least one")
        if self.activation_bytes_per_token < 1:
            raise ValueError("activation_bytes_per_token must be positive")

    def estimate(
        self,
        inventory: GroupInventory,
        precision: PrecisionFormat,
    ) -> LatencyEstimate:
        shape_elements = (
            sum(prod(shape) for shape in inventory.operator_shapes.values())
            if inventory.operator_shapes
            else inventory.parameter_count
        )
        weight_bytes = shape_elements * _bits(precision) / 8
        weight_bytes *= self.metadata_factor if precision is not PrecisionFormat.BF16 else 1.0
        # One decode token is the conservative default.  A backend plugin can
        # replace this provider when batch/prefill shape is known precisely.
        activation_bytes = self.activation_bytes_per_token * _activation_bits(precision) / 8
        total_bytes = weight_bytes + activation_bytes
        memory_seconds = total_bytes / (
            self.profile.memory_bandwidth_gbps
            * 1_000_000_000
            * self.profile.memory_efficiency
        )
        # A shape-independent FLOP fallback: two operations per parameter for
        # one GEMV token.  This is only relevant for unusually low bandwidth.
        flops = shape_elements * 2
        compute_seconds = flops / (
            self.profile.compute_tflops
            * 1_000_000_000_000
            * self.profile.compute_efficiency
        )
        ideal = max(memory_seconds, compute_seconds)
        launch = self.profile.launch_overhead_us / 1_000_000
        qdq = (
            self.profile.qdq_overhead_us / 1_000_000
            if precision is not PrecisionFormat.BF16
            else 0.0
        )
        layout = (
            self.profile.layout_overhead_us / 1_000_000
            if precision is not PrecisionFormat.BF16
            else 0.0
        )
        # A non-independent domain is a fused contract.  Without a backend
        # capability plugin we add a small uncertainty rather than pretending
        # that every internal Linear can be independently accelerated.
        domain_warning = ()
        domain_penalty = 0.0
        if inventory.quantization_domain != "independent":
            domain_warning = (
                "fused quantization domain estimated atomically; backend resolver validation required",
            )
            domain_penalty = self.profile.fallback_overhead_us / 1_000_000
        seconds = ideal + launch + qdq + layout + domain_penalty
        hardware_warning = (
            ("hardware bandwidth/compute are proxy defaults; provide a target profile",)
            if self.proxy_hardware_parameters
            else ()
        )
        uncertainty = self.profile.uncertainty + (0.15 if domain_warning else 0.0)
        uncertainty += 0.20 if self.proxy_hardware_parameters else 0.0
        uncertainty = min(0.90, uncertainty)
        return LatencyEstimate(
            seconds=seconds,
            lower_bound=seconds * (1 - uncertainty),
            upper_bound=seconds * (1 + uncertainty),
            confidence=max(0.05, 1 - uncertainty),
            model=self.model_id,
            components={
                "ideal_memory_seconds": memory_seconds,
                "ideal_compute_seconds": compute_seconds,
                "launch_seconds": launch,
                "qdq_seconds": qdq,
                "layout_seconds": layout,
                "fallback_seconds": domain_penalty,
            },
            warnings=(
                f"deployment_mode={self.deployment_mode}",
                "static estimate; no target-engine execution",
                *hardware_warning,
                *(
                    ("operator shapes supplied by runtime inventory",)
                    if inventory.operator_shapes
                    else ("shape metadata unavailable; parameter count used",)
                ),
                *domain_warning,
            ),
        )

    def summary(self) -> Mapping[str, Any]:
        return {
            "schema_version": LATENCY_SCHEMA,
            "model": self.model_id,
            "mode": "static",
            "deployment_mode": self.deployment_mode,
            "hardware_profile": self.profile.name,
            "memory_bandwidth_gbps": self.profile.memory_bandwidth_gbps,
            "compute_tflops": self.profile.compute_tflops,
            "uncertainty": self.profile.uncertainty,
        }


@dataclass(frozen=True)
class CalibratedLatencyProvider:
    """Roofline provider corrected by sparse operator calibration entries."""

    base: RooflineLatencyProvider
    entries: Mapping[str, float]
    model_id: str = "quantplan.calibrated_roofline/v1"

    def estimate(
        self,
        inventory: GroupInventory,
        precision: PrecisionFormat,
    ) -> LatencyEstimate:
        key = f"{inventory.group_id}|{precision.value}"
        base = self.base.estimate(inventory, precision)
        calibrated = self.entries.get(key)
        if calibrated is None:
            return LatencyEstimate(
                seconds=base.seconds,
                lower_bound=base.lower_bound,
                upper_bound=base.upper_bound,
                confidence=max(0.05, base.confidence * 0.75),
                model=self.model_id,
                components=base.components,
                warnings=(*base.warnings, f"no calibration entry for {key}"),
            )
        if calibrated < 0 or not isfinite(calibrated):
            raise ValueError(f"calibration entry {key!r} must be finite and non-negative")
        uncertainty = 0.10
        return LatencyEstimate(
            seconds=calibrated,
            lower_bound=calibrated * (1 - uncertainty),
            upper_bound=calibrated * (1 + uncertainty),
            confidence=0.90,
            model=self.model_id,
            components={**base.components, "calibrated_seconds": calibrated},
            warnings=(*base.warnings, f"calibration_key={key}"),
        )

    def summary(self) -> Mapping[str, Any]:
        return {
            **self.base.summary(),
            "mode": "calibrated",
            "model": self.model_id,
            "calibration_entries": len(self.entries),
        }


def load_calibration(path: str | Path) -> Mapping[str, float]:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read latency calibration {source}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise TypeError("latency calibration must be a JSON object")
    if document.get("schema_version") != CALIBRATION_SCHEMA:
        raise ValueError(
            f"unsupported latency calibration schema {document.get('schema_version')!r}"
        )
    raw_entries = document.get("entries")
    if not isinstance(raw_entries, Mapping):
        raise TypeError("latency calibration entries must be a mapping")
    return {str(key): float(value) for key, value in raw_entries.items()}


def build_latency_provider(
    *,
    mode: str,
    deployment_mode: str,
    hardware_profile: str,
    memory_bandwidth_gbps: float | None = None,
    compute_tflops: float | None = None,
    memory_efficiency: float = 0.70,
    compute_efficiency: float = 0.50,
    launch_overhead_us: float = 2.0,
    qdq_overhead_us: float = 0.0,
    layout_overhead_us: float = 0.0,
    fallback_overhead_us: float = 0.0,
    uncertainty: float = 0.25,
    calibration_path: str | Path | None = None,
) -> LatencyProvider | None:
    normalized = mode.strip().lower()
    if normalized in {"off", "none", "disabled"}:
        return None
    if normalized not in {"static", "calibrated"}:
        raise ValueError("latency mode must be off, static or calibrated")
    profile = RooflineProfile(
        name=hardware_profile.strip() or "unknown",
        memory_bandwidth_gbps=memory_bandwidth_gbps or 1000.0,
        compute_tflops=compute_tflops or 100.0,
        memory_efficiency=memory_efficiency,
        compute_efficiency=compute_efficiency,
        launch_overhead_us=launch_overhead_us,
        qdq_overhead_us=qdq_overhead_us,
        layout_overhead_us=layout_overhead_us,
        fallback_overhead_us=fallback_overhead_us,
        uncertainty=uncertainty,
    )
    base = RooflineLatencyProvider(
        profile,
        deployment_mode=deployment_mode,
        proxy_hardware_parameters=(memory_bandwidth_gbps is None or compute_tflops is None),
    )
    if normalized == "static":
        return base
    if calibration_path is None:
        raise ValueError("calibrated latency mode requires calibration_path")
    return CalibratedLatencyProvider(base, load_calibration(calibration_path))


__all__ = [
    "CALIBRATION_SCHEMA",
    "LATENCY_SCHEMA",
    "CalibratedLatencyProvider",
    "LatencyEstimate",
    "LatencyProvider",
    "RooflineLatencyProvider",
    "RooflineProfile",
    "build_latency_provider",
    "load_calibration",
]
