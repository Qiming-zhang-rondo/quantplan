from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import isclose, isfinite


class PrecisionFormat(str, Enum):
    """Logical candidates. A backend adapter decides whether each one is executable."""

    BF16 = "bf16"
    W8A8 = "w8a8"
    W4A8 = "w4a8"
    W4A4 = "w4a4"


class SupportStatus(str, Enum):
    SUPPORTED_VERIFIED = "supported_verified"
    SUPPORTED_EXPERIMENTAL = "supported_experimental"
    KNOWN_UNSUPPORTED = "known_unsupported"
    UNKNOWN = "unknown"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class DecisionGroup:
    """The smallest set of modules that must receive one precision decision."""

    group_id: str
    module_names: tuple[str, ...]
    kind: str
    allowed_formats: tuple[PrecisionFormat, ...]
    locked: bool = False
    lock_reason: str | None = None
    fixed_format: PrecisionFormat | None = None

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id must not be empty")
        if not self.module_names:
            raise ValueError(f"decision group {self.group_id!r} has no modules")
        if not self.allowed_formats:
            raise ValueError(f"decision group {self.group_id!r} has no allowed formats")
        if self.locked:
            if not self.lock_reason:
                raise ValueError("locked groups must include a lock_reason")
            if self.fixed_format is None:
                raise ValueError("locked groups must include a fixed_format")
            if self.fixed_format not in self.allowed_formats:
                raise ValueError("fixed_format must be present in allowed_formats")
        elif self.fixed_format is not None:
            raise ValueError("fixed_format is only valid for locked groups")


@dataclass(frozen=True)
class CandidateEvidence:
    """Evidence for one candidate in one explicit analysis context.

    ``risk`` is a proposal signal, not a prediction of end-to-end NLL change.
    """

    group_id: str
    precision: PrecisionFormat
    context_id: str
    risk: float
    estimated_cost: float
    post_norm_p95: float | None = None
    route_flip_rate: float | None = None
    clipping_rate: float | None = None
    confidence: float | None = None
    warnings: tuple[str, ...] = ()
    latency_seconds: float | None = None
    latency_lower_bound: float | None = None
    latency_upper_bound: float | None = None
    latency_confidence: float | None = None
    latency_model: str | None = None
    latency_components: Mapping[str, float] = field(default_factory=dict)
    latency_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isfinite(self.risk) or self.risk < 0:
            raise ValueError("risk must be finite and non-negative")
        if not isfinite(self.estimated_cost) or self.estimated_cost < 0:
            raise ValueError("estimated_cost must be finite and non-negative")
        if self.post_norm_p95 is not None and (
            not isfinite(self.post_norm_p95) or self.post_norm_p95 < 0
        ):
            raise ValueError("post_norm_p95 must be finite and non-negative")
        for name in ("route_flip_rate", "clipping_rate", "confidence"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in (
            "latency_seconds",
            "latency_lower_bound",
            "latency_upper_bound",
        ):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            self.latency_lower_bound is not None
            and self.latency_upper_bound is not None
            and self.latency_lower_bound > self.latency_upper_bound
        ):
            raise ValueError("latency_lower_bound must not exceed latency_upper_bound")
        if self.latency_confidence is not None and (
            not isfinite(self.latency_confidence)
            or not 0 <= self.latency_confidence <= 1
        ):
            raise ValueError("latency_confidence must be finite and in [0, 1]")
        if self.latency_model is not None and not self.latency_model.strip():
            raise ValueError("latency_model must be null or a non-empty string")
        if not isinstance(self.latency_components, Mapping):
            raise TypeError("latency_components must be a mapping")
        for key, value in self.latency_components.items():
            if not isinstance(key, str) or not key:
                raise ValueError("latency_components keys must be non-empty strings")
            if not isfinite(float(value)) or float(value) < 0:
                raise ValueError("latency_components values must be finite and non-negative")


@dataclass(frozen=True)
class GroupInventory:
    """A runtime-discovered decision group and its quantizable module inventory."""

    group_id: str
    module_names: tuple[str, ...]
    parameter_count: int
    kind: str = "transformer_block"
    warnings: tuple[str, ...] = ()
    backend_patterns: tuple[str, ...] = ()
    quantization_domain: str = "independent"
    allowed_formats: tuple[PrecisionFormat, ...] | None = None
    source_precision: PrecisionFormat | None = None
    operator_shapes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id must not be empty")
        if not self.module_names:
            raise ValueError(f"group inventory {self.group_id!r} has no modules")
        if self.parameter_count < 1:
            raise ValueError("parameter_count must be positive")
        if self.backend_patterns and (
            len(self.backend_patterns) != len(self.module_names)
            or any(not pattern for pattern in self.backend_patterns)
        ):
            raise ValueError(
                "backend_patterns must be empty or align one-to-one with module_names"
            )
        if not self.quantization_domain:
            raise ValueError("quantization_domain must not be empty")
        if self.allowed_formats is not None:
            if not self.allowed_formats:
                raise ValueError("allowed_formats must not be empty when provided")
            if any(not isinstance(item, PrecisionFormat) for item in self.allowed_formats):
                raise ValueError("allowed_formats must contain PrecisionFormat values")
            if len(set(self.allowed_formats)) != len(self.allowed_formats):
                raise ValueError("allowed_formats must not contain duplicates")
        if self.source_precision is not None and not isinstance(
            self.source_precision, PrecisionFormat
        ):
            raise ValueError("source_precision must be a PrecisionFormat or None")
        if not isinstance(self.operator_shapes, Mapping):
            raise TypeError("operator_shapes must be a mapping")
        for name, shape in self.operator_shapes.items():
            if not isinstance(name, str) or not name:
                raise ValueError("operator_shapes keys must be non-empty strings")
            if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
                raise TypeError("operator_shapes values must be shape sequences")
            normalized_shape = tuple(int(dimension) for dimension in shape)
            if not normalized_shape or any(dimension < 1 for dimension in normalized_shape):
                raise ValueError("operator_shapes dimensions must be positive")
        object.__setattr__(
            self,
            "operator_shapes",
            {str(name): tuple(int(dimension) for dimension in shape) for name, shape in self.operator_shapes.items()},
        )

    @property
    def effective_backend_patterns(self) -> tuple[str, ...]:
        return self.backend_patterns or self.module_names


@dataclass(frozen=True)
class InterventionMeasurement:
    """Whole-model evidence from quantizing one group with one candidate.

    ``candidate_nll`` is measured after the complete model forward, so the
    value includes propagation through residual paths, normalization layers
    and all downstream blocks.  It is still conditional on the declared
    proxy/backend quantizer and workload.
    """

    group_id: str
    precision: PrecisionFormat
    context_id: str
    module_names: tuple[str, ...]
    parameter_count: int
    baseline_nll: float
    candidate_nll: float
    token_count: int
    elapsed_seconds: float
    quantizer_id: str
    kl_divergence: float | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.group_id or not self.context_id or not self.quantizer_id:
            raise ValueError("group_id, context_id and quantizer_id must not be empty")
        if not self.module_names:
            raise ValueError("module_names must not be empty")
        if self.parameter_count < 1:
            raise ValueError("parameter_count must be positive")
        if self.token_count < 1:
            raise ValueError("token_count must be positive")
        for name in ("baseline_nll", "candidate_nll", "elapsed_seconds"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.kl_divergence is not None and (
            not isfinite(self.kl_divergence) or self.kl_divergence < 0
        ):
            raise ValueError("kl_divergence must be finite and non-negative")

    @property
    def delta_nll(self) -> float:
        """Observed candidate-minus-BF16 teacher-forced NLL per token."""

        return self.candidate_nll - self.baseline_nll


@dataclass(frozen=True)
class ApproximateMeasurement:
    """Fast proposal evidence; never an observed whole-model loss change.

    ``estimated_risk`` is produced from a local candidate perturbation and a
    downstream loss gradient.  It may rank candidates, but it must be followed
    by :class:`PlanEvaluation` before a deployable YAML can be emitted.
    """

    group_id: str
    precision: PrecisionFormat
    context_id: str
    module_names: tuple[str, ...]
    parameter_count: int
    estimated_risk: float
    token_count: int
    elapsed_seconds: float
    metric_id: str
    mean_first_order_effect: float
    rms_first_order_effect: float
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.group_id or not self.context_id or not self.metric_id:
            raise ValueError("group_id, context_id and metric_id must not be empty")
        if not self.module_names:
            raise ValueError("module_names must not be empty")
        if self.parameter_count < 1:
            raise ValueError("parameter_count must be positive")
        if self.token_count < 1:
            raise ValueError("token_count must be positive")
        for name in (
            "estimated_risk",
            "elapsed_seconds",
            "rms_first_order_effect",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isfinite(self.mean_first_order_effect):
            raise ValueError("mean_first_order_effect must be finite")


@dataclass(frozen=True)
class StatisticalMeasurement:
    """No-replay gradient-moment ranking evidence, not Fisher or delta NLL.

    Separate from ApproximateMeasurement: no signed Taylor effect or local
    candidate output was measured. Components are nonnegative factorized
    noise energies at the declared quantization boundaries.
    """

    group_id: str
    precision: PrecisionFormat
    context_id: str
    module_names: tuple[str, ...]
    parameter_count: int
    estimated_risk: float
    token_count: int
    elapsed_seconds: float
    metric_id: str
    weight_risk: float
    activation_risk: float
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.group_id or not self.context_id or not self.metric_id:
            raise ValueError("group_id, context_id and metric_id must not be empty")
        if not self.module_names or self.parameter_count < 1 or self.token_count < 1:
            raise ValueError("module_names and positive parameter/token counts are required")
        for name in ("estimated_risk", "weight_risk", "activation_risk", "elapsed_seconds"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isclose(
            self.estimated_risk, self.weight_risk + self.activation_risk,
            rel_tol=1e-9, abs_tol=1e-15,
        ):
            raise ValueError("estimated_risk must equal weight_risk + activation_risk")


@dataclass(frozen=True)
class DirectionalMeasurement:
    """Signed shared-projection Taylor evidence, not observed loss or Fisher.

    A common nonnegative offset for every candidate in a group preserves all
    within-group differences for the nonnegative-risk solver. Cross-group
    sensitivity comparisons must use mean_first_order_effect, not shifted risk.
    """

    group_id: str
    precision: PrecisionFormat
    context_id: str
    module_names: tuple[str, ...]
    parameter_count: int
    estimated_risk: float
    token_count: int
    elapsed_seconds: float
    metric_id: str
    mean_first_order_effect: float
    ranking_offset: float
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.group_id or not self.context_id or not self.metric_id:
            raise ValueError("group_id, context_id and metric_id must not be empty")
        if not self.module_names or self.parameter_count < 1 or self.token_count < 1:
            raise ValueError("module_names and positive parameter/token counts are required")
        for name in ("estimated_risk", "ranking_offset", "elapsed_seconds"):
            if not isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isfinite(self.mean_first_order_effect):
            raise ValueError("mean_first_order_effect must be finite")
        if not isclose(
            self.estimated_risk, self.mean_first_order_effect + self.ranking_offset,
            rel_tol=1e-9, abs_tol=1e-15,
        ):
            raise ValueError("estimated_risk must equal signed effect plus ranking_offset")
        if self.precision is PrecisionFormat.BF16 and self.mean_first_order_effect != 0:
            raise ValueError("BF16 reference signed effect must be zero")


@dataclass(frozen=True)
class PlanEvaluation:
    """Whole-model validation of one complete mixed-precision assignment."""

    plan_id: str
    context_id: str
    baseline_nll: float
    candidate_nll: float
    token_count: int
    elapsed_seconds: float
    kl_divergence: float | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.plan_id or not self.context_id:
            raise ValueError("plan_id and context_id must not be empty")
        if self.token_count < 1:
            raise ValueError("token_count must be positive")
        for name in ("baseline_nll", "candidate_nll", "elapsed_seconds"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.kl_divergence is not None and (
            not isfinite(self.kl_divergence) or self.kl_divergence < 0
        ):
            raise ValueError("kl_divergence must be finite and non-negative")

    @property
    def delta_nll(self) -> float:
        return self.candidate_nll - self.baseline_nll

    def passes(self, *, max_delta_nll: float, tolerance: float = 1e-12) -> bool:
        if not isfinite(max_delta_nll) or max_delta_nll < 0:
            raise ValueError("max_delta_nll must be finite and non-negative")
        return self.delta_nll < max_delta_nll or isclose(
            self.delta_nll,
            max_delta_nll,
            abs_tol=tolerance,
        )


@dataclass(frozen=True)
class QuantPlan:
    assignments: Mapping[str, PrecisionFormat]
    context_id: str
    estimated_cost: float
    proposal_score: float
    evidence_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def validate_groups(self, groups: Sequence[DecisionGroup]) -> None:
        expected = {group.group_id for group in groups}
        actual = set(self.assignments)
        if expected != actual:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"plan/group mismatch: missing={missing}, extra={extra}")
        for group in groups:
            selected = self.assignments[group.group_id]
            if selected not in group.allowed_formats:
                raise ValueError(
                    f"{group.group_id!r} does not allow precision {selected.value!r}"
                )
            if group.locked and selected is not group.fixed_format:
                raise ValueError(
                    f"{group.group_id!r} is locked to {group.fixed_format.value!r}"
                )


@dataclass(frozen=True)
class SearchResult:
    candidates: tuple[QuantPlan, ...]
    rejected_reasons: tuple[str, ...] = field(default_factory=tuple)
