from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Mapping, Sequence


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
