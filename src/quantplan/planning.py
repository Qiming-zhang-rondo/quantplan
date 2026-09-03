"""Convert whole-model intervention measurements into bounded plan candidates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite

from quantplan.domain import (
    ApproximateMeasurement,
    CandidateEvidence,
    DecisionGroup,
    GroupInventory,
    InterventionMeasurement,
    PlanEvaluation,
    PrecisionFormat,
    QuantPlan,
)
from quantplan.search import RiskGuidedSearch, SearchPolicy

DEFAULT_RELATIVE_COSTS: Mapping[PrecisionFormat, float] = {
    PrecisionFormat.W4A4: 0.75,
    PrecisionFormat.W4A8: 1.0,
    PrecisionFormat.W8A8: 1.5,
    PrecisionFormat.BF16: 2.5,
}


@dataclass(frozen=True)
class PlanningPolicy:
    """Conservative defaults for the first runnable planning loop."""

    base_precision: PrecisionFormat = PrecisionFormat.W4A8
    max_high_precision_fraction: float = 0.25
    max_candidates: int = 8
    kl_weight: float = 0.0
    max_delta_nll: float = 0.02

    def __post_init__(self) -> None:
        if not 0 <= self.max_high_precision_fraction <= 1:
            raise ValueError("max_high_precision_fraction must be in [0, 1]")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if not isfinite(self.kl_weight) or self.kl_weight < 0:
            raise ValueError("kl_weight must be finite and non-negative")
        if not isfinite(self.max_delta_nll) or self.max_delta_nll < 0:
            raise ValueError("max_delta_nll must be finite and non-negative")


@dataclass(frozen=True)
class CandidateSet:
    """Search inputs and resulting whole-model proposals."""

    groups: tuple[DecisionGroup, ...]
    evidence: tuple[CandidateEvidence, ...]
    plans: tuple[QuantPlan, ...]
    max_estimated_cost: float
    rejected_reasons: tuple[str, ...] = ()


def build_candidate_set(
    inventories: Sequence[GroupInventory],
    measurements: Iterable[InterventionMeasurement | ApproximateMeasurement],
    *,
    allowed_formats: Sequence[PrecisionFormat],
    policy: PlanningPolicy | None = None,
    relative_costs: Mapping[PrecisionFormat, float] = DEFAULT_RELATIVE_COSTS,
) -> CandidateSet:
    """Build cost-bounded plans from one homogeneous evidence family.

    A negative observed delta NLL is treated as measurement noise for planning
    and clipped to zero.  The raw signed value remains in the measurement
    artifact and is never rewritten as a predicted loss.
    """

    if policy is None:
        policy = PlanningPolicy()
    inventory_by_id = _index_inventory(inventories)
    formats = tuple(dict.fromkeys(allowed_formats))
    if policy.base_precision not in formats:
        raise ValueError("base_precision must be included in allowed_formats")
    if PrecisionFormat.BF16 not in formats:
        formats += (PrecisionFormat.BF16,)
    _validate_costs(formats, relative_costs)

    measured = tuple(measurements)
    if not measured:
        raise ValueError("at least one intervention measurement is required")
    context_ids = {item.context_id for item in measured}
    if len(context_ids) != 1:
        raise ValueError(f"measurements use different contexts: {sorted(context_ids)}")
    context_id = next(iter(context_ids))
    evidence_types = {type(item) for item in measured}
    if len(evidence_types) != 1:
        raise ValueError("exact and approximate measurements cannot be mixed")
    token_count = measured[0].token_count
    if isinstance(measured[0], InterventionMeasurement):
        baseline_nll = measured[0].baseline_nll
        for item in measured[1:]:
            assert isinstance(item, InterventionMeasurement)
            if not isclose(item.baseline_nll, baseline_nll, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "measurements with the same context_id have different baseline NLL values"
                )
    for item in measured[1:]:
        if item.token_count != token_count:
            raise ValueError(
                "measurements with the same context_id have different token counts"
            )

    by_key: dict[
        tuple[str, PrecisionFormat], InterventionMeasurement | ApproximateMeasurement
    ] = {}
    evidence: list[CandidateEvidence] = []
    for item in measured:
        inventory = inventory_by_id.get(item.group_id)
        if inventory is None:
            raise ValueError(f"measurement refers to unknown group {item.group_id!r}")
        if item.module_names != inventory.module_names:
            raise ValueError(f"module inventory drift for group {item.group_id!r}")
        if item.parameter_count != inventory.parameter_count:
            raise ValueError(f"parameter count drift for group {item.group_id!r}")
        if (
            inventory.allowed_formats is not None
            and item.precision not in inventory.allowed_formats
        ):
            continue
        if item.precision not in formats:
            raise ValueError(
                f"measurement precision {item.precision.value!r} is not allowed"
            )
        key = (item.group_id, item.precision)
        if key in by_key:
            raise ValueError(
                f"duplicate measurement for {item.group_id}/{item.precision.value}"
            )
        by_key[key] = item
        warnings = list(item.warnings)
        if isinstance(item, InterventionMeasurement):
            signed_delta = item.delta_nll
            if signed_delta < 0:
                warnings.append("negative delta NLL clipped to zero for proposal ranking")
            kl = item.kl_divergence or 0.0
            risk = max(0.0, signed_delta) + policy.kl_weight * kl
        else:
            if policy.kl_weight:
                warnings.append("KL weight ignored for approximate fast evidence")
            risk = item.estimated_risk
        evidence.append(
            CandidateEvidence(
                group_id=item.group_id,
                precision=item.precision,
                context_id=context_id,
                risk=risk,
                estimated_cost=(
                    item.parameter_count * float(relative_costs[item.precision])
                ),
                warnings=tuple(warnings),
            )
        )

    groups = tuple(
        DecisionGroup(
            group_id=inventory.group_id,
            module_names=inventory.module_names,
            kind=inventory.kind,
            allowed_formats=_inventory_formats(inventory, formats),
        )
        for inventory in inventories
    )
    groups_by_id = {group.group_id: group for group in groups}
    for inventory in inventories:
        group = groups_by_id[inventory.group_id]
        base_precision = (
            policy.base_precision
            if policy.base_precision in group.allowed_formats
            else _cheapest_allowed_format(group, relative_costs)
        )
        if (inventory.group_id, base_precision) not in by_key:
            raise ValueError(
                f"missing base measurement for {inventory.group_id}/"
                f"{base_precision.value}"
            )
        if (inventory.group_id, PrecisionFormat.BF16) not in by_key:
            evidence.append(
                CandidateEvidence(
                    group_id=inventory.group_id,
                    precision=PrecisionFormat.BF16,
                    context_id=context_id,
                    risk=0.0,
                    estimated_cost=(
                        inventory.parameter_count
                        * float(relative_costs[PrecisionFormat.BF16])
                    ),
                    warnings=("BF16 anchor synthesized from the shared baseline",),
                )
            )

    available_by_group = {
        group_id: {precision for (candidate_group, precision) in by_key if candidate_group == group_id}
        | {PrecisionFormat.BF16}
        for group_id in inventory_by_id
    }
    groups = tuple(
        DecisionGroup(
            group_id=group.group_id,
            module_names=group.module_names,
            kind=group.kind,
            allowed_formats=tuple(
                precision
                for precision in formats
                if precision in available_by_group[group.group_id]
            ),
        )
        for group in groups
    )

    base_cost = sum(
        inventory.parameter_count
        * float(
            relative_costs[
                (
                    policy.base_precision
                    if group.allowed_formats and policy.base_precision in group.allowed_formats
                    else _cheapest_allowed_format(group, relative_costs)
                )
            ]
        )
        for inventory, group in zip(inventories, groups, strict=True)
    )
    ceiling_cost = sum(
        inventory.parameter_count
        * max(float(relative_costs[item]) for item in group.allowed_formats)
        for inventory, group in zip(inventories, groups, strict=True)
    )
    max_cost = base_cost + policy.max_high_precision_fraction * (
        ceiling_cost - base_cost
    )
    search_result = RiskGuidedSearch().propose(
        groups,
        evidence,
        SearchPolicy(
            max_estimated_cost=max_cost,
            max_candidates=policy.max_candidates,
        ),
    )
    return CandidateSet(
        groups=groups,
        evidence=tuple(evidence),
        plans=search_result.candidates,
        max_estimated_cost=max_cost,
        rejected_reasons=search_result.rejected_reasons,
    )


def plan_id(plan: QuantPlan) -> str:
    """Return a stable short ID for an assignment and its evidence context."""

    payload = {
        "assignments": {
            key: value.value for key, value in sorted(plan.assignments.items())
        },
        "context_id": plan.context_id,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"plan-{digest[:12]}"


def choose_lowest_cost_passing_plan(
    plans: Sequence[QuantPlan],
    evaluations: Iterable[PlanEvaluation],
    *,
    max_delta_nll: float,
) -> tuple[QuantPlan, PlanEvaluation] | None:
    """Choose the cheapest measured plan that passes the frozen NLL gate."""

    measured = tuple(evaluations)
    by_id = {item.plan_id: item for item in measured}
    if len(by_id) != len(measured):
        raise ValueError("duplicate plan evaluation")
    expected_ids = {plan_id(plan) for plan in plans}
    actual_ids = set(by_id)
    if actual_ids != expected_ids:
        raise ValueError(
            "plan evaluations must cover the candidate plans exactly: "
            f"missing={sorted(expected_ids - actual_ids)}, "
            f"extra={sorted(actual_ids - expected_ids)}"
        )
    passing: list[tuple[QuantPlan, PlanEvaluation]] = []
    for plan in plans:
        evaluation = by_id.get(plan_id(plan))
        if evaluation is None:
            raise AssertionError("validated plan ID disappeared from the evaluation index")
        if evaluation.context_id != plan.context_id:
            raise ValueError(
                f"evaluation context mismatch for {evaluation.plan_id}: "
                f"plan={plan.context_id!r}, evaluation={evaluation.context_id!r}"
            )
        if evaluation.passes(max_delta_nll=max_delta_nll):
            passing.append((plan, evaluation))
    if not passing:
        return None
    return min(passing, key=lambda item: (item[0].estimated_cost, item[1].delta_nll))


def _index_inventory(
    inventories: Sequence[GroupInventory],
) -> dict[str, GroupInventory]:
    if not inventories:
        raise ValueError("at least one group inventory is required")
    indexed: dict[str, GroupInventory] = {}
    for item in inventories:
        if item.group_id in indexed:
            raise ValueError(f"duplicate group inventory {item.group_id!r}")
        indexed[item.group_id] = item
    return indexed


def _inventory_formats(
    inventory: GroupInventory,
    formats: Sequence[PrecisionFormat],
) -> tuple[PrecisionFormat, ...]:
    allowed = tuple(
        precision
        for precision in formats
        if inventory.allowed_formats is None or precision in inventory.allowed_formats
    )
    if PrecisionFormat.BF16 not in allowed:
        allowed += (PrecisionFormat.BF16,)
    return allowed


def _cheapest_allowed_format(
    group: DecisionGroup,
    relative_costs: Mapping[PrecisionFormat, float],
) -> PrecisionFormat:
    return min(
        group.allowed_formats,
        key=lambda precision: (float(relative_costs[precision]), precision.value),
    )


def _validate_costs(
    formats: Sequence[PrecisionFormat],
    costs: Mapping[PrecisionFormat, float],
) -> None:
    for precision in formats:
        if precision not in costs:
            raise ValueError(f"missing relative cost for {precision.value}")
        value = float(costs[precision])
        if not isfinite(value) or value <= 0:
            raise ValueError(f"relative cost for {precision.value} must be positive")
