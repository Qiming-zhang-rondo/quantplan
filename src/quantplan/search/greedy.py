from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping, Sequence

from quantplan.domain.models import (
    CandidateEvidence,
    DecisionGroup,
    PrecisionFormat,
    QuantPlan,
    SearchResult,
)


@dataclass(frozen=True)
class SearchPolicy:
    """Cost and diagnostic guards for a proposal pass.

    Risk is not treated as an additive loss budget. ``max_estimated_cost`` is
    the comparable constraint; the search uses within-group risk reduction to
    prioritize upgrades and produces proposals for whole-model validation.
    """

    max_estimated_cost: float
    max_candidates: int = 3
    max_post_norm_p95: float | None = None
    max_route_flip_rate: float | None = None
    max_clipping_rate: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.max_estimated_cost) or self.max_estimated_cost < 0:
            raise ValueError("max_estimated_cost must be finite and non-negative")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least one")
        for name in (
            "max_post_norm_p95",
            "max_route_flip_rate",
            "max_clipping_rate",
        ):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("max_route_flip_rate", "max_clipping_rate"):
            value = getattr(self, name)
            if value is not None and value > 1:
                raise ValueError(f"{name} must be at most one")


@dataclass(frozen=True)
class _Upgrade:
    group_id: str
    source: PrecisionFormat
    target: PrecisionFormat
    risk_reduction: float
    cost_increase: float

    @property
    def value(self) -> float:
        if self.cost_increase == 0:
            return float("inf") if self.risk_reduction > 0 else 0.0
        return self.risk_reduction / self.cost_increase


class RiskGuidedSearch:
    """Experimental gain-per-cost proposer with structural decision groups.

    This class deliberately does not claim global optimality. It builds a small
    cost-bounded candidate set for downstream KL/NLL validation.
    """

    _ORDER = (
        PrecisionFormat.W4A4,
        PrecisionFormat.W4A8,
        PrecisionFormat.W8A8,
        PrecisionFormat.BF16,
    )

    def propose(
        self,
        groups: Sequence[DecisionGroup],
        evidence: Iterable[CandidateEvidence],
        policy: SearchPolicy,
    ) -> SearchResult:
        evidence_by_group = self._index_evidence(evidence)
        assignment: dict[str, PrecisionFormat] = {}
        risks: dict[str, float] = {}
        costs: dict[str, float] = {}
        rejected: list[str] = []

        for group in groups:
            available = evidence_by_group.get(group.group_id, {})
            if group.locked:
                legal = (
                    [group.fixed_format]
                    if group.fixed_format in available
                    and self._passes_guards(available[group.fixed_format], policy)
                    else []
                )
            else:
                legal = [
                    fmt
                    for fmt in self._ORDER
                    if fmt in group.allowed_formats
                    and fmt in available
                    and self._passes_guards(available[fmt], policy)
                ]
            if not legal:
                rejected.append(f"{group.group_id}: no candidate passes support/evidence guards")
                continue
            selected = min(
                legal,
                key=lambda fmt: (available[fmt].estimated_cost, self._ORDER.index(fmt)),
            )
            assignment[group.group_id] = selected
            risks[group.group_id] = available[selected].risk
            costs[group.group_id] = available[selected].estimated_cost

        if len(assignment) != len(groups):
            return SearchResult(candidates=(), rejected_reasons=tuple(rejected))

        if sum(costs.values()) > policy.max_estimated_cost:
            rejected.append("minimum legal plan exceeds the estimated-cost budget")
            return SearchResult(candidates=(), rejected_reasons=tuple(rejected))

        plans = [self._make_plan(assignment, risks, costs, evidence_by_group)]
        while True:
            remaining_cost = policy.max_estimated_cost - sum(costs.values())
            upgrades = self._upgrades(
                groups,
                assignment,
                evidence_by_group,
                policy,
                remaining_cost,
            )
            if not upgrades:
                break
            best = max(upgrades, key=lambda item: (item.value, item.risk_reduction))
            candidate = evidence_by_group[best.group_id][best.target]
            assignment[best.group_id] = best.target
            risks[best.group_id] = candidate.risk
            costs[best.group_id] = candidate.estimated_cost
            plans.append(self._make_plan(assignment, risks, costs, evidence_by_group))

        selected_plans = self._select_representative_plans(plans, policy.max_candidates)
        return SearchResult(candidates=tuple(selected_plans), rejected_reasons=tuple(rejected))

    @staticmethod
    def _index_evidence(
        evidence: Iterable[CandidateEvidence],
    ) -> dict[str, dict[PrecisionFormat, CandidateEvidence]]:
        indexed: dict[str, dict[PrecisionFormat, CandidateEvidence]] = {}
        contexts: set[str] = set()
        for item in evidence:
            contexts.add(item.context_id)
            group_items = indexed.setdefault(item.group_id, {})
            if item.precision in group_items:
                raise ValueError(
                    f"duplicate evidence for {item.group_id}/{item.precision.value}"
                )
            group_items[item.precision] = item
        if len(contexts) > 1:
            raise ValueError(f"evidence from different contexts cannot be mixed: {contexts}")
        return indexed

    @staticmethod
    def _passes_guards(item: CandidateEvidence, policy: SearchPolicy) -> bool:
        checks = (
            (item.post_norm_p95, policy.max_post_norm_p95),
            (item.route_flip_rate, policy.max_route_flip_rate),
            (item.clipping_rate, policy.max_clipping_rate),
        )
        return all(
            limit is None or (value is not None and value <= limit)
            for value, limit in checks
        )

    def _upgrades(
        self,
        groups: Sequence[DecisionGroup],
        assignment: Mapping[str, PrecisionFormat],
        evidence: Mapping[str, Mapping[PrecisionFormat, CandidateEvidence]],
        policy: SearchPolicy,
        remaining_cost: float,
    ) -> list[_Upgrade]:
        upgrades: list[_Upgrade] = []
        for group in groups:
            if group.locked:
                continue
            current = assignment[group.group_id]
            current_index = self._ORDER.index(current)
            available = evidence[group.group_id]
            current_item = available[current]
            current_risk = current_item.risk
            for target in self._ORDER[current_index + 1 :]:
                if target not in group.allowed_formats or target not in available:
                    continue
                target_item = available[target]
                if not self._passes_guards(target_item, policy):
                    continue
                target_risk = target_item.risk
                reduction = current_risk - target_risk
                cost_increase = target_item.estimated_cost - current_item.estimated_cost
                if reduction > 0 and 0 <= cost_increase <= remaining_cost:
                    upgrades.append(
                        _Upgrade(
                            group_id=group.group_id,
                            source=current,
                            target=target,
                            risk_reduction=reduction,
                            cost_increase=cost_increase,
                        )
                    )
        return upgrades

    @staticmethod
    def _make_plan(
        assignment: Mapping[str, PrecisionFormat],
        risks: Mapping[str, float],
        costs: Mapping[str, float],
        evidence: Mapping[str, Mapping[PrecisionFormat, CandidateEvidence]],
    ) -> QuantPlan:
        context_ids = {
            evidence[group_id][precision].context_id
            for group_id, precision in assignment.items()
        }
        context_id = next(iter(context_ids)) if context_ids else "unknown"
        return QuantPlan(
            assignments=dict(assignment),
            context_id=context_id,
            estimated_cost=sum(costs.values()),
            proposal_score=sum(risks.values()),
            warnings=(
                "proposal_score is a non-calibrated aggregate, not predicted loss",
                "proposal-only: whole-model KL/NLL validation is required",
            ),
        )

    @staticmethod
    def _select_representative_plans(plans: Sequence[QuantPlan], limit: int) -> list[QuantPlan]:
        if len(plans) <= limit:
            return list(plans)
        if limit == 1:
            # A one-candidate request should return the most risk-recovered
            # proposal, not the cheapest unvalidated starting point.
            return [plans[-1]]
        indexes = {0, len(plans) - 1}
        if limit > 2:
            for step in range(1, limit - 1):
                indexes.add(round(step * (len(plans) - 1) / (limit - 1)))
        return [plans[index] for index in sorted(indexes)[:limit]]
