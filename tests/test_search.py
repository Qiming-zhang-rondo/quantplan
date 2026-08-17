from quantplan.domain.models import CandidateEvidence, DecisionGroup, PrecisionFormat
from quantplan.search import RiskGuidedSearch, SearchPolicy


def _evidence(group: str, precision: PrecisionFormat, risk: float, cost: float):
    return CandidateEvidence(
        group_id=group,
        precision=precision,
        context_id="bf16-anchor-v1",
        risk=risk,
        estimated_cost=cost,
    )


def test_search_upgrades_best_risk_recovery_per_cost_first():
    groups = [
        DecisionGroup(
            group_id="a",
            module_names=("layers.0.mlp",),
            kind="mlp",
            allowed_formats=(PrecisionFormat.W4A8, PrecisionFormat.W8A8),
        ),
        DecisionGroup(
            group_id="b",
            module_names=("layers.1.mlp",),
            kind="mlp",
            allowed_formats=(PrecisionFormat.W4A8, PrecisionFormat.W8A8),
        ),
    ]
    evidence = [
        _evidence("a", PrecisionFormat.W4A8, 0.8, 4),
        _evidence("a", PrecisionFormat.W8A8, 0.1, 8),
        _evidence("b", PrecisionFormat.W4A8, 0.5, 4),
        _evidence("b", PrecisionFormat.W8A8, 0.3, 8),
    ]

    result = RiskGuidedSearch().propose(
        groups,
        evidence,
        SearchPolicy(max_estimated_cost=12, max_candidates=3),
    )

    assert result.candidates
    passing = result.candidates[-1]
    assert passing.assignments["a"] is PrecisionFormat.W8A8
    assert passing.estimated_cost <= 12


def test_search_refuses_to_mix_analysis_contexts():
    group = DecisionGroup(
        group_id="a",
        module_names=("layers.0.mlp",),
        kind="mlp",
        allowed_formats=(PrecisionFormat.W4A8, PrecisionFormat.W8A8),
    )
    evidence = [
        _evidence("a", PrecisionFormat.W4A8, 0.8, 4),
        CandidateEvidence(
            group_id="a",
            precision=PrecisionFormat.W8A8,
            context_id="different-context",
            risk=0.1,
            estimated_cost=8,
        ),
    ]

    try:
        RiskGuidedSearch().propose(
            [group], evidence, SearchPolicy(max_estimated_cost=8)
        )
    except ValueError as exc:
        assert "different contexts" in str(exc)
    else:
        raise AssertionError("mixed evidence contexts must fail")


def test_single_candidate_request_returns_most_recovered_plan():
    group = DecisionGroup(
        group_id="a",
        module_names=("layers.0.mlp",),
        kind="mlp",
        allowed_formats=(PrecisionFormat.W4A8, PrecisionFormat.W8A8),
    )
    evidence = [
        _evidence("a", PrecisionFormat.W4A8, 0.8, 4),
        _evidence("a", PrecisionFormat.W8A8, 0.1, 8),
    ]

    result = RiskGuidedSearch().propose(
        [group],
        evidence,
        SearchPolicy(max_estimated_cost=8, max_candidates=1),
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].assignments["a"] is PrecisionFormat.W8A8


def test_locked_group_keeps_its_explicit_precision():
    group = DecisionGroup(
        group_id="router",
        module_names=("layers.0.mlp.router",),
        kind="router",
        allowed_formats=(PrecisionFormat.W4A8, PrecisionFormat.BF16),
        locked=True,
        lock_reason="routing path remains high precision in P0",
        fixed_format=PrecisionFormat.BF16,
    )
    evidence = [
        _evidence("router", PrecisionFormat.W4A8, 0.01, 4),
        _evidence("router", PrecisionFormat.BF16, 0.0, 16),
    ]

    result = RiskGuidedSearch().propose(
        [group],
        evidence,
        SearchPolicy(max_estimated_cost=16),
    )

    assert result.candidates[0].assignments["router"] is PrecisionFormat.BF16


def test_search_can_skip_a_non_monotonic_intermediate_precision():
    group = DecisionGroup(
        group_id="a",
        module_names=("layers.0.mlp",),
        kind="mlp",
        allowed_formats=(
            PrecisionFormat.W4A4,
            PrecisionFormat.W4A8,
            PrecisionFormat.W8A8,
        ),
    )
    evidence = [
        _evidence("a", PrecisionFormat.W4A4, 1.0, 4),
        _evidence("a", PrecisionFormat.W4A8, 1.1, 6),
        _evidence("a", PrecisionFormat.W8A8, 0.1, 8),
    ]

    result = RiskGuidedSearch().propose(
        [group], evidence, SearchPolicy(max_estimated_cost=8)
    )

    assert result.candidates[-1].assignments["a"] is PrecisionFormat.W8A8


def test_search_returns_no_candidate_when_minimum_plan_exceeds_budget():
    group = DecisionGroup(
        group_id="a",
        module_names=("layers.0.mlp",),
        kind="mlp",
        allowed_formats=(PrecisionFormat.W4A8,),
    )

    result = RiskGuidedSearch().propose(
        [group],
        [_evidence("a", PrecisionFormat.W4A8, 0.8, 4)],
        SearchPolicy(max_estimated_cost=3),
    )

    assert not result.candidates
    assert "minimum legal plan" in result.rejected_reasons[-1]


def test_candidate_evidence_rejects_nan() -> None:
    try:
        _evidence("a", PrecisionFormat.W4A8, float("nan"), 4)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("NaN evidence must not enter search")


def test_configured_guard_requires_corresponding_evidence() -> None:
    group = DecisionGroup(
        group_id="a",
        module_names=("layers.0.mlp",),
        kind="mlp",
        allowed_formats=(PrecisionFormat.W4A8,),
    )

    result = RiskGuidedSearch().propose(
        [group],
        [_evidence("a", PrecisionFormat.W4A8, 0.8, 4)],
        SearchPolicy(max_estimated_cost=4, max_route_flip_rate=0.01),
    )

    assert not result.candidates
    assert "evidence guards" in result.rejected_reasons[0]
