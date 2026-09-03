from dataclasses import replace

import pytest

from quantplan.domain import (
    ApproximateMeasurement,
    GroupInventory,
    InterventionMeasurement,
    PlanEvaluation,
    PrecisionFormat,
)
from quantplan.planning import (
    PlanningPolicy,
    build_candidate_set,
    choose_lowest_cost_passing_plan,
    plan_id,
)


def _measurement(
    group_id: str,
    precision: PrecisionFormat,
    candidate_nll: float,
    *,
    modules: tuple[str, ...],
) -> InterventionMeasurement:
    return InterventionMeasurement(
        group_id=group_id,
        precision=precision,
        context_id="ctx",
        module_names=modules,
        parameter_count=100,
        baseline_nll=2.0,
        candidate_nll=candidate_nll,
        token_count=128,
        elapsed_seconds=1.0,
        quantizer_id="torch-uniform-proxy-v1",
    )


def test_candidate_set_prioritizes_the_sensitive_group() -> None:
    inventories = (
        GroupInventory("block_0", ("layers.0.fc",), 100),
        GroupInventory("block_1", ("layers.1.fc",), 100),
    )
    measurements = (
        _measurement(
            "block_0", PrecisionFormat.W4A8, 2.2, modules=("layers.0.fc",)
        ),
        _measurement(
            "block_0", PrecisionFormat.W8A8, 2.01, modules=("layers.0.fc",)
        ),
        _measurement(
            "block_1", PrecisionFormat.W4A8, 2.04, modules=("layers.1.fc",)
        ),
        _measurement(
            "block_1", PrecisionFormat.W8A8, 2.02, modules=("layers.1.fc",)
        ),
    )

    result = build_candidate_set(
        inventories,
        measurements,
        allowed_formats=(PrecisionFormat.W4A8, PrecisionFormat.W8A8),
        policy=PlanningPolicy(max_high_precision_fraction=0.2, max_candidates=3),
    )

    assert result.plans
    best = result.plans[-1]
    assert best.assignments["block_0"] is PrecisionFormat.W8A8
    assert best.assignments["block_1"] is PrecisionFormat.W4A8


def test_negative_delta_is_kept_only_as_a_warning_for_planning() -> None:
    inventory = GroupInventory("block_0", ("layers.0.fc",), 100)
    result = build_candidate_set(
        (inventory,),
        (
            _measurement(
                "block_0",
                PrecisionFormat.W4A8,
                1.99,
                modules=inventory.module_names,
            ),
        ),
        allowed_formats=(PrecisionFormat.W4A8,),
    )

    w4 = next(
        item for item in result.evidence if item.precision is PrecisionFormat.W4A8
    )
    assert w4.risk == 0
    assert "negative delta NLL" in w4.warnings[0]


def test_candidate_set_uses_fast_risk_without_inventing_delta_nll() -> None:
    inventory = GroupInventory("block_0", ("layers.0.fc",), 100)
    measurement = ApproximateMeasurement(
        group_id="block_0",
        precision=PrecisionFormat.W4A8,
        context_id="ctx",
        module_names=inventory.module_names,
        parameter_count=100,
        estimated_risk=0.125,
        token_count=128,
        elapsed_seconds=1.0,
        metric_id="fast-v1",
        mean_first_order_effect=-0.02,
        rms_first_order_effect=0.105,
    )

    result = build_candidate_set(
        (inventory,),
        (measurement,),
        allowed_formats=(PrecisionFormat.W4A8,),
    )

    w4 = next(item for item in result.evidence if item.precision is PrecisionFormat.W4A8)
    assert w4.risk == pytest.approx(0.125)
    assert not hasattr(measurement, "delta_nll")


def test_domain_allowed_formats_filter_unsupported_w4_candidate() -> None:
    inventory = GroupInventory(
        "mla_preprocess",
        ("model.layers.0.q_proj",),
        100,
        quantization_domain="mla_preprocess",
        allowed_formats=(PrecisionFormat.W8A8, PrecisionFormat.BF16),
    )
    result = build_candidate_set(
        (inventory,),
        (
            _measurement(
                "mla_preprocess",
                PrecisionFormat.W4A8,
                2.5,
                modules=inventory.module_names,
            ),
            _measurement(
                "mla_preprocess",
                PrecisionFormat.W8A8,
                2.1,
                modules=inventory.module_names,
            ),
        ),
        allowed_formats=(PrecisionFormat.W4A8, PrecisionFormat.W8A8),
    )

    assert all(item.precision is not PrecisionFormat.W4A8 for item in result.evidence)
    assert all(
        assignment is not PrecisionFormat.W4A8
        for plan in result.plans
        for assignment in plan.assignments.values()
    )


def test_choose_lowest_cost_plan_uses_whole_plan_validation() -> None:
    inventory = GroupInventory("block_0", ("layers.0.fc",), 100)
    result = build_candidate_set(
        (inventory,),
        (
            _measurement(
                "block_0",
                PrecisionFormat.W4A8,
                2.2,
                modules=inventory.module_names,
            ),
            _measurement(
                "block_0",
                PrecisionFormat.W8A8,
                2.01,
                modules=inventory.module_names,
            ),
        ),
        allowed_formats=(PrecisionFormat.W4A8, PrecisionFormat.W8A8),
        policy=PlanningPolicy(max_high_precision_fraction=1.0, max_candidates=3),
    )
    evaluations = tuple(
        PlanEvaluation(
            plan_id=plan_id(plan),
            context_id="ctx",
            baseline_nll=2.0,
            candidate_nll=(2.1 if index == 0 else 2.01),
            token_count=128,
            elapsed_seconds=1.0,
        )
        for index, plan in enumerate(result.plans)
    )

    chosen = choose_lowest_cost_passing_plan(
        result.plans,
        evaluations,
        max_delta_nll=0.02,
    )

    assert chosen is not None
    assert chosen[1].delta_nll <= 0.02


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("baseline_nll", 2.1, "different baseline NLL"),
        ("token_count", 127, "different token counts"),
    ],
)
def test_candidate_set_rejects_measurements_that_reuse_a_context_inconsistently(
    field: str,
    value: float,
    message: str,
) -> None:
    inventory = GroupInventory("block_0", ("layers.0.fc",), 100)
    first = _measurement(
        "block_0",
        PrecisionFormat.W4A8,
        2.2,
        modules=inventory.module_names,
    )
    second = replace(
        _measurement(
            "block_0",
            PrecisionFormat.W8A8,
            2.01,
            modules=inventory.module_names,
        ),
        **{field: value},
    )

    with pytest.raises(ValueError, match=message):
        build_candidate_set(
            (inventory,),
            (first, second),
            allowed_formats=(PrecisionFormat.W4A8, PrecisionFormat.W8A8),
        )


def test_plan_selection_rejects_mismatched_evaluation_context() -> None:
    inventory = GroupInventory("block_0", ("layers.0.fc",), 100)
    candidate_set = build_candidate_set(
        (inventory,),
        (
            _measurement(
                "block_0",
                PrecisionFormat.W4A8,
                2.01,
                modules=inventory.module_names,
            ),
        ),
        allowed_formats=(PrecisionFormat.W4A8,),
    )
    evaluations = tuple(
        PlanEvaluation(
            plan_id=plan_id(plan),
            context_id="different-context",
            baseline_nll=2.0,
            candidate_nll=2.01,
            token_count=128,
            elapsed_seconds=1.0,
        )
        for plan in candidate_set.plans
    )

    with pytest.raises(ValueError, match="evaluation context mismatch"):
        choose_lowest_cost_passing_plan(
            candidate_set.plans,
            evaluations,
            max_delta_nll=0.02,
        )
