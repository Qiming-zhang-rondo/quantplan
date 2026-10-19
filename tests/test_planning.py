from dataclasses import replace

import pytest

from quantplan.domain import (
    ApproximateMeasurement,
    DirectionalMeasurement,
    GroupInventory,
    InterventionMeasurement,
    PlanEvaluation,
    PrecisionFormat,
    StatisticalMeasurement,
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


def test_statistical_risk_is_a_separate_uncalibrated_evidence_family() -> None:
    inventory = GroupInventory("block_0", ("layers.0.fc",), 100)
    measurement = StatisticalMeasurement(
        group_id="block_0",
        precision=PrecisionFormat.W4A8,
        context_id="ctx",
        module_names=inventory.module_names,
        parameter_count=100,
        estimated_risk=0.125,
        token_count=128,
        elapsed_seconds=1.0,
        metric_id="stats-v1",
        weight_risk=0.1,
        activation_risk=0.025,
    )

    result = build_candidate_set(
        (inventory,),
        (measurement,),
        allowed_formats=(PrecisionFormat.W4A8,),
        policy=PlanningPolicy(kl_weight=1.0),
    )

    w4 = next(item for item in result.evidence if item.precision is PrecisionFormat.W4A8)
    assert w4.risk == pytest.approx(measurement.estimated_risk)
    assert any("KL weight ignored" in warning for warning in w4.warnings)
    assert not hasattr(measurement, "delta_nll")
    assert not hasattr(measurement, "mean_first_order_effect")

    exact = _measurement(
        "block_0", PrecisionFormat.W8A8, 2.01, modules=inventory.module_names
    )
    replay = ApproximateMeasurement(
        group_id="block_0",
        precision=PrecisionFormat.W8A8,
        context_id="ctx",
        module_names=inventory.module_names,
        parameter_count=100,
        estimated_risk=0.01,
        token_count=128,
        elapsed_seconds=1.0,
        metric_id="fast-v1",
        mean_first_order_effect=0.0,
        rms_first_order_effect=0.01,
    )
    directional = DirectionalMeasurement(
        group_id="block_0",
        precision=PrecisionFormat.W8A8,
        context_id="ctx",
        module_names=inventory.module_names,
        parameter_count=100,
        estimated_risk=0.01,
        token_count=128,
        elapsed_seconds=1.0,
        metric_id="taylor-v1",
        mean_first_order_effect=0.01,
        ranking_offset=0.0,
    )
    for other in (exact, replay, directional):
        with pytest.raises(ValueError, match="measurements cannot be mixed"):
            build_candidate_set(
                (inventory,),
                (measurement, other),
                allowed_formats=(PrecisionFormat.W4A8, PrecisionFormat.W8A8),
            )


@pytest.mark.parametrize(
    "family", [ApproximateMeasurement, StatisticalMeasurement, DirectionalMeasurement]
)
def test_candidate_set_rejects_different_approximate_metric_versions(family) -> None:
    inventory = GroupInventory("block_0", ("layers.0.fc",), 100)
    components = {
        StatisticalMeasurement: {"weight_risk": 0.1, "activation_risk": 0.025},
        ApproximateMeasurement: {
            "mean_first_order_effect": 0.0,
            "rms_first_order_effect": 0.125,
        },
        DirectionalMeasurement: {"mean_first_order_effect": 0.125, "ranking_offset": 0.0},
    }[family]
    first = family(
        group_id="block_0",
        precision=PrecisionFormat.W4A8,
        context_id="ctx",
        module_names=inventory.module_names,
        parameter_count=100,
        estimated_risk=0.125,
        token_count=128,
        elapsed_seconds=1.0,
        metric_id="metric-v1",
        **components,
    )
    second = replace(first, precision=PrecisionFormat.W8A8, metric_id="metric-v2")

    with pytest.raises(ValueError, match="different metric_id values cannot be mixed"):
        build_candidate_set(
            (inventory,),
            (first, second),
            allowed_formats=(PrecisionFormat.W4A8, PrecisionFormat.W8A8),
        )


def _directional_measurement(
    inventory: GroupInventory,
    precision: PrecisionFormat,
    effect: float,
    offset: float,
) -> DirectionalMeasurement:
    return DirectionalMeasurement(
        group_id=inventory.group_id,
        precision=precision,
        context_id="ctx",
        module_names=inventory.module_names,
        parameter_count=inventory.parameter_count,
        estimated_risk=effect + offset,
        token_count=128,
        elapsed_seconds=1.0,
        metric_id="taylor-v1",
        mean_first_order_effect=effect,
        ranking_offset=offset,
    )


@pytest.mark.parametrize("explicit_bf16", [False, True])
def test_directional_offsets_preserve_signed_order_including_bf16(explicit_bf16) -> None:
    inventories = (
        GroupInventory("block_0", ("layers.0.fc",), 100),
        GroupInventory("block_1", ("layers.1.fc",), 100),
    )
    signed_effects = {
        ("block_0", PrecisionFormat.W4A8): -0.1,
        ("block_0", PrecisionFormat.W8A8): -0.2,
        ("block_1", PrecisionFormat.W4A8): 0.3,
        ("block_1", PrecisionFormat.W8A8): 0.1,
    }
    offsets = {"block_0": 0.2, "block_1": 0.0}
    measurements = [
        _directional_measurement(
            inventory, precision, signed_effects[inventory.group_id, precision],
            offsets[inventory.group_id],
        )
        for inventory in inventories
        for precision in (PrecisionFormat.W4A8, PrecisionFormat.W8A8)
    ]
    if explicit_bf16:
        measurements.extend(
            _directional_measurement(
                inventory, PrecisionFormat.BF16, 0.0, offsets[inventory.group_id]
            )
            for inventory in inventories
        )

    result = build_candidate_set(
        inventories,
        measurements,
        allowed_formats=(PrecisionFormat.W4A8, PrecisionFormat.W8A8),
        policy=PlanningPolicy(max_high_precision_fraction=1.0),
    )

    risks = {(item.group_id, item.precision): item.risk for item in result.evidence}
    assert risks["block_0", PrecisionFormat.W4A8] == pytest.approx(0.1)
    assert risks["block_0", PrecisionFormat.W8A8] == pytest.approx(0.0)
    assert risks["block_0", PrecisionFormat.BF16] == pytest.approx(0.2)
    assert any(plan.assignments["block_0"] is PrecisionFormat.W8A8 for plan in result.plans)
    assert all(plan.assignments["block_0"] is not PrecisionFormat.BF16 for plan in result.plans)
    for plan in result.plans:
        signed_total = sum(
            signed_effects.get((group_id, precision), 0.0)
            for group_id, precision in plan.assignments.items()
        )
        assert plan.proposal_score == pytest.approx(signed_total + sum(offsets.values()))


def test_directional_measurements_reject_inconsistent_group_offsets() -> None:
    inventory = GroupInventory("block_0", ("layers.0.fc",), 100)
    first = _directional_measurement(inventory, PrecisionFormat.W4A8, -0.1, 0.2)
    second = _directional_measurement(inventory, PrecisionFormat.W8A8, -0.2, 0.3)

    with pytest.raises(ValueError, match="different ranking_offset values"):
        build_candidate_set(
            (inventory,),
            (first, second),
            allowed_formats=(PrecisionFormat.W4A8, PrecisionFormat.W8A8),
        )


def test_directional_planning_rechecks_corrupt_risk_evidence() -> None:
    inventory = GroupInventory("block_0", ("layers.0.fc",), 100)
    measurement = _directional_measurement(inventory, PrecisionFormat.W4A8, -0.1, 0.2)
    object.__setattr__(measurement, "estimated_risk", 0.3)

    with pytest.raises(ValueError, match="estimated_risk is inconsistent"):
        build_candidate_set(
            (inventory,), (measurement,), allowed_formats=(PrecisionFormat.W4A8,)
        )


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


def test_source_locked_domain_does_not_invent_bf16_fallback() -> None:
    inventory = GroupInventory(
        "mla_preprocess",
        ("model.layers.0.q_proj",),
        100,
        quantization_domain="mla_preprocess",
        allowed_formats=(PrecisionFormat.W8A8,),
        source_precision=PrecisionFormat.W8A8,
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
        policy=PlanningPolicy(max_high_precision_fraction=1.0),
    )

    assert result.plans
    assert all(
        plan.assignments["mla_preprocess"] is PrecisionFormat.W8A8
        for plan in result.plans
    )
    assert all(item.precision is not PrecisionFormat.BF16 for item in result.evidence)


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
