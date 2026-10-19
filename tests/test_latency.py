import json

import pytest

from quantplan.domain import GroupInventory, InterventionMeasurement, PrecisionFormat
from quantplan.latency import (
    CALIBRATION_SCHEMA,
    CalibratedLatencyProvider,
    RooflineLatencyProvider,
    RooflineProfile,
    build_latency_provider,
)
from quantplan.planning import PlanningPolicy, build_candidate_set


def test_roofline_estimate_prefers_lower_weight_bits() -> None:
    provider = RooflineLatencyProvider(
        RooflineProfile(
            name="test",
            memory_bandwidth_gbps=1000,
            compute_tflops=100,
            launch_overhead_us=0.01,
        )
    )
    group = GroupInventory("g", ("layers.0.fc",), 1_000_000)
    w4 = provider.estimate(group, PrecisionFormat.W4A8)
    w8 = provider.estimate(group, PrecisionFormat.W8A8)
    assert w4.seconds < w8.seconds
    assert w4.lower_bound <= w4.seconds <= w4.upper_bound
    assert "static estimate" in w4.warnings[1]


def test_calibrated_provider_overrides_sparse_entry(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": CALIBRATION_SCHEMA,
                "entries": {"g|w4a8": 0.000123},
            }
        ),
        encoding="utf-8",
    )
    provider = build_latency_provider(
        mode="calibrated",
        deployment_mode="decode_bs1",
        hardware_profile="a5",
        calibration_path=path,
    )
    assert isinstance(provider, CalibratedLatencyProvider)
    estimate = provider.estimate(
        GroupInventory("g", ("layers.0.fc",), 1_000_000), PrecisionFormat.W4A8
    )
    assert estimate.seconds == pytest.approx(0.000123)
    assert estimate.confidence == pytest.approx(0.9)


def test_latency_provider_becomes_planner_cost() -> None:
    group = GroupInventory("g", ("layers.0.fc",), 1_000_000)
    measurements = tuple(
        InterventionMeasurement(
            group_id="g",
            precision=precision,
            context_id="ctx",
            module_names=group.module_names,
            parameter_count=group.parameter_count,
            baseline_nll=2.0,
            candidate_nll=2.01,
            token_count=16,
            elapsed_seconds=0.1,
            quantizer_id="test",
        )
        for precision in (PrecisionFormat.W4A8, PrecisionFormat.W8A8)
    )
    provider = build_latency_provider(
        mode="static",
        deployment_mode="decode_bs1",
        hardware_profile="test",
        memory_bandwidth_gbps=1000,
        compute_tflops=100,
        launch_overhead_us=0.01,
    )
    result = build_candidate_set(
        (group,),
        measurements,
        allowed_formats=(PrecisionFormat.W4A8, PrecisionFormat.W8A8),
        policy=PlanningPolicy(max_high_precision_fraction=1.0),
        latency_provider=provider,
    )
    assert all(item.latency_seconds is not None for item in result.evidence)
    assert all(item.estimated_cost == item.latency_seconds for item in result.evidence)
