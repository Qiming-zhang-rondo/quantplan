import numpy as np
import pytest

from quantplan.metrics.risk import (
    aggregate_prompt_risk,
    compute_prompt_risk,
)


def test_compute_single_prompt_with_token_mask_matches_definition() -> None:
    h_ref = np.array([[1.0, 2.0], [3.0, 4.0]])
    h_q = np.array([[1.5, 1.0], [99.0, 99.0]])
    gradient = np.array([[2.0, -1.0], [7.0, 8.0]])
    mask = np.array([True, False])

    result = compute_prompt_risk(h_ref, h_q, gradient, mask, epsilon=1e-9)

    # Only the first token contributes: delta=[0.5, -1], a=[2, -1].
    expected_t1 = 2.0
    expected_n = 2.0
    expected_d = 8.0
    np.testing.assert_allclose(result.t1, [expected_t1])
    np.testing.assert_allclose(result.n, [expected_n])
    np.testing.assert_allclose(result.d, [expected_d])
    np.testing.assert_allclose(result.s, [expected_n / (expected_d + 1e-9)])
    np.testing.assert_allclose(
        result.kappa, [expected_t1**2 / (expected_n + 1e-9)]
    )
    expected_relative_mse = (0.5**2 + 1.0**2) / (1.0**2 + 2.0**2 + 1e-9)
    np.testing.assert_allclose(result.residual_relative_mse, [expected_relative_mse])
    np.testing.assert_array_equal(result.valid_element_count, [2])
    np.testing.assert_array_equal(result.low_denominator, [False])
    np.testing.assert_array_equal(result.status, ["ok"])


def test_batch_metrics_and_p50_p95_p99_aggregation() -> None:
    h_ref = np.ones((4, 1, 1), dtype=np.float64)
    delta = np.arange(1.0, 5.0).reshape(4, 1, 1)
    h_q = h_ref + delta
    gradient = np.ones_like(h_ref)

    result = compute_prompt_risk(h_ref, h_q, gradient, epsilon=1e-12)
    summary = result.aggregate()

    expected_energy = np.square(np.arange(1.0, 5.0))
    np.testing.assert_allclose(result.t1, np.arange(1.0, 5.0))
    np.testing.assert_allclose(result.n, expected_energy)
    np.testing.assert_allclose(result.d, np.ones(4))
    np.testing.assert_allclose(result.s, expected_energy / (1.0 + 1e-12))
    np.testing.assert_allclose(
        result.kappa, expected_energy / (expected_energy + 1e-12)
    )

    assert summary.prompt_count == 4
    assert summary.valid_prompt_count == 4
    assert summary.low_denominator_count == 0
    assert summary.t1.p50 == pytest.approx(np.percentile(np.arange(1.0, 5.0), 50))
    assert summary.n.p95 == pytest.approx(np.percentile(expected_energy, 95))
    assert summary.s.p99 == pytest.approx(
        np.percentile(expected_energy / (1.0 + 1e-12), 99)
    )


def test_low_denominator_is_returned_and_explicitly_flagged() -> None:
    h_ref = np.zeros((1, 2, 2))
    h_q = np.ones_like(h_ref)
    gradient = np.ones_like(h_ref)

    result = compute_prompt_risk(
        h_ref,
        h_q,
        gradient,
        denominator_threshold=1e-6,
    )

    np.testing.assert_allclose(result.d, [0.0])
    np.testing.assert_array_equal(result.low_denominator, [True])
    np.testing.assert_array_equal(result.status, ["low_denominator"])
    assert result.has_low_denominator
    assert np.isfinite(result.s[0])
    assert result.aggregate().low_denominator_count == 1


def test_all_masked_prompt_is_nan_and_excluded_from_aggregation() -> None:
    h_ref = np.ones((2, 2, 1))
    h_q = h_ref + np.array([[[1.0], [1.0]], [[3.0], [3.0]]])
    gradient = np.ones_like(h_ref)
    mask = np.array([[False, False], [True, True]])

    result = compute_prompt_risk(h_ref, h_q, gradient, mask)
    summary = aggregate_prompt_risk(result)

    assert np.isnan(result.t1[0])
    assert np.isnan(result.s[0])
    assert result.status.tolist() == ["no_valid_elements", "ok"]
    assert result.valid_element_count.tolist() == [0, 2]
    assert summary.prompt_count == 2
    assert summary.valid_prompt_count == 1
    assert summary.t1.p50 == pytest.approx(6.0)
    assert summary.n.p95 == pytest.approx(18.0)


def test_full_element_mask_is_supported() -> None:
    h_ref = np.ones((1, 1, 2))
    h_q = np.array([[[2.0, 11.0]]])
    gradient = np.ones_like(h_ref)
    mask = np.array([[[True, False]]])

    result = compute_prompt_risk(h_ref, h_q, gradient, mask)

    np.testing.assert_allclose(result.t1, [1.0])
    np.testing.assert_allclose(result.n, [1.0])
    np.testing.assert_array_equal(result.valid_element_count, [1])


@pytest.mark.parametrize(
    ("h_ref", "h_q", "gradient", "message"),
    [
        (np.ones((2, 2)), np.ones((2, 3)), np.ones((2, 2)), "identical shapes"),
        (np.array([1.0, np.nan]), np.ones(2), np.ones(2), "finite"),
        (np.array(["x"]), np.array(["x"]), np.array(["x"]), "real numeric"),
    ],
)
def test_invalid_arrays_are_rejected(h_ref, h_q, gradient, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        compute_prompt_risk(h_ref, h_q, gradient)


@pytest.mark.parametrize("mask", [np.array([0.5, 1.0]), np.array([0.0, np.nan])])
def test_non_binary_or_non_finite_mask_is_rejected(mask: np.ndarray) -> None:
    with pytest.raises(ValueError, match="0/1"):
        compute_prompt_risk(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)), mask)


def test_unbroadcastable_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="not broadcastable"):
        compute_prompt_risk(
            np.ones((2, 3, 4)),
            np.ones((2, 3, 4)),
            np.ones((2, 3, 4)),
            np.ones((2, 2), dtype=bool),
        )


@pytest.mark.parametrize("epsilon", [0.0, -1.0, np.inf, np.nan])
def test_invalid_epsilon_is_rejected(epsilon: float) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        compute_prompt_risk(np.ones(1), np.ones(1), np.ones(1), epsilon=epsilon)


def test_aggregate_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="PromptRiskResult"):
        aggregate_prompt_risk(object())  # type: ignore[arg-type]
