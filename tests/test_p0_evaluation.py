import pytest

from quantplan.p0 import evaluate_ranking


def test_perfect_ranking_passes_point_metric_thresholds():
    truth = {"a": 0.1, "b": 0.9, "c": 0.4, "d": 0.7}
    proxy = {"a": 1.0, "b": 9.0, "c": 4.0, "d": 7.0}

    result = evaluate_ranking(proxy, truth, top_k=2)

    assert result.spearman_rho == pytest.approx(1.0)
    assert result.dangerous_recall == 1.0
    assert result.ndcg == pytest.approx(1.0)
    assert result.passes_point_metrics()


def test_group_mismatch_fails_closed():
    with pytest.raises(ValueError, match="group mismatch"):
        evaluate_ranking({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "d": 3})


def test_inverse_ranking_does_not_pass():
    truth = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
    proxy = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}

    result = evaluate_ranking(proxy, truth, top_k=2)

    assert result.spearman_rho == pytest.approx(-1.0)
    assert not result.passes_point_metrics()


def test_non_finite_scores_fail_closed():
    with pytest.raises(ValueError, match="finite"):
        evaluate_ranking(
            {"a": 1.0, "b": float("nan"), "c": 3.0},
            {"a": 1.0, "b": 2.0, "c": 3.0},
        )
