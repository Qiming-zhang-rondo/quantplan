from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log2, sqrt
from typing import Mapping


@dataclass(frozen=True)
class RankingEvaluation:
    spearman_rho: float
    dangerous_recall: float
    ndcg: float
    count: int
    top_k: int

    def passes_point_metrics(
        self,
        *,
        min_spearman: float = 0.60,
        min_dangerous_recall: float = 0.75,
        min_ndcg: float = 0.75,
    ) -> bool:
        """Check only point-estimate thresholds, not the complete P0 gate."""

        return (
            self.spearman_rho >= min_spearman
            and self.dangerous_recall >= min_dangerous_recall
            and self.ndcg >= min_ndcg
        )


def evaluate_ranking(
    proxy_scores: Mapping[str, float],
    truth_damage: Mapping[str, float],
    *,
    top_k: int | None = None,
) -> RankingEvaluation:
    """Compare a proposal score with held-out single-intervention damage.

    Higher values must mean more dangerous for both mappings. Evaluation units
    are decision groups (normally blocks in P0), never individual tokens.
    """

    if set(proxy_scores) != set(truth_damage):
        missing_proxy = sorted(set(truth_damage) - set(proxy_scores))
        missing_truth = sorted(set(proxy_scores) - set(truth_damage))
        raise ValueError(
            "score/truth group mismatch: "
            f"missing_proxy={missing_proxy}, missing_truth={missing_truth}"
        )
    if len(proxy_scores) < 3:
        raise ValueError("at least three decision groups are required")

    group_ids = sorted(proxy_scores)
    count = len(group_ids)
    if top_k is None:
        top_k = max(1, round(count / 4))
    if not 1 <= top_k <= count:
        raise ValueError("top_k must be between one and the number of groups")

    proxy_values = [float(proxy_scores[group_id]) for group_id in group_ids]
    truth_values = [float(truth_damage[group_id]) for group_id in group_ids]
    if not all(isfinite(value) for value in (*proxy_values, *truth_values)):
        raise ValueError("score and truth values must all be finite")
    rho = _pearson(_average_ranks(proxy_values), _average_ranks(truth_values))

    proxy_order = sorted(group_ids, key=lambda key: (-proxy_scores[key], key))
    truth_order = sorted(group_ids, key=lambda key: (-truth_damage[key], key))
    dangerous_recall = len(set(proxy_order[:top_k]) & set(truth_order[:top_k])) / top_k
    ndcg = _ndcg(proxy_order, truth_damage, top_k)
    return RankingEvaluation(
        spearman_rho=rho,
        dangerous_recall=dangerous_recall,
        ndcg=ndcg,
        count=count,
        top_k=top_k,
    )


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        average_rank = (position + 1 + end) / 2
        for index in range(position, end):
            ranks[indexed[index][0]] = average_rank
        position = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_norm = sum((value - left_mean) ** 2 for value in left)
    right_norm = sum((value - right_mean) ** 2 for value in right)
    denominator = sqrt(left_norm * right_norm)
    return numerator / denominator if denominator else 0.0


def _ndcg(order: list[str], relevance: Mapping[str, float], top_k: int) -> float:
    minimum = min(relevance.values())
    shifted = {key: value - minimum for key, value in relevance.items()}

    def dcg(ranking: list[str]) -> float:
        return sum(
            shifted[group_id] / log2(rank + 2)
            for rank, group_id in enumerate(ranking[:top_k])
        )

    ideal = sorted(relevance, key=lambda key: (-relevance[key], key))
    ideal_dcg = dcg(ideal)
    return dcg(order) / ideal_dcg if ideal_dcg else 1.0
