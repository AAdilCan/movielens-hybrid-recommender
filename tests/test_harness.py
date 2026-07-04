"""Tests for the evaluation harness on in-memory data (no download)."""

from __future__ import annotations

import pandas as pd

from recsys.config import EvalConfig
from recsys.eval.harness import evaluate_model
from recsys.models.popularity import PopularityRecommender


def _split_frames():
    """Build a tiny train/test pair with a known correct answer.

    Every user's most popular held-out item is one that popularity ranks high,
    so the popularity model should score non-zero precision.
    """
    train = pd.DataFrame(
        [
            (1, 100, 5.0, 1),
            (1, 101, 4.0, 2),
            (2, 100, 5.0, 1),
            (2, 102, 4.0, 2),
            (3, 100, 4.0, 1),
            (3, 103, 5.0, 2),
        ],
        columns=["userId", "movieId", "rating", "timestamp"],
    )
    # Held-out positives: users 1 and 2 both go on to like movie 104.
    test_positive = pd.DataFrame(
        [
            (1, 104, 5.0, 3),
            (2, 104, 5.0, 3),
        ],
        columns=["userId", "movieId", "rating", "timestamp"],
    )
    # Add 104 to train (from other users) so popularity can rank it.
    extra = pd.DataFrame(
        [(4, 104, 5.0, 1), (5, 104, 5.0, 1), (6, 104, 4.0, 1)],
        columns=["userId", "movieId", "rating", "timestamp"],
    )
    return pd.concat([train, extra], ignore_index=True), test_positive


def test_evaluate_model_returns_expected_keys():
    train, test_positive = _split_frames()
    config = EvalConfig(k_values=(1, 3), min_user_ratings=1)
    result = evaluate_model(PopularityRecommender(), train, test_positive, config)

    for metric in ("precision", "recall", "map", "ndcg"):
        for k in config.k_values:
            assert f"{metric}@{k}" in result
    assert "coverage" in result and "n_users" in result


def test_evaluate_model_scores_positive_when_hit_possible():
    train, test_positive = _split_frames()
    config = EvalConfig(k_values=(3,), min_user_ratings=1)
    result = evaluate_model(PopularityRecommender(), train, test_positive, config)
    # Movie 104 is popular and unseen by users 1 and 2, so it should be
    # recommendable and land in the top-3 -> non-zero recall.
    assert result["n_users"] == 2.0
    assert result["recall@3"] > 0.0


def test_coverage_in_unit_range():
    train, test_positive = _split_frames()
    config = EvalConfig(k_values=(3,), min_user_ratings=1)
    result = evaluate_model(PopularityRecommender(), train, test_positive, config)
    assert 0.0 <= result["coverage"] <= 1.0
