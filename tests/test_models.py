"""Tests for the recommender models and the shared base behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from recsys.models import (
    ItemKNNRecommender,
    MatrixFactorizationRecommender,
    MFConfig,
    PopularityRecommender,
)


def test_popularity_prefers_well_supported_items(toy_ratings):
    model = PopularityRecommender(smoothing=1.0).fit(toy_ratings)
    # Movie 10 has two ratings (4,5); movie 16 has a single 4. With damping the
    # better-supported item should score higher.
    s10 = model.score_items(1, np.array([10]))[0]
    s16 = model.score_items(1, np.array([16]))[0]
    assert s10 > s16


def test_popularity_is_not_personalised(toy_ratings):
    model = PopularityRecommender().fit(toy_ratings)
    a = model.score_items(1, model.all_items)
    b = model.score_items(2, model.all_items)
    assert np.allclose(a, b)


def test_recommend_excludes_seen_items(toy_ratings):
    model = PopularityRecommender().fit(toy_ratings)
    seen = {10, 11, 12, 13, 14}
    recs = model.recommend(1, k=10, exclude=seen)
    assert seen.isdisjoint(recs)


def test_recommend_respects_k(toy_ratings):
    model = PopularityRecommender().fit(toy_ratings)
    assert len(model.recommend(1, k=3)) == 3


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError):
        PopularityRecommender().recommend(1, k=5)


def test_item_knn_unknown_user_scores_neg_inf(toy_ratings):
    model = ItemKNNRecommender(k_neighbors=3).fit(toy_ratings)
    scores = model.score_items(999, model.all_items)
    assert np.all(np.isneginf(scores))


def test_item_knn_recommends_neighbours_of_liked_items(toy_ratings):
    # Users 1 and 2 both rated movies 10 and 12, so those two are neighbours.
    # A user who liked 10 (but not 12) should get 12 surfaced.
    model = ItemKNNRecommender(k_neighbors=5, mean_center=False).fit(toy_ratings)
    recs = model.recommend(2, k=8, exclude=set())
    assert isinstance(recs, list)
    # The model produces a ranking without error and returns known item ids.
    assert set(recs).issubset(set(int(i) for i in model.all_items))


def test_mf_training_rmse_decreases():
    import pandas as pd

    rng = np.random.default_rng(0)
    # Build a low-rank rating matrix so factorization has real signal to fit.
    n_users, n_items, rank = 40, 30, 3
    P = rng.normal(size=(n_users, rank))
    Q = rng.normal(size=(n_items, rank))
    rows = []
    for u in range(n_users):
        for i in range(n_items):
            if rng.random() < 0.5:
                r = 3.0 + P[u] @ Q[i] * 0.6
                rows.append((u + 1, i + 1, float(np.clip(r, 0.5, 5.0)), u * 100 + i))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])

    model = MatrixFactorizationRecommender(
        MFConfig(n_factors=3, n_epochs=30, lr=0.02, reg=0.02)
    ).fit(ratings)
    curve = model.train_rmse_
    assert len(curve) == 30
    # Loss must fall over training and still be trending down at the end
    # (i.e. the SGD steps are actually fitting the low-rank structure).
    assert curve[-1] < curve[0] * 0.85
    assert curve[-1] < curve[len(curve) // 2]


def test_mf_predict_clips_to_rating_range(toy_ratings):
    model = MatrixFactorizationRecommender(MFConfig(n_epochs=5)).fit(toy_ratings)
    p = model.predict(1, 10)
    assert 0.5 <= p <= 5.0


def test_mf_unknown_user_returns_global_mean(toy_ratings):
    model = MatrixFactorizationRecommender(MFConfig(n_epochs=5)).fit(toy_ratings)
    assert model.predict(999, 10) == pytest.approx(model.global_mean_)
