"""Tests for the data layer: temporal split, filtering and matrix building."""

from __future__ import annotations

import numpy as np
import pytest

from recsys.config import EvalConfig
from recsys.data.matrix import build_interaction_matrix
from recsys.data.split import (
    filter_min_interactions,
    positive_interactions,
    temporal_split,
)


def test_filter_min_interactions_drops_short_users(toy_ratings):
    kept = filter_min_interactions(toy_ratings, min_user_ratings=4)
    # user 3 has only 2 ratings and should be removed.
    assert set(kept["userId"]) == {1, 2}


def test_filter_min_interactions_noop_when_threshold_low(toy_ratings):
    kept = filter_min_interactions(toy_ratings, min_user_ratings=1)
    assert len(kept) == len(toy_ratings)


def test_temporal_split_holds_out_latest_items(toy_ratings):
    config = EvalConfig(test_fraction=0.2, min_user_ratings=1)
    train, test = temporal_split(toy_ratings, config)

    # No overlap between train and test rows.
    train_pairs = set(zip(train["userId"], train["movieId"]))
    test_pairs = set(zip(test["userId"], test["movieId"]))
    assert train_pairs.isdisjoint(test_pairs)

    # Every test item is strictly newer than that user's newest train item:
    # this is the core no-leakage guarantee.
    for user in test["userId"].unique():
        latest_train = train.loc[train["userId"] == user, "timestamp"].max()
        earliest_test = test.loc[test["userId"] == user, "timestamp"].min()
        assert earliest_test > latest_train


def test_temporal_split_always_leaves_one_in_train(toy_ratings):
    # A high test fraction must still leave at least one training row per user.
    config = EvalConfig(test_fraction=0.9, min_user_ratings=1)
    train, _ = temporal_split(toy_ratings, config)
    counts = train.groupby("userId").size()
    assert (counts >= 1).all()
    assert set(counts.index) == set(toy_ratings["userId"].unique())


def test_temporal_split_rejects_bad_fraction(toy_ratings):
    with pytest.raises(ValueError):
        temporal_split(toy_ratings, EvalConfig(test_fraction=1.5))


def test_positive_interactions_threshold(toy_ratings):
    config = EvalConfig(positive_threshold=4.0)
    pos = positive_interactions(toy_ratings, config)
    assert (pos["rating"] >= 4.0).all()
    # Ratings below 4.0 (the 3.0, 2.0 and 1.0 rows) are excluded.
    assert len(pos) == 8


def test_build_interaction_matrix_shape_and_values(toy_ratings):
    im = build_interaction_matrix(toy_ratings)
    assert im.n_users == 3
    assert im.n_items == toy_ratings["movieId"].nunique()

    # Spot-check that a known rating landed in the right cell.
    row = im.user_index[1]
    col = im.item_index[10]
    assert im.matrix[row, col] == pytest.approx(5.0)


def test_build_interaction_matrix_mappings_roundtrip(toy_ratings):
    im = build_interaction_matrix(toy_ratings)
    for uid, idx in im.user_index.items():
        assert im.user_ids[idx] == uid
    for mid, idx in im.item_index.items():
        assert im.item_ids[idx] == mid


def test_seen_items_matches_input(toy_ratings):
    im = build_interaction_matrix(toy_ratings)
    assert im.seen_items(1) == {10, 11, 12, 13, 14}
    assert im.seen_items(3) == {11, 17}
    # Unknown user yields an empty set rather than raising.
    assert im.seen_items(999) == set()


def test_matrix_item_order_is_sorted(toy_ratings):
    im = build_interaction_matrix(toy_ratings)
    assert np.all(np.diff(im.item_ids) > 0)
