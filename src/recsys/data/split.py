"""Temporal train/test splitting for offline recommender evaluation.

A random split leaks the future into the past: if I train on a rating a user
made in 2018 and test on one from 2015, the model has seen information it could
never have at prediction time. To avoid that I split *per user* in time order,
holding out each user's most recent interactions as the test set. This mirrors
how a deployed system is actually evaluated — predict the next things a user
will like given everything they have done so far.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.config import EvalConfig
from recsys.logging_utils import get_logger

logger = get_logger(__name__)


def filter_min_interactions(
    ratings: pd.DataFrame, min_user_ratings: int
) -> pd.DataFrame:
    """Drop users with fewer than ``min_user_ratings`` ratings.

    Users with only a handful of interactions cannot be split into a meaningful
    train and test portion, and they add noise to ranking metrics, so I remove
    them before splitting rather than special-casing them everywhere downstream.
    """
    if min_user_ratings <= 1:
        return ratings.reset_index(drop=True)

    counts = ratings.groupby("userId")["movieId"].transform("size")
    kept = ratings[counts >= min_user_ratings].reset_index(drop=True)
    dropped_users = ratings["userId"].nunique() - kept["userId"].nunique()
    if dropped_users:
        logger.info(
            "Dropped %d users with fewer than %d ratings",
            dropped_users,
            min_user_ratings,
        )
    return kept


def temporal_split(
    ratings: pd.DataFrame, config: EvalConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings into train/test by holding out each user's latest items.

    For every user the rows are ordered by ``(timestamp, movieId)`` and the last
    ``ceil(n * test_fraction)`` of them go to the test set, with the guarantee
    that at least one interaction always stays in train (so no user becomes a
    pure cold-start in their own split).

    Args:
        ratings: Cleaned ratings table from :func:`recsys.data.loader.load_ratings`.
        config: Evaluation knobs; defaults to a fresh :class:`EvalConfig`.

    Returns:
        ``(train, test)`` DataFrames with the same columns as the input.
    """
    config = config or EvalConfig()
    if not 0.0 < config.test_fraction < 1.0:
        raise ValueError(
            f"test_fraction must be in (0, 1), got {config.test_fraction}."
        )

    ratings = filter_min_interactions(ratings, config.min_user_ratings)
    ordered = ratings.sort_values(["userId", "timestamp", "movieId"]).reset_index(
        drop=True
    )

    # Position of each row within its user, and the user's total count.
    rank = ordered.groupby("userId").cumcount()
    counts = ordered.groupby("userId")["movieId"].transform("size").to_numpy()

    n_test = np.ceil(counts * config.test_fraction).astype(int)
    n_test = np.minimum(n_test, counts - 1)  # always leave >=1 in train
    n_train = counts - n_test
    is_test = rank.to_numpy() >= n_train

    train = ordered[~is_test].reset_index(drop=True)
    test = ordered[is_test].reset_index(drop=True)
    logger.info(
        "Temporal split: %d train / %d test rows over %d users",
        len(train),
        len(test),
        ordered["userId"].nunique(),
    )
    return train, test


def positive_interactions(
    ratings: pd.DataFrame, config: EvalConfig | None = None
) -> pd.DataFrame:
    """Filter ratings down to positive (relevant) interactions.

    Ranking metrics treat a recommendation as a hit only when the user actually
    liked the item, so the held-out test set is reduced to ratings at or above
    :attr:`EvalConfig.positive_threshold`.
    """
    config = config or EvalConfig()
    return ratings[ratings["rating"] >= config.positive_threshold].reset_index(
        drop=True
    )
