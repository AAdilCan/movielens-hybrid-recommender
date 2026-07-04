"""Recommendation models (collaborative, content-based, hybrid)."""

from __future__ import annotations

import pandas as pd

from recsys.config import EvalConfig, HybridWeights
from recsys.models.base import BaseRecommender
from recsys.models.content import ContentRecommender
from recsys.models.hybrid import HybridRecommender
from recsys.models.item_knn import ItemKNNRecommender
from recsys.models.matrix_factorization import (
    MatrixFactorizationRecommender,
    MFConfig,
)
from recsys.models.popularity import PopularityRecommender

__all__ = [
    "BaseRecommender",
    "PopularityRecommender",
    "ItemKNNRecommender",
    "MatrixFactorizationRecommender",
    "MFConfig",
    "ContentRecommender",
    "HybridRecommender",
    "build_default_hybrid",
]


def build_default_hybrid(
    movies: pd.DataFrame,
    weights: HybridWeights | None = None,
    eval_config: EvalConfig | None = None,
) -> HybridRecommender:
    """Assemble the standard item-kNN + MF + content ensemble.

    The popularity model is wired in as the cold-start fallback rather than as a
    weighted component so it only kicks in when personalisation is impossible.
    """
    weights = weights or HybridWeights()
    eval_config = eval_config or EvalConfig()
    components: dict[str, BaseRecommender] = {
        "item_knn": ItemKNNRecommender(),
        "matrix_factorization": MatrixFactorizationRecommender(),
        "content": ContentRecommender(
            movies, positive_threshold=eval_config.positive_threshold
        ),
    }
    mixing = {
        "item_knn": weights.item_knn,
        "matrix_factorization": weights.matrix_factorization,
        "content": weights.content,
    }
    return HybridRecommender(
        components, weights=mixing, fallback=PopularityRecommender()
    )
