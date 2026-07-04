"""Popularity baseline recommender.

The most-rated movies, recommended to everyone. It ignores personalisation
entirely, which is exactly why it makes a good baseline: any collaborative or
content model that cannot beat "just show everyone the popular stuff" is not
earning its complexity. It also doubles as the cold-start fallback for users
the personalised models have never seen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.logging_utils import get_logger
from recsys.models.base import BaseRecommender

logger = get_logger(__name__)


class PopularityRecommender(BaseRecommender):
    """Rank items by a damped popularity score.

    Popularity is the interaction count shrunk toward zero by a small additive
    smoothing term, which stops a movie with a single 5-star rating from
    outranking one with hundreds of ratings.
    """

    name = "popularity"

    def __init__(self, smoothing: float = 5.0) -> None:
        super().__init__()
        self.smoothing = smoothing
        self._items: np.ndarray = np.empty(0, dtype=np.int64)
        self._scores: dict[int, float] = {}

    def fit(self, ratings: pd.DataFrame) -> "PopularityRecommender":
        counts = ratings.groupby("movieId").size()
        means = ratings.groupby("movieId")["rating"].mean()
        # Popularity = mean rating weighted by how many ratings back it up.
        # count / (count + smoothing) pulls low-support items toward 0.
        support = counts / (counts + self.smoothing)
        score = (means * support).sort_values(ascending=False)

        self._items = score.index.to_numpy(dtype=np.int64)
        self._scores = score.to_dict()
        self._fitted = True
        logger.info("Fit popularity model over %d items", len(self._items))
        return self

    @property
    def all_items(self) -> np.ndarray:
        return self._items

    def _score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        return np.array(
            [self._scores.get(int(i), -np.inf) for i in item_ids], dtype=np.float64
        )
