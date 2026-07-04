"""Weighted hybrid ensemble of recommenders.

Each component model scores candidates on its own scale — popularity returns a
damped mean rating, item-kNN a similarity-weighted average, matrix
factorization a predicted rating, content a cosine in [0, 1]. Summing those
directly would let whichever model has the largest raw range dominate, so I
min-max normalise every model's scores over the *candidate set* before mixing
them with configurable weights.

Two design choices matter here:

* **Cold-start fallback.** If none of the personalised models can score a user
  (a brand-new user with no training interactions), the ensemble falls back to
  the popularity model so it always returns something reasonable.
* **Normalise per request, not globally.** Ranking only cares about the order
  within the candidate pool, so normalising over exactly the items being ranked
  keeps every model on equal footing regardless of catalogue-wide score spread.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.logging_utils import get_logger
from recsys.models.base import BaseRecommender
from recsys.models.popularity import PopularityRecommender

logger = get_logger(__name__)


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Scale finite scores to [0, 1]; non-finite entries become 0.

    A model that cannot score an item returns ``-inf``; those map to 0 so the
    item simply gets no vote from that model rather than poisoning the sum.
    """
    out = np.zeros_like(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    if not finite.any():
        return out
    vals = scores[finite]
    lo, hi = vals.min(), vals.max()
    if hi > lo:
        out[finite] = (vals - lo) / (hi - lo)
    else:
        out[finite] = 1.0  # all equal -> all fully relevant
    return out


class HybridRecommender(BaseRecommender):
    """Linearly blend several recommenders with per-model weights.

    Args:
        models: Mapping of model name to a (fitted or unfitted) recommender.
        weights: Per-model mixing weights; missing entries default to 1.0.
            Only relative magnitude matters — they are used as-is on normalised
            scores.
        fallback: Model used when no component can score a user. Defaults to a
            fresh :class:`PopularityRecommender`.
    """

    name = "hybrid"

    def __init__(
        self,
        models: dict[str, BaseRecommender],
        weights: dict[str, float] | None = None,
        fallback: BaseRecommender | None = None,
    ) -> None:
        super().__init__()
        if not models:
            raise ValueError("HybridRecommender needs at least one component model.")
        self.models = models
        self.weights = weights or {}
        self.fallback = fallback or PopularityRecommender()
        self._all_items: np.ndarray = np.empty(0, dtype=np.int64)

    def fit(self, ratings: pd.DataFrame) -> "HybridRecommender":
        item_union: set[int] = set()
        for name, model in self.models.items():
            if not model._fitted:
                model.fit(ratings)
            item_union.update(int(i) for i in model.all_items)
        if not self.fallback._fitted:
            self.fallback.fit(ratings)
        item_union.update(int(i) for i in self.fallback.all_items)

        self._all_items = np.array(sorted(item_union), dtype=np.int64)
        self._fitted = True
        logger.info(
            "Fit hybrid over %d models: %s",
            len(self.models),
            ", ".join(self.models),
        )
        return self

    @property
    def all_items(self) -> np.ndarray:
        return self._all_items

    def _score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        item_ids = np.asarray(item_ids)
        combined = np.zeros(len(item_ids), dtype=np.float64)
        any_contribution = False

        for name, model in self.models.items():
            raw = model.score_items(user_id, item_ids)
            if not np.isfinite(raw).any():
                continue  # this model knows nothing about this user
            weight = float(self.weights.get(name, 1.0))
            combined += weight * _minmax(raw)
            any_contribution = True

        if not any_contribution:
            # Cold start: nobody could personalise, so rank by popularity.
            return _minmax(self.fallback.score_items(user_id, item_ids))
        return combined
