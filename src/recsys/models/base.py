"""Common interface shared by every recommender.

Every model — the popularity baseline, the collaborative ones and the hybrid
ensemble — implements the same tiny contract: ``fit`` on a ratings DataFrame and
``recommend`` a ranked list of movie ids for a user. Keeping the surface this
small is what lets the evaluation harness and the ensemble treat all models
interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """Abstract base class for all recommenders.

    Subclasses must implement :meth:`fit` and :meth:`_score`. The public
    :meth:`recommend` and :meth:`score_items` methods are provided here so the
    already-seen filtering and top-k logic live in one place.
    """

    #: Human-readable name used in results tables and logs.
    name: str = "base"

    def __init__(self) -> None:
        self._fitted = False

    @abstractmethod
    def fit(self, ratings: pd.DataFrame) -> "BaseRecommender":
        """Train the model on a ratings table and return ``self``."""

    @abstractmethod
    def _score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        """Return an affinity score for ``user_id`` against each item in ``item_ids``.

        Higher means more relevant. Items the model cannot score (e.g. unknown
        to a collaborative model) should receive ``-inf`` so they never surface.
        """

    @property
    def all_items(self) -> np.ndarray:
        """Every item id the model can recommend. Set during :meth:`fit`."""
        raise NotImplementedError

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name} must be fit before use.")

    def score_items(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        """Public wrapper around :meth:`_score` with a fitted-state check."""
        self._check_fitted()
        return self._score(user_id, np.asarray(item_ids))

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude: set[int] | None = None,
        candidates: np.ndarray | None = None,
    ) -> list[int]:
        """Return the top-``k`` movie ids for ``user_id``, best first.

        Args:
            user_id: MovieLens user id to recommend for.
            k: Number of items to return.
            exclude: Movie ids to drop from the ranking (typically the user's
                training interactions, so we don't recommend what they've seen).
            candidates: Restrict scoring to this pool of item ids; defaults to
                :attr:`all_items`.

        Returns:
            A list of at most ``k`` movie ids ordered by descending score.
            Items scoring ``-inf`` are never returned.
        """
        self._check_fitted()
        items = self.all_items if candidates is None else np.asarray(candidates)
        if exclude:
            mask = ~np.isin(items, list(exclude))
            items = items[mask]
        if items.size == 0:
            return []

        scores = self._score(user_id, items)
        finite = np.isfinite(scores)
        if not finite.any():
            return []
        items, scores = items[finite], scores[finite]

        # argpartition for the top-k, then sort just that slice — cheaper than a
        # full sort when the catalogue is large.
        k = min(k, items.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [int(i) for i in items[top]]
