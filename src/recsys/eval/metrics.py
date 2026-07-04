"""Top-k ranking metrics for offline recommender evaluation.

All metrics take a ranked list of recommended item ids and a set of relevant
(held-out, positively-rated) item ids for one user, and return a scalar. The
harness averages them over users. I implemented them directly rather than
pulling in a metrics library so the definitions are explicit and testable — the
exact form of NDCG and MAP varies between sources, and here they are pinned:

* **precision@k** — fraction of the top-k that are relevant.
* **recall@k** — fraction of the user's relevant items that appear in the top-k.
* **average precision@k** — mean of precision evaluated at each relevant hit,
  normalised by ``min(k, n_relevant)`` (so a user with few relevant items can
  still reach 1.0). MAP is this averaged over users.
* **NDCG@k** — DCG with binary gains and log2 discount, divided by the ideal DCG.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of the top-``k`` recommendations that are relevant."""
    if k <= 0:
        return 0.0
    top = recommended[:k]
    if not top:
        return 0.0
    hits = sum(1 for item in top if item in relevant)
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant items captured in the top-``k``."""
    if not relevant:
        return 0.0
    top = recommended[:k]
    hits = sum(1 for item in top if item in relevant)
    return hits / len(relevant)


def average_precision_at_k(
    recommended: Sequence[int], relevant: set[int], k: int
) -> float:
    """Average precision at ``k`` for a single user."""
    if not relevant:
        return 0.0
    top = recommended[:k]
    hits = 0
    score = 0.0
    for rank, item in enumerate(top, start=1):
        if item in relevant:
            hits += 1
            score += hits / rank
    return score / min(k, len(relevant))


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Normalised discounted cumulative gain at ``k`` with binary relevance."""
    if not relevant:
        return 0.0
    top = recommended[:k]
    gains = np.array([1.0 if item in relevant else 0.0 for item in top])
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
    dcg = float((gains * discounts).sum())

    ideal_hits = min(len(relevant), k)
    idcg = float((1.0 / np.log2(np.arange(2, ideal_hits + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """1.0 if any of the top-``k`` is relevant, else 0.0."""
    top = recommended[:k]
    return 1.0 if any(item in relevant for item in top) else 0.0
