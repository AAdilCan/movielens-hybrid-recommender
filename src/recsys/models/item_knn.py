"""Item-item k-nearest-neighbours collaborative filtering.

The idea: two movies are similar if the same users tend to rate them alike. I
build an item-item cosine similarity matrix from the user-item matrix, keep only
each item's top-``k`` neighbours (which sparsifies the model and cuts noise from
weakly-correlated pairs), and score a candidate for a user by how similar it is
to the items that user already rated highly.

Item-based CF is the classic choice for this shape of data because there are far
fewer items than user-item pairs to compare at predict time, and item
similarities are more stable over time than user similarities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from recsys.data.matrix import InteractionMatrix, build_interaction_matrix
from recsys.logging_utils import get_logger
from recsys.models.base import BaseRecommender

logger = get_logger(__name__)


class ItemKNNRecommender(BaseRecommender):
    """Top-k item-item collaborative filter with cosine similarity.

    Args:
        k_neighbors: How many nearest items to keep per item. Smaller values
            denoise the similarity matrix and speed up scoring.
        mean_center: Subtract each item's mean rating before computing
            similarity, which removes popularity/scale effects so cosine
            reflects co-rating *patterns* rather than absolute levels.
    """

    name = "item_knn"

    def __init__(self, k_neighbors: int = 40, mean_center: bool = True) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors
        self.mean_center = mean_center
        self._im: InteractionMatrix | None = None
        self._sim: sparse.csr_matrix | None = None
        self._sim_abs_sum: np.ndarray = np.empty(0)
        self._item_means: np.ndarray = np.empty(0)

    def fit(self, ratings: pd.DataFrame) -> "ItemKNNRecommender":
        im = build_interaction_matrix(ratings)
        self._im = im
        user_item = im.matrix

        # Item vectors live in the columns; transpose so each row is an item.
        item_user = user_item.T.tocsr()
        if self.mean_center:
            item_user, self._item_means = _mean_center_rows(item_user)
        else:
            self._item_means = np.zeros(im.n_items)

        sim = cosine_similarity(item_user, dense_output=False)
        sim.setdiag(0.0)  # an item is not its own neighbour
        sim = _keep_top_k_per_row(sim.tocsr(), self.k_neighbors)
        self._sim = sim.tocsr()
        # Per-item sum of |similarity| to its neighbours; the score denominator.
        self._sim_abs_sum = np.asarray(abs(self._sim).sum(axis=1)).ravel()
        self._fitted = True
        logger.info(
            "Fit item-kNN: %d items, top-%d neighbours, %d similarity edges",
            im.n_items,
            self.k_neighbors,
            self._sim.nnz,
        )
        return self

    @property
    def all_items(self) -> np.ndarray:
        assert self._im is not None
        return self._im.item_ids

    def _score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        assert self._im is not None and self._sim is not None
        im = self._im
        if not im.has_user(user_id):
            return np.full(len(item_ids), -np.inf)

        # The user's rated items as a dense item-length vector of ratings.
        user_row = im.user_row(user_id)
        rated_cols = user_row.indices
        rated_vals = user_row.data
        if rated_cols.size == 0:
            return np.full(len(item_ids), -np.inf)

        prefs = np.zeros(im.n_items)
        prefs[rated_cols] = rated_vals

        # Score every item at once: (S @ prefs)_i is the similarity-weighted sum
        # of the user's ratings over item i's neighbours; divide by the neighbour
        # |similarity| mass to get a weighted average. One sparse mat-vec beats a
        # Python loop over candidates.
        numer = self._sim.dot(prefs)
        full = np.divide(
            numer,
            self._sim_abs_sum,
            out=np.full(im.n_items, -np.inf),
            where=self._sim_abs_sum > 0,
        )

        cols = np.array(
            [im.item_index.get(int(m), -1) for m in item_ids], dtype=np.int64
        )
        scores = np.full(len(item_ids), -np.inf)
        known = cols >= 0
        scores[known] = full[cols[known]]
        return scores


def _mean_center_rows(mat: sparse.csr_matrix) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Subtract each row's mean over its *observed* (non-zero) entries."""
    mat = mat.tocsr(copy=True)
    sums = np.asarray(mat.sum(axis=1)).ravel()
    counts = np.diff(mat.indptr)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    # Subtract the row mean only from stored entries, keeping structural zeros.
    mat.data -= np.repeat(means, counts)
    return mat, means


def _keep_top_k_per_row(mat: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    """Zero out all but the ``k`` largest entries in each row."""
    mat = mat.tocsr(copy=True)
    for row in range(mat.shape[0]):
        start, end = mat.indptr[row], mat.indptr[row + 1]
        n = end - start
        if n <= k:
            continue
        segment = mat.data[start:end]
        # Indices of the smallest (n-k) entries within this row -> drop them.
        drop = np.argpartition(segment, n - k)[: n - k]
        mat.data[start:end][drop] = 0.0
    mat.eliminate_zeros()
    return mat
