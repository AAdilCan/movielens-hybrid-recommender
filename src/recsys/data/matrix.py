"""Sparse user-item interaction matrix.

Collaborative models need a compact numeric view of the ratings: rows are
users, columns are movies, and the stored value is the rating. MovieLens ids
are sparse and non-contiguous (movie ids run past 190000 in the full release),
so I map them to dense 0-based indices and keep the mapping around for turning
model output back into real movie ids.

The matrix is CSR because item-item similarity and matrix factorisation both
iterate over user rows, and CSR gives cheap row slicing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class InteractionMatrix:
    """A user-item rating matrix plus the id<->index mappings.

    Attributes:
        matrix: CSR matrix of shape ``(n_users, n_items)`` holding ratings.
        user_ids: Array where ``user_ids[i]`` is the MovieLens id of row ``i``.
        item_ids: Array where ``item_ids[j]`` is the MovieLens id of column ``j``.
        user_index: Reverse map from MovieLens user id to row index.
        item_index: Reverse map from MovieLens movie id to column index.
    """

    matrix: sparse.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_index: dict[int, int]
    item_index: dict[int, int]

    @property
    def n_users(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_items(self) -> int:
        return self.matrix.shape[1]

    def has_user(self, user_id: int) -> bool:
        return user_id in self.user_index

    def has_item(self, item_id: int) -> bool:
        return item_id in self.item_index

    def user_row(self, user_id: int) -> sparse.csr_matrix:
        """Return the (1, n_items) rating row for ``user_id``."""
        return self.matrix.getrow(self.user_index[user_id])

    def seen_items(self, user_id: int) -> set[int]:
        """Return the set of MovieLens movie ids a user has already rated."""
        if user_id not in self.user_index:
            return set()
        cols = self.user_row(user_id).indices
        return {int(self.item_ids[c]) for c in cols}


def build_interaction_matrix(ratings: pd.DataFrame) -> InteractionMatrix:
    """Build a CSR user-item matrix from a ratings table.

    Args:
        ratings: DataFrame with ``userId``, ``movieId`` and ``rating`` columns
            (typically the training split).

    Returns:
        An :class:`InteractionMatrix`. Item ordering follows sorted movie id so
        the layout is deterministic across runs.
    """
    user_ids = np.sort(ratings["userId"].unique())
    item_ids = np.sort(ratings["movieId"].unique())
    user_index = {int(u): i for i, u in enumerate(user_ids)}
    item_index = {int(m): j for j, m in enumerate(item_ids)}

    rows = ratings["userId"].map(user_index).to_numpy()
    cols = ratings["movieId"].map(item_index).to_numpy()
    data = ratings["rating"].to_numpy(dtype=np.float64)

    matrix = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(user_ids), len(item_ids))
    )
    # Duplicate (user, item) pairs would sum; the loader guarantees one rating
    # per pair, but summing duplicates defensively keeps the matrix well-formed.
    matrix.sum_duplicates()
    logger.info(
        "Built %d x %d interaction matrix with %d non-zeros (%.2f%% dense)",
        matrix.shape[0],
        matrix.shape[1],
        matrix.nnz,
        100.0 * matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
    )
    return InteractionMatrix(
        matrix=matrix,
        user_ids=user_ids,
        item_ids=item_ids,
        user_index=user_index,
        item_index=item_index,
    )
