"""Biased matrix factorization trained with SGD, implemented from scratch.

This is the FunkSVD-style model that won a lot of the Netflix Prize: predict a
rating as

    r_hat(u, i) = mu + b_u + b_i + p_u . q_i

where ``mu`` is the global mean, ``b_u`` / ``b_i`` are user and item biases, and
``p_u`` / ``q_i`` are latent factor vectors. Everything is learned by stochastic
gradient descent on the regularised squared error over observed ratings. I wrote
the training loop by hand (rather than reaching for ``surprise`` or ``implicit``)
because the point of the project is to show I understand what these libraries do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recsys.data.matrix import InteractionMatrix, build_interaction_matrix
from recsys.logging_utils import get_logger
from recsys.models.base import BaseRecommender

logger = get_logger(__name__)


@dataclass
class MFConfig:
    """Hyper-parameters for :class:`MatrixFactorizationRecommender`."""

    n_factors: int = 32
    n_epochs: int = 30
    lr: float = 0.01
    reg: float = 0.05
    init_scale: float = 0.1
    random_seed: int = 42


class MatrixFactorizationRecommender(BaseRecommender):
    """Biased SGD matrix factorization.

    The latent factors and biases are updated one observed rating at a time.
    Training reports RMSE on the fitted ratings each epoch so convergence is
    visible in the logs and testable.
    """

    name = "matrix_factorization"

    def __init__(self, config: MFConfig | None = None) -> None:
        super().__init__()
        self.config = config or MFConfig()
        self._im: InteractionMatrix | None = None
        self.global_mean_: float = 0.0
        self.user_bias_: np.ndarray = np.empty(0)
        self.item_bias_: np.ndarray = np.empty(0)
        self.user_factors_: np.ndarray = np.empty((0, 0))
        self.item_factors_: np.ndarray = np.empty((0, 0))
        self.train_rmse_: list[float] = []

    def fit(self, ratings: pd.DataFrame) -> "MatrixFactorizationRecommender":
        cfg = self.config
        im = build_interaction_matrix(ratings)
        self._im = im
        coo = im.matrix.tocoo()
        users = coo.row.astype(np.int64)
        items = coo.col.astype(np.int64)
        vals = coo.data.astype(np.float64)

        rng = np.random.default_rng(cfg.random_seed)
        self.global_mean_ = float(vals.mean())
        self.user_bias_ = np.zeros(im.n_users)
        self.item_bias_ = np.zeros(im.n_items)
        self.user_factors_ = rng.normal(
            0.0, cfg.init_scale, size=(im.n_users, cfg.n_factors)
        )
        self.item_factors_ = rng.normal(
            0.0, cfg.init_scale, size=(im.n_items, cfg.n_factors)
        )

        n = len(vals)
        self.train_rmse_ = []
        for epoch in range(cfg.n_epochs):
            order = rng.permutation(n)
            sq_err = 0.0
            for idx in order:
                u, i, r = users[idx], items[idx], vals[idx]
                pred = (
                    self.global_mean_
                    + self.user_bias_[u]
                    + self.item_bias_[i]
                    + self.user_factors_[u] @ self.item_factors_[i]
                )
                err = r - pred
                sq_err += err * err

                # Cache factor rows before the in-place update so both use the
                # pre-update values (the coupled SGD step).
                pu = self.user_factors_[u].copy()
                qi = self.item_factors_[i]

                self.user_bias_[u] += cfg.lr * (err - cfg.reg * self.user_bias_[u])
                self.item_bias_[i] += cfg.lr * (err - cfg.reg * self.item_bias_[i])
                self.user_factors_[u] += cfg.lr * (err * qi - cfg.reg * pu)
                self.item_factors_[i] += cfg.lr * (err * pu - cfg.reg * qi)

            rmse = float(np.sqrt(sq_err / n))
            self.train_rmse_.append(rmse)
            if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == cfg.n_epochs - 1:
                logger.info("MF epoch %2d/%d  train RMSE %.4f", epoch + 1, cfg.n_epochs, rmse)

        self._fitted = True
        return self

    @property
    def all_items(self) -> np.ndarray:
        assert self._im is not None
        return self._im.item_ids

    def predict(self, user_id: int, item_id: int) -> float:
        """Predicted rating for one (user, item) pair, clipped to [0.5, 5]."""
        self._check_fitted()
        assert self._im is not None
        u = self._im.user_index.get(int(user_id))
        i = self._im.item_index.get(int(item_id))
        if u is None or i is None:
            return self.global_mean_
        pred = (
            self.global_mean_
            + self.user_bias_[u]
            + self.item_bias_[i]
            + self.user_factors_[u] @ self.item_factors_[i]
        )
        return float(np.clip(pred, 0.5, 5.0))

    def _score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        assert self._im is not None
        u = self._im.user_index.get(int(user_id))
        if u is None:
            return np.full(len(item_ids), -np.inf)

        cols = np.array(
            [self._im.item_index.get(int(m), -1) for m in item_ids], dtype=np.int64
        )
        scores = np.full(len(item_ids), -np.inf)
        known = cols >= 0
        if known.any():
            qi = self.item_factors_[cols[known]]
            bi = self.item_bias_[cols[known]]
            dot = qi @ self.user_factors_[u]
            scores[known] = self.global_mean_ + self.user_bias_[u] + bi + dot
        return scores
