"""Offline evaluation harness.

Trains every model on the temporal train split and scores its top-k
recommendations against each user's held-out positive interactions. The output
is a tidy results table (one row per model) plus, optionally, a CSV and
comparison plots written under ``reports/``.

The evaluation protocol matches the split: for a user we recommend from the
whole catalogue minus the items they interacted with in *train*, then check how
many of their held-out *positive* test items we ranked into the top-k. This is
the standard all-items ranking protocol — harder and more honest than scoring
against a handful of sampled negatives.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from recsys.config import EvalConfig, REPORTS_DIR, ensure_dirs
from recsys.data.loader import load_movies, load_ratings
from recsys.data.split import positive_interactions, temporal_split
from recsys.eval.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recsys.logging_utils import get_logger
from recsys.models import (
    BaseRecommender,
    ItemKNNRecommender,
    MatrixFactorizationRecommender,
    PopularityRecommender,
    build_default_hybrid,
)

logger = get_logger(__name__)


def build_model_zoo(movies: pd.DataFrame) -> dict[str, BaseRecommender]:
    """Instantiate every model that should appear in the comparison table."""
    from recsys.models import ContentRecommender

    return {
        "Popularity": PopularityRecommender(),
        "ItemKNN": ItemKNNRecommender(),
        "MatrixFactorization": MatrixFactorizationRecommender(),
        "Content": ContentRecommender(movies),
        "Hybrid": build_default_hybrid(movies),
    }


def evaluate_model(
    model: BaseRecommender,
    train: pd.DataFrame,
    test_positive: pd.DataFrame,
    config: EvalConfig,
) -> dict[str, float]:
    """Fit ``model`` and compute averaged ranking metrics over test users.

    Args:
        model: An unfitted recommender.
        train: Training interactions.
        test_positive: Held-out positive interactions (the ground truth).
        config: Evaluation knobs (``k_values`` etc.).

    Returns:
        A flat dict of ``metric@k`` -> value plus ``coverage`` and ``n_users``.
    """
    model.fit(train)
    max_k = max(config.k_values)

    seen_by_user = train.groupby("userId")["movieId"].agg(set).to_dict()
    relevant_by_user = test_positive.groupby("userId")["movieId"].agg(set).to_dict()

    metric_fns: dict[str, Callable] = {
        "precision": precision_at_k,
        "recall": recall_at_k,
        "map": average_precision_at_k,
        "ndcg": ndcg_at_k,
    }
    sums: dict[str, float] = {
        f"{m}@{k}": 0.0 for m in metric_fns for k in config.k_values
    }
    recommended_catalogue: set[int] = set()
    n_users = 0

    for user_id, relevant in relevant_by_user.items():
        if not relevant:
            continue
        seen = seen_by_user.get(user_id, set())
        recs = model.recommend(user_id, k=max_k, exclude=seen)
        if not recs:
            continue
        recommended_catalogue.update(recs[: max(config.k_values)])
        n_users += 1
        for k in config.k_values:
            for name, fn in metric_fns.items():
                sums[f"{name}@{k}"] += fn(recs, relevant, k)

    results = {
        key: (total / n_users if n_users else 0.0) for key, total in sums.items()
    }
    n_catalogue = len(model.all_items)
    results["coverage"] = (
        len(recommended_catalogue) / n_catalogue if n_catalogue else 0.0
    )
    results["n_users"] = float(n_users)
    logger.info("Evaluated %s over %d users", model.name, n_users)
    return results


def run_evaluation(
    dataset: str = "small",
    config: EvalConfig | None = None,
    write_report: bool = True,
) -> pd.DataFrame:
    """Run the full model comparison and return a results DataFrame.

    Args:
        dataset: Dataset key to load.
        config: Evaluation configuration; defaults to :class:`EvalConfig`.
        write_report: If True, save ``reports/results.csv`` and plots.

    Returns:
        A DataFrame indexed by model name with one column per metric.
    """
    config = config or EvalConfig()
    ratings = load_ratings(dataset)
    movies = load_movies(dataset)
    train, test = temporal_split(ratings, config)
    test_positive = positive_interactions(test, config)
    logger.info(
        "Evaluation set: %d train, %d positive test interactions",
        len(train),
        len(test_positive),
    )

    rows = []
    for label, model in build_model_zoo(movies).items():
        metrics = evaluate_model(model, train, test_positive, config)
        rows.append({"model": label, **metrics})

    results = pd.DataFrame(rows).round(4)
    results = results.sort_values(f"ndcg@{config.k_values[1]}", ascending=False)

    if write_report:
        _write_report(results, config)
    return results


def _write_report(results: pd.DataFrame, config: EvalConfig) -> None:
    """Persist the results table as CSV and draw comparison plots."""
    ensure_dirs()
    csv_path = REPORTS_DIR / "results.csv"
    results.to_csv(csv_path, index=False)
    logger.info("Wrote %s", csv_path)

    try:
        from recsys.eval.plots import plot_metric_comparison

        plot_metric_comparison(results, config, REPORTS_DIR)
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        logger.warning("Skipped plots: %s", exc)
