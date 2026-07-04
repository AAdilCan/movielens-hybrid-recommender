"""Comparison plots for the evaluation results.

Kept separate from the harness so the metric computation has no hard dependency
on matplotlib and the plots can be regenerated from a saved results table.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from recsys.config import EvalConfig  # noqa: E402
from recsys.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def plot_metric_comparison(
    results: pd.DataFrame, config: EvalConfig, out_dir: Path
) -> list[Path]:
    """Draw grouped bar charts comparing models on each ranking metric.

    Produces one subplot per metric family (precision, recall, MAP, NDCG),
    grouped by k. Returns the list of written file paths.
    """
    metrics = ["precision", "recall", "map", "ndcg"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    models = results["model"].tolist()

    for ax, metric in zip(axes.ravel(), metrics):
        width = 0.8 / len(config.k_values)
        x = range(len(models))
        for i, k in enumerate(config.k_values):
            col = f"{metric}@{k}"
            offsets = [xi + i * width for xi in x]
            ax.bar(offsets, results[col].to_numpy(), width=width, label=f"@{k}")
        ax.set_title(metric.upper())
        ax.set_xticks([xi + width for xi in x])
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Model comparison on MovieLens (temporal split)")
    fig.tight_layout()
    ranking_path = out_dir / "metric_comparison.png"
    fig.savefig(ranking_path, dpi=120)
    plt.close(fig)

    # A second, simpler figure: NDCG@10 vs coverage — accuracy/diversity tradeoff.
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    k_mid = config.k_values[len(config.k_values) // 2]
    ax2.scatter(results["coverage"], results[f"ndcg@{k_mid}"], s=80)
    for _, row in results.iterrows():
        ax2.annotate(
            row["model"],
            (row["coverage"], row[f"ndcg@{k_mid}"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    ax2.set_xlabel("Catalogue coverage")
    ax2.set_ylabel(f"NDCG@{k_mid}")
    ax2.set_title("Accuracy vs. coverage")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    tradeoff_path = out_dir / "accuracy_coverage.png"
    fig2.savefig(tradeoff_path, dpi=120)
    plt.close(fig2)

    logger.info("Wrote plots: %s, %s", ranking_path.name, tradeoff_path.name)
    return [ranking_path, tradeoff_path]
