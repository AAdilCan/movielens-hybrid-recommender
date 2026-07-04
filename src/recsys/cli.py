"""Command-line interface for training and querying the recommender.

Two subcommands:

* ``recsys recommend --user 1`` — fit the hybrid model on a temporal train split
  and print top-N recommendations for a user, with titles and genres.
* ``recsys evaluate`` — run the evaluation harness across all models and print
  the results table (delegates to :mod:`recsys.eval`).

Keeping the CLI thin — it only wires together the data, model and eval layers —
means the same code paths are exercised by the tests and by a human at the shell.
"""

from __future__ import annotations

import click
import pandas as pd

from recsys.config import DEFAULT_DATASET, EvalConfig
from recsys.data.loader import load_movies, load_ratings
from recsys.data.split import temporal_split
from recsys.logging_utils import get_logger
from recsys.models import build_default_hybrid

logger = get_logger(__name__)


@click.group()
def cli() -> None:
    """MovieLens hybrid recommender."""


@cli.command()
@click.option("--user", "user_id", type=int, required=True, help="MovieLens user id.")
@click.option("--dataset", default=DEFAULT_DATASET, help="Dataset key (small/1m).")
@click.option("--top-n", default=10, show_default=True, help="How many to recommend.")
def recommend(user_id: int, dataset: str, top_n: int) -> None:
    """Print top-N recommendations for a user from the hybrid model."""
    config = EvalConfig()
    ratings = load_ratings(dataset)
    movies = load_movies(dataset)
    train, _ = temporal_split(ratings, config)

    model = build_default_hybrid(movies, eval_config=config)
    model.fit(train)

    seen = set(train.loc[train["userId"] == user_id, "movieId"])
    recs = model.recommend(user_id, k=top_n, exclude=seen)
    if not recs:
        click.echo(f"No recommendations available for user {user_id}.")
        return

    meta = movies.set_index("movieId")
    click.echo(f"\nTop {len(recs)} recommendations for user {user_id}:\n")
    for rank, movie_id in enumerate(recs, start=1):
        row = meta.loc[movie_id]
        click.echo(f"{rank:>2}. {row['title']}  [{row['genres']}]")


@cli.command()
@click.option("--dataset", default=DEFAULT_DATASET, help="Dataset key (small/1m).")
@click.option(
    "--report/--no-report",
    default=True,
    show_default=True,
    help="Write a CSV/plots report under reports/.",
)
def evaluate(dataset: str, report: bool) -> None:
    """Evaluate all models on a temporal split and print a metrics table."""
    # Imported lazily so `recsys recommend` does not pay the eval import cost.
    from recsys.eval.harness import run_evaluation

    results = run_evaluation(dataset=dataset, write_report=report)
    click.echo("\n" + results.to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    cli()
