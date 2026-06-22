"""Quick exploratory pass over the MovieLens ratings.

Run with ``python scripts/explore_data.py`` after installing the package. It
downloads the small release if needed and prints the headline statistics I used
to make modelling decisions (sparsity, rating distribution, long-tail of item
popularity). Nothing here is imported by the library; it is a scratch pad I keep
in the repo so the numbers in the README are reproducible.
"""

from __future__ import annotations

import pandas as pd

from recsys.config import DEFAULT_DATASET, get_dataset
from recsys.data.download import download_dataset


def main() -> None:
    raw_path = download_dataset(DEFAULT_DATASET)
    dataset = get_dataset(DEFAULT_DATASET)

    ratings = pd.read_csv(raw_path / dataset.ratings_file)
    movies = pd.read_csv(raw_path / dataset.movies_file)

    n_users = ratings["userId"].nunique()
    n_items = ratings["movieId"].nunique()
    n_ratings = len(ratings)
    density = n_ratings / (n_users * n_items)

    print("=" * 60)
    print(f"MovieLens '{DEFAULT_DATASET}' release")
    print("=" * 60)
    print(f"users            : {n_users:,}")
    print(f"movies (rated)   : {n_items:,}")
    print(f"movies (catalog) : {len(movies):,}")
    print(f"ratings          : {n_ratings:,}")
    print(f"matrix density   : {density:.4%}")
    print()

    print("rating distribution")
    print(ratings["rating"].value_counts().sort_index().to_string())
    print(f"mean rating: {ratings['rating'].mean():.3f}")
    print()

    per_user = ratings.groupby("userId").size()
    per_item = ratings.groupby("movieId").size()
    print("ratings per user : "
          f"min={per_user.min()} median={per_user.median():.0f} max={per_user.max()}")
    print("ratings per item : "
          f"min={per_item.min()} median={per_item.median():.0f} max={per_item.max()}")

    # Long-tail check: share of ratings captured by the top 10% of movies.
    top_decile = per_item.sort_values(ascending=False)
    cut = max(1, len(top_decile) // 10)
    head_share = top_decile.iloc[:cut].sum() / n_ratings
    print(f"top 10% of movies hold {head_share:.1%} of all ratings")


if __name__ == "__main__":
    main()
