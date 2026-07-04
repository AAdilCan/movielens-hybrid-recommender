"""Shared pytest fixtures.

These build small, deterministic ratings tables in memory so the data-layer
tests never touch the network or the real MovieLens download.
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def toy_ratings() -> pd.DataFrame:
    """A tiny ratings table with known structure.

    Three users, monotonically increasing timestamps per user so the temporal
    order is unambiguous:

    - user 1 rated 5 movies
    - user 2 rated 4 movies
    - user 3 rated 2 movies
    """
    rows = [
        # user, movie, rating, timestamp
        (1, 10, 5.0, 100),
        (1, 11, 4.0, 101),
        (1, 12, 3.0, 102),
        (1, 13, 5.0, 103),
        (1, 14, 2.0, 104),
        (2, 10, 4.0, 200),
        (2, 12, 5.0, 201),
        (2, 15, 1.0, 202),
        (2, 16, 4.0, 203),
        (3, 11, 5.0, 300),
        (3, 17, 4.0, 301),
    ]
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


@pytest.fixture
def toy_movies() -> pd.DataFrame:
    """Movie metadata matching the ids used in :func:`toy_ratings`."""
    rows = [
        (10, "Toy Story (1995)", "Adventure|Animation|Children"),
        (11, "Heat (1995)", "Action|Crime|Thriller"),
        (12, "GoldenEye (1995)", "Action|Adventure|Thriller"),
        (13, "Casino (1995)", "Crime|Drama"),
        (14, "Sabrina (1995)", "Comedy|Romance"),
        (15, "Sudden Death (1995)", "Action"),
        (16, "Nixon (1995)", "Drama"),
        (17, "Cutthroat Island (1995)", "Action|Adventure|Romance"),
    ]
    return pd.DataFrame(rows, columns=["movieId", "title", "genres"])
