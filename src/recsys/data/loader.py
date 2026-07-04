"""Load and validate the MovieLens ratings and movies tables.

The two MovieLens releases I support have slightly different on-disk formats:
the "small" release ships comma-separated CSVs with a header row, while the 1M
release uses ``::``-separated ``.dat`` files with no header and latin-1 text in
the titles. Both quirks are absorbed here so the rest of the pipeline only ever
sees clean, typed DataFrames with a stable schema.
"""

from __future__ import annotations

import pandas as pd

from recsys.config import DatasetConfig, get_dataset
from recsys.data.download import download_dataset
from recsys.logging_utils import get_logger

logger = get_logger(__name__)

RATINGS_COLUMNS = ["userId", "movieId", "rating", "timestamp"]
MOVIES_COLUMNS = ["movieId", "title", "genres"]

# The genre placeholder MovieLens uses when a film has no listed genres.
NO_GENRES = "(no genres listed)"


class SchemaError(ValueError):
    """Raised when a loaded table does not match the expected schema."""


def _read_table(path, sep: str, columns: list[str]) -> pd.DataFrame:
    """Read a delimited MovieLens file into a DataFrame.

    A single-character separator means the modern CSV format with a header
    row; a multi-character separator (``::``) means the legacy ``.dat`` format,
    which has no header and needs the slower Python parser plus latin-1 decoding
    for accented titles.
    """
    if len(sep) == 1:
        return pd.read_csv(path, sep=sep)
    return pd.read_csv(
        path,
        sep=sep,
        names=columns,
        header=None,
        engine="python",
        encoding="latin-1",
    )


def _validate(df: pd.DataFrame, required: list[str], table: str) -> None:
    """Check that ``required`` columns are present and key columns are non-null."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SchemaError(
            f"{table} table is missing columns {missing}; got {list(df.columns)}."
        )
    nulls = df[required].isna().any()
    bad = nulls[nulls].index.tolist()
    if bad:
        raise SchemaError(f"{table} table has null values in columns {bad}.")


def load_ratings(
    dataset: str | DatasetConfig = "small", *, download: bool = True
) -> pd.DataFrame:
    """Return the ratings table with columns ``[userId, movieId, rating, timestamp]``.

    Args:
        dataset: Dataset key or a :class:`DatasetConfig`.
        download: Fetch the release first if it is not already on disk.

    Returns:
        A DataFrame with integer ids, float ratings and an integer Unix
        timestamp, sorted by ``(userId, timestamp)`` for deterministic splitting.
    """
    cfg = dataset if isinstance(dataset, DatasetConfig) else get_dataset(dataset)
    if download:
        download_dataset(cfg.name)

    path = cfg.raw_path / cfg.ratings_file
    df = _read_table(path, cfg.sep, RATINGS_COLUMNS)
    _validate(df, RATINGS_COLUMNS, "ratings")

    df = df[RATINGS_COLUMNS].copy()
    df["userId"] = df["userId"].astype("int64")
    df["movieId"] = df["movieId"].astype("int64")
    df["rating"] = df["rating"].astype("float64")
    df["timestamp"] = df["timestamp"].astype("int64")

    if (df["rating"] < 0).any() or (df["rating"] > 5).any():
        raise SchemaError("ratings table has values outside the 0-5 range.")

    df = df.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
    logger.info(
        "Loaded %d ratings from %d users on %d movies",
        len(df),
        df["userId"].nunique(),
        df["movieId"].nunique(),
    )
    return df


def load_movies(
    dataset: str | DatasetConfig = "small", *, download: bool = True
) -> pd.DataFrame:
    """Return the movies table with columns ``[movieId, title, genres]``.

    Genres arrive as a pipe-delimited string (e.g. ``"Action|Sci-Fi"``); they
    are kept as text here and tokenised later by the content-based model.
    """
    cfg = dataset if isinstance(dataset, DatasetConfig) else get_dataset(dataset)
    if download:
        download_dataset(cfg.name)

    path = cfg.raw_path / cfg.movies_file
    df = _read_table(path, cfg.sep, MOVIES_COLUMNS)
    _validate(df, MOVIES_COLUMNS, "movies")

    df = df[MOVIES_COLUMNS].copy()
    df["movieId"] = df["movieId"].astype("int64")
    df["title"] = df["title"].astype("string")
    df["genres"] = df["genres"].astype("string").fillna(NO_GENRES)

    if df["movieId"].duplicated().any():
        raise SchemaError("movies table contains duplicate movieId values.")

    logger.info("Loaded %d movies", len(df))
    return df
