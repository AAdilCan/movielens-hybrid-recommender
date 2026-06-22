"""Project paths and dataset configuration.

All tunable knobs live here so the rest of the code can stay free of magic
numbers. Paths are resolved relative to the repository root, which keeps the
package importable from anywhere (tests, notebooks, the CLI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Repository root is two parents up from this file: src/recsys/config.py
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
ARTIFACTS_DIR: Path = ROOT_DIR / "artifacts"
REPORTS_DIR: Path = ROOT_DIR / "reports"


@dataclass(frozen=True)
class DatasetConfig:
    """Where to fetch a MovieLens release and what files it contains.

    The "small" (100k) release is the default because it downloads in seconds
    and trains on a laptop in well under a minute, which keeps the evaluation
    loop fast. The 1M release is wired up for anyone who wants stronger numbers.
    """

    name: str
    url: str
    # Directory created inside the zip, e.g. "ml-latest-small".
    archive_subdir: str
    ratings_file: str = "ratings.csv"
    movies_file: str = "movies.csv"
    sep: str = ","

    @property
    def raw_path(self) -> Path:
        return RAW_DIR / self.archive_subdir


DATASETS: dict[str, DatasetConfig] = {
    "small": DatasetConfig(
        name="small",
        url="https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        archive_subdir="ml-latest-small",
    ),
    "1m": DatasetConfig(
        name="1m",
        url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        archive_subdir="ml-1m",
        ratings_file="ratings.dat",
        movies_file="movies.dat",
        sep="::",
    ),
}

DEFAULT_DATASET = "small"


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation knobs shared across models."""

    # Ratings at or above this value count as a positive (relevant) interaction.
    positive_threshold: float = 4.0
    # Cut-offs for ranking metrics.
    k_values: tuple[int, ...] = (5, 10, 20)
    # Fraction of the most recent interactions held out per user.
    test_fraction: float = 0.2
    # Users with fewer than this many ratings are dropped before splitting.
    min_user_ratings: int = 5
    random_seed: int = 42


@dataclass(frozen=True)
class HybridWeights:
    """Default mixing weights for the ensemble (normalised at fusion time)."""

    item_knn: float = 1.0
    matrix_factorization: float = 1.0
    content: float = 0.5
    weights: dict[str, float] = field(default_factory=dict)


def get_dataset(name: str = DEFAULT_DATASET) -> DatasetConfig:
    """Return the dataset config for ``name`` or raise a helpful error."""
    if name not in DATASETS:
        available = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset {name!r}. Available: {available}.")
    return DATASETS[name]


def ensure_dirs() -> None:
    """Create the data/artifact/report directories if they do not exist."""
    for directory in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
