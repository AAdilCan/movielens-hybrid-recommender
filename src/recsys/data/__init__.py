"""Data acquisition and preparation for MovieLens."""

from recsys.data.download import download_dataset, is_downloaded
from recsys.data.loader import load_movies, load_ratings
from recsys.data.matrix import InteractionMatrix, build_interaction_matrix
from recsys.data.split import (
    filter_min_interactions,
    positive_interactions,
    temporal_split,
)

__all__ = [
    "download_dataset",
    "is_downloaded",
    "load_ratings",
    "load_movies",
    "InteractionMatrix",
    "build_interaction_matrix",
    "filter_min_interactions",
    "positive_interactions",
    "temporal_split",
]
