"""Content-based recommender over movie genres and titles.

Collaborative models only know about co-rating patterns; they say nothing about
*what* a movie is, and they fall apart for items nobody has rated yet. The
content model fixes both: it represents every movie by a TF-IDF vector built
from its genres and title tokens, builds a user profile as the rating-weighted
average of the vectors of movies that user liked, and scores candidates by
cosine similarity to that profile. Because item vectors come from metadata, a
brand-new movie can be scored the moment it has genres — no ratings required.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recsys.logging_utils import get_logger
from recsys.models.base import BaseRecommender

logger = get_logger(__name__)

# Strip a trailing "(1994)" style year and lowercase; genres use "|" as sep.
_YEAR_RE = re.compile(r"\(\d{4}\)\s*$")


def _movie_document(title: str, genres: str) -> str:
    """Turn a movie's title and genres into one bag-of-words string."""
    title = _YEAR_RE.sub("", str(title)).strip()
    genre_tokens = str(genres).replace("|", " ").replace("(no genres listed)", "")
    return f"{title} {genre_tokens}".lower()


class ContentRecommender(BaseRecommender):
    """TF-IDF content-based recommender.

    Args:
        movies: Movie metadata with ``movieId``, ``title`` and ``genres``.
        positive_threshold: Ratings at or above this build the user profile;
            lower ratings are ignored so the profile reflects what a user likes.
        max_features: Cap on the TF-IDF vocabulary size.
    """

    name = "content"

    def __init__(
        self,
        movies: pd.DataFrame,
        positive_threshold: float = 4.0,
        max_features: int = 5000,
    ) -> None:
        super().__init__()
        self.movies = movies.reset_index(drop=True)
        self.positive_threshold = positive_threshold
        self.max_features = max_features
        self._item_ids: np.ndarray = np.empty(0, dtype=np.int64)
        self._item_index: dict[int, int] = {}
        self._item_vectors: sparse.csr_matrix | None = None
        self._user_profiles: dict[int, np.ndarray] = {}

    def fit(self, ratings: pd.DataFrame) -> "ContentRecommender":
        docs = [
            _movie_document(t, g)
            for t, g in zip(self.movies["title"], self.movies["genres"])
        ]
        vectorizer = TfidfVectorizer(
            max_features=self.max_features, token_pattern=r"[a-z0-9]+"
        )
        item_vectors = vectorizer.fit_transform(docs)
        item_vectors = normalize(item_vectors)  # unit rows -> dot == cosine

        self._item_ids = self.movies["movieId"].to_numpy(dtype=np.int64)
        self._item_index = {int(m): i for i, m in enumerate(self._item_ids)}
        self._item_vectors = item_vectors.tocsr()

        self._user_profiles = self._build_user_profiles(ratings)
        self._fitted = True
        logger.info(
            "Fit content model: %d items, %d-term vocab, %d user profiles",
            item_vectors.shape[0],
            item_vectors.shape[1],
            len(self._user_profiles),
        )
        return self

    def _build_user_profiles(self, ratings: pd.DataFrame) -> dict[int, np.ndarray]:
        """Rating-weighted average of the vectors of each user's liked movies."""
        assert self._item_vectors is not None
        liked = ratings[ratings["rating"] >= self.positive_threshold]
        profiles: dict[int, np.ndarray] = {}
        for user_id, group in liked.groupby("userId"):
            rows, weights = [], []
            for movie_id, rating in zip(group["movieId"], group["rating"]):
                idx = self._item_index.get(int(movie_id))
                if idx is not None:
                    rows.append(idx)
                    weights.append(float(rating))
            if not rows:
                continue
            w = np.asarray(weights)
            profile = self._item_vectors[rows].multiply(w[:, None]).sum(axis=0)
            profile = np.asarray(profile).ravel() / w.sum()
            profiles[int(user_id)] = profile
        return profiles

    @property
    def all_items(self) -> np.ndarray:
        return self._item_ids

    def item_vector(self, movie_id: int) -> np.ndarray | None:
        """Return the TF-IDF vector for a movie, or ``None`` if unknown."""
        assert self._item_vectors is not None
        idx = self._item_index.get(int(movie_id))
        if idx is None:
            return None
        return np.asarray(self._item_vectors[idx].todense()).ravel()

    def _score(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        assert self._item_vectors is not None
        profile = self._user_profiles.get(int(user_id))
        if profile is None:
            return np.full(len(item_ids), -np.inf)

        cols = np.array(
            [self._item_index.get(int(m), -1) for m in item_ids], dtype=np.int64
        )
        scores = np.full(len(item_ids), -np.inf)
        known = cols >= 0
        if known.any():
            vecs = self._item_vectors[cols[known]]
            scores[known] = vecs.dot(profile)
        return scores
