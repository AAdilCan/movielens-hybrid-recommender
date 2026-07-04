"""Tests for the content-based model and the hybrid ensemble."""

from __future__ import annotations

import numpy as np
import pytest

from recsys.models.content import ContentRecommender, _movie_document
from recsys.models.hybrid import HybridRecommender, _minmax
from recsys.models.popularity import PopularityRecommender


def test_movie_document_strips_year_and_lowercases():
    doc = _movie_document("Toy Story (1995)", "Adventure|Animation")
    assert "1995" not in doc
    assert doc == "toy story adventure animation"


def test_movie_document_handles_no_genres():
    doc = _movie_document("Mystery (2000)", "(no genres listed)")
    assert doc.strip() == "mystery"


def test_content_profiles_reflect_liked_genres(toy_ratings, toy_movies):
    model = ContentRecommender(toy_movies, positive_threshold=4.0).fit(toy_ratings)
    # User 2 liked movie 10 (Adventure/Animation/Children) and movie 16 (Drama).
    # Movie 17 (Action/Adventure/Romance) shares 'adventure' -> positive score.
    score = model.score_items(2, np.array([17]))[0]
    assert np.isfinite(score)
    assert score > 0


def test_content_unknown_user_scores_neg_inf(toy_ratings, toy_movies):
    model = ContentRecommender(toy_movies).fit(toy_ratings)
    assert np.all(np.isneginf(model.score_items(999, model.all_items)))


def test_content_scores_cold_item_from_metadata(toy_ratings, toy_movies):
    # Movie 15 (Action) is present in metadata; content can score it even though
    # user 1 never rated it, because scoring only needs the item's vector.
    model = ContentRecommender(toy_movies).fit(toy_ratings)
    s = model.score_items(1, np.array([15]))[0]
    assert np.isfinite(s)


def test_minmax_maps_to_unit_range():
    out = _minmax(np.array([1.0, 3.0, 5.0]))
    assert out.min() == 0.0 and out.max() == 1.0


def test_minmax_treats_neg_inf_as_zero():
    out = _minmax(np.array([-np.inf, 2.0, 4.0]))
    assert out[0] == 0.0
    assert out[2] == pytest.approx(1.0)


def test_minmax_all_equal_maps_to_one():
    out = _minmax(np.array([2.0, 2.0, 2.0]))
    assert np.allclose(out, 1.0)


def test_hybrid_requires_a_component():
    with pytest.raises(ValueError):
        HybridRecommender({})


def test_hybrid_blends_components(toy_ratings, toy_movies):
    models = {
        "popularity": PopularityRecommender(),
        "content": ContentRecommender(toy_movies),
    }
    hybrid = HybridRecommender(models, weights={"popularity": 1.0, "content": 1.0})
    hybrid.fit(toy_ratings)
    recs = hybrid.recommend(1, k=5, exclude=set())
    assert len(recs) > 0
    assert set(recs).issubset(set(int(i) for i in hybrid.all_items))


def test_hybrid_cold_start_falls_back_to_popularity(toy_ratings, toy_movies):
    # Content alone cannot score an unknown user; the hybrid should still return
    # results via the popularity fallback.
    models = {"content": ContentRecommender(toy_movies)}
    hybrid = HybridRecommender(models, fallback=PopularityRecommender())
    hybrid.fit(toy_ratings)
    recs = hybrid.recommend(9999, k=3)
    assert len(recs) == 3
