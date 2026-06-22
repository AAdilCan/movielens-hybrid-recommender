"""Hybrid movie recommender on the MovieLens dataset.

Combines a popularity baseline, item-item collaborative filtering, matrix
factorization and a content-based model into a weighted ensemble, evaluated
with ranking metrics (precision@k, recall@k, MAP, NDCG) on a temporal split.
"""

__version__ = "0.1.0"
