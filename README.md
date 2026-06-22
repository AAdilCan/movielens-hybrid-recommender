# MovieLens Hybrid Recommender

A movie recommendation system on the MovieLens dataset that compares four
approaches under one evaluation harness and blends them into a hybrid ensemble:

- **Popularity** — a non-personalised baseline (most-rated, highest-rated).
- **Item-item collaborative filtering** — cosine similarity over the user-item matrix.
- **Matrix factorization** — latent factors learned with SGD on observed ratings.
- **Content-based** — TF-IDF over movie genres and titles, ranked by user profile.
- **Hybrid** — a weighted fusion of the personalised models with cold-start fallback.

Everything is evaluated with ranking metrics (**precision@k, recall@k, MAP, NDCG**)
plus catalogue **coverage**, using a **temporal leave-last-out** split so we never
train on a user's future.

> Status: work in progress. Day 1 ships the scaffold, data layer and exploration.
> The model implementations, evaluation tables and plots land over the week.

## Dataset

I use the [MovieLens](https://grouplens.org/datasets/movielens/) `latest-small`
release by default (it downloads in seconds and trains on a laptop). The 1M
release is wired up via `--dataset 1m`. Raw files are downloaded on demand into
`data/raw/` and never committed.

Headline stats for the small release:

| metric | value |
| --- | --- |
| users | 610 |
| movies (rated) | 9,724 |
| ratings | 100,836 |
| matrix density | 1.70% |
| mean rating | 3.50 |
| top 10% of movies hold | 60% of all ratings |

The last row is why a pure popularity model is a deceptively strong baseline and
why content-based signal matters for the long tail.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

# download the data and print the exploration stats above
python scripts/explore_data.py
```

## Project layout

```
src/recsys/
├── config.py           # paths, dataset registry, eval/hybrid settings
├── logging_utils.py    # shared logger
├── data/
│   └── download.py      # fetch + unpack MovieLens releases
├── models/              # popularity, item-kNN, MF, content, hybrid (coming)
├── eval/                # ranking metrics + evaluation harness (coming)
└── cli.py               # train / evaluate / recommend (coming)
scripts/explore_data.py  # reproducible dataset exploration
tests/                   # pytest suite (coming)
```

## Roadmap

- [x] Scaffold, packaging, config, logging, data download + exploration
- [ ] Data layer: loading, temporal split, schema validation
- [ ] Models: popularity, item-kNN, matrix factorization, content-based
- [ ] Hybrid ensemble + CLI
- [ ] Evaluation harness, results table, plots
- [ ] Test suite + cold-start handling
- [ ] Documentation + final results

## License

MIT
