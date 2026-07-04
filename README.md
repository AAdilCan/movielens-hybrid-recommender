# MovieLens Hybrid Recommender

A movie recommendation system on the MovieLens dataset that compares five
approaches under one evaluation harness and blends the best of them into a
hybrid ensemble:

- **Popularity** — a non-personalised baseline (damped mean rating).
- **Item-item collaborative filtering** — top-k cosine similarity over the user-item matrix.
- **Matrix factorization** — biased latent factors learned with SGD from scratch.
- **Content-based** — TF-IDF over movie genres and titles, ranked by a user profile.
- **Hybrid** — a weighted, rank-normalised fusion of the personalised models with a cold-start fallback.

Everything is evaluated with ranking metrics (**precision@k, recall@k, MAP, NDCG**)
plus catalogue **coverage**, using a **temporal leave-last-out** split so a model
is never trained on a user's future.

## Results

Small MovieLens release, temporal split (last 20% of each user's history held
out), all-items ranking protocol, evaluated over 592 users. Sorted by NDCG@10.

| Model | Precision@10 | Recall@10 | MAP@10 | NDCG@10 | Coverage |
| --- | --- | --- | --- | --- | --- |
| **Hybrid** | **0.072** | **0.067** | **0.045** | **0.095** | 0.115 |
| ItemKNN | 0.058 | 0.060 | 0.032 | 0.076 | **0.163** |
| Popularity | 0.059 | 0.048 | 0.035 | 0.074 | 0.014 |
| MatrixFactorization | 0.021 | 0.014 | 0.009 | 0.025 | 0.082 |
| Content | 0.005 | 0.004 | 0.002 | 0.005 | 0.093 |

![Model comparison](reports/metric_comparison.png)

The hybrid tops every accuracy metric. Two results are worth calling out because
they are easy to get wrong:

- **Popularity is a strong baseline.** On MovieLens the top 10% of movies hold
  60% of all ratings, so "recommend what's popular" is hard to beat. Any model
  that loses to it is not earning its complexity.
- **Rating-prediction MF underperforms at ranking.** The SGD factorization is
  trained to minimise rating error, not ranking loss, so its top-N lists are
  dominated by high-bias items. This is a known limitation of explicit-feedback
  MF and the reason implicit-feedback objectives (BPR, ALS) exist — see
  [DOCUMENTATION.md](DOCUMENTATION.md) for the full discussion.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# download the data and print the dataset exploration stats
python scripts/explore_data.py

# recommend for a single user (fits the hybrid on the train split first)
recsys recommend --user 1 --top-n 10

# run the full model comparison and write reports/results.csv + plots
recsys evaluate
```

Example recommendation output:

```
Top 5 recommendations for user 1:

 1. Terminator 2: Judgment Day (1991)  [Action|Sci-Fi]
 2. Snow White and the Seven Dwarfs (1937)  [Animation|Children|Drama|Fantasy|Musical]
 3. Cinderella (1950)  [Animation|Children|Fantasy|Musical|Romance]
 4. Aliens (1986)  [Action|Adventure|Horror|Sci-Fi]
 5. Dragonslayer (1981)  [Action|Adventure|Fantasy]
```

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

## Project layout

```
src/recsys/
├── config.py              # paths, dataset registry, eval/hybrid settings
├── logging_utils.py       # shared logger
├── data/
│   ├── download.py        # fetch + unpack MovieLens releases
│   ├── loader.py          # typed, schema-validated ratings/movies tables
│   ├── split.py           # temporal leave-last-out split + filtering
│   └── matrix.py          # sparse CSR user-item matrix + id mappings
├── models/
│   ├── base.py            # BaseRecommender interface (fit / recommend)
│   ├── popularity.py      # damped popularity baseline + cold-start fallback
│   ├── item_knn.py        # top-k item-item cosine CF
│   ├── matrix_factorization.py  # biased SGD MF, written from scratch
│   ├── content.py         # TF-IDF content model with user profiles
│   └── hybrid.py          # rank-normalised weighted ensemble
├── eval/
│   ├── metrics.py         # precision/recall/MAP/NDCG at k
│   ├── harness.py         # fit-and-score every model, build results table
│   └── plots.py           # comparison bar charts + accuracy/coverage scatter
└── cli.py                 # `recsys recommend` / `recsys evaluate`
scripts/explore_data.py    # reproducible dataset exploration
tests/                     # 47 pytest cases (metrics, split, models, CLI)
```

## Testing

```bash
pytest                     # 47 tests
pytest --cov=recsys        # with coverage
```

## Documentation

[DOCUMENTATION.md](DOCUMENTATION.md) is the deep dive: architecture, the
evaluation protocol, methodology and the engineering tradeoffs behind each model
choice.

## License

MIT
