# Model Evaluation Results

Split strategy **per_user_time** (train window ends 1998-04-22 23:10:38+00:00) - train 80,000 / test 20,000 ratings. Relevant = test rating >= 4. Items already rated in train are excluded from candidates.

## Main results - warm users (n = 907), K = 10

| model | precision@10 | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | diversity@10 | novelty@10 | rmse | mae |
|---|---|---|---|---|---|---|---|---|---|
| Popularity (baseline) | 0.0760 | 0.0652 | 0.0940 | 0.3870 | 0.0446 | 0.6636 | 8.0003 | 1.0929 | 0.8812 |
| Item-kNN CF | 0.1158 | 0.1376 | 0.1524 | 0.5645 | 0.1213 | 0.7420 | 8.2388 | 0.9771 | 0.7721 |
| Matrix Factorization (SVD) | 0.0438 | 0.0387 | 0.0501 | 0.2922 | 0.1522 | 0.7283 | 9.6273 | 0.9767 | 0.7714 |

## K = 5 (warm users)

| model | precision@5 | recall@5 | ndcg@5 |
|---|---|---|---|
| Popularity (baseline) | 0.0928 | 0.0402 | 0.1003 |
| Item-kNN CF | 0.1288 | 0.0755 | 0.1440 |
| Matrix Factorization (SVD) | 0.0485 | 0.0202 | 0.0495 |

## All test users incl. cold start (n = 907), K = 10

Cold-start users have no training history, so every model serves them the popularity list.

| model | precision@10 | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | diversity@10 | novelty@10 |
|---|---|---|---|---|---|---|---|
| Popularity (baseline) | 0.0760 | 0.0652 | 0.0940 | 0.3870 | 0.0446 | 0.6636 | 8.0003 |
| Item-kNN CF | 0.1158 | 0.1376 | 0.1524 | 0.5645 | 0.1213 | 0.7420 | 8.2388 |
| Matrix Factorization (SVD) | 0.0438 | 0.0387 | 0.0501 | 0.2922 | 0.1522 | 0.7283 | 9.6273 |

RMSE/MAE are computed on the test ratings of user-item pairs where both the user and the item appear in training. The popularity model predicts each item's Bayesian-average rating.
