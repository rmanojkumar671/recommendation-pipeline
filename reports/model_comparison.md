# Model Evaluation Results

Split strategy **global_time** (train window ends 1998-03-07 02:21:09+00:00) - train 80,001 / test 19,999 ratings. Relevant = test rating >= 4. Items already rated in train are excluded from candidates.

## Main results - warm users (n = 100), K = 10

| model | precision@10 | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | diversity@10 | novelty@10 | rmse | mae |
|---|---|---|---|---|---|---|---|---|---|
| Popularity (baseline) | 0.1230 | 0.1024 | 0.1532 | 0.4600 | 0.0422 | 0.6786 | 8.2522 | 1.0327 | 0.8362 |
| Item-kNN CF | 0.1200 | 0.0577 | 0.1355 | 0.3500 | 0.0868 | 0.7227 | 8.4624 | 0.9842 | 0.7757 |
| Matrix Factorization (SVD) | 0.0680 | 0.0697 | 0.0909 | 0.3600 | 0.0904 | 0.7136 | 9.7487 | 0.9786 | 0.7685 |

## K = 5 (warm users)

| model | precision@5 | recall@5 | ndcg@5 |
|---|---|---|---|
| Popularity (baseline) | 0.1300 | 0.0729 | 0.1487 |
| Item-kNN CF | 0.1160 | 0.0322 | 0.1331 |
| Matrix Factorization (SVD) | 0.0640 | 0.0370 | 0.0804 |

## All test users incl. cold start (n = 292), K = 10

Cold-start users have no training history, so every model serves them the popularity list.

| model | precision@10 | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | diversity@10 | novelty@10 |
|---|---|---|---|---|---|---|---|
| Popularity (baseline) | 0.2329 | 0.0745 | 0.2611 | 0.6062 | 0.0422 | 0.6326 | 8.0280 |
| Item-kNN CF | 0.2318 | 0.0592 | 0.2550 | 0.5685 | 0.0874 | 0.6477 | 8.1000 |
| Matrix Factorization (SVD) | 0.2140 | 0.0633 | 0.2397 | 0.5719 | 0.0910 | 0.6446 | 8.5405 |

RMSE/MAE are computed on the test ratings of user-item pairs where both the user and the item appear in training. The popularity model predicts each item's Bayesian-average rating.
