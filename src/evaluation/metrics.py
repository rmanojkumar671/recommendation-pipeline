"""Ranking and rating metrics (pure functions, unit-tested in tests/)."""
from __future__ import annotations

import numpy as np


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    return len(set(recommended[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    gains = [1.0 / np.log2(i + 2) for i, it in enumerate(recommended[:k]) if it in relevant]
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return float(sum(gains) / ideal) if ideal > 0 else 0.0


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    return float(len(set(recommended[:k]) & relevant) > 0)


def catalog_coverage(all_recs: list[list], n_catalog: int) -> float:
    """Share of the catalogue that appears in at least one user's list."""
    return len({i for recs in all_recs for i in recs}) / n_catalog


def intra_list_diversity(recs: list, item_vectors: dict) -> float:
    """1 - mean pairwise cosine similarity of the items' genre vectors."""
    vecs = [item_vectors[i] for i in recs if i in item_vectors]
    if len(vecs) < 2:
        return 0.0
    V = np.asarray(vecs, dtype=float)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    S = V @ V.T
    n = len(V)
    return float(1 - (S.sum() - np.trace(S)) / (n * (n - 1)))


def novelty(recs: list, item_popularity_share: dict) -> float:
    """Mean self-information -log2(p(i)); higher = less obvious recommendations."""
    vals = [-np.log2(item_popularity_share.get(i, 0) + 1e-12) for i in recs]
    return float(np.mean(vals)) if vals else 0.0


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))
