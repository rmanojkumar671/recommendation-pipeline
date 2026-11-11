"""Recommender models.

* PopularityRecommender   - non-personalised baseline built from item features
* ItemKNNRecommender      - item-based collaborative filtering (binary cosine for ranking,
                            shrunk adjusted cosine for rating prediction)
* MatrixFactorization     - biased MF (Funk-SVD) trained with SGD; predicts explicit ratings

All models share one interface (fit / recommend / predict / save / load) and
fall back to the popularity ranking for users with no training history
(cold start), which is how a production system would serve them.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


class BaseRecommender:
    name = "base"

    def __init__(self, **params):
        self.params = params
        self.fit_seconds = None

    # mapping helpers --------------------------------------------------------
    def _set_index(self, user_index: pd.DataFrame, item_index: pd.DataFrame):
        self.user_ids = user_index["user_id"].to_numpy()
        self.item_ids = item_index["item_id"].to_numpy()
        self.u2r = dict(zip(user_index["user_id"], user_index["row"]))
        self.i2c = dict(zip(item_index["item_id"], item_index["col"]))

    def _set_fallback(self, item_features: pd.DataFrame, min_ratings: int = 10):
        f = item_features.set_index("item_id").reindex(self.item_ids)
        score = f["item_bayesian_avg"] * np.log1p(f["item_popularity"])
        score[f["item_popularity"] < min_ratings] = -np.inf
        self.fallback_scores = score.to_numpy(dtype=float)
        self.item_bayes = f["item_bayesian_avg"].to_numpy(dtype=float)

    def fit(self, train: pd.DataFrame, matrix: sp.csr_matrix, user_index, item_index, item_features):
        t0 = time.time()
        self._set_index(user_index, item_index)
        self._set_fallback(item_features, self.params.get("min_ratings", 10))
        self.seen = matrix.copy().tocsr()
        self.seen.data[:] = 1
        self.global_mean = float(train["rating"].mean())
        self._fit(train, matrix, item_features)
        self.fit_seconds = round(time.time() - t0, 2)
        return self

    def _fit(self, train, matrix, item_features):  # pragma: no cover - abstract
        raise NotImplementedError

    def _scores(self, row: int) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def is_known_user(self, user_id) -> bool:
        r = self.u2r.get(user_id)
        return r is not None and self.seen[r].nnz > 0

    # public API -------------------------------------------------------------
    def recommend(self, user_id, k: int = 10, exclude_seen: bool = True) -> list[int]:
        row = self.u2r.get(user_id)
        if row is None or self.seen[row].nnz == 0:
            scores = self.fallback_scores.copy()
        else:
            scores = self._scores(row).astype(float)
        if exclude_seen and row is not None:
            scores[self.seen[row].indices] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return self.item_ids[top].tolist()

    def predict(self, user_ids, item_ids) -> np.ndarray:
        """Predicted explicit rating for each (user, item) pair."""
        return np.array([self._predict_one(u, i) for u, i in zip(user_ids, item_ids)])

    def _predict_one(self, user_id, item_id) -> float:
        c = self.i2c.get(item_id)
        return float(self.item_bayes[c]) if c is not None else self.global_mean

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> "BaseRecommender":
        with open(path, "rb") as fh:
            return pickle.load(fh)


# =========================================================================== baseline
class PopularityRecommender(BaseRecommender):
    """Ranks items by bayesian_avg * log(1 + n_ratings) - popular *and* well liked.
    Every user gets the same list (minus items they have already seen).
    Rating prediction = item Bayesian average."""
    name = "popularity"

    def _fit(self, train, matrix, item_features):
        self.scores_ = self.fallback_scores

    def _scores(self, row):
        return self.scores_.copy()


# =========================================================================== item-kNN CF
class ItemKNNRecommender(BaseRecommender):
    name = "item_knn"

    def _fit(self, train, matrix, item_features):
        k, lam = self.params.get("n_neighbors", 40), self.params.get("shrinkage", 10)
        R = matrix.tocsr().astype(np.float64)
        B = (R > 0).astype(np.float64)
        co = (B.T @ B).toarray()                            # co-occurrence counts

        # (a) ranking similarity: cosine on binary interactions ("users who watched i also watched j")
        n = np.sqrt(np.diag(co)) + 1e-9
        S_rank = co / np.outer(n, n)
        # (b) rating similarity: adjusted cosine on user-mean-centred ratings, shrunk by support
        counts = np.diff(R.indptr)
        means = np.divide(np.asarray(R.sum(axis=1)).ravel(), counts, out=np.zeros(R.shape[0]), where=counts > 0)
        C = R.copy()
        C.data -= np.repeat(means, counts)
        norms = np.sqrt(np.asarray(C.multiply(C).sum(axis=0)).ravel()) + 1e-9
        S_rate = (C.T @ C).toarray() / np.outer(norms, norms) * (co / (co + lam))

        self.S_rank = self._prune(S_rank, k)
        self.S_rate = self._prune(S_rate, k)
        self.user_means, self.C = means, C.tocsr()

        # baseline estimates b_ui = mu + b_u + b_i (regularised, 10 ALS sweeps) for kNN-baseline prediction
        mu, reg = self.global_mean, self.params.get("baseline_reg", 10)
        rows = train["user_id"].map(self.u2r).to_numpy()
        cols = train["item_id"].map(self.i2c).to_numpy()
        r = train["rating"].to_numpy(float)
        bu, bi = np.zeros(R.shape[0]), np.zeros(R.shape[1])
        for _ in range(10):
            bi = np.bincount(cols, r - mu - bu[rows], R.shape[1]) / (np.bincount(cols, minlength=R.shape[1]) + reg)
            bu = np.bincount(rows, r - mu - bi[cols], R.shape[0]) / (np.bincount(rows, minlength=R.shape[0]) + reg)
        self.bu, self.bi = bu, bi
        # residual matrix (r_uj - b_uj) used in the neighbourhood term
        self.Res = sp.csr_matrix((r - (mu + bu[rows] + bi[cols]), (rows, cols)), shape=R.shape)

    @staticmethod
    def _prune(S: np.ndarray, k: int) -> sp.csr_matrix:
        """Zero the diagonal and keep only the k most similar neighbours of each item."""
        S = S.copy()
        np.fill_diagonal(S, 0.0)
        if k < S.shape[1]:
            idx = np.argpartition(-S, k, axis=1)[:, k:]
            np.put_along_axis(S, idx, 0.0, axis=1)
        return sp.csr_matrix(S)

    def _scores(self, row):
        # score(i) = sum of similarities between i and the items the user interacted with
        return np.asarray((self.seen[row] @ self.S_rank).todense()).ravel()

    def _predict_one(self, user_id, item_id):
        """kNN-baseline: b_ui + sum_j s_ij (r_uj - b_uj) / (sum_j |s_ij| + 1)."""
        r, c = self.u2r.get(user_id), self.i2c.get(item_id)
        if r is None or c is None or self.seen[r].nnz == 0:
            return super()._predict_one(user_id, item_id)
        base = self.global_mean + self.bu[r] + self.bi[c]
        res = self.Res[r]
        sims = self.S_rate[:, c].toarray().ravel()[res.indices]
        mask = sims > 0
        if not mask.any():
            return float(np.clip(base, 1, 5))
        adj = sims[mask] @ res.data[mask] / (sims[mask].sum() + 1.0)   # +1 damps thin neighbourhoods
        return float(np.clip(base + adj, 1, 5))


# =========================================================================== matrix factorisation
class MatrixFactorization(BaseRecommender):
    """r_hat(u,i) = mu + b_u + b_i + p_u . q_i, trained with SGD on observed ratings."""
    name = "matrix_factorization"

    def _fit(self, train, matrix, item_features):
        p = self.params
        f, epochs, lr, reg = p["n_factors"], p["n_epochs"], p["learning_rate"], p["reg"]
        rng = np.random.default_rng(p.get("seed", 42))
        n_u, n_i = len(self.user_ids), len(self.item_ids)
        self.P = rng.normal(0, p.get("init_std", 0.1), (n_u, f))
        self.Q = rng.normal(0, p.get("init_std", 0.1), (n_i, f))
        self.bu, self.bi = np.zeros(n_u), np.zeros(n_i)
        mu = self.global_mean
        rows = train["user_id"].map(self.u2r).to_numpy()
        cols = train["item_id"].map(self.i2c).to_numpy()
        vals = train["rating"].to_numpy(dtype=float)
        P, Q, bu, bi = self.P, self.Q, self.bu, self.bi
        self.history = []
        for ep in range(epochs):
            order = rng.permutation(len(vals))
            for idx in order:
                u, i, r = rows[idx], cols[idx], vals[idx]
                pu, qi = P[u], Q[i]
                err = r - (mu + bu[u] + bi[i] + pu @ qi)
                bu[u] += lr * (err - reg * bu[u])
                bi[i] += lr * (err - reg * bi[i])
                P[u] = pu + lr * (err * qi - reg * pu)
                Q[i] = qi + lr * (err * pu - reg * qi)
            pred = mu + bu[rows] + bi[cols] + np.einsum("ij,ij->i", P[rows], Q[cols])
            self.history.append(float(np.sqrt(np.mean((vals - pred) ** 2))))
        self.train_rmse = self.history[-1]

    def _scores(self, row):
        return self.global_mean + self.bu[row] + self.bi + self.Q @ self.P[row]

    def _predict_one(self, user_id, item_id):
        r, c = self.u2r.get(user_id), self.i2c.get(item_id)
        known_u = r is not None and self.seen[r].nnz > 0
        if c is None:
            return self.global_mean
        if not known_u:
            return super()._predict_one(user_id, item_id)
        return float(np.clip(self.global_mean + self.bu[r] + self.bi[c] + self.P[r] @ self.Q[c], 1, 5))


MODEL_REGISTRY = {
    "popularity": PopularityRecommender,
    "item_knn": ItemKNNRecommender,
    "matrix_factorization": MatrixFactorization,
}
