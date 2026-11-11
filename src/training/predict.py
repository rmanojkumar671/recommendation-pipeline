"""Serving / prediction: top-K recommendations for a user, reusing the feature store.

    python -m src.training.predict --user 42 --model item_knn --k 10

The same FeatureStore that fed training supplies the item features (titles,
genres, popularity) and the user profile used to explain the recommendations,
and `compute_online_user_features` recomputes the user's features from the
latest history with the training-time definitions (no training/serving skew).
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import pandas as pd

from src.features.feature_store import FeatureStore
from src.training.models import BaseRecommender
from src.utils.common import load_config, load_parquet


def recommend_for_user(user_id: int, model_name: str = "item_knn", k: int = 10, config_path=None) -> dict:
    cfg = load_config(config_path)
    model = BaseRecommender.load(cfg["paths"]["models"] / model_name / "model.pkl")
    fs = FeatureStore(cfg)
    proc = cfg["paths"]["processed"]
    history = load_parquet(proc / "interactions_clean.parquet")
    users, items = load_parquet(proc / "users.parquet"), load_parquet(proc / "items.parquet")

    offline = fs.get_user_features([user_id])
    online = fs.compute_online_user_features(user_id, history, users, items)
    recs = model.recommend(user_id, k)
    itf = fs.get_item_features(recs, ["title", "genres", "item_popularity", "item_bayesian_avg"]).set_index("item_id").loc[recs]
    return {
        "user_id": user_id,
        "model": model_name,
        "served_by": "personalised" if model.is_known_user(user_id) else "popularity fallback (cold start)",
        "offline_profile": offline.iloc[0][["user_activity_count", "user_avg_rating", "user_preferred_genre"]].to_dict() if len(offline) else {},
        "online_profile": online.iloc[0][["user_activity_count", "user_avg_rating", "user_preferred_genre",
                                          "user_days_since_last_interaction"]].to_dict() if len(online) else {},
        "recommendations": itf.reset_index().assign(rank=range(1, len(recs) + 1)).to_dict("records"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--model", default="item_knn", choices=["popularity", "item_knn", "matrix_factorization"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    out = recommend_for_user(a.user, a.model, a.k, a.config)
    print(f"User {out['user_id']} - model {out['model']} - served by {out['served_by']}")
    print("Offline features (train window):", out["offline_profile"])
    print("Online features (latest history):", out["online_profile"])
    print(pd.DataFrame(out["recommendations"])[["rank", "item_id", "title", "genres", "item_popularity"]].to_string(index=False))


if __name__ == "__main__":
    main()
