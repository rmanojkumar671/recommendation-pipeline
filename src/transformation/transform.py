"""Stage 4 - Data transformation.

Builds derived datasets from the TRAIN split only (so nothing from the test
period leaks into features or models):

* user_activity_summary  - one row per user
* item_popularity        - one row per item, with popularity rank
* genre_summary          - rating volume / mean per genre
* interaction matrix     - sparse user x item rating matrix (CSR, .npz) + id maps
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.ingestion.ingest import GENRES
from src.utils.common import get_logger, load_config, load_parquet, save_parquet, write_json


def user_activity_summary(train: pd.DataFrame, items: pd.DataFrame, pos_thr: int) -> pd.DataFrame:
    g = train.groupby("user_id")
    out = pd.DataFrame({
        "n_ratings": g.size(),
        "avg_rating": g["rating"].mean(),
        "std_rating": g["rating"].std().fillna(0.0),
        "n_positive": g["rating"].apply(lambda r: int((r >= pos_thr).sum())),
        "first_interaction": g["event_time"].min(),
        "last_interaction": g["event_time"].max(),
    })
    out["active_days"] = (out["last_interaction"] - out["first_interaction"]).dt.days + 1
    genres = train.merge(items[["item_id", "genres"]], on="item_id")
    out["n_distinct_genres"] = genres.assign(g=genres["genres"].str.split("|")).explode("g").groupby("user_id")["g"].nunique()
    return out.reset_index()


def item_popularity(train: pd.DataFrame, items: pd.DataFrame, pos_thr: int, m: int) -> pd.DataFrame:
    g = train.groupby("item_id")
    out = pd.DataFrame({
        "n_ratings": g.size(), "avg_rating": g["rating"].mean(),
        "n_positive": g["rating"].apply(lambda r: int((r >= pos_thr).sum())),
        "last_interaction": g["event_time"].max(),
    }).reset_index()
    c = train["rating"].mean()
    out["bayesian_avg"] = (out["n_ratings"] * out["avg_rating"] + m * c) / (out["n_ratings"] + m)
    out["popularity_rank"] = out["n_ratings"].rank(ascending=False, method="min").astype(int)
    out = items[["item_id", "title", "genres"]].merge(out, on="item_id", how="left")
    out["n_ratings"] = out["n_ratings"].fillna(0).astype(int)
    out["n_positive"] = out["n_positive"].fillna(0).astype(int)
    return out.sort_values("n_ratings", ascending=False).reset_index(drop=True)


def genre_summary(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    x = train.merge(items[["item_id", "genres"]], on="item_id")
    x = x.assign(genre=x["genres"].str.split("|")).explode("genre")
    return (x.groupby("genre").agg(n_ratings=("rating", "size"), avg_rating=("rating", "mean"),
                                   n_users=("user_id", "nunique")).reset_index()
            .sort_values("n_ratings", ascending=False))


def build_interaction_matrix(train: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame):
    """Rows/cols cover the full user/item catalogue so ids map consistently at prediction time."""
    user_ids = np.sort(users["user_id"].unique())
    item_ids = np.sort(items["item_id"].unique())
    uidx = pd.Series(np.arange(len(user_ids)), index=user_ids)
    iidx = pd.Series(np.arange(len(item_ids)), index=item_ids)
    mat = sp.csr_matrix((train["rating"].astype(np.float32),
                         (uidx.loc[train["user_id"]].values, iidx.loc[train["item_id"]].values)),
                        shape=(len(user_ids), len(item_ids)))
    return mat, pd.DataFrame({"user_id": user_ids, "row": uidx.values}), pd.DataFrame({"item_id": item_ids, "col": iidx.values})


def run(config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    logger = get_logger("transformation", cfg)
    proc = cfg["paths"]["processed"]
    fcfg = cfg["features"]
    train = load_parquet(proc / "train.parquet")
    users, items = load_parquet(proc / "users.parquet"), load_parquet(proc / "items.parquet")

    uas = user_activity_summary(train, items, fcfg["positive_threshold"])
    pop = item_popularity(train, items, fcfg["positive_threshold"], fcfg["bayesian_prior_count"])
    gen = genre_summary(train, items)
    mat, umap, imap = build_interaction_matrix(train, users, items)

    save_parquet(uas, proc / "user_activity_summary.parquet")
    save_parquet(pop, proc / "item_popularity.parquet")
    save_parquet(gen, proc / "genre_summary.parquet")
    sp.save_npz(proc / "interaction_matrix.npz", mat)
    save_parquet(umap, proc / "user_index.parquet")
    save_parquet(imap, proc / "item_index.parquet")
    meta = {"matrix_shape": list(mat.shape), "nnz": int(mat.nnz),
            "density_pct": round(100 * mat.nnz / (mat.shape[0] * mat.shape[1]), 3),
            "users_summarised": len(uas), "items_summarised": len(pop),
            "items_never_rated_in_train": int((pop["n_ratings"] == 0).sum())}
    write_json(meta, proc / "transformation_metadata.json")
    logger.info("Interaction matrix %s, nnz=%d (density %.2f%%)", mat.shape, mat.nnz, meta["density_pct"])
    return meta


if __name__ == "__main__":
    run()
