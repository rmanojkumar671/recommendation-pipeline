"""Stage 3 - Data preparation.

Cleans and standardises validated data, resolves the warnings raised by the
validation stage, de-duplicates repeated user-item interactions and creates a
*time-based* train/test split (no future information leaks into training).
"""
from __future__ import annotations

import pandas as pd

from src.ingestion.ingest import GENRES
from src.utils.common import get_logger, load_config, load_parquet, save_parquet, write_json

AGE_BINS = [0, 17, 24, 34, 44, 54, 64, 120]
AGE_LABELS = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    u = users.copy()
    u["gender"] = u["gender"].str.strip().str.upper()
    u["occupation"] = u["occupation"].str.strip().str.lower()
    # Non-US / alphanumeric zips (e.g. Canadian "T8H1N") are kept but flagged
    u["zip_code"] = u["zip_code"].astype(str).str.strip().str.upper()
    u["zip_is_us"] = u["zip_code"].str.fullmatch(r"\d{5}")
    u["zip_region"] = u["zip_code"].where(u["zip_is_us"], "X").str[0]   # first digit = US region
    u["age_group"] = pd.cut(u["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    return u[["user_id", "age", "age_group", "gender", "occupation", "zip_code", "zip_is_us", "zip_region"]]


def clean_items(items: pd.DataFrame) -> pd.DataFrame:
    it = items.copy()
    it["title"] = it["title"].str.strip()
    it["title_year"] = pd.to_numeric(it["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
    it["release_date"] = pd.to_datetime(it["release_date"], format="%d-%b-%Y", errors="coerce")
    # impute missing release dates: title year -> 1 Jan of that year, else median date
    from_title = pd.to_datetime(it["title_year"].astype("Int64").astype(str) + "-01-01", errors="coerce")
    it["release_date_imputed"] = it["release_date"].isna()
    it["release_date"] = it["release_date"].fillna(from_title).fillna(it["release_date"].median())
    it["release_year"] = it["release_date"].dt.year.astype(int)
    it["imdb_url"] = it["imdb_url"].fillna("unknown")
    it = it.drop(columns=["video_release_date"])  # 100 % null
    genre_cols = [f"genre_{g}" for g in GENRES]
    it["genres"] = it[genre_cols].apply(lambda r: "|".join(g for g, c in zip(GENRES, genre_cols) if r[c] == 1), axis=1)
    it["n_genres"] = it[genre_cols].sum(axis=1)
    it["title_clean"] = it["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    return it


def clean_interactions(inter: pd.DataFrame, strategy: str, logger) -> pd.DataFrame:
    x = inter.copy()
    before = len(x)
    if strategy == "keep_latest":
        x = x.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="last")
    else:  # keep_first
        x = x.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="first")
    logger.info("De-duplicated repeated user-item pairs: %d -> %d rows", before, len(x))
    x["rating"] = x["rating"].astype("int8")
    x["event_time"] = pd.to_datetime(x["timestamp"], unit="s", utc=True)
    return x.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)


def time_split(inter: pd.DataFrame, test_fraction: float):
    """Global temporal split: the most recent `test_fraction` of events go to test."""
    cutoff_ts = int(inter["timestamp"].quantile(1 - test_fraction))
    train = inter[inter["timestamp"] <= cutoff_ts].copy()
    test = inter[inter["timestamp"] > cutoff_ts].copy()
    return train, test, cutoff_ts


def per_user_time_split(inter: pd.DataFrame, test_fraction: float, min_train: int = 5):
    """Each user's most recent `test_fraction` of ratings go to test (users with fewer
    than `min_train` ratings stay fully in train). Still chronological per user, but
    the global timeline overlaps between train and test - used as a sensitivity check."""
    x = inter.sort_values(["user_id", "timestamp"]).copy()
    x["_rank"] = x.groupby("user_id").cumcount()
    x["_n"] = x.groupby("user_id")["item_id"].transform("size")
    n_test = (x["_n"] * test_fraction).round().astype(int)
    is_test = (x["_rank"] >= x["_n"] - n_test) & (x["_n"] - n_test >= min_train)
    train, test = x[~is_test].drop(columns=["_rank", "_n"]), x[is_test].drop(columns=["_rank", "_n"])
    return train, test, int(train["timestamp"].max())


def run(config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    logger = get_logger("preparation", cfg)
    proc = cfg["paths"]["processed"]
    users = clean_users(load_parquet(proc / "validated_users.parquet"))
    items = clean_items(load_parquet(proc / "validated_items.parquet"))
    inter = clean_interactions(load_parquet(proc / "validated_interactions.parquet"),
                               cfg["preparation"]["dedupe_strategy"], logger)

    strategy = cfg["preparation"].get("split_strategy", "global_time")
    split_fn = time_split if strategy == "global_time" else per_user_time_split
    train, test, cutoff = split_fn(inter, cfg["preparation"]["test_fraction"])
    warm = test["user_id"].isin(train["user_id"]) & test["item_id"].isin(train["item_id"])
    meta = {
        "split": strategy,
        "cutoff_timestamp": cutoff,
        "cutoff_datetime_utc": str(pd.Timestamp(cutoff, unit="s", tz="UTC")),
        "train_rows": len(train), "test_rows": len(test),
        "train_period": [str(train["event_time"].min()), str(train["event_time"].max())],
        "test_period": [str(test["event_time"].min()), str(test["event_time"].max())],
        "train_users": int(train["user_id"].nunique()), "test_users": int(test["user_id"].nunique()),
        "test_users_cold_start": int((~test["user_id"].drop_duplicates().isin(train["user_id"])).sum()),
        "test_rows_warm": int(warm.sum()),
        "test_rows_cold_user_or_item": int((~warm).sum()),
        "items_with_imputed_release_date": int(items["release_date_imputed"].sum()),
    }
    save_parquet(users, proc / "users.parquet")
    save_parquet(items, proc / "items.parquet")
    save_parquet(inter, proc / "interactions_clean.parquet")
    save_parquet(train, proc / "train.parquet")
    save_parquet(test, proc / "test.parquet")
    write_json(meta, proc / "split_metadata.json")
    logger.info("Split at %s: train=%d test=%d (cold-start test users=%d)",
                meta["cutoff_datetime_utc"], len(train), len(test), meta["test_users_cold_start"])
    return meta


if __name__ == "__main__":
    run()
