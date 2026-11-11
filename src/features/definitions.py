"""Feature definitions: the single source of truth for every feature.

Each feature is produced by a pure function of (interactions, users, items,
reference_time). The same functions are called offline (to materialise the
training feature tables) and at prediction time (to compute fresh features
for a user), which avoids training/serving skew.

`FEATURE_REGISTRY` documents every feature and is rendered into
reports/feature_catalog.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.ingestion.ingest import GENRES


@dataclass(frozen=True)
class FeatureDef:
    name: str
    entity: str          # user | item | user_item
    dtype: str
    description: str
    transformation: str
    source: str
    used_by: str

    def as_dict(self):
        return asdict(self)


_U, _I, _X = "user", "item", "user_item"
FEATURE_REGISTRY: list[FeatureDef] = [
    # ---------------- user features
    FeatureDef("user_activity_count", _U, "int", "Number of ratings the user has given", "count(rating) per user", "train interactions", "popularity fallback, evaluation slicing"),
    FeatureDef("user_avg_rating", _U, "float", "Mean rating given by the user", "mean(rating) per user", "train interactions", "MF cold-item fallback, rating normalisation"),
    FeatureDef("user_rating_std", _U, "float", "Std-dev of the user's ratings (0 if one rating)", "std(rating) per user", "train interactions", "analysis"),
    FeatureDef("user_positive_ratio", _U, "float", "Share of the user's ratings that are positive (>= threshold)", "mean(rating >= 4)", "train interactions", "analysis"),
    FeatureDef("user_preferred_genre", _U, "category", "Genre with most positive ratings from the user", "argmax over genres of count(rating>=4)", "train interactions + items", "explanations, diversity analysis"),
    FeatureDef("user_genre_entropy", _U, "float", "Shannon entropy of the user's genre distribution (taste breadth)", "-sum p_g log2 p_g", "train interactions + items", "analysis"),
    FeatureDef("user_days_since_last_interaction", _U, "int", "Days between the user's last rating and the reference time", "(reference_time - max(event_time)).days", "train interactions", "recency analysis, serving"),
    FeatureDef("user_tenure_days", _U, "int", "Days between the user's first and last rating", "(max - min event_time).days", "train interactions", "analysis"),
    FeatureDef("user_age", _U, "int", "Age in years (demographic)", "as-is", "users", "segmentation"),
    FeatureDef("user_age_group", _U, "category", "Age bucket (<18, 18-24, ...)", "pd.cut on age", "users", "segmentation"),
    FeatureDef("user_gender", _U, "category", "Gender (M/F)", "upper-cased", "users", "segmentation"),
    FeatureDef("user_occupation", _U, "category", "Occupation (21 levels)", "lower-cased", "users", "segmentation"),
    FeatureDef("user_aff_<genre>", _U, "float (x19)", "Share of the user's ratings that fall in each genre", "count(genre)/count(all) per user", "train interactions + items", "genre-aware re-ranking, diversity"),
    # ---------------- item features
    FeatureDef("item_popularity", _I, "int", "Number of ratings the item received", "count(rating) per item", "train interactions", "popularity model"),
    FeatureDef("item_avg_rating", _I, "float", "Mean rating of the item", "mean(rating) per item", "train interactions", "popularity model, MF fallback"),
    FeatureDef("item_bayesian_avg", _I, "float", "Damped mean rating, shrunk to the global mean", "(n*avg + m*C)/(n+m), m=20", "train interactions", "popularity model score + rating prediction"),
    FeatureDef("item_rating_std", _I, "float", "Std-dev of the item's ratings", "std(rating) per item", "train interactions", "analysis"),
    FeatureDef("item_positive_ratio", _I, "float", "Share of the item's ratings that are positive", "mean(rating >= 4)", "train interactions", "analysis"),
    FeatureDef("item_popularity_rank", _I, "int", "Rank of the item by number of ratings (1 = most popular)", "rank(desc) of item_popularity", "train interactions", "novelty metric"),
    FeatureDef("item_recent_popularity", _I, "float", "Exponentially time-decayed rating count (half-life 30 days)", "sum 0.5^(age_days/30)", "train interactions", "trend-aware popularity"),
    FeatureDef("item_days_since_last_interaction", _I, "int", "Days since the item was last rated (NaN if never rated)", "(reference_time - max(event_time)).days", "train interactions", "staleness analysis"),
    FeatureDef("item_age_years", _I, "int", "Years between release and the reference time", "reference_year - release_year", "items", "analysis"),
    FeatureDef("item_n_genres", _I, "int", "Number of genres the item belongs to", "sum of genre flags", "items", "analysis"),
    FeatureDef("item_primary_genre", _I, "category", "First listed genre of the item", "first of genres", "items", "explanations"),
    FeatureDef("genre_<genre>", _I, "int 0/1 (x19)", "One-hot genre flags", "as-is from u.item", "items", "diversity metric, genre affinity"),
    # ---------------- interaction features
    FeatureDef("rating_dev_user", _X, "float", "Rating minus the user's mean rating", "rating - user_avg_rating", "train interactions", "analysis / implicit weighting"),
    FeatureDef("rating_dev_item", _X, "float", "Rating minus the item's mean rating", "rating - item_avg_rating", "train interactions", "analysis"),
    FeatureDef("is_positive", _X, "int 0/1", "1 if rating >= positive threshold", "rating >= 4", "train interactions", "implicit-feedback models"),
    FeatureDef("days_before_reference", _X, "int", "Age of the interaction in days at the reference time", "(reference_time - event_time).days", "train interactions", "recency weighting"),
    FeatureDef("recency_weight", _X, "float", "Exponential decay weight of the interaction", "0.5^(days_before_reference/30)", "train interactions", "recency-weighted training"),
    FeatureDef("genre_match_preferred", _X, "int 0/1", "1 if the item contains the user's preferred genre", "user_preferred_genre in item genres", "train interactions + features", "analysis"),
]


def _genre_long(inter: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    x = inter[["user_id", "item_id", "rating"]].merge(items[["item_id", "genres"]], on="item_id")
    return x.assign(genre=x["genres"].str.split("|")).explode("genre")


def compute_user_features(inter: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame,
                          reference_time: pd.Timestamp, pos_thr: int = 4) -> pd.DataFrame:
    g = inter.groupby("user_id")
    f = pd.DataFrame({
        "user_activity_count": g.size(),
        "user_avg_rating": g["rating"].mean(),
        "user_rating_std": g["rating"].std().fillna(0.0),
        "user_positive_ratio": g["rating"].apply(lambda r: (r >= pos_thr).mean()),
        "user_days_since_last_interaction": (reference_time - g["event_time"].max()).dt.days,
        "user_tenure_days": (g["event_time"].max() - g["event_time"].min()).dt.days,
    })
    gl = _genre_long(inter, items)
    counts = gl.groupby(["user_id", "genre"]).size().unstack(fill_value=0).reindex(columns=GENRES, fill_value=0)
    share = counts.div(counts.sum(axis=1), axis=0)
    pos = gl[gl["rating"] >= pos_thr].groupby(["user_id", "genre"]).size().unstack(fill_value=0)
    pref = pos.idxmax(axis=1).reindex(f.index)
    f["user_preferred_genre"] = pref.fillna(counts.idxmax(axis=1).reindex(f.index))
    p = share.where(share > 0)
    f["user_genre_entropy"] = (-(p * np.log2(p)).sum(axis=1)).clip(lower=0).reindex(f.index)
    f = f.join(share.add_prefix("user_aff_"))
    demo = users.set_index("user_id")[["age", "age_group", "gender", "occupation"]].add_prefix("user_")
    # left-join from the full user table so users with no history still get demographics
    out = demo.join(f, how="left")
    out["user_activity_count"] = out["user_activity_count"].fillna(0).astype(int)
    return out.reset_index()


def compute_item_features(inter: pd.DataFrame, items: pd.DataFrame, reference_time: pd.Timestamp,
                          pos_thr: int = 4, prior_m: int = 20, half_life: float = 30.0) -> pd.DataFrame:
    g = inter.groupby("item_id")
    C = inter["rating"].mean()
    age_days = (reference_time - inter["event_time"]).dt.total_seconds() / 86400
    decay = 0.5 ** (age_days / half_life)
    f = pd.DataFrame({
        "item_popularity": g.size(),
        "item_avg_rating": g["rating"].mean(),
        "item_rating_std": g["rating"].std().fillna(0.0),
        "item_positive_ratio": g["rating"].apply(lambda r: (r >= pos_thr).mean()),
        "item_recent_popularity": decay.groupby(inter["item_id"]).sum(),
        "item_days_since_last_interaction": (reference_time - g["event_time"].max()).dt.days,
    })
    base = items.set_index("item_id")
    out = base[["title", "genres", "release_year", "n_genres"] + [f"genre_{x}" for x in GENRES]].join(f, how="left")
    out["item_popularity"] = out["item_popularity"].fillna(0).astype(int)
    out["item_recent_popularity"] = out["item_recent_popularity"].fillna(0.0)
    n, avg = out["item_popularity"], out["item_avg_rating"].fillna(C)
    out["item_bayesian_avg"] = (n * avg + prior_m * C) / (n + prior_m)
    out["item_popularity_rank"] = out["item_popularity"].rank(ascending=False, method="min").astype(int)
    out["item_age_years"] = reference_time.year - out["release_year"]
    out["item_primary_genre"] = out["genres"].str.split("|").str[0]
    out = out.rename(columns={"n_genres": "item_n_genres"})
    return out.reset_index()


def compute_interaction_features(inter: pd.DataFrame, user_f: pd.DataFrame, item_f: pd.DataFrame,
                                 items: pd.DataFrame, reference_time: pd.Timestamp,
                                 pos_thr: int = 4, half_life: float = 30.0) -> pd.DataFrame:
    x = inter[["user_id", "item_id", "rating", "timestamp", "event_time"]].merge(
        user_f[["user_id", "user_avg_rating", "user_preferred_genre"]], on="user_id", how="left").merge(
        item_f[["item_id", "item_avg_rating", "genres"]], on="item_id", how="left")
    x["rating_dev_user"] = x["rating"] - x["user_avg_rating"]
    x["rating_dev_item"] = x["rating"] - x["item_avg_rating"]
    x["is_positive"] = (x["rating"] >= pos_thr).astype(int)
    x["days_before_reference"] = (reference_time - x["event_time"]).dt.days
    x["recency_weight"] = 0.5 ** (x["days_before_reference"] / half_life)
    x["genre_match_preferred"] = [int(isinstance(p, str) and p in str(g).split("|"))
                                  for p, g in zip(x["user_preferred_genre"], x["genres"])]
    return x[["user_id", "item_id", "rating", "timestamp", "rating_dev_user", "rating_dev_item",
              "is_positive", "days_before_reference", "recency_weight", "genre_match_preferred"]]
