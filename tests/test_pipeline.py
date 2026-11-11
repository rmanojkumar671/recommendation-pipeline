"""Unit tests for metrics, validation rules, splitting and features.

Run with:  python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics as M
from src.features.definitions import compute_item_features, compute_user_features
from src.pipeline import TASKS, execution_order
from src.preparation.prepare import per_user_time_split, time_split
from src.validation.validate import validate_interactions

VCFG = {"rating_min": 1, "rating_max": 5, "timestamp_min": "1995-01-01", "timestamp_max": "now"}


# ----------------------------------------------------------------- metrics
def test_precision_recall_ndcg():
    recs, rel = [1, 2, 3, 4, 5], {2, 5, 9}
    assert M.precision_at_k(recs, rel, 5) == pytest.approx(2 / 5)
    assert M.recall_at_k(recs, rel, 5) == pytest.approx(2 / 3)
    ideal = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert M.ndcg_at_k(recs, rel, 5) == pytest.approx((1 / np.log2(3) + 1 / np.log2(6)) / ideal)
    assert M.ndcg_at_k([2, 5, 9], rel, 3) == pytest.approx(1.0)


def test_coverage_and_diversity():
    assert M.catalog_coverage([[1, 2], [2, 3]], 10) == pytest.approx(0.3)
    vecs = {1: np.array([1, 0]), 2: np.array([0, 1]), 3: np.array([1, 0])}
    assert M.intra_list_diversity([1, 2], vecs) == pytest.approx(1.0)
    assert M.intra_list_diversity([1, 3], vecs) == pytest.approx(0.0, abs=1e-9)


def test_rmse_mae():
    assert M.rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))
    assert M.mae([1, 2, 3], [2, 2, 2]) == pytest.approx(2 / 3)


# ----------------------------------------------------------------- validation
def test_validation_rules_catch_bad_rows():
    df = pd.DataFrame({
        "user_id": ["1", "1", "1", "2", "99", "1", "1"],
        "item_id": ["10", "10", "11", "10", "10", "12", "13"],
        "rating": ["4", "4", "7", "five", "3", None, "3"],
        "timestamp": ["880000000", "880000000", "880000001", "880000002", "880000003", "880000004", "-5"],
    })
    eng = validate_interactions(df, VCFG, user_ids={1, 2}, item_ids={10, 11, 12, 13})
    ok, err = eng.split()
    assert list(ok.index) == [0]
    reasons = dict(zip(err.index, err["failed_rules"]))
    assert "exact_duplicate" in reasons[1]
    assert "invalid_rating" in reasons[2]
    assert "bad_type_rating" in reasons[3]
    assert "orphan_user" in reasons[4]
    assert "missing_required" in reasons[5]
    assert "invalid_timestamp" in reasons[6]


# ----------------------------------------------------------------- splitting
def _toy_inter(n=100):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"user_id": rng.integers(1, 6, n), "item_id": rng.integers(1, 30, n),
                       "rating": rng.integers(1, 6, n), "timestamp": np.arange(n) + 880_000_000})
    df["event_time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def test_global_time_split_has_no_leakage():
    train, test, cutoff = time_split(_toy_inter(), 0.2)
    assert train["timestamp"].max() <= cutoff < test["timestamp"].min()
    assert len(train) + len(test) == 100


def test_per_user_split_is_chronological_per_user():
    train, test, _ = per_user_time_split(_toy_inter(200), 0.2)
    for u, t in test.groupby("user_id"):
        assert train[train["user_id"] == u]["timestamp"].max() < t["timestamp"].min()


# ----------------------------------------------------------------- features
def test_feature_functions():
    inter = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [10, 11, 10], "rating": [5, 3, 4],
                          "timestamp": [0, 0, 0]})
    inter["event_time"] = pd.to_datetime(["1998-01-01", "1998-01-11", "1998-01-05"], utc=True)
    items = pd.DataFrame({"item_id": [10, 11, 12], "title": ["A", "B", "C"], "genres": ["Drama", "Comedy|Drama", "Action"],
                          "release_year": [1990, 1995, 1997], "n_genres": [1, 2, 1]})
    for g in ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
              "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]:
        items[f"genre_{g}"] = items["genres"].str.contains(g).astype(int)
    users = pd.DataFrame({"user_id": [1, 2, 3], "age": [20, 30, 40], "age_group": ["18-24", "25-34", "35-44"],
                          "gender": ["M", "F", "M"], "occupation": ["student"] * 3})
    ref = pd.Timestamp("1998-01-21", tz="UTC")
    uf = compute_user_features(inter, users, items, ref).set_index("user_id")
    assert uf.loc[1, "user_activity_count"] == 2
    assert uf.loc[1, "user_avg_rating"] == pytest.approx(4.0)
    assert uf.loc[1, "user_days_since_last_interaction"] == 10
    assert uf.loc[1, "user_preferred_genre"] == "Drama"
    assert uf.loc[3, "user_activity_count"] == 0          # cold user still gets a row
    itf = compute_item_features(inter, items, ref, prior_m=0).set_index("item_id")
    assert itf.loc[10, "item_popularity"] == 2
    assert itf.loc[10, "item_bayesian_avg"] == pytest.approx(4.5)
    assert itf.loc[12, "item_popularity"] == 0


# ----------------------------------------------------------------- orchestration
def test_dag_order_respects_dependencies():
    order = execution_order()
    for task, (_, deps) in TASKS.items():
        for d in deps:
            assert order.index(d) < order.index(task)
