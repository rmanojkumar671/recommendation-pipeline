"""Stage 7 - Model evaluation and comparison.

Protocol
--------
* Test set = interactions after the time cutoff. A test item is *relevant*
  if its rating >= relevance_threshold (default 4).
* Candidates for a user = every catalogue item the user has not rated in train.
* Two cohorts are reported:
    - warm  : test users who also appear in train (personalisation is possible)
    - all   : every test user with >= 1 relevant item; cold users are served
              by the popularity fallback (realistic production behaviour)
* Ranking metrics @K for K in config (5 and 10): Precision, Recall, NDCG,
  HitRate; list metrics: Coverage, Diversity (intra-list, genre vectors), Novelty.
* Rating metrics: RMSE and MAE on test ratings of warm user-item pairs.

Metrics are written to reports/model_comparison.{csv,md}, a figure, and back
to each model's MLflow run.
"""
from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.features.feature_store import FeatureStore
from src.ingestion.ingest import GENRES
from src.training.models import MODEL_REGISTRY, BaseRecommender
from src.training.train import setup_mlflow
from src.utils.common import df_to_markdown, get_logger, load_config, load_parquet, read_json, write_json

DISPLAY = {"popularity": "Popularity (baseline)", "item_knn": "Item-kNN CF",
           "matrix_factorization": "Matrix Factorization (SVD)"}


def evaluate_model(model: BaseRecommender, test: pd.DataFrame, train: pd.DataFrame, item_vecs: dict,
                   pop_share: dict, n_catalog: int, k_values: list[int], thr: int) -> dict:
    kmax = max(k_values)
    rel = test[test["rating"] >= thr].groupby("user_id")["item_id"].apply(set)
    train_users = set(train["user_id"])
    out = {}
    for cohort in ("warm", "all"):
        users = [u for u in rel.index if (u in train_users or cohort == "all")]
        recs = {u: model.recommend(u, kmax) for u in users}
        res = {"n_users": len(users)}
        for k in k_values:
            res[f"precision@{k}"] = np.mean([M.precision_at_k(recs[u], rel[u], k) for u in users])
            res[f"recall@{k}"] = np.mean([M.recall_at_k(recs[u], rel[u], k) for u in users])
            res[f"ndcg@{k}"] = np.mean([M.ndcg_at_k(recs[u], rel[u], k) for u in users])
            res[f"hit_rate@{k}"] = np.mean([M.hit_rate_at_k(recs[u], rel[u], k) for u in users])
            lists = [recs[u][:k] for u in users]
            res[f"coverage@{k}"] = M.catalog_coverage(lists, n_catalog)
            res[f"diversity@{k}"] = np.mean([M.intra_list_diversity(l, item_vecs) for l in lists])
            res[f"novelty@{k}"] = np.mean([M.novelty(l, pop_share) for l in lists])
        out[cohort] = res
    warm_pairs = test[test["user_id"].isin(train_users) & test["item_id"].isin(set(train["item_id"]))]
    pred = model.predict(warm_pairs["user_id"].to_numpy(), warm_pairs["item_id"].to_numpy())
    out["rating"] = {"rmse": M.rmse(warm_pairs["rating"], pred), "mae": M.mae(warm_pairs["rating"], pred),
                     "n_ratings": len(warm_pairs)}
    return out, recs


def run(config_path: str | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    logger = get_logger("evaluation", cfg)
    proc, rep = cfg["paths"]["processed"], cfg["paths"]["reports"]
    ecfg = cfg["evaluation"]
    train, test = load_parquet(proc / "train.parquet"), load_parquet(proc / "test.parquet")
    fs = FeatureStore(cfg)
    itf = fs.get_item_features()
    gmat = itf[[f"genre_{g}" for g in GENRES]].to_numpy()
    item_vecs = dict(zip(itf["item_id"], gmat))
    total = itf["item_popularity"].sum()
    pop_share = dict(zip(itf["item_id"], itf["item_popularity"] / total))
    n_catalog = len(itf)

    setup_mlflow(cfg)
    rows, all_recs = [], {}
    for name in MODEL_REGISTRY:
        mdir = cfg["paths"]["models"] / name
        if not (mdir / "model.pkl").exists():
            logger.warning("Model %s not trained - skipped", name)
            continue
        model = BaseRecommender.load(mdir / "model.pkl")
        res, recs = evaluate_model(model, test, train, item_vecs, pop_share, n_catalog,
                                   ecfg["k_values"], ecfg["relevance_threshold"])
        all_recs[name] = recs
        for cohort in ("warm", "all"):
            rows.append({"model": DISPLAY[name], "cohort": cohort, **res[cohort],
                         "rmse": res["rating"]["rmse"], "mae": res["rating"]["mae"]})
        info = read_json(mdir / "model_info.json")
        with mlflow.start_run(run_id=info["mlflow_run_id"]):
            flat = {f"{c}.{k.replace('@', '_at_')}": v for c in ("warm", "all") for k, v in res[c].items()}
            flat.update({"test_rmse": res["rating"]["rmse"], "test_mae": res["rating"]["mae"]})
            mlflow.log_metrics({k: float(v) for k, v in flat.items()})
        logger.info("%s warm: P@10=%.4f R@10=%.4f NDCG@10=%.4f RMSE=%.4f", name, res["warm"]["precision@10"],
                    res["warm"]["recall@10"], res["warm"]["ndcg@10"], res["rating"]["rmse"])

    df = pd.DataFrame(rows)
    df.to_csv(rep / "model_comparison.csv", index=False)
    _write_markdown(df, cfg, read_json(proc / "split_metadata.json"))
    _plot(df, rep / "figures" / "model_comparison.png", ecfg["primary_k"])
    _sample_recs(all_recs, itf, rep / "sample_recommendations.csv")
    with mlflow.start_run(run_name="evaluation_summary"):
        mlflow.set_tag("stage", "evaluation")
        for f in ("model_comparison.csv", "model_comparison.md", "figures/model_comparison.png"):
            mlflow.log_artifact(str(rep / f))
    write_json({"status": "ok", "models": list(all_recs)}, rep / "evaluation_status.json")
    return df


def _write_markdown(df: pd.DataFrame, cfg: dict, split: dict) -> None:
    k = cfg["evaluation"]["primary_k"]
    cols5 = ["precision@5", "recall@5", "ndcg@5"]
    colsk = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"hit_rate@{k}", f"coverage@{k}", f"diversity@{k}", f"novelty@{k}"]
    warm = df[df["cohort"] == "warm"]
    allc = df[df["cohort"] == "all"]
    md = ["# Model Evaluation Results", "",
          f"Split strategy **{split['split']}** (train window ends {split['cutoff_datetime_utc']}) - train {split['train_rows']:,} / test {split['test_rows']:,} ratings. "
          f"Relevant = test rating >= {cfg['evaluation']['relevance_threshold']}. Items already rated in train are excluded from candidates.", "",
          f"## Main results - warm users (n = {int(warm['n_users'].iat[0])}), K = {k}", "",
          df_to_markdown(warm[["model"] + colsk + ["rmse", "mae"]]), "",
          "## K = 5 (warm users)", "", df_to_markdown(warm[["model"] + cols5]), "",
          f"## All test users incl. cold start (n = {int(allc['n_users'].iat[0])}), K = {k}", "",
          "Cold-start users have no training history, so every model serves them the popularity list.", "",
          df_to_markdown(allc[["model"] + colsk]), "",
          "RMSE/MAE are computed on the test ratings of user-item pairs where both the user and the item "
          "appear in training. The popularity model predicts each item's Bayesian-average rating.", ""]
    (cfg["paths"]["reports"] / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")


def _plot(df: pd.DataFrame, path, k: int) -> None:
    warm = df[df["cohort"] == "warm"].set_index("model")
    metrics = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"coverage@{k}", f"diversity@{k}"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [3, 1]})
    x = np.arange(len(metrics))
    w = 0.26
    colors = ["#9aa5b1", "#2a7ab0", "#d9822b"]
    for j, (m, row) in enumerate(warm.iterrows()):
        axes[0].bar(x + (j - 1) * w, row[metrics].astype(float), w, label=m, color=colors[j % 3])
    axes[0].set_xticks(x, metrics)
    axes[0].set_title(f"Ranking metrics, warm users (K={k})")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)
    for j, (m, row) in enumerate(warm.iterrows()):
        axes[1].bar(j, row["rmse"], color=colors[j % 3])
    axes[1].set_xticks(range(len(warm)), [{"Popularity (baseline)": "Popularity", "Item-kNN CF": "Item-kNN",
                                          "Matrix Factorization (SVD)": "MF"}.get(m, m) for m in warm.index], fontsize=8)
    axes[1].set_title("Test RMSE (lower is better)")
    axes[1].set_ylim(0.8, max(1.2, warm["rmse"].max() + 0.05))
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _sample_recs(all_recs: dict, itf: pd.DataFrame, path) -> None:
    titles = dict(zip(itf["item_id"], itf["title"]))
    users = sorted(next(iter(all_recs.values())).keys())[:5]
    rows = [{"user_id": u, "model": DISPLAY[m], "rank": r + 1, "item_id": i, "title": titles.get(i)}
            for m, recs in all_recs.items() for u in users if u in recs for r, i in enumerate(recs[u][:5])]
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    run()
