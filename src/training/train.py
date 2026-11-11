"""Stage 6 - Model training with MLflow experiment tracking.

`run(model_name)` trains one model (so the orchestrator can run them as
independent, parallelisable tasks), logs params / training metrics /
artifacts to MLflow and saves the model to models/<name>/.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import numpy as np
import scipy.sparse as sp

from src.features.feature_store import FeatureStore
from src.training.models import MODEL_REGISTRY
from src.utils.common import (PROJECT_ROOT, get_logger, load_config, load_parquet, read_json,
                              set_seed, sha256_file, write_json)


def setup_mlflow(cfg: dict) -> None:
    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{(PROJECT_ROOT / uri.replace('sqlite:///', '')).as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])


def model_params(cfg: dict, name: str) -> dict:
    t = cfg["training"]
    if name == "popularity":
        return dict(t["popularity"])
    if name == "item_knn":
        return dict(t["item_knn"], min_ratings=t["popularity"]["min_ratings"])
    if name == "matrix_factorization":
        return dict(t["matrix_factorization"], seed=cfg["seed"], min_ratings=t["popularity"]["min_ratings"])
    raise KeyError(name)


def run(model_name: str, config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    logger = get_logger(f"training.{model_name}", cfg)
    set_seed(cfg["seed"])
    proc = cfg["paths"]["processed"]
    train = load_parquet(proc / "train.parquet")
    matrix = sp.load_npz(proc / "interaction_matrix.npz")
    uidx, iidx = load_parquet(proc / "user_index.parquet"), load_parquet(proc / "item_index.parquet")
    fs = FeatureStore(cfg)
    item_features = fs.get_item_features(columns=["item_bayesian_avg", "item_popularity"])

    params = model_params(cfg, model_name)
    model = MODEL_REGISTRY[model_name](**params)
    setup_mlflow(cfg)
    with mlflow.start_run(run_name=model_name) as run_:
        mlflow.set_tags({"model": model_name, "stage": "training", "dataset": cfg["ingestion"]["dataset"],
                         "feature_set_version": fs.metadata()["feature_set_version"]})
        mlflow.log_params({**{f"model.{k}": v for k, v in params.items()},
                           "data.train_rows": len(train), "data.n_users": matrix.shape[0],
                           "data.n_items": matrix.shape[1], "split.cutoff": read_json(proc / "split_metadata.json")["cutoff_datetime_utc"],
                           "seed": cfg["seed"]})
        model.fit(train, matrix, uidx, iidx, item_features)
        train_pred = model.predict(train["user_id"].to_numpy(), train["item_id"].to_numpy())
        train_rmse = float(np.sqrt(np.mean((train["rating"].to_numpy() - train_pred) ** 2)))
        mlflow.log_metrics({"train_rmse": train_rmse, "fit_seconds": model.fit_seconds})
        if hasattr(model, "history"):
            for ep, v in enumerate(model.history):
                mlflow.log_metric("train_rmse_epoch", v, step=ep)

        out_dir = cfg["paths"]["models"] / model_name
        model_path = out_dir / "model.pkl"
        model.save(model_path)
        info = {"model": model_name, "mlflow_run_id": run_.info.run_id, "params": params,
                "train_rmse": train_rmse, "fit_seconds": model.fit_seconds,
                "model_sha256": sha256_file(model_path),
                "feature_set_version": fs.metadata()["feature_set_version"]}
        write_json(info, out_dir / "model_info.json")
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(out_dir / "model_info.json"), artifact_path="model")
    logger.info("Trained %s in %.1fs (train RMSE %.4f), MLflow run %s", model_name, model.fit_seconds,
                train_rmse, info["mlflow_run_id"])
    return info


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "popularity")
