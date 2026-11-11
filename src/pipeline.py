"""Pipeline definition shared by every orchestrator.

`TASKS` maps a task id to (callable, upstream task ids). Both the Airflow DAG
(dags/recommendation_pipeline_dag.py) and the plain-Python runner
(run_pipeline.py) are built from this single graph, so the dependency
structure is defined exactly once.

Every task reads its inputs from and writes its outputs to disk
(data/, models/, reports/), so tasks are idempotent and can be retried or
re-run individually.
"""
from __future__ import annotations

from graphlib import TopologicalSorter
from typing import Callable

from src.evaluation import evaluate
from src.features import feature_store
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train
from src.transformation import transform
from src.validation import validate


def _train(name: str) -> Callable:
    def fn(config_path: str | None = None):
        return train.run(name, config_path)
    fn.__name__ = f"train_{name}"
    return fn


TASKS: dict[str, tuple[Callable, list[str]]] = {
    "ingest_data":                (ingest.run, []),
    "validate_data":              (validate.run, ["ingest_data"]),
    "prepare_data":               (prepare.run, ["validate_data"]),
    "transform_data":             (transform.run, ["prepare_data"]),
    "build_features":             (feature_store.run, ["transform_data"]),
    "train_popularity":           (_train("popularity"), ["build_features"]),
    "train_item_knn":             (_train("item_knn"), ["build_features"]),
    "train_matrix_factorization": (_train("matrix_factorization"), ["build_features"]),
    "evaluate_models":            (evaluate.run, ["train_popularity", "train_item_knn", "train_matrix_factorization"]),
}


def execution_order() -> list[str]:
    return list(TopologicalSorter({t: deps for t, (_, deps) in TASKS.items()}).static_order())


def run_task(task_id: str, config_path: str | None = None) -> dict:
    """Run one task. Returns a small JSON-able status (safe for Airflow XCom);
    the real outputs live on disk."""
    fn, _ = TASKS[task_id]
    fn(config_path)
    return {"task": task_id, "status": "success"}
