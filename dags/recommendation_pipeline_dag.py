"""Apache Airflow DAG for the MovieLens recommendation pipeline.

ingest_data >> validate_data >> prepare_data >> transform_data >> build_features
    >> [train_popularity, train_item_knn, train_matrix_factorization] >> evaluate_models

The task graph comes from src/pipeline.py (single source of truth). Each task
is idempotent (reads/writes files), retried once on failure, and the three
training tasks run in parallel when the executor allows it.

Setup: point Airflow at this folder, e.g.
    export AIRFLOW__CORE__DAGS_FOLDER=/path/to/recommendation-pipeline/dags
    export RECSYS_HOME=/path/to/recommendation-pipeline
Works with Airflow 2.x and 3.x.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("RECSYS_HOME", Path(__file__).resolve().parents[1]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("RECSYS_HOME", str(PROJECT_ROOT))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from airflow import DAG

try:  # Airflow 3.x
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:  # Airflow 2.x
    from airflow.operators.python import PythonOperator

from src.pipeline import TASKS, run_task

CONFIG_PATH = os.environ.get("RECSYS_CONFIG", str(PROJECT_ROOT / "configs" / "config.yaml"))

default_args = {
    "owner": "dm4ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=30),
}

with DAG(
    dag_id="movielens_recommendation_pipeline",
    description="End-to-end data management pipeline for a MovieLens recommender",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",          # retrain weekly; trigger manually for ad-hoc runs
    catchup=False,
    max_active_runs=1,           # runs share the data/ folder, so never overlap
    tags=["dm4ml", "recsys", "movielens"],
    doc_md=__doc__,
) as dag:
    ops = {
        task_id: PythonOperator(
            task_id=task_id,
            python_callable=run_task,
            op_kwargs={"task_id": task_id, "config_path": CONFIG_PATH},
        )
        for task_id in TASKS
    }
    for task_id, (_, upstream) in TASKS.items():
        for up in upstream:
            ops[up] >> ops[task_id]
