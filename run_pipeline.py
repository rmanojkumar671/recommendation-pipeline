"""Run the full pipeline (or selected tasks) without Airflow.

    python run_pipeline.py                       # whole pipeline
    python run_pipeline.py --from build_features # resume from a task
    python run_pipeline.py --only evaluate_models
    python run_pipeline.py --config configs/config_per_user_split.yaml

Tasks run in dependency (topological) order from src/pipeline.py - the same
graph the Airflow DAG uses. A run log is written to reports/pipeline_run_log.json.
For scheduling without Airflow, call this script from cron, e.g.
    0 2 * * 0  cd /path/to/recommendation-pipeline && python run_pipeline.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timezone

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src.pipeline import TASKS, execution_order, run_task
from src.utils.common import get_logger, load_config, write_json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--from", dest="start", default=None, help="start from this task")
    ap.add_argument("--only", nargs="+", default=None, help="run only these tasks")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("pipeline", cfg)
    order = execution_order()
    if args.start:
        order = order[order.index(args.start):]
    if args.only:
        order = [t for t in order if t in args.only]

    log = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "config": cfg["_config_path"], "tasks": []}
    status = 0
    logger.info("Pipeline start - tasks: %s", " -> ".join(order))
    for task in order:
        t0 = time.time()
        try:
            run_task(task, args.config)
            entry = {"task": task, "upstream": TASKS[task][1], "status": "success"}
        except Exception as exc:  # stop on first failure, like an orchestrator would
            entry = {"task": task, "upstream": TASKS[task][1], "status": "failed", "error": repr(exc)}
            logger.error("Task %s failed:\n%s", task, traceback.format_exc())
            status = 1
        entry["seconds"] = round(time.time() - t0, 2)
        log["tasks"].append(entry)
        logger.info("[%s] %s (%.1fs)", entry["status"].upper(), task, entry["seconds"])
        if status:
            break
    log["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    log["status"] = "success" if status == 0 else "failed"
    write_json(log, cfg["paths"]["reports"] / "pipeline_run_log.json")
    logger.info("Pipeline %s", log["status"].upper())
    return status


if __name__ == "__main__":
    sys.exit(main())
