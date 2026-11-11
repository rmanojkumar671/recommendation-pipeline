"""Stage 5 - Feature management: a lightweight, file-based feature store.

* `materialize()` computes all feature tables from the training window and
  writes them to data/features/ with a metadata file (feature-set version,
  reference time, row counts, null rates).
* `get_user_features()` / `get_item_features()` serve the same tables to the
  training code (offline) and to `src/training/predict.py` (serving).
* `compute_online_user_features()` recomputes a user's features from the
  latest history using the *same* definitions, for real-time prediction.
* The feature catalog (reports/feature_catalog.md/.csv) is generated from
  the registry plus statistics of the materialised data.

This mirrors the core ideas of Feast (entities, feature views, offline vs
online retrieval, point-in-time reference) without needing a server.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from src.features.definitions import (FEATURE_REGISTRY, compute_interaction_features,
                                      compute_item_features, compute_user_features)
from src.utils.common import (df_to_markdown, get_logger, load_config, load_parquet, read_json,
                              save_parquet, write_json)


class FeatureStore:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dir = cfg["paths"]["features"]
        self.fcfg = cfg["features"]
        self._cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ offline
    def materialize(self, train, users, items, reference_time: pd.Timestamp) -> dict:
        kw = dict(pos_thr=self.fcfg["positive_threshold"])
        uf = compute_user_features(train, users, items, reference_time, **kw)
        itf = compute_item_features(train, items, reference_time, prior_m=self.fcfg["bayesian_prior_count"],
                                    half_life=self.fcfg["recency_half_life_days"], **kw)
        xf = compute_interaction_features(train, uf, itf, items, reference_time,
                                          half_life=self.fcfg["recency_half_life_days"], **kw)
        tables = {"user_features": uf, "item_features": itf, "interaction_features": xf}
        for name, df in tables.items():
            save_parquet(df, self.dir / f"{name}.parquet")
        version = hashlib.sha256(json.dumps([f.as_dict() for f in FEATURE_REGISTRY]).encode()
                                 + json.dumps(self.fcfg, sort_keys=True).encode()).hexdigest()[:12]
        meta = {
            "feature_set_version": version,
            "materialized_at_utc": datetime.now(timezone.utc).isoformat(),
            "reference_time": str(reference_time),
            "params": self.fcfg,
            "tables": {n: {"rows": len(d), "columns": len(d.columns),
                           "null_pct": round(100 * d.isna().mean().mean(), 3)} for n, d in tables.items()},
        }
        write_json(meta, self.dir / "feature_metadata.json")
        self._cache = tables
        return meta

    # ------------------------------------------------------------------ retrieval
    def _table(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            self._cache[name] = load_parquet(self.dir / f"{name}.parquet")
        return self._cache[name]

    def get_user_features(self, user_ids=None, columns=None) -> pd.DataFrame:
        df = self._table("user_features")
        if user_ids is not None:
            df = df[df["user_id"].isin(list(user_ids))]
        return df if columns is None else df[["user_id"] + list(columns)]

    def get_item_features(self, item_ids=None, columns=None) -> pd.DataFrame:
        df = self._table("item_features")
        if item_ids is not None:
            df = df[df["item_id"].isin(list(item_ids))]
        return df if columns is None else df[["item_id"] + list(columns)]

    def get_interaction_features(self) -> pd.DataFrame:
        return self._table("interaction_features")

    def metadata(self) -> dict:
        return read_json(self.dir / "feature_metadata.json")

    # ------------------------------------------------------------------ online
    def compute_online_user_features(self, user_id: int, history: pd.DataFrame, users, items,
                                     reference_time: pd.Timestamp | None = None) -> pd.DataFrame:
        """Fresh features for one user at serving time, using the same definitions."""
        reference_time = reference_time or pd.Timestamp.now(tz="UTC")
        h = history[history["user_id"] == user_id]
        return compute_user_features(h, users[users["user_id"] == user_id], items, reference_time,
                                     pos_thr=self.fcfg["positive_threshold"])

    # ------------------------------------------------------------------ catalog
    def write_catalog(self) -> pd.DataFrame:
        tables = {"user": self._table("user_features"), "item": self._table("item_features"),
                  "user_item": self._table("interaction_features")}
        rows = []
        for f in FEATURE_REGISTRY:
            df = tables[f.entity]
            if "<genre>" in f.name:
                prefix = f.name.split("<")[0]
                cols = [c for c in df.columns if c.startswith(prefix)]
                stat = f"{len(cols)} columns"
                null = round(100 * df[cols].isna().mean().mean(), 2)
            else:
                s = df[f.name]
                null = round(100 * s.isna().mean(), 2)
                stat = (f"mean={s.mean():.3f}, min={s.min():.2f}, max={s.max():.2f}"
                        if pd.api.types.is_numeric_dtype(s) else f"{s.nunique()} levels, top={s.mode().iat[0]}")
            rows.append({"feature": f.name, "entity": f.entity, "dtype": f.dtype, "description": f.description,
                         "transformation": f.transformation, "source": f.source, "used_by": f.used_by,
                         "null_pct": null, "stats (train window)": stat})
        cat = pd.DataFrame(rows)
        rep = self.cfg["paths"]["reports"]
        cat.to_csv(rep / "feature_catalog.csv", index=False)
        meta = self.metadata()
        md = ["# Feature Catalog", "",
              f"Feature-set version `{meta['feature_set_version']}` - reference time {meta['reference_time']} "
              "(end of the training window). Entities: `user_id`, `item_id`, (`user_id`, `item_id`).", "",
              "Tables: `data/features/user_features.parquet`, `item_features.parquet`, `interaction_features.parquet`. "
              "All features are defined once in `src/features/definitions.py` and reused for training and prediction.", "",
              df_to_markdown(cat, ".2f"), ""]
        (rep / "feature_catalog.md").write_text("\n".join(md), encoding="utf-8")
        return cat


def run(config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    logger = get_logger("features", cfg)
    proc = cfg["paths"]["processed"]
    train = load_parquet(proc / "train.parquet")
    users, items = load_parquet(proc / "users.parquet"), load_parquet(proc / "items.parquet")
    ref = pd.Timestamp(read_json(proc / "split_metadata.json")["cutoff_timestamp"], unit="s", tz="UTC")
    fs = FeatureStore(cfg)
    meta = fs.materialize(train, users, items, ref)
    fs.write_catalog()
    logger.info("Materialised feature set %s: %s", meta["feature_set_version"],
                {k: v["rows"] for k, v in meta["tables"].items()})
    return meta


if __name__ == "__main__":
    run()
