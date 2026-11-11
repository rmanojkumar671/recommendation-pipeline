"""Stage 2 - Data validation.

Two layers:
1. Row-level rule engine. Each rule is a vectorised boolean check with a
   severity. Rows that fail an `error` rule are quarantined into
   data/errors/<table>_errors.csv together with the list of failed rules;
   `warning` rules are reported but the row is kept (fixed in preparation).
2. Schema contract (Pandera). The rows that survive are type-coerced and
   validated against a strict Pandera schema; the pipeline stops if the
   contract fails, so downstream stages can trust their inputs.

Raw files are read as strings so that type errors are detected instead of
silently coerced.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pandera.pandas as pa

from src.utils.common import df_to_markdown, get_logger, load_config, save_parquet, write_json


# --------------------------------------------------------------------------- helpers
def _is_int_like(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    return num.notna() & (num == np.floor(num))


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


class RuleEngine:
    def __init__(self, table: str, df: pd.DataFrame):
        self.table, self.df = table, df
        self.results: list[dict] = []
        self.fail = pd.DataFrame(index=df.index)

    def check(self, name: str, mask_bad: pd.Series, severity: str, description: str):
        mask_bad = mask_bad.fillna(True).astype(bool)
        self.fail[name] = mask_bad
        self.results.append({"table": self.table, "rule": name, "severity": severity,
                             "description": description, "failed_rows": int(mask_bad.sum()),
                             "failed_pct": round(100 * mask_bad.mean(), 3)})
        self._sev = getattr(self, "_sev", {}) | {name: severity}

    def split(self):
        err_rules = [r for r, s in self._sev.items() if s == "error"]
        bad = self.fail[err_rules].any(axis=1)
        failed = self.fail[err_rules].apply(lambda r: ";".join(r.index[r.values]), axis=1)
        errors = self.df[bad].copy()
        errors["failed_rules"] = failed[bad]
        errors["validated_at_utc"] = datetime.now(timezone.utc).isoformat()
        return self.df[~bad].copy(), errors


# --------------------------------------------------------------------------- per-table rules
def validate_users(df: pd.DataFrame, vcfg: dict, occupations: set[str]) -> RuleEngine:
    e = RuleEngine("users", df)
    req = ["user_id", "age", "gender", "occupation", "zip_code"]
    e.check("missing_required", df[req].isna().any(axis=1) | (df[req].astype(str).apply(lambda c: c.str.strip()) == "").any(axis=1),
            "error", "Any required field is null/blank")
    e.check("bad_type_user_id", ~_is_int_like(df["user_id"]), "error", "user_id must be an integer")
    e.check("duplicate_user_id", df["user_id"].duplicated(keep="first"), "error", "user_id must be unique (later copies quarantined)")
    age = _num(df["age"])
    e.check("invalid_age", ~_is_int_like(df["age"]) | (age < vcfg["age_min"]) | (age > vcfg["age_max"]),
            "error", f"age must be an integer in [{vcfg['age_min']}, {vcfg['age_max']}]")
    e.check("invalid_gender", ~df["gender"].isin(vcfg["allowed_genders"]), "error",
            f"gender must be one of {vcfg['allowed_genders']}")
    e.check("unknown_occupation", ~df["occupation"].isin(occupations), "error", "occupation must be in u.occupation")
    e.check("non_numeric_zip", ~df["zip_code"].astype(str).str.fullmatch(r"\d{5}"), "warning",
            "zip_code is not a 5-digit US code (kept; standardised in preparation)")
    return e


def validate_items(df: pd.DataFrame) -> RuleEngine:
    e = RuleEngine("items", df)
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    e.check("missing_required", df[["item_id", "title"]].isna().any(axis=1), "error", "item_id and title are required")
    e.check("bad_type_item_id", ~_is_int_like(df["item_id"]), "error", "item_id must be an integer")
    e.check("duplicate_item_id", df["item_id"].duplicated(keep="first"), "error", "item_id must be unique")
    e.check("invalid_genre_flag", ~df[genre_cols].isin(["0", "1", 0, 1]).all(axis=1), "error", "genre flags must be 0/1")
    rd = pd.to_datetime(df["release_date"], format="%d-%b-%Y", errors="coerce")
    e.check("missing_release_date", rd.isna(), "warning", "release_date missing/unparseable (imputed from title year)")
    e.check("missing_imdb_url", df["imdb_url"].isna(), "warning", "imdb_url missing (filled with 'unknown')")
    e.check("empty_video_release_date", df["video_release_date"].isna(), "warning",
            "video_release_date is null (column is 100% empty and dropped)")
    e.check("no_genre", df[genre_cols].astype(str).eq("0").all(axis=1), "warning", "item has no genre flag set")
    return e


def validate_interactions(df: pd.DataFrame, vcfg: dict, user_ids: set[int], item_ids: set[int]) -> RuleEngine:
    e = RuleEngine("interactions", df)
    req = ["user_id", "item_id", "rating", "timestamp"]
    e.check("missing_required", df[req].isna().any(axis=1), "error", "user_id, item_id, rating, timestamp are required")
    e.check("bad_type_ids", (df["user_id"].notna() & ~_is_int_like(df["user_id"])) |
            (df["item_id"].notna() & ~_is_int_like(df["item_id"])), "error", "user_id/item_id must be integers")
    rating = _num(df["rating"])
    e.check("bad_type_rating", df["rating"].notna() & rating.isna(), "error", "rating must be numeric")
    e.check("invalid_rating", rating.notna() & (~_is_int_like(df["rating"]) | (rating < vcfg["rating_min"]) |
            (rating > vcfg["rating_max"])), "error",
            f"rating must be an integer in [{vcfg['rating_min']}, {vcfg['rating_max']}]")
    ts = _num(df["timestamp"])
    tmin = pd.Timestamp(vcfg["timestamp_min"], tz="UTC").timestamp()
    tmax = datetime.now(timezone.utc).timestamp() if vcfg["timestamp_max"] == "now" else \
        pd.Timestamp(vcfg["timestamp_max"], tz="UTC").timestamp()
    e.check("invalid_timestamp", df["timestamp"].notna() & (ts.isna() | (ts < tmin) | (ts > tmax)), "error",
            f"timestamp must be unix seconds between {vcfg['timestamp_min']} and {vcfg['timestamp_max']}")
    e.check("exact_duplicate", df.duplicated(subset=req, keep="first"), "error",
            "exact duplicate of an earlier row (first occurrence kept)")
    uid, iid = _num(df["user_id"]), _num(df["item_id"])
    e.check("orphan_user", uid.notna() & ~uid.isin(user_ids), "error", "user_id not found in validated users (referential integrity)")
    e.check("orphan_item", iid.notna() & ~iid.isin(item_ids), "error", "item_id not found in validated items (referential integrity)")
    e.check("repeat_user_item", df.duplicated(subset=["user_id", "item_id"], keep=False) &
            ~df.duplicated(subset=req, keep=False), "warning",
            "same user rated the same item more than once (latest kept in preparation)")
    return e


# --------------------------------------------------------------------------- pandera contracts
USERS_SCHEMA = pa.DataFrameSchema({
    "user_id": pa.Column(int, pa.Check.gt(0), unique=True),
    "age": pa.Column(int, pa.Check.in_range(5, 100)),
    "gender": pa.Column(str, pa.Check.isin(["M", "F"])),
    "occupation": pa.Column(str),
    "zip_code": pa.Column(str, nullable=False),
}, strict=False, coerce=True)

ITEMS_SCHEMA = pa.DataFrameSchema({
    "item_id": pa.Column(int, pa.Check.gt(0), unique=True),
    "title": pa.Column(str, nullable=False),
}, strict=False, coerce=True)

INTERACTIONS_SCHEMA = pa.DataFrameSchema({
    "user_id": pa.Column(int, pa.Check.gt(0)),
    "item_id": pa.Column(int, pa.Check.gt(0)),
    "rating": pa.Column(int, pa.Check.in_range(1, 5)),
    "timestamp": pa.Column(int, pa.Check.gt(0)),
}, strict=False, coerce=True, unique=["user_id", "item_id", "timestamp"])


def _cast(df: pd.DataFrame, ints: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in ints:
        df[c] = pd.to_numeric(df[c]).astype("int64")
    return df


# --------------------------------------------------------------------------- stage entry point
def run(config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    logger = get_logger("validation", cfg)
    raw, err_dir, proc = cfg["paths"]["raw"], cfg["paths"]["errors"], cfg["paths"]["processed"]
    vcfg = cfg["validation"]

    users = pd.read_csv(raw / "users.csv", dtype=str, keep_default_na=False, na_values=[""])
    items = pd.read_csv(raw / "items.csv", dtype=str)
    # keep_default_na=False so strings like "N/A" are seen as bad values, not silently as nulls
    inter = pd.read_csv(raw / "interactions.csv", dtype=str, keep_default_na=False, na_values=[""])
    occupations = set((raw / "ml-100k" / "u.occupation").read_text(encoding="latin-1").split())

    # ground-truth tags from the noise injector are kept aside for scoring only
    truth = {name: df.pop("_injected") if "_injected" in df else pd.Series("none", index=df.index)
             for name, df in (("users", users), ("items", items), ("interactions", inter))}

    report = {"run_at_utc": datetime.now(timezone.utc).isoformat(), "tables": {}, "rules": [], "contracts": {}}
    outputs = {}

    eu = validate_users(users, vcfg, occupations)
    users_ok, users_err = eu.split()
    ei = validate_items(items)
    items_ok, items_err = ei.split()
    valid_uids = set(pd.to_numeric(users_ok["user_id"]).astype(int))
    valid_iids = set(pd.to_numeric(items_ok["item_id"]).astype(int))
    ex = validate_interactions(inter, vcfg, valid_uids, valid_iids)
    inter_ok, inter_err = ex.split()

    for name, eng, ok, err in (("users", eu, users_ok, users_err), ("items", ei, items_ok, items_err),
                               ("interactions", ex, inter_ok, inter_err)):
        err_with_truth = err.assign(_injected=truth[name].loc[err.index])
        err_with_truth.to_csv(err_dir / f"{name}_errors.csv", index=False)
        report["rules"] += eng.results
        injected = truth[name] != "none"
        caught = injected & truth[name].index.isin(err.index)
        # "repeat_rating" injections are intentionally *kept* (they are a warning, resolved in preparation)
        expected_err = injected & (truth[name] != "repeat_rating")
        report["tables"][name] = {
            "input_rows": len(eng.df), "valid_rows": len(ok), "error_rows": len(err),
            "error_pct": round(100 * len(err) / max(len(eng.df), 1), 3),
            "injected_bad_rows": int(expected_err.sum()),
            "injected_bad_rows_caught": int((caught & expected_err).sum()),
            "clean_rows_wrongly_quarantined": int((~injected & truth[name].index.isin(err.index)).sum()),
        }
        outputs[name] = ok
        logger.info("%s: %d in, %d valid, %d quarantined", name, len(eng.df), len(ok), len(err))

    # ---- schema contracts on the surviving rows
    users_ok = _cast(users_ok, ["user_id", "age"])
    items_ok = _cast(items_ok, ["item_id"] + [c for c in items_ok.columns if c.startswith("genre_")])
    inter_ok = _cast(inter_ok, ["user_id", "item_id", "rating", "timestamp"])
    for name, df, schema in (("users", users_ok, USERS_SCHEMA), ("items", items_ok, ITEMS_SCHEMA),
                             ("interactions", inter_ok, INTERACTIONS_SCHEMA)):
        try:
            schema.validate(df, lazy=True)
            report["contracts"][name] = {"status": "passed", "failures": 0}
        except pa.errors.SchemaErrors as exc:
            report["contracts"][name] = {"status": "failed", "failures": len(exc.failure_cases)}
            logger.error("Pandera contract failed for %s:\n%s", name, exc.failure_cases.head())
    if any(c["status"] != "passed" for c in report["contracts"].values()):
        write_json(report, cfg["paths"]["reports"] / "validation_report.json")
        raise RuntimeError("Schema contract failed - see reports/validation_report.json")

    save_parquet(users_ok, proc / "validated_users.parquet")
    save_parquet(items_ok, proc / "validated_items.parquet")
    save_parquet(inter_ok, proc / "validated_interactions.parquet")
    write_json(report, cfg["paths"]["reports"] / "validation_report.json")
    write_markdown_report(report, cfg)
    logger.info("Validation complete; contracts passed")
    return report


def write_markdown_report(report: dict, cfg: dict) -> None:
    tables = pd.DataFrame(report["tables"]).T.reset_index().rename(columns={"index": "table"})
    rules = pd.DataFrame(report["rules"])
    contracts = pd.DataFrame(report["contracts"]).T.reset_index().rename(columns={"index": "table"})
    err_dir = cfg["paths"]["errors"]
    sample = pd.read_csv(err_dir / "interactions_errors.csv", dtype=str).fillna("<null>").head(12)[
        ["user_id", "item_id", "rating", "timestamp", "failed_rules"]]
    md = [
        "# Data Validation Report", "",
        f"Generated: {report['run_at_utc']}", "",
        "Quarantined rows are written to `data/errors/<table>_errors.csv` with the list of failed rules. "
        "Rows that fail only `warning` rules are kept and fixed in the preparation stage.", "",
        "## Summary by table", "", df_to_markdown(tables), "",
        "`injected_bad_rows` is the number of deliberately corrupted rows added at ingestion "
        "(ground truth); `injected_bad_rows_caught` shows how many the rules detected.", "",
        "## Results by rule", "", df_to_markdown(rules[["table", "rule", "severity", "failed_rows", "failed_pct", "description"]], ".3f"), "",
        "## Pandera schema contracts (run on the rows that passed)", "", df_to_markdown(contracts), "",
        "## Sample of quarantined interactions", "", df_to_markdown(sample), "",
    ]
    (cfg["paths"]["reports"] / "validation_report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    run()
