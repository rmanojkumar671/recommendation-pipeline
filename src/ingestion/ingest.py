"""Stage 1 - Data ingestion.

Downloads MovieLens 100K (official GroupLens archive first, GitHub mirror as a
fallback), verifies checksums, parses the three entity files into tidy CSVs
(users, items, interactions) in data/raw/, optionally appends a documented set
of corrupted records so that the validation stage has something to catch, and
writes an ingestion manifest for lineage.
"""
from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger, load_config, set_seed, sha256_file, write_json

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


# --------------------------------------------------------------------------- download
def _download(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "recsys-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_movielens(cfg: dict, logger) -> Path:
    """Ensure data/raw/ml-100k contains the source files; return the directory."""
    icfg = cfg["ingestion"]
    target = cfg["paths"]["raw"] / "ml-100k"
    target.mkdir(parents=True, exist_ok=True)

    if all((target / f).exists() for f in icfg["files"]) and _checksums_ok(target, icfg, logger):
        logger.info("Source files already present and verified in %s (cache hit)", target)
        return target

    # 1) official archive
    try:
        logger.info("Downloading official archive %s", icfg["official_zip_url"])
        blob = _download(icfg["official_zip_url"])
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for f in icfg["files"]:
                with zf.open(f"ml-100k/{f}") as src, open(target / f, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        source = icfg["official_zip_url"]
    except Exception as exc:  # network blocked, timeout, ...
        logger.warning("Official download failed (%s); falling back to mirror", exc)
        for f in icfg["files"]:
            (target / f).write_bytes(_download(f"{icfg['mirror_base_url']}/{f}"))
        source = icfg["mirror_base_url"]

    if not _checksums_ok(target, icfg, logger):
        raise RuntimeError("Checksum mismatch - downloaded MovieLens files are corrupt or altered")
    (target / "SOURCE.txt").write_text(source)
    logger.info("Downloaded MovieLens 100K from %s", source)
    return target


def _checksums_ok(folder: Path, icfg: dict, logger) -> bool:
    for fname, expected in icfg.get("checksums", {}).items():
        actual = sha256_file(folder / fname)
        if actual != expected:
            logger.error("Checksum mismatch for %s: %s != %s", fname, actual, expected)
            return False
    return True


# --------------------------------------------------------------------------- parsing
def parse_users(folder: Path) -> pd.DataFrame:
    return pd.read_csv(folder / "u.user", sep="|", header=None,
                       names=["user_id", "age", "gender", "occupation", "zip_code"],
                       dtype={"zip_code": str}, encoding="latin-1")


def parse_items(folder: Path) -> pd.DataFrame:
    cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + \
           [f"genre_{g}" for g in GENRES]
    return pd.read_csv(folder / "u.item", sep="|", header=None, names=cols, encoding="latin-1")


def parse_interactions(folder: Path) -> pd.DataFrame:
    return pd.read_csv(folder / "u.data", sep="\t", header=None,
                       names=["user_id", "item_id", "rating", "timestamp"])


# --------------------------------------------------------------------------- noise injection
def inject_noise(users: pd.DataFrame, items: pd.DataFrame, inter: pd.DataFrame,
                 ncfg: dict, rng: np.random.Generator):
    """Append corrupted records. Every injected row is tagged in `_injected` so the
    validation report can show precision/recall of the checks against ground truth."""
    inter = inter.copy().astype({"user_id": object, "item_id": object,
                                 "rating": object, "timestamp": object})
    inter["_injected"] = "none"
    pieces = [inter]

    def sample(n):
        return inter.iloc[rng.choice(len(inter), size=n, replace=False)].copy()

    d = sample(ncfg["duplicate_interactions"]); d["_injected"] = "duplicate"; pieces.append(d)

    r = sample(ncfg["repeat_interactions"])
    r["timestamp"] = r["timestamp"].astype(int) + rng.integers(3600, 86400 * 5, len(r))
    r["rating"] = rng.integers(1, 6, len(r)); r["_injected"] = "repeat_rating"; pieces.append(r)

    m = sample(ncfg["missing_values"])
    col = rng.choice(["rating", "user_id", "timestamp"], len(m))
    for i, c in enumerate(col):
        m.iloc[i, m.columns.get_loc(c)] = None
    m["_injected"] = "missing_value"; pieces.append(m)

    b = sample(ncfg["invalid_ratings"])
    b["rating"] = rng.choice([0, 6, 7, -1, 3.5, 10], len(b)); b["_injected"] = "invalid_rating"; pieces.append(b)

    t = sample(ncfg["invalid_timestamps"])
    future = int(datetime(2035, 1, 1, tzinfo=timezone.utc).timestamp())
    t["timestamp"] = rng.choice([-1, 0, 123456, future, future + 999], len(t)); t["_injected"] = "invalid_timestamp"
    pieces.append(t)

    o = sample(ncfg["orphan_interactions"])
    half = len(o) // 2
    o.iloc[:half, o.columns.get_loc("user_id")] = rng.integers(5000, 6000, half)
    o.iloc[half:, o.columns.get_loc("item_id")] = rng.integers(9000, 9999, len(o) - half)
    o["_injected"] = "orphan"; pieces.append(o)

    s = sample(ncfg["bad_type_values"])
    s["rating"] = rng.choice(["five", "N/A", "4 stars", "?"], len(s)); s["_injected"] = "bad_type"; pieces.append(s)

    # Scatter injected rows through the file, but always *after* the row they were
    # copied from, so "first occurrence" in a duplicate pair is the genuine record.
    n = len(inter)
    inter["_order"] = np.arange(n, dtype=float)
    for piece in pieces[1:]:
        pos = piece.index.to_numpy(dtype=float)
        piece["_order"] = pos + rng.uniform(0.01, 1.0, len(piece)) * (n - pos)
    noisy_inter = (pd.concat(pieces, ignore_index=True).sort_values("_order")
                   .drop(columns="_order").reset_index(drop=True))

    # users: invalid age, invalid gender, duplicate id
    users = users.copy().astype({"age": object})
    users["_injected"] = "none"
    bad = users.sample(ncfg["bad_users"], random_state=int(rng.integers(1e9))).copy()
    n3 = len(bad) // 3
    bad.iloc[:n3, bad.columns.get_loc("age")] = rng.choice([-3, 0, 150, 999], n3)
    bad.iloc[n3:2 * n3, bad.columns.get_loc("gender")] = rng.choice(["X", "U", "male"], n3)
    # invalid-age / invalid-gender users get fresh ids so they test *only* that rule
    bad.iloc[:2 * n3, bad.columns.get_loc("user_id")] = np.arange(2000, 2000 + 2 * n3)
    bad["_injected"] = ["invalid_age"] * n3 + ["invalid_gender"] * n3 + ["duplicate_user_id"] * (len(bad) - 2 * n3)
    users = pd.concat([users, bad], ignore_index=True)

    items = items.copy()
    items["_injected"] = "none"
    return users, items, noisy_inter


# --------------------------------------------------------------------------- stage entry point
def run(config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    logger = get_logger("ingestion", cfg)
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    src = fetch_movielens(cfg, logger)
    users, items, inter = parse_users(src), parse_items(src), parse_interactions(src)
    logger.info("Parsed users=%d items=%d interactions=%d", len(users), len(items), len(inter))

    if cfg["ingestion"]["inject_noise"]:
        users, items, inter = inject_noise(users, items, inter, cfg["ingestion"]["noise"], rng)
        logger.info("Injected noise -> users=%d interactions=%d", len(users), len(inter))

    raw = cfg["paths"]["raw"]
    users.to_csv(raw / "users.csv", index=False)
    items.to_csv(raw / "items.csv", index=False)
    inter.to_csv(raw / "interactions.csv", index=False)

    manifest = {
        "stage": "ingestion",
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": cfg["ingestion"]["dataset"],
        "source": (src / "SOURCE.txt").read_text() if (src / "SOURCE.txt").exists() else "cache",
        "source_checksums": {f: sha256_file(src / f) for f in cfg["ingestion"]["files"]},
        "noise_injected": cfg["ingestion"]["inject_noise"],
        "outputs": {
            "users.csv": {"rows": len(users), "sha256": sha256_file(raw / "users.csv")},
            "items.csv": {"rows": len(items), "sha256": sha256_file(raw / "items.csv")},
            "interactions.csv": {"rows": len(inter), "sha256": sha256_file(raw / "interactions.csv")},
        },
    }
    write_json(manifest, raw / "ingestion_manifest.json")
    logger.info("Ingestion complete; manifest written")
    return manifest


if __name__ == "__main__":
    run()
