"""Draws docs/architecture.png (run: python docs/make_architecture_diagram.py)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "architecture.png"
fig, ax = plt.subplots(figsize=(15, 8.2))
ax.set_xlim(0, 150); ax.set_ylim(0, 82); ax.axis("off")

C = {"stage": "#e8f1fa", "edge": "#2a6f9e", "store": "#fff4e5", "sedge": "#c27a1a",
     "ops": "#eef7ee", "oedge": "#3c8a3c", "txt": "#1d2733", "sub": "#4a5868"}

def box(x, y, w, h, title, lines, fc, ec, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=fs, weight="bold", color=C["txt"])
    ax.text(x + w / 2, y + h - 6.6, "\n".join(lines), ha="center", va="top", fontsize=8.1, color=C["sub"], linespacing=1.35)

def arrow(x1, y1, x2, y2, color="#55606e", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13, color=color, lw=1.4, linestyle=ls))

ax.text(75, 80, "End-to-End Data Management Pipeline - MovieLens 100K Recommender", ha="center", fontsize=15, weight="bold", color=C["txt"])

# --- main pipeline row(s)
W, H, y1, y2 = 24, 19, 50, 20
top = [("1. Ingestion", ["GroupLens zip / mirror", "SHA-256 checksums", "seeded noise injection", "manifest.json"]),
       ("2. Validation", ["rule engine: nulls, dupes,", "types, ranges, timestamps,", "referential integrity", "+ Pandera contracts"]),
       ("3. Preparation", ["standardise & impute", "dedupe (keep latest)", "time-based split", "train / test parquet"]),
       ("4. Transformation", ["user activity summary", "item popularity", "genre summary", "sparse user-item matrix"])]
xs = [4, 40, 76, 112]
for x, (t, l) in zip(xs, top):
    box(x, y1, W, H, t, l, C["stage"], C["edge"])
for a, b in zip(xs[:-1], xs[1:]):
    arrow(a + W + 0.8, y1 + H / 2, b - 0.8, y1 + H / 2)

bottom = [("5. Feature Store", ["31 documented features", "user / item / interaction", "offline + online retrieval", "versioned, catalog.md"]),
          ("6. Training (x3)", ["Popularity baseline", "Item-kNN CF", "Matrix Factorization", "runs in parallel"]),
          ("7. Evaluation", ["P/R/NDCG/HitRate @5,@10", "coverage, diversity,", "novelty, RMSE, MAE", "warm + all cohorts"]),
          ("Serving / predict", ["top-K per user", "cold-start fallback", "same feature defs", "src/training/predict.py"])]
xb = [112, 76, 40, 4]
for x, (t, l) in zip(xb, bottom):
    box(x, y2, W, H, t, l, C["stage"], C["edge"])
arrow(xs[3] + W / 2, y1 - 0.8, xb[0] + W / 2, y2 + H + 0.8)
for a, b in zip(xb[:-2], xb[1:-1]):
    arrow(a - 0.8, y2 + H / 2, b + W + 0.8, y2 + H / 2)
arrow(xb[2] - 0.8, y2 + H / 2, xb[3] + W + 0.8, y2 + H / 2)
ax.text((xb[2] + xb[3] + W) / 2, y2 + H / 2 + 1.5, "best model", fontsize=7.5, color=C["sub"], ha="center")

# --- storage layer
stores = [("data/raw", 4), ("data/errors", 40), ("data/processed", 76), ("data/features", 112)]
for (name, x) in stores:
    ax.add_patch(FancyBboxPatch((x + 3, 71.5), W - 6, 4.2, boxstyle="round,pad=0.3", fc=C["store"], ec=C["sedge"], lw=1.2))
    ax.text(x + W / 2, 73.6, name, ha="center", va="center", fontsize=8.6, color=C["sedge"], weight="bold")
    ax.plot([x + W / 2, x + W / 2], [y1 + H + 0.6, 71.3], color=C["sedge"], lw=1.3, ls=":")
    ax.text(x + W / 2 + 1, (y1 + H + 71.3) / 2, "writes", fontsize=6.8, color=C["sedge"], va="center")

# --- cross-cutting layer
box(4, 1.5, 70, 11.5, "Orchestration - Apache Airflow DAG (dags/recommendation_pipeline_dag.py)",
    ["same task graph as run_pipeline.py (src/pipeline.py)", "@weekly, retries=1, max_active_runs=1, idempotent file-based tasks"],
    C["ops"], C["oedge"], fs=10)
box(78, 1.5, 68, 11.5, "Tracking & config - MLflow + configs/config.yaml",
    ["params, metrics and model artifacts per run (MLflow)", "seed, feature-set version, data checksums, logs/pipeline.log"],
    C["ops"], C["oedge"], fs=10)
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print("saved", OUT)
