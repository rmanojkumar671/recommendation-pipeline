# data/raw

Written by **stage 1 (ingestion)** when you run `python run_pipeline.py`:

- `ml-100k/` - original MovieLens 100K files (u.data, u.item, u.user, u.genre, u.occupation), SHA-256 verified
- `users.csv`, `items.csv`, `interactions.csv` - parsed tables (+ documented, labelled injected defects)
- `ingestion_manifest.json` - source, checksums and row counts

Not committed: the MovieLens licence does not allow redistribution. See `data/samples/1_ingestion/` for examples.
