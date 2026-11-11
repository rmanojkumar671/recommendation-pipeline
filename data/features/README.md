# data/features

Written by **stage 5 (feature store)**:

- `user_features.parquet`, `item_features.parquet`, `interaction_features.parquet` - materialised feature tables
- `feature_metadata.json` - feature-set version, reference time, row counts, null rates

Feature definitions live in `src/features/definitions.py`; documentation is in `reports/feature_catalog.md`.
Regenerated on every run, so not committed. See `data/samples/5_features/` for examples.
