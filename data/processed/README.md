# data/processed

Written by **stages 2-4 (validation, preparation, transformation)**:

- `validated_*.parquet` - rows that passed all validation rules and Pandera contracts
- `users.parquet`, `items.parquet`, `interactions_clean.parquet` - cleaned and standardised entities
- `train.parquet`, `test.parquet`, `split_metadata.json` - time-based train/test split
- `user_activity_summary.parquet`, `item_popularity.parquet`, `genre_summary.parquet` - derived summaries
- `interaction_matrix.npz`, `user_index.parquet`, `item_index.parquet` - sparse user-item matrix + id maps

Regenerated on every run, so not committed. See `data/samples/2_validation` ... `4_transformation` for examples.
