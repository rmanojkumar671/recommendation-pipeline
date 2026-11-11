# Feature Catalog

Feature-set version `19a6bee73fbf` - reference time 1998-03-07 02:21:09+00:00 (end of the training window). Entities: `user_id`, `item_id`, (`user_id`, `item_id`).

Tables: `data/features/user_features.parquet`, `item_features.parquet`, `interaction_features.parquet`. All features are defined once in `src/features/definitions.py` and reused for training and prediction.

| feature | entity | dtype | description | transformation | source | used_by | null_pct | stats (train window) |
|---|---|---|---|---|---|---|---|---|
| user_activity_count | user | int | Number of ratings the user has given | count(rating) per user | train interactions | popularity fallback, evaluation slicing | 0 | mean=84.837, min=0.00, max=737.00 |
| user_avg_rating | user | float | Mean rating given by the user | mean(rating) per user | train interactions | MF cold-item fallback, rating normalisation | 20.36 | mean=3.583, min=1.49, max=4.87 |
| user_rating_std | user | float | Std-dev of the user's ratings (0 if one rating) | std(rating) per user | train interactions | analysis | 20.36 | mean=1.014, min=0.00, max=1.75 |
| user_positive_ratio | user | float | Share of the user's ratings that are positive (>= threshold) | mean(rating >= 4) | train interactions | analysis | 20.36 | mean=0.576, min=0.00, max=1.00 |
| user_preferred_genre | user | category | Genre with most positive ratings from the user | argmax over genres of count(rating>=4) | train interactions + items | explanations, diversity analysis | 20.36 | 9 levels, top=Drama |
| user_genre_entropy | user | float | Shannon entropy of the user's genre distribution (taste breadth) | -sum p_g log2 p_g | train interactions + items | analysis | 20.36 | mean=3.351, min=-0.00, max=3.73 |
| user_days_since_last_interaction | user | int | Days between the user's last rating and the reference time | (reference_time - max(event_time)).days | train interactions | recency analysis, serving | 20.36 | mean=73.807, min=0.00, max=166.00 |
| user_tenure_days | user | int | Days between the user's first and last rating | (max - min event_time).days | train interactions | analysis | 20.36 | mean=19.971, min=0.00, max=167.00 |
| user_age | user | int | Age in years (demographic) | as-is | users | segmentation | 0 | mean=34.052, min=7.00, max=73.00 |
| user_age_group | user | category | Age bucket (<18, 18-24, ...) | pd.cut on age | users | segmentation | 0 | 7 levels, top=25-34 |
| user_gender | user | category | Gender (M/F) | upper-cased | users | segmentation | 0 | 2 levels, top=M |
| user_occupation | user | category | Occupation (21 levels) | lower-cased | users | segmentation | 0 | 21 levels, top=student |
| user_aff_<genre> | user | float (x19) | Share of the user's ratings that fall in each genre | count(genre)/count(all) per user | train interactions + items | genre-aware re-ranking, diversity | 20.36 | 19 columns |
| item_popularity | item | int | Number of ratings the item received | count(rating) per item | train interactions | popularity model | 0 | mean=47.563, min=0.00, max=473.00 |
| item_avg_rating | item | float | Mean rating of the item | mean(rating) per item | train interactions | popularity model, MF fallback | 3.92 | mean=3.042, min=1.00, max=5.00 |
| item_bayesian_avg | item | float | Damped mean rating, shrunk to the global mean | (n*avg + m*C)/(n+m), m=20 | train interactions | popularity model score + rating prediction | 0 | mean=3.409, min=2.47, max=4.41 |
| item_rating_std | item | float | Std-dev of the item's ratings | std(rating) per item | train interactions | analysis | 3.92 | mean=0.965, min=0.00, max=2.83 |
| item_positive_ratio | item | float | Share of the item's ratings that are positive | mean(rating >= 4) | train interactions | analysis | 3.92 | mean=0.398, min=0.00, max=1.00 |
| item_popularity_rank | item | int | Rank of the item by number of ratings (1 = most popular) | rank(desc) of item_popularity | train interactions | novelty metric | 0 | mean=825.043, min=1.00, max=1617.00 |
| item_recent_popularity | item | float | Exponentially time-decayed rating count (half-life 30 days) | sum 0.5^(age_days/30) | train interactions | trend-aware popularity | 0 | mean=10.681, min=0.00, max=97.17 |
| item_days_since_last_interaction | item | int | Days since the item was last rated (NaN if never rated) | (reference_time - max(event_time)).days | train interactions | staleness analysis | 3.92 | mean=16.501, min=0.00, max=167.00 |
| item_age_years | item | int | Years between release and the reference time | reference_year - release_year | items | analysis | 0 | mean=8.611, min=0.00, max=76.00 |
| item_n_genres | item | int | Number of genres the item belongs to | sum of genre flags | items | analysis | 0 | mean=1.720, min=1.00, max=6.00 |
| item_primary_genre | item | category | First listed genre of the item | first of genres | items | explanations | 0 | 19 levels, top=Drama |
| genre_<genre> | item | int 0/1 (x19) | One-hot genre flags | as-is from u.item | items | diversity metric, genre affinity | 0 | 19 columns |
| rating_dev_user | user_item | float | Rating minus the user's mean rating | rating - user_avg_rating | train interactions | analysis / implicit weighting | 0 | mean=-0.000, min=-3.66, max=3.51 |
| rating_dev_item | user_item | float | Rating minus the item's mean rating | rating - item_avg_rating | train interactions | analysis | 0 | mean=-0.000, min=-3.54, max=3.17 |
| is_positive | user_item | int 0/1 | 1 if rating >= positive threshold | rating >= 4 | train interactions | implicit-feedback models | 0 | mean=0.551, min=0.00, max=1.00 |
| days_before_reference | user_item | int | Age of the interaction in days at the reference time | (reference_time - event_time).days | train interactions | recency weighting | 0 | mean=88.362, min=0.00, max=167.00 |
| recency_weight | user_item | float | Exponential decay weight of the interaction | 0.5^(days_before_reference/30) | train interactions | recency-weighted training | 0 | mean=0.227, min=0.02, max=1.00 |
| genre_match_preferred | user_item | int 0/1 | 1 if the item contains the user's preferred genre | user_preferred_genre in item genres | train interactions + features | analysis | 0 | mean=0.434, min=0.00, max=1.00 |
