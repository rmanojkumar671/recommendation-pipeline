# Data Validation Report

Generated: 2026-11-10T13:36:17.733939+00:00

Quarantined rows are written to `data/errors/<table>_errors.csv` with the list of failed rules. Rows that fail only `warning` rules are kept and fixed in the preparation stage.

## Summary by table

| table | input_rows | valid_rows | error_rows | error_pct | injected_bad_rows | injected_bad_rows_caught | clean_rows_wrongly_quarantined |
|---|---|---|---|---|---|---|---|
| users | 953 | 943 | 10 | 1.0490 | 10 | 10 | 0 |
| items | 1682 | 1682 | 0 | 0 | 0 | 0 | 0 |
| interactions | 100410 | 100100 | 310 | 0.3090 | 310 | 310 | 0 |

`injected_bad_rows` is the number of deliberately corrupted rows added at ingestion (ground truth); `injected_bad_rows_caught` shows how many the rules detected.

## Results by rule

| table | rule | severity | failed_rows | failed_pct | description |
|---|---|---|---|---|---|
| users | missing_required | error | 0 | 0 | Any required field is null/blank |
| users | bad_type_user_id | error | 0 | 0 | user_id must be an integer |
| users | duplicate_user_id | error | 4 | 0.420 | user_id must be unique (later copies quarantined) |
| users | invalid_age | error | 3 | 0.315 | age must be an integer in [5, 100] |
| users | invalid_gender | error | 3 | 0.315 | gender must be one of ['M', 'F'] |
| users | unknown_occupation | error | 0 | 0 | occupation must be in u.occupation |
| users | non_numeric_zip | warning | 18 | 1.889 | zip_code is not a 5-digit US code (kept; standardised in preparation) |
| items | missing_required | error | 0 | 0 | item_id and title are required |
| items | bad_type_item_id | error | 0 | 0 | item_id must be an integer |
| items | duplicate_item_id | error | 0 | 0 | item_id must be unique |
| items | invalid_genre_flag | error | 0 | 0 | genre flags must be 0/1 |
| items | missing_release_date | warning | 1 | 0.059 | release_date missing/unparseable (imputed from title year) |
| items | missing_imdb_url | warning | 3 | 0.178 | imdb_url missing (filled with 'unknown') |
| items | empty_video_release_date | warning | 1682 | 100 | video_release_date is null (column is 100% empty and dropped) |
| items | no_genre | warning | 0 | 0 | item has no genre flag set |
| interactions | missing_required | error | 40 | 0.040 | user_id, item_id, rating, timestamp are required |
| interactions | bad_type_ids | error | 0 | 0 | user_id/item_id must be integers |
| interactions | bad_type_rating | error | 20 | 0.020 | rating must be numeric |
| interactions | invalid_rating | error | 40 | 0.040 | rating must be an integer in [1, 5] |
| interactions | invalid_timestamp | error | 30 | 0.030 | timestamp must be unix seconds between 1995-01-01 and now |
| interactions | exact_duplicate | error | 150 | 0.149 | exact duplicate of an earlier row (first occurrence kept) |
| interactions | orphan_user | error | 15 | 0.015 | user_id not found in validated users (referential integrity) |
| interactions | orphan_item | error | 15 | 0.015 | item_id not found in validated items (referential integrity) |
| interactions | repeat_user_item | warning | 438 | 0.436 | same user rated the same item more than once (latest kept in preparation) |

## Pandera schema contracts (run on the rows that passed)

| table | status | failures |
|---|---|---|
| users | passed | 0 |
| items | passed | 0 |
| interactions | passed | 0 |

## Sample of quarantined interactions

| user_id | item_id | rating | timestamp | failed_rules |
|---|---|---|---|---|
| 93 | 820 | 3 | 888705966 | exact_duplicate |
| 130 | 622 | 3 | 875802173 | exact_duplicate |
| 109 | 1039 | 2 | 880579418 | exact_duplicate |
| 201 | 1227 | 1 | 884140787 | exact_duplicate |
| 249 | 218 | 3 | 879641322 | exact_duplicate |
| 215 | 216 | -1.0 | 891435782 | invalid_rating |
| 262 | 419 | 3 | 879791710 | exact_duplicate |
| 1 | 9433 | 1 | 875241619 | orphan_item |
| 297 | 116 | 4 | 874954260 | exact_duplicate |
| 314 | 1150 | -1.0 | 877887002 | invalid_rating |
| 401 | 71 | 2 | <null> | missing_required |
| 291 | 428 | ? | 874871766 | bad_type_rating |
