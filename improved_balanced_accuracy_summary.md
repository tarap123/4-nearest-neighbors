# Improved Balanced Accuracy Experiment

Implemented in `improved_balanced_accuracy_experiment.py`.

Changes included:

- Time-aware validation split option using the latest 20% of incidents as validation.
- Out-of-fold target encoding for high-cardinality categorical columns.
- Tuned Random Forest and HistGradientBoosting models.
- Missingness indicators, datetime features, domain interactions, numeric clipping, median imputation, and class-weight tuning.

## Apples-to-Apples Random Split

This uses the same style of stratified 80/20 random validation split as the existing notebook.

| Model | Threshold | Balanced accuracy | Accuracy | Damage recall | Damage precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.13 | 0.825942 | 0.827772 | 0.823846 | 0.245436 |
| HistGradientBoosting | 0.27 | 0.829297 | 0.822865 | 0.836667 | 0.241847 |
| RF + HGB ensemble | 0.19 | 0.831711 | 0.822686 | 0.842051 | 0.242451 |

Best random-split balanced accuracy: **0.831711**.

Previous notebook RF threshold result: **0.820018**.

## Time-Aware Split

This trains on older incidents and validates on the latest 20% of incidents. It is harsher because validation is mostly 2023-2026 and has a lower damage rate.

| Model | Threshold | Balanced accuracy | Accuracy | Damage recall | Damage precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.13 | 0.803565 | 0.856006 | 0.746857 | 0.172784 |
| HistGradientBoosting | 0.22 | 0.815172 | 0.851491 | 0.775899 | 0.172397 |
| RF + HGB ensemble | 0.20 | 0.816556 | 0.847741 | 0.782835 | 0.169657 |

Best time-aware balanced accuracy: **0.816556**.
