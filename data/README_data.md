# Data

## Source
Credit Card Fraud Detection dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Files

| File | Rows | Description |
|------|------|-------------|
| `creditcard.csv` | 284,807 | Original training data (2013 transactions) |
| `drift_1.csv` | 15,000 | Production batch 1 |
| `drift_2.csv` | 15,000 | Production batch 2 |
| `drift_3.csv` | 18,000 | Production batch 3 |
| `drift_4.csv` | 15,000 | Production batch 4 |
| `drift_5.csv` | 15,000 | Production batch 5 |

## Schema

- `Time` — seconds elapsed since first transaction
- `V1–V28` — PCA-transformed features (anonymised)
- `Amount` — transaction amount in EUR
- `Class` — target label (0 = legitimate, 1 = fraud)
- `day` — present in drift batches; simulates production day index

## Notes

- Raw data files are **not committed** to the repo (see `.gitignore`).
- Place all CSV files in this `data/` folder before running any notebooks.
- The dataset is highly imbalanced (~0.17% fraud). Handle accordingly.
