# Detecting Data Drift — Credit Card Fraud

> MLOps challenge: detecting, diagnosing, and responding to data drift in a fraud detection model.

---

## Team

| Name | Student number |
|------|---------------|
| Jorrit | 245486 |
| _(teammate)_ | |
| _(teammate)_ | |

**Branch:** `team-jorrit-drift`  
**Contact:** Martha Nikolaou

---

## Project Overview

Real-world ML models degrade silently as data distributions shift over time. This project trains a fraud classifier on historical credit card data, then monitors how the model behaves as production data drifts across 5 simulated batches.

### Tasks

| # | Task | Notebook |
|---|------|----------|
| 1 | Train a baseline fraud classifier | `02_baseline_model.ipynb` |
| 2 | Compare training vs production distributions | `01_eda.ipynb` |
| 3 | Detect drift (PSI, KS test, Evidently) | `03_drift_detection.ipynb` |
| 4 | Assess model performance degradation | `04_impact_analysis.ipynb` |
| 5 | Monitoring & retraining strategy | `src/drift_monitor.py` |

---

## Repository Structure

```
.
├── data/
│   ├── README_data.md         # Data documentation
│   ├── creditcard.csv         # Training data (not committed)
│   ├── drift_1.csv            # Production batch 1 (not committed)
│   └── ...                    # drift_2 through drift_5
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_baseline_model.ipynb # Baseline model training
│   ├── 03_drift_detection.ipynb# Drift detection analysis
│   └── 04_impact_analysis.ipynb# Performance impact assessment
├── src/
│   ├── drift_monitor.py        # Automated drift detection logic
│   └── retrain_trigger.py      # Retraining trigger conditions
├── dashboard/
│   └── app.py                  # Streamlit monitoring dashboard
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo and switch to your branch

```bash
git clone <repo-url>
cd <repo-name>
git checkout team-[name]-drift
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add data files

Place all CSV files in the `data/` folder (they are gitignored due to size).  
See `data/README_data.md` for the full file list.

---

## Running the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard shows:
- Feature distribution plots (train vs production batches)
- Drift scores per feature with threshold indicators
- Model performance metrics over time
- Alert / health status indicator

---

## Running the Notebooks

Notebooks are numbered and should be run in order:

```bash
jupyter notebook
```

Each notebook is self-contained and runs top-to-bottom without errors.

---

## Key Findings

> _To be filled in after completing the analysis._

- **Drifted features:** ...
- **Worst batch:** ...
- **Performance degradation:** AUC-ROC dropped from X to Y on batch Z
- **Retraining threshold:** PSI > 0.2 on any feature triggers alert

---

## Drift Detection Methods Used

| Method | What it detects | Threshold |
|--------|----------------|-----------|
| KS test | Covariate shift in continuous features | p < 0.05 |
| PSI (Population Stability Index) | Feature distribution shift | PSI > 0.2 |
| Chi-squared | Categorical / label drift | p < 0.05 |
| Evidently AI | Automated full-suite drift report | — |

---

## License

Educational use only. Dataset from [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
