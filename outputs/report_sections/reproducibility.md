# Reproducibility

## Random seed
- Global `RANDOM_STATE = 42` (`src/config.py`)
- Used for train/val/test splits, cross-validation folds, and randomized hyperparameter search.

## Environment
- **Platform:** Darwin 23.6.0 (arm64)
- **Python:** 3.11.7 (main, Dec 15 2023, 12:09:56) [Clang 14.0.6 ]
- **Run date (UTC):** 2026-06-01 19:28:25

## Package versions
- **python:** 3.11.7 (main, Dec 15 2023, 12:09:56) [Clang 14.0.6 ]
- **pandas:** 2.0.3
- **numpy:** 1.24.3
- **sklearn:** 1.8.0
- **matplotlib:** 3.7.2
- **seaborn:** 0.13.2
- **shap:** 0.48.0

## Data
- **Path:** `WA_Fn-UseC_-Telco-Customer-Churn.csv` in project root
- **Source:** IBM Telco Customer Churn (Kaggle)

## Experimental protocol
| Setting | Value |
|---------|--------|
| Train / Val / Test | 60% / 20% / 20% (stratified) |
| CV folds (tuning & uncertainty) | 5 |
| Hyperparameter search | RandomizedSearchCV on train only |
| Model selection metric | Validation F1 |
| Final refit | Train + validation before test evaluation |

## How to reproduce

```bash
cd ML_Homework
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main
```

All figures are written to `outputs/figures/` and metrics to `outputs/results/`.
