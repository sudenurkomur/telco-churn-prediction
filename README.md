# Telco Customer Churn Prediction

[![Repository](https://img.shields.io/badge/GitHub-telco--churn--prediction-blue)](https://github.com/sudenurkomur/telco-churn-prediction)

**Course:** CME 4403 Introduction to Machine Learning  
**Task:** Supervised binary classification — predict whether a telecom customer will churn.  
**Dataset:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (also distributed on Kaggle as `telco-customer-churn`).

## Problem definition

- **Target:** `Churn` (`Yes` / `No`)
- **Goal:** Learn a model from customer demographics, services, contract, and billing features that generalizes to unseen customers.
- **Why it matters:** Retaining customers is cheaper than acquisition; early churn risk scores support targeted retention.

## Dataset

| Property | Value |
|----------|--------|
| Instances | 7,043 customers |
| Features | 19 predictors (after dropping `customerID`) |
| Target | Binary churn (~26.5% positive class) |
| Source | IBM / Kaggle (modern, documented; not UCI) |

Place the CSV in the project root:

```
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Project structure

```
ML_Homework/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # data (not redistributed in repo)
├── requirements.txt
├── README.md
├── src/
│   ├── config.py          # paths, seeds, column definitions
│   ├── preprocess.py      # cleaning, feature engineering, splits
│   ├── models.py          # 3 model pipelines + hyperparameter spaces
│   ├── train.py           # training and model selection
│   ├── evaluate.py        # metrics, subgroup error analysis
│   ├── visualize.py       # EDA and result plots
│   └── main.py            # end-to-end entry point
└── outputs/
    ├── figures/           # PNG plots
    └── results/           # CSV/JSON metrics
```

## Models (assignment minimum: 3)

| Model | Role |
|-------|------|
| Logistic Regression | Baseline (linear, class-weighted) |
| Random Forest | Substantive (ensemble trees) |
| Hist Gradient Boosting | Substantive (gradient boosted trees) |

## Experimental design

1. **Stratified split:** 60% train / 20% validation / 20% test  
2. **No leakage:** All imputation, scaling, and one-hot encoding live inside `sklearn.Pipeline` and are fit only on training folds.  
3. **Tuning:** `RandomizedSearchCV` with 5-fold stratified CV on the **train** split only (`scoring=f1`).  
4. **Selection:** Best model chosen by **validation F1**.  
5. **Final evaluation:** Best model refit on train+val, evaluated once on the **held-out test** set.  
6. **Metrics:** Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, subgroup error analysis.

## Setup

```bash
cd ML_Homework
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python -m src.main
```

Runtime is typically a few minutes (hyperparameter search on ~4k training rows).

## Outputs

After running, inspect:

- `outputs/figures/` — EDA plots, ROC/PR curves, confusion matrices, model comparison bar chart  
- `outputs/results/` — `test_metrics_all_models.csv`, `validation_metrics.csv`, `subgroup_error_analysis.csv`, `run_summary.json`

### Explainable AI & feature selection (Step 7)

| Figure | Description |
|--------|-------------|
| `confusion_matrices_all_models.png` | All models: counts + row-normalized % |
| `confusion_matrix_best_detailed.png` | Best model with TN/FP/FN/TP metrics |
| `feature_selection_mutual_info.png` | Top features by mutual information |
| `permutation_importance_*.png` | Raw & encoded permutation importance |
| `shap_summary_*.png` / `shap_bar_*.png` | SHAP beeswarm & bar plots |
| `rf_feature_importance_*.png` | Random Forest Gini importances |

| CSV | Description |
|-----|-------------|
| `feature_selection_mutual_info.csv` | Full MI ranking |
| `feature_selection_top25.csv` | SelectKBest top features |
| `shap_importance_*.csv` | Mean \|SHAP\| per feature |
| `permutation_importance_*.csv` | Permutation scores |

Run only XAI step (after full train): `python -m src.run_explainability`  
Regenerate rubric tables/report sections: `python -m src.run_reporting`

### Rubric reporting (Step 8)

| Output | Description |
|--------|-------------|
| `results/cv_summary.csv` | Mean ± std CV F1 and ROC-AUC per model |
| `results/cv_summary_table.md` | Markdown table for PDF report |
| `results/classification_report_detailed.csv` | Per-class precision/recall/F1/support |
| `results/pr_auc_results.csv` | PR-AUC on test set |
| `results/final_results_table.csv` | All metrics, all models |
| `figures/precision_recall_curves.png` | Dedicated PR curves |
| `report_sections/limitations.md` | Limitations section |
| `report_sections/ethics.md` | Ethics section |
| `report_sections/reproducibility.md` | Seeds, versions, run instructions |
| `report_sections/model_interpretation.md` | Why LR won; recall discussion |
| `results/rubric_compliance_check.md` | Assignment checklist |
| `results/final_project_audit.md` | Readiness scores (/10 per category) |

## Reproducibility

- Global random seed: `42` (see `src/config.py`)
- Fixed stratified splits and documented protocol in this README

## Report

Write a 5–8 page PDF report covering: problem motivation, data provenance, preprocessing, models, experimental setup, results table, at least one figure, discussion, limitations/ethics, references.

## Academic integrity

This implementation is an original pipeline for the term project. If you use public Kaggle notebooks or generative AI assistance, cite them in your report as required by the course.

## License

Dataset terms follow the Kaggle / IBM source. Code in `src/` is provided for course submission.
