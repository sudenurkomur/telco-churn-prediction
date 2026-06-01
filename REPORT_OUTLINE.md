# Report Outline (5–8 pages PDF)

Use this outline for your written report. Numbers below are from a reference run (`python -m src.main`); re-run and update if your results differ slightly.

## 1. Introduction
- Predict telecom customer churn to enable proactive retention.
- Binary classification; imbalanced classes (~26.5% churn).

## 2. Data
- **Source:** IBM Telco Customer Churn, Kaggle (cite URL and access date).
- **Size:** 7,043 rows, 21 columns; 19 features after dropping `customerID`.
- **Target:** `Churn` (Yes/No).
- **Limitations:** Snapshot data (no time series), synthetic telco scenario, class imbalance, 11 `TotalCharges` missing for `tenure=0` (imputed as 0).

## 3. Preprocessing & EDA
- `TotalCharges` coerced to float; new customers (`tenure=0`) → 0.
- Engineered: `avg_monthly_spend`, `has_auto_payment`, `num_optional_services`.
- One-hot encoding for categoricals; median imputation + scaling for linear model.
- Include figures from `outputs/figures/eda_*.png`.

## 4. Methods
| Model | Role |
|-------|------|
| Logistic Regression | Baseline |
| Random Forest | Substantive |
| Hist Gradient Boosting | Substantive |

- **Split:** Stratified 60/20/20 train/val/test.
- **Tuning:** RandomizedSearchCV, 5-fold CV on train, optimize F1.
- **Leakage control:** sklearn `Pipeline` (preprocessing inside folds only).
- **Selection:** Best validation F1; refit on train+val; final metrics on test.

## 5. Experimental Setup
- Python 3, scikit-learn, pandas; seed = 42.
- Metrics: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC.

## 6. Results (example table — update from `outputs/results/test_metrics_all_models.csv`)

| Model | Accuracy | Bal. Acc. | F1 | ROC-AUC |
|-------|----------|-----------|-----|---------|
| Logistic Regression | 0.74 | 0.76 | 0.62 | 0.83 |
| Random Forest | 0.76 | 0.75 | 0.62 | 0.83 |
| Hist Gradient Boosting | 0.79 | 0.70 | 0.56 | 0.84 |

- Include `confusion_matrix_final.png`, `roc_pr_curves.png`, `model_comparison_test.png`.
- Discuss subgroup errors (`subgroup_error_analysis.csv`): month-to-month contracts, low tenure.

## 7. Discussion
- Tree models may maximize accuracy but hurt recall on churn without threshold tuning.
- Strong predictors align with EDA: tenure, contract type, fiber, electronic check.
- Errors: false positives (unnecessary retention spend) vs false negatives (lost customers).

## 8. Limitations & Ethics
- Model may reflect historical bias; careful use of `gender` in deployment.
- Churn scores should not be used to discriminate; transparency and human review.

## 9. Conclusion
- Summarize best model and whether it meets business needs (recall vs precision trade-off).

## References
- Dataset: IBM Telco Customer Churn on Kaggle.
- scikit-learn documentation.
- Course assignment PDF.
