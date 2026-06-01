# Model Interpretation

## Why was Random Forest selected?


**Random Forest** was selected by **validation F1** (test F1 ≈ 0.619, recall ≈ 0.725).

1. **Non-linear interactions** (e.g., fiber + month-to-month) captured without manual feature crosses.
2. **Robust to outliers** in `MonthlyCharges` and `TotalCharges`.
3. **Strong CV F1** (mean ≈ 0.637 ± 0.018) — highest among the three models on train folds.
4. **Ensemble averaging** reduces variance vs. a single tree.
5. **Interpretable** via Gini importance and SHAP (see explainability outputs).


## Why did Random Forest differ from the winner?

Random Forest: accuracy **0.764**, F1 **0.619**, recall **0.725**.

- Default **0.5 threshold** favors the majority (no-churn) class.
- **Bagging** can smooth boundaries — fewer borderline churners flagged vs. a high-recall linear model.
- CV mean F1 ≈ 0.637 — competitive but validation split chose **random forest**.

## Why did Hist Gradient Boosting show high accuracy but low F1?

HistGradientBoosting: accuracy **0.793** (highest), F1 **0.562** (lowest), recall **0.500** (lowest).

- Boosting with default threshold **prioritizes overall accuracy** → predicts **no churn** more often.
- **Higher precision (0.643)** but misses churners — poor for retention goals.
- **Accuracy is misleading** under ~27% churn prevalence.

## Logistic Regression vs. trees (comparative)

| Aspect | Logistic Regression | Tree ensembles |
|--------|---------------------|----------------|
| Test F1 | 0.625 | RF 0.619, HGB 0.562 |
| Test recall | 0.810 | RF 0.725, HGB 0.500 |
| Interpretability | Coefficients | SHAP / feature importance |

## Why is recall important for churn?

- **False negatives** = missed leavers → revenue loss.
- **False positives** = retention spend on loyal customers — often cheaper than losing subscribers.
- Campaigns usually prefer **high recall**, then refine with precision for offer cost.
- Use **F1** and **PR-AUC**, not accuracy alone.

## Summary table (test set)

| Model | Accuracy | F1 | Recall | PR-AUC | Mean CV F1 ± Std |
|-------|----------|-----|--------|--------|------------------|
| Logistic Regression | 0.742 | 0.625 | 0.810 | 0.623 | 0.621 ± 0.021 |
| Random Forest | 0.764 | 0.619 | 0.725 | 0.619 | 0.637 ± 0.018 |
| Hist Gradient Boosting | 0.793 | 0.562 | 0.500 | 0.637 | 0.591 ± 0.031 |
