# Rubric Compliance Check (CME 4403 Term Project)

Generated: 2026-06-01 19:28 UTC

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Supervised classification or regression stated | ✓ Satisfied | Binary churn classification in README and code docstrings |
| Dataset from modern approved source (not UCI) | ✓ Satisfied | IBM Telco Churn via Kaggle cited in README |
| Clearly defined target variable | ✓ Satisfied | Target: Churn (Yes/No); 7043 labeled rows |
| ≥500 labeled examples | ✓ Satisfied | 7043 instances |
| One baseline + two substantive models | ✓ Satisfied | Logistic Regression, Random Forest, HistGradientBoosting |
| Train/val/test or sound CV protocol | ✓ Satisfied | 60/20/20 stratified + 5-fold CV on train |
| No data leakage in preprocessing | ✓ Satisfied | sklearn Pipeline; fit on train/CV folds only |
| Hyperparameter tuning not on test set | ✓ Satisfied | RandomizedSearchCV on train; selection on val |
| Stratified splits for classification | ✓ Satisfied | stratified_train_val_test_split |
| Reproducibility (random seeds) | ✓ Satisfied | RANDOM_STATE=42; reproducibility.md |
| Uncertainty reporting (CV mean ± std) | ✓ Satisfied | cv_summary.csv with F1 and ROC-AUC |
| Classification metrics: accuracy/balanced acc, P/R/F1 | ✓ Satisfied | final_results_table.csv |
| Confusion matrix | ✓ Satisfied | Multiple confusion matrix figures |
| ROC-AUC and PR-AUC | ✓ Satisfied | ROC/PR curves and pr_auc_results.csv |
| Error analysis beyond single metric | ✓ Satisfied | Contract and tenure subgroup errors |
| Source code + README + requirements | ✓ Satisfied | README.md, requirements.txt, src/ |
| Explainability / interpretation | ✓ Satisfied | SHAP, permutation importance, MI feature selection |
| Limitations and ethics discussion | ✓ Satisfied | report_sections/limitations.md, ethics.md |
| Written PDF report (5–8 pages) | ✗ Missing | Student must compile PDF; REPORT_OUTLINE.md provided |

**Summary:** 18 satisfied, 0 partial, 1 missing (of 19 items).

**Note:** PDF report is the main remaining deliverable for full submission.