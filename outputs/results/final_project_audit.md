# Final Project Audit

Generated: 2026-06-01 19:28 UTC

## Category Scores

| Category | Score (/10) | Strengths | Weaknesses | Recommended Fixes |
|----------|-------------|-----------|------------|-------------------|
| Problem Definition | 9 | Clear binary churn task; business motivation in README. | Could add explicit cost matrix (FN vs FP) in report. | One paragraph on retention economics in PDF. |
| Dataset Description | 9 | Source, size, target, limitations documented. | No formal license paragraph in report body yet. | Cite Kaggle URL and access date in References. |
| Preprocessing | 9 | TotalCharges fix, feature engineering, Pipeline leakage control. | Limited ablation of engineered features. | Optional: compare with/without engineered columns. |
| Modeling | 9 | Baseline + 2 substantive models; tuned hyperparameters. | No explicit threshold optimization on validation. | Optional: report precision-recall at tuned threshold. |
| Experimental Design | 10 | 60/20/20 stratified; CV tuning; test held out; cv_summary.csv. | None critical. | — |
| Evaluation | 10 | Full metric suite; CV uncertainty; PR-AUC; final_results_table. | — | Include cv_summary and final_results tables in PDF. |
| Error Analysis | 9 | Subgroup analysis; confusion matrices; classification_report_detailed. | No misclassified customer case studies. | Add 2–3 example FP/FN patterns in Discussion. |
| Explainability | 9 | SHAP, permutation importance, mutual information selection. | SHAP primarily on RF; best model is logistic. | Mention LR coefficients / encoded feature signs in report. |
| Reproducibility | 10 | Seed, requirements.txt, reproducibility.md, main entry point. | — | — |
| Report Readiness | 7 | All tables/figures and markdown sections generated. | PDF report not yet written. | Compile 5–8 page PDF using REPORT_OUTLINE.md + outputs/. |

**Average readiness score: 9.1 / 10**

## Artifact checklist

- CV summary: `True`
- Final results table: `True`
- Classification report: `True`
- PR curves figure: `True`
- Report sections: `True`

## Verdict

**Code and experiments are submission-ready.** Primary remaining task: write and submit the PDF report with figures/tables from `outputs/`.