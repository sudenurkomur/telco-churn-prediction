#!/usr/bin/env python3
"""
Telco Customer Churn — end-to-end supervised learning pipeline.

CME 4403 Term Project
Problem: Binary classification of customer churn (Churn = Yes/No)
Dataset: IBM Telco Customer Churn (Kaggle)

Usage (from project root):
    python -m src.main
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as script or module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, OUTPUT_DIR, RESULTS_DIR, RANDOM_STATE
from src.evaluate import (
    evaluate_all_on_test,
    evaluate_final_model,
    plot_roc_pr_curves,
    subgroup_error_analysis,
)
from src.train import (
    prepare_splits,
    refit_on_train_val,
    save_training_artifacts,
    select_best_model,
    train_all_models,
)
from src.explainability import run_full_explainability
from src.reporting import run_full_reporting
from src.visualize import plot_confusion_matrix, plot_eda, plot_model_comparison


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print_header("Telco Customer Churn — ML Term Project")
    print(f"Random seed: {RANDOM_STATE}")

    # --- EDA ---
    print_header("Step 1: Exploratory Data Analysis")
    plot_eda()
    print(f"EDA figures saved to: {FIGURES_DIR}")

    # --- Splits & training ---
    print_header("Step 2: Data Splits (60/20/20 stratified)")
    splits = prepare_splits()
    print(f"  Train: {len(splits.y_train)} | Val: {len(splits.y_val)} | Test: {len(splits.y_test)}")
    print(f"  Train churn rate: {splits.y_train.mean():.3f}")

    print_header("Step 3: Train 3 Models (baseline + 2 substantive)")
    print("  - Logistic Regression (baseline)")
    print("  - Random Forest")
    print("  - Hist Gradient Boosting")
    print("  Hyperparameter tuning: RandomizedSearchCV, 5-fold CV on train only")

    trained = train_all_models(splits)
    best_name = select_best_model(trained)
    save_training_artifacts(splits, trained, best_name)
    print(f"\n  Best model on validation F1: {best_name}")

    # --- Test evaluation (all models, untouched test) ---
    print_header("Step 4: Test Set Evaluation (all models)")
    test_df = evaluate_all_on_test(trained, splits)
    print(test_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Refit best on train+val, final test ---
    print_header("Step 5: Refit Best Model on Train+Val → Final Test")
    final_model = refit_on_train_val(trained, splits, best_name)
    final_eval = evaluate_final_model(final_model, splits, model_name=best_name)
    plot_confusion_matrix(
        final_eval["confusion_matrix"],
        title=f"Confusion Matrix — {best_name} (Test)",
        filename="confusion_matrix_final.png",
    )

    # --- Diagnostics ---
    print_header("Step 6: Error Analysis & Figures")
    subgroup_error_analysis(final_model, splits)
    plot_model_comparison(test_df)
    plot_roc_pr_curves(trained, splits, final_model=final_model, final_name=f"{best_name}_refit")

    for name, tm in trained.items():
        from src.evaluate import get_predictions
        from sklearn.metrics import confusion_matrix

        y_pred, _ = get_predictions(tm.pipeline, splits.X_test)
        cm = confusion_matrix(splits.y_test, y_pred)
        plot_confusion_matrix(
            cm,
            title=f"Confusion Matrix — {name} (Test)",
            filename=f"confusion_matrix_{name}.png",
        )

    # --- Explainability: confusion matrices, feature selection, SHAP ---
    print_header("Step 7: Feature Selection & Explainable AI (XAI)")
    run_full_explainability(trained, splits, best_name, final_model)

    # --- Rubric reporting: CV uncertainty, PR-AUC, report sections, audit ---
    print_header("Step 8: Rubric Reporting & Submission Audit")
    run_full_reporting(trained, splits, best_name, final_model, final_eval, test_df)

    # --- Summary ---
    print_header("Final Test Metrics (refit best model)")
    for k, v in final_eval["metrics"].items():
        print(f"  {k}: {v:.4f}")

    summary = {
        "best_model": best_name,
        "test_metrics_all_models": test_df.to_dict(orient="records"),
        "final_test_metrics": final_eval["metrics"],
        "outputs": {
            "figures": str(FIGURES_DIR),
            "results": str(RESULTS_DIR),
        },
    }
    with open(RESULTS_DIR / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_header("Done")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Summary: {RESULTS_DIR / 'run_summary.json'}")


if __name__ == "__main__":
    main()
