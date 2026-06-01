"""Metrics, test evaluation, and subgroup error analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from .config import FIGURES_DIR, RESULTS_DIR


def get_predictions(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pred, y_proba_positive) for a fitted classifier pipeline."""
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = model.decision_function(X)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-12)
    return y_pred, y_proba


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    model,
    X: pd.DataFrame,
) -> dict[str, float]:
    """Compute classification metrics appropriate for imbalanced churn data."""
    y_pred, y_proba = get_predictions(model, X)
    y_true = np.asarray(y_true)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def evaluate_all_on_test(
    trained: dict,
    splits,
) -> pd.DataFrame:
    """Evaluate each tuned model on the held-out test set."""
    rows = []
    for name, tm in trained.items():
        metrics = compute_metrics(splits.y_test, tm.pipeline, splits.X_test)
        rows.append({"model": name, **metrics})
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "test_metrics_all_models.csv", index=False)
    return df


def evaluate_final_model(
    model,
    splits,
    model_name: str = "final_best",
) -> dict:
    """Full test evaluation for the refit best model including classification report."""
    y_test = splits.y_test
    X_test = splits.X_test
    y_pred, y_proba = get_predictions(model, X_test)

    metrics = compute_metrics(y_test, model, X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        output_dict=True,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "test_metrics_final.json", "w") as f:
        json.dump({"model": model_name, **metrics, "classification_report": report}, f, indent=2)

    pd.DataFrame(cm, index=["true_no", "true_yes"], columns=["pred_no", "pred_yes"]).to_csv(
        RESULTS_DIR / "confusion_matrix_final.csv"
    )

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classification_report": report,
    }


def subgroup_error_analysis(
    model,
    splits,
) -> pd.DataFrame:
    """
    Error rates by contract type and tenure bin on the test set.

    Helps satisfy the assignment requirement for analysis beyond a single metric.
    """
    X_test = splits.X_test.copy()
    y_test = np.asarray(splits.y_test)
    y_pred, _ = get_predictions(model, X_test)

    analysis = X_test[["Contract", "tenure"]].copy()
    analysis["y_true"] = y_test
    analysis["y_pred"] = y_pred
    analysis["correct"] = analysis["y_true"] == analysis["y_pred"]
    analysis["tenure_bin"] = pd.cut(
        analysis["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    )

    rows = []
    for col in ["Contract", "tenure_bin"]:
        for group, grp in analysis.groupby(col, observed=True):
            n = len(grp)
            churn_rate = grp["y_true"].mean()
            error_rate = 1.0 - grp["correct"].mean()
            false_churn = ((grp["y_true"] == 0) & (grp["y_pred"] == 1)).sum()
            missed_churn = ((grp["y_true"] == 1) & (grp["y_pred"] == 0)).sum()
            rows.append(
                {
                    "group_type": col,
                    "group": str(group),
                    "n": n,
                    "churn_rate": round(churn_rate, 4),
                    "error_rate": round(error_rate, 4),
                    "false_churn": int(false_churn),
                    "missed_churn": int(missed_churn),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "subgroup_error_analysis.csv", index=False)
    return df


def plot_roc_pr_curves(
    trained: dict,
    splits,
    final_model=None,
    final_name: str = "final_best",
) -> None:
    """Save ROC and PR curve plots for all models on the test set."""
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    y_test = splits.y_test

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, tm in trained.items():
        _, y_proba = get_predictions(tm.pipeline, splits.X_test)
        RocCurveDisplay.from_predictions(y_test, y_proba, name=name.replace("_", " "), ax=axes[0])
        PrecisionRecallDisplay.from_predictions(
            y_test, y_proba, name=name.replace("_", " "), ax=axes[1]
        )

    if final_model is not None:
        _, y_proba = get_predictions(final_model, splits.X_test)
        RocCurveDisplay.from_predictions(
            y_test, y_proba, name=final_name.replace("_", " "), ax=axes[0], linestyle="--"
        )
        PrecisionRecallDisplay.from_predictions(
            y_test, y_proba, name=final_name.replace("_", " "), ax=axes[1], linestyle="--"
        )

    axes[0].set_title("ROC Curves (Test Set)")
    axes[1].set_title("Precision-Recall Curves (Test Set)")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
