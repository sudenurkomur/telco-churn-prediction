"""
Feature selection, confusion-matrix visuals, and explainable AI (SHAP, permutation).

All analysis uses train data for fitting selectors/explainers and test for final plots
where appropriate to avoid leakage.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

from .config import FIGURES_DIR, RANDOM_STATE, RESULTS_DIR
from .evaluate import get_predictions

sns.set_theme(style="whitegrid", palette="muted")

TOP_K_FEATURES = 25
SHAP_SAMPLE_SIZE = 400


def _shap_values_for_positive_class(shap_values) -> np.ndarray:
    """Normalize SHAP output to (n_samples, n_features) for the positive class."""
    if isinstance(shap_values, list):
        # Binary classifiers: [class0, class1]
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    return shap_values


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_preprocessor(pipeline):
    """Return the fitted preprocessing step from a sklearn Pipeline."""
    return pipeline.named_steps["preprocess"]


def get_feature_names(pipeline) -> list[str]:
    """Human-readable feature names after ColumnTransformer."""
    preprocessor = get_preprocessor(pipeline)
    return list(preprocessor.get_feature_names_out())


def transform_features(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply fitted preprocessor to raw features."""
    return get_preprocessor(pipeline).transform(X)


def plot_all_confusion_matrices(
    trained: dict,
    splits,
    final_model=None,
    final_name: str = "final_refit",
) -> None:
    """
  Plot confusion matrices for all models (counts + row-normalized %) in one figure.
    """
    _ensure_dirs()
    models_to_plot = [(name, tm.pipeline) for name, tm in trained.items()]
    if final_model is not None:
        models_to_plot.append((final_name, final_model))

    n = len(models_to_plot)
    fig, axes = plt.subplots(n, 2, figsize=(11, 4 * n))
    if n == 1:
        axes = np.array([axes])

    y_test = np.asarray(splits.y_test)
    labels = ["No Churn", "Churn"]

    for row, (name, model) in enumerate(models_to_plot):
        y_pred, _ = get_predictions(model, splits.X_test)
        cm = confusion_matrix(y_test, y_pred)

        # Counts
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[row, 0],
            cbar_kws={"label": "Count"},
        )
        axes[row, 0].set_xlabel("Predicted")
        axes[row, 0].set_ylabel("Actual")
        axes[row, 0].set_title(f"{name.replace('_', ' ').title()} — Counts")

        # Row-normalized (recall per true class)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".1%",
            cmap="Oranges",
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[row, 1],
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Proportion"},
        )
        axes[row, 1].set_xlabel("Predicted")
        axes[row, 1].set_ylabel("Actual")
        axes[row, 1].set_title(f"{name.replace('_', ' ').title()} — Row %")

    fig.suptitle("Confusion Matrices (Test Set)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices_all_models.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Single large matrix for final best model
    if final_model is not None:
        _plot_single_confusion_detailed(
            final_model,
            splits.X_test,
            y_test,
            title=f"Best Model ({final_name}) — Test Set",
            filename="confusion_matrix_best_detailed.png",
        )


def _plot_single_confusion_detailed(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    title: str,
    filename: str,
) -> None:
    """Detailed confusion matrix with counts, percentages, and metrics annotation."""
    y_pred, _ = get_predictions(model, X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = ["No Churn", "Churn"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
    )
    axes[0].set_title("Counts")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")

    cm_pct = cm / cm.sum() * 100
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Purples",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
    )
    axes[1].set_title("% of All Predictions")
    axes[1].set_ylabel("Actual")
    axes[1].set_xlabel("Predicted")

    tn, fp, fn, tp = cm.ravel()
    text = (
        f"TN={tn}  FP={fp}\nFN={fn}  TP={tp}\n"
        f"Recall (Churn)={tp/(tp+fn):.2%}\n"
        f"Precision (Churn)={tp/(tp+fp):.2%}"
    )
    fig.text(0.5, -0.02, text, ha="center", fontsize=10, family="monospace")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_mutual_info_feature_selection(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = TOP_K_FEATURES,
) -> pd.DataFrame:
    """
    Rank preprocessed features with mutual information (train only).

    This is filter-based feature selection for interpretation — not retraining
    the main models unless separately requested.
    """
    _ensure_dirs()
    feature_names = get_feature_names(pipeline)
    X_tr = transform_features(pipeline, X_train)

    mi_scores = mutual_info_classif(X_tr, y_train, random_state=RANDOM_STATE)
    df = pd.DataFrame(
        {"feature": feature_names, "mutual_info": mi_scores}
    ).sort_values("mutual_info", ascending=False)

    df.to_csv(RESULTS_DIR / "feature_selection_mutual_info.csv", index=False)

    # SelectKBest for top-k list (filter-based selection)
    k = min(top_k, X_tr.shape[1])
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X_tr, y_train)
    selected_names = [
        feature_names[i] for i, keep in enumerate(selector.get_support()) if keep
    ]
    selected = df[df["feature"].isin(selected_names)]
    selected.to_csv(RESULTS_DIR / f"feature_selection_top{top_k}.csv", index=False)

    # Plot top features
    top = df.head(top_k)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("viridis", len(top))
    ax.barh(range(len(top)), top["mutual_info"].values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].str.replace("__", " | "), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mutual Information")
    ax.set_title(f"Top {top_k} Features (Mutual Information — Train Set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_selection_mutual_info.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return df


def plot_permutation_importance(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
    top_k: int = 20,
) -> pd.DataFrame:
    """Permutation importance on validation set (model-agnostic XAI)."""
    _ensure_dirs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=15,
            random_state=RANDOM_STATE,
            scoring="f1",
            n_jobs=-1,
        )

    # Raw feature importances (aggregated by original column prefix)
    raw_names = X_val.columns.tolist()
    imp_raw = pd.DataFrame(
        {"feature": raw_names, "importance_mean": result.importances_mean}
    ).sort_values("importance_mean", ascending=False)
    imp_raw.to_csv(
        RESULTS_DIR / f"permutation_importance_raw_{model_name}.csv", index=False
    )

    top = imp_raw.head(top_k)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=top, y="feature", x="importance_mean", hue="feature", legend=False, ax=ax)
    ax.set_title(f"Permutation Importance (F1 drop) — {model_name}\nValidation Set")
    ax.set_xlabel("Mean importance")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / f"permutation_importance_{model_name}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    return imp_raw


def plot_preprocessed_permutation_importance(
    pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
    top_k: int = 20,
) -> pd.DataFrame:
    """Permutation on preprocessed feature space (after one-hot)."""
    _ensure_dirs()
    preprocessor = get_preprocessor(pipeline)
    classifier = pipeline.named_steps["model"]

    X_val_t = preprocessor.transform(X_val)
    feature_names = get_feature_names(pipeline)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            classifier,
            X_val_t,
            y_val,
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="f1",
            n_jobs=-1,
        )

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    df.to_csv(
        RESULTS_DIR / f"permutation_importance_encoded_{model_name}.csv", index=False
    )

    top = df.head(top_k)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        range(len(top)),
        top["importance_mean"],
        xerr=top["importance_std"],
        color=sns.color_palette("mako", len(top)),
    )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].str.replace("__", " | "), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation importance (F1 decrease)")
    ax.set_title(f"Encoded Feature Importance — {model_name}")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / f"permutation_importance_encoded_{model_name}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    return df


def plot_shap_analysis(
    pipeline,
    model_name: str,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    sample_size: int = SHAP_SAMPLE_SIZE,
) -> None:
    """SHAP summary and bar plots for tree or linear models."""
    try:
        import shap
    except ImportError:
        print("  [WARN] shap not installed — skip SHAP plots. pip install shap")
        return

    _ensure_dirs()
    preprocessor = get_preprocessor(pipeline)
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline)

    rng = np.random.RandomState(RANDOM_STATE)
    n_bg = min(200, len(X_background))
    n_ex = min(sample_size, len(X_explain))
    bg_idx = rng.choice(len(X_background), n_bg, replace=False)
    ex_idx = rng.choice(len(X_explain), n_ex, replace=False)

    X_bg_t = preprocessor.transform(X_background.iloc[bg_idx])
    X_ex_t = preprocessor.transform(X_explain.iloc[ex_idx])

    # Choose explainer by model type
    model_class = type(model).__name__
    if "Forest" in model_class or "GradientBoosting" in model_class or "HistGradient" in model_class:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_ex_t)
    elif "LogisticRegression" in model_class:
        explainer = shap.LinearExplainer(model, X_bg_t, feature_names=feature_names)
        shap_values = explainer.shap_values(X_ex_t)
    else:
        explainer = shap.Explainer(model.predict, X_bg_t)
        shap_values = explainer(X_ex_t).values

    shap_values = _shap_values_for_positive_class(shap_values)

    X_ex_df = pd.DataFrame(X_ex_t, columns=feature_names)

    # Summary beeswarm
    plt.figure(figsize=(11, 8))
    shap.summary_plot(
        shap_values,
        X_ex_df,
        show=False,
        max_display=20,
    )
    plt.title(f"SHAP Summary — {model_name.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / f"shap_summary_{model_name}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Bar plot (mean |SHAP|)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_ex_df,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.title(f"SHAP Feature Importance — {model_name.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / f"shap_bar_{model_name}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Save mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0).ravel()
    n_feat = min(len(feature_names), len(mean_shap))
    shap_df = pd.DataFrame(
        {
            "feature": feature_names[:n_feat],
            "mean_abs_shap": mean_shap[:n_feat],
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(RESULTS_DIR / f"shap_importance_{model_name}.csv", index=False)


def plot_rf_feature_importance(pipeline, model_name: str, top_k: int = 20) -> None:
    """Native feature importances from Random Forest on encoded features."""
    _ensure_dirs()
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return

    feature_names = get_feature_names(pipeline)
    imp = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values(
        "importance", ascending=False
    )
    df.to_csv(RESULTS_DIR / f"model_feature_importance_{model_name}.csv", index=False)

    top = df.head(top_k)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        range(len(top)),
        top["importance"],
        color=sns.color_palette("flare", len(top)),
    )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].str.replace("__", " | "), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Random Forest Feature Importance — Encoded Features")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / f"rf_feature_importance_{model_name}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def run_full_explainability(
    trained: dict,
    splits,
    best_name: str,
    final_model,
) -> None:
    """Run feature selection, confusion matrices, permutation & SHAP for report."""
    _ensure_dirs()
    best_tm = trained[best_name]
    best_pipe = best_tm.pipeline

    print("  → Confusion matrices (all models + detailed best)")
    plot_all_confusion_matrices(trained, splits, final_model=final_model, final_name=best_name)

    print("  → Feature selection (mutual information, train set)")
    run_mutual_info_feature_selection(best_pipe, splits.X_train, splits.y_train)

    print("  → Permutation importance (validation set)")
    plot_permutation_importance(best_pipe, splits.X_val, splits.y_val, best_name)
    plot_preprocessed_permutation_importance(
        best_pipe, splits.X_val, splits.y_val, best_name
    )

    if best_name == "random_forest":
        print("  → Random Forest Gini importances")
        plot_rf_feature_importance(best_pipe, best_name)

    # SHAP on best + tree model (RF) for richer XAI
    print("  → SHAP explainability")
    X_train_val = pd.concat([splits.X_train, splits.X_val])
    plot_shap_analysis(best_pipe, best_name, X_train_val, splits.X_test)

    if "random_forest" in trained and best_name != "random_forest":
        plot_shap_analysis(
            trained["random_forest"].pipeline,
            "random_forest",
            X_train_val,
            splits.X_test,
        )

    summary = {
        "feature_selection": str(RESULTS_DIR / "feature_selection_mutual_info.csv"),
        "figures": [
            "confusion_matrices_all_models.png",
            "confusion_matrix_best_detailed.png",
            "feature_selection_mutual_info.png",
            f"permutation_importance_{best_name}.png",
            f"permutation_importance_encoded_{best_name}.png",
            f"shap_summary_{best_name}.png",
            f"shap_bar_{best_name}.png",
        ],
    }
    with open(RESULTS_DIR / "explainability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
