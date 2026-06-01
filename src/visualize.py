"""Exploratory data analysis and result visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, TARGET_COL
from .preprocess import clean_and_engineer, load_raw_data

sns.set_theme(style="whitegrid", palette="muted")


def _ensure_figures_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def plot_eda() -> None:
    """Generate exploratory plots saved to outputs/figures/."""
    _ensure_figures_dir()
    df = clean_and_engineer(load_raw_data())

    # 1. Churn distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    churn_counts = df[TARGET_COL].value_counts()
    churn_counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_title("Churn Class Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    for i, v in enumerate(churn_counts.values):
        ax.text(i, v + 50, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_churn_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Tenure by churn
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(
        data=df,
        x="tenure",
        hue=TARGET_COL,
        fill=True,
        common_norm=False,
        ax=ax,
    )
    ax.set_title("Tenure Distribution by Churn")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_tenure_by_churn.png", dpi=150)
    plt.close(fig)

    # 3. Churn rate by contract
    churn_rate = df.groupby("Contract")[TARGET_COL].apply(lambda s: (s == "Yes").mean())
    fig, ax = plt.subplots(figsize=(7, 4))
    churn_rate.sort_values(ascending=False).plot(kind="bar", ax=ax, color="#C44E52")
    ax.set_title("Churn Rate by Contract Type")
    ax.set_ylabel("Churn Rate")
    ax.set_ylim(0, 0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_churn_by_contract.png", dpi=150)
    plt.close(fig)

    # 4. Monthly charges by churn
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x=TARGET_COL, y="MonthlyCharges", ax=ax)
    ax.set_title("Monthly Charges vs Churn")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_monthly_charges.png", dpi=150)
    plt.close(fig)

    # 5. Correlation heatmap (numeric)
    numeric = df.select_dtypes(include=[np.number])
    if TARGET_COL in df.columns:
        numeric = numeric.copy()
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric Feature Correlations")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, title: str, filename: str) -> None:
    """Plot and save a confusion matrix heatmap."""
    _ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred No", "Pred Yes"],
        yticklabels=["True No", "True Yes"],
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)


def plot_model_comparison(test_metrics: pd.DataFrame) -> None:
    """Bar chart comparing key metrics across models on the test set."""
    _ensure_figures_dir()
    metrics = ["f1", "balanced_accuracy", "roc_auc", "pr_auc"]
    plot_df = test_metrics.melt(
        id_vars=["model"],
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="metric", y="score", hue="model", ax=ax)
    ax.set_title("Test Set Model Comparison")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison_test.png", dpi=150)
    plt.close(fig)
