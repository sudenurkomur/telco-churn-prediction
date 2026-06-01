"""Model definitions and hyperparameter search spaces."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .preprocess import build_preprocessor


def build_model_pipelines() -> dict[str, Pipeline]:
    """
    Return named sklearn Pipelines: preprocessor + classifier.

    Three models per assignment:
      1. Logistic Regression (baseline)
      2. Random Forest (substantive)
      3. Histogram Gradient Boosting (substantive, gradient boosted trees family)
    """
    preprocessor = build_preprocessor(scale_numeric=True)

    pipelines = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(scale_numeric=False)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        random_state=42,
                        early_stopping=True,
                    ),
                ),
            ]
        ),
    }
    return pipelines


def get_param_distributions() -> dict[str, dict]:
    """Randomized search spaces per model (tuned on train via CV)."""
    return {
        "logistic_regression": {
            "model__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        },
        "random_forest": {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": ["sqrt", "log2"],
        },
        "hist_gradient_boosting": {
            "model__max_depth": [3, 5, 8, 12],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_iter": [200, 300, 400],
            "model__min_samples_leaf": [20, 50, 100],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        },
    }
