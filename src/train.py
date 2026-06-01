"""Model training with nested CV on train split and validation-based selection."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import (
    CV_FOLDS,
    N_ITER_RANDOM_SEARCH,
    RANDOM_STATE,
    RESULTS_DIR,
    TRAIN_SIZE,
    TEST_SIZE,
    VAL_SIZE,
)
from .models import build_model_pipelines, get_param_distributions
from .preprocess import (
    clean_and_engineer,
    get_feature_target,
    load_raw_data,
    stratified_train_val_test_split,
)


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    best_params: dict
    cv_best_score: float
    val_metrics: dict


def prepare_splits() -> SplitData:
    """Load data, engineer features, and create stratified splits."""
    raw = load_raw_data()
    cleaned = clean_and_engineer(raw)
    X, y = get_feature_target(cleaned)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
        X,
        y,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    return SplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def train_all_models(splits: SplitData) -> dict[str, TrainedModel]:
    """
    Train three models with RandomizedSearchCV on the training set only.

    Inner stratified k-fold CV is used for hyperparameter tuning (no leakage
    from val/test). Validation metrics are computed afterward for comparison.
    """
    from .evaluate import compute_metrics

    pipelines = build_model_pipelines()
    param_dists = get_param_distributions()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    trained: dict[str, TrainedModel] = {}

    for name, pipeline in pipelines.items():
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dists[name],
            n_iter=N_ITER_RANDOM_SEARCH,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
            verbose=1,
        )
        search.fit(splits.X_train, splits.y_train)

        best_pipe = search.best_estimator_
        val_metrics = compute_metrics(splits.y_val, best_pipe, splits.X_val)

        trained[name] = TrainedModel(
            name=name,
            pipeline=best_pipe,
            best_params=search.best_params_,
            cv_best_score=float(search.best_score_),
            val_metrics=val_metrics,
        )

    return trained


def select_best_model(trained: dict[str, TrainedModel]) -> str:
    """Pick model with highest validation F1."""
    return max(trained, key=lambda k: trained[k].val_metrics["f1"])


def refit_on_train_val(
    trained: dict[str, TrainedModel],
    splits: SplitData,
    best_name: str,
) -> Pipeline:
    """Refit the best tuned pipeline on combined train+val before final test eval."""
    X_combined = pd.concat([splits.X_train, splits.X_val], axis=0)
    y_combined = pd.concat([splits.y_train, splits.y_val], axis=0)

    best = trained[best_name]
    final_pipe = clone(best.pipeline)
    final_pipe.fit(X_combined, y_combined)
    return final_pipe


def save_training_artifacts(
    splits: SplitData,
    trained: dict[str, TrainedModel],
    best_name: str,
) -> None:
    """Persist split sizes, best params, and validation comparison table."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    split_info = {
        "train_size": len(splits.y_train),
        "val_size": len(splits.y_val),
        "test_size": len(splits.y_test),
        "train_churn_rate": float(splits.y_train.mean()),
        "val_churn_rate": float(splits.y_val.mean()),
        "test_churn_rate": float(splits.y_test.mean()),
    }

    val_comparison = []
    for name, tm in trained.items():
        row = {"model": name, "cv_best_f1": tm.cv_best_score, **tm.val_metrics}
        val_comparison.append(row)

    with open(RESULTS_DIR / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    pd.DataFrame(val_comparison).to_csv(RESULTS_DIR / "validation_metrics.csv", index=False)

    best_params_all = {name: tm.best_params for name, tm in trained.items()}
    with open(RESULTS_DIR / "best_hyperparameters.json", "w") as f:
        json.dump(best_params_all, f, indent=2)

    with open(RESULTS_DIR / "selected_best_model.txt", "w") as f:
        f.write(best_name)
