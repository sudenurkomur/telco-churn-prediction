"""Data loading, cleaning, feature engineering, and sklearn preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_COLS,
    DATA_PATH,
    ENGINEERED_COLS,
    ID_COL,
    NUMERIC_COLS,
    TARGET_COL,
)

SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

AUTO_PAYMENT_KEYWORDS = ("automatic",)


def load_raw_data(path=None) -> pd.DataFrame:
    """Load the Telco churn CSV from disk."""
    path = path or DATA_PATH
    return pd.read_csv(path)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean TotalCharges, engineer features, and return a modeling-ready frame.

    TotalCharges is empty for new customers (tenure=0); we impute with 0.0
    because they have not yet accumulated billed charges.
    """
    data = df.copy()

    # Drop duplicate customer IDs if any
    if ID_COL in data.columns:
        data = data.drop_duplicates(subset=[ID_COL])

    # Fix TotalCharges: coerce to float, empty strings -> NaN -> 0 for tenure==0
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    new_customer_mask = data["tenure"] == 0
    data.loc[new_customer_mask & data["TotalCharges"].isna(), "TotalCharges"] = 0.0
    # Remaining NaNs (if any) filled with median on full frame before split is leakage;
    # we only fix tenure==0 here; pipeline imputer handles any other NaNs on train only.
    data["TotalCharges"] = data["TotalCharges"].fillna(0.0)

    # Average monthly spend proxy (guards divide-by-zero)
    data["avg_monthly_spend"] = data["TotalCharges"] / data["tenure"].replace(0, np.nan)
    data["avg_monthly_spend"] = data["avg_monthly_spend"].fillna(data["MonthlyCharges"])

    # Automatic payment indicator
    data["has_auto_payment"] = (
        data["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)
    )

    # Count of subscribed optional internet services
    def _is_subscribed(value: str) -> int:
        return int(value == "Yes")

    data["num_optional_services"] = data[SERVICE_COLS].apply(
        lambda row: sum(_is_subscribed(v) for v in row), axis=1
    )

    return data


def get_feature_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split features X and binary target y (1 = churn)."""
    data = df.drop(columns=[ID_COL], errors="ignore")
    y = (data[TARGET_COL] == "Yes").astype(int)
    X = data.drop(columns=[TARGET_COL])
    return X, y


def get_feature_column_groups() -> tuple[list[str], list[str]]:
    """Return numeric and categorical column names used by the preprocessor."""
    numeric = NUMERIC_COLS + ENGINEERED_COLS
    categorical = CATEGORICAL_COLS
    return numeric, categorical


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """
    Build a ColumnTransformer for numeric and categorical features.

    When scale_numeric=True, numeric features are imputed and standardized
  (needed for logistic regression). Tree models are scale-invariant but
    sharing the same preprocessor keeps comparisons fair and leakage-safe.
    """
    numeric_cols, categorical_cols = get_feature_column_groups()

    if scale_numeric:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def stratified_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
):
    """Two-stage stratified split into train, validation, and test sets."""
    from sklearn.model_selection import train_test_split

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1.0 - train_size),
        stratify=y,
        random_state=random_state,
    )

    relative_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
