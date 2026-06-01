"""Project configuration and reproducibility settings."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"
REPORT_SECTIONS_DIR = OUTPUT_DIR / "report_sections"

# Reproducibility
RANDOM_STATE = 42

# Stratified split ratios (train / val / test)
TRAIN_SIZE = 0.60
VAL_SIZE = 0.20
TEST_SIZE = 0.20

# Target and columns to drop from features
TARGET_COL = "Churn"
ID_COL = "customerID"
POSITIVE_CLASS = "Yes"

# Raw categorical columns (excluding target)
CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERIC_COLS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# Engineered feature names (created in preprocess)
ENGINEERED_COLS = [
    "avg_monthly_spend",
    "has_auto_payment",
    "num_optional_services",
]

# Hyperparameter search
N_ITER_RANDOM_SEARCH = 25
CV_FOLDS = 5  # inner CV on training split only
