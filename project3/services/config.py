from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT3_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = PROJECT3_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_pipeline.joblib"
BASELINE_METRICS_PATH = METRICS_DIR / "baseline_metrics.json"
BASELINE_CONFUSION_MATRIX_PATH = FIGURES_DIR / "baseline_confusion_matrix.png"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATASET_NAME = "fancyzhx/ag_news"

CLASS_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

EXPECTED_COLUMNS = {"text", "label"}
EXPECTED_SPLITS = {"train", "test"}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

TFIDF_CONFIG = {
    "lowercase": True,
    "strip_accents": "unicode",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.98,
    "max_features": 100_000,
    "sublinear_tf": True,
}


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------

LOGISTIC_REGRESSION_CONFIG = {
    "C": 4.0,
    "solver": "lbfgs",
    "max_iter": 1_000,
    "random_state": RANDOM_STATE,
}