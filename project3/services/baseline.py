"""
Baseline classifier for AG News.

The classifier is trained on the complete official training split and
evaluated on the untouched official test split.
"""

from pathlib import Path
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data_loader import load_ag_news
from .evaluation import evaluate_predictions


PROJECT3_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT3_DIR / "artifacts" / "models"
METRICS_DIR = PROJECT3_DIR / "artifacts" / "metrics"

MODEL_PATH = MODEL_DIR / "baseline_model.joblib"
METRICS_PATH = METRICS_DIR / "baseline_metrics.json"


def build_baseline_pipeline() -> Pipeline:
    """
    Build the TF-IDF and Logistic Regression pipeline.
    """

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=4.0,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )
