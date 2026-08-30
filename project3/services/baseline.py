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

def train_and_evaluate_baseline() -> dict:
    """
    Train the baseline on the complete AG News training set and evaluate it
    on the official test set.
    """

    dataset = load_ag_news()

    pipeline = build_baseline_pipeline()

    pipeline.fit(
        dataset.train["text"],
        dataset.train["label"],
    )

    predictions = pipeline.predict(dataset.test["text"])

    metrics = evaluate_predictions(
        dataset.test["label"],
        predictions,
    )

    result = {
        "model_name": "TF-IDF with Logistic Regression",
        "train_samples": int(len(dataset.train)),
        "test_samples": int(len(dataset.test)),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "classification_report": metrics["classification_report"],
        "confusion_matrix": metrics["confusion_matrix"],
        "class_names": metrics["class_names"],
        "configuration": {
            "ngram_range": [1, 2],
            "max_features": 100_000,
            "minimum_document_frequency": 2,
            "regularization_c": 4.0,
            "random_state": 42,
        },
    }

    save_baseline_artifacts(pipeline, result)

    return result


def save_baseline_artifacts(
    pipeline: Pipeline,
    result: dict,
) -> None:
    """
    Save the trained model and evaluation results.
    """

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_baseline_results() -> dict | None:
    """
    Load previously generated baseline results.
    """

    if not METRICS_PATH.exists():
        return None

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)
    
def load_baseline_model() -> Pipeline | None:
    """
    Load the previously trained baseline classifier.

    Returns None when the baseline experiment has not been run yet.
    """

    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)