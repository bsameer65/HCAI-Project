"""
Baseline classifier for AG News.

The classifier is trained on the complete official training split and
evaluated on the untouched official test split.
"""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data_loader import load_ag_news
from .evaluation import evaluate_predictions


PROJECT3_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT3_DIR / "artifacts" / "models"
METRICS_DIR = PROJECT3_DIR / "artifacts" / "metrics"

FIGURE_DIR = (
    PROJECT3_DIR
    / "static"
    / "project3"
    / "figures"
    / "baseline"
)

MODEL_PATH = MODEL_DIR / "baseline_model.joblib"
METRICS_PATH = METRICS_DIR / "baseline_metrics.json"

PER_CLASS_F1_FIGURE_PATH = (
    FIGURE_DIR / "baseline_per_class_f1.png"
)

PER_CLASS_F1_STATIC_PATH = (
    "project3/figures/baseline/"
    "baseline_per_class_f1.png"
)


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


def _get_f1_score(class_metrics: dict) -> float:
    """
    Return the F1-score regardless of whether the stored key uses
    an underscore or a hyphen.
    """

    if "f1_score" in class_metrics:
        return float(class_metrics["f1_score"])

    if "f1-score" in class_metrics:
        return float(class_metrics["f1-score"])

    raise KeyError(
        "F1-score was not found in the classification report."
    )


def create_baseline_per_class_plot(result: dict) -> None:
    """
    Create a bar chart showing the percentage of test examples
    from each AG News class that were classified correctly.

    For a single-label multiclass classifier, this corresponds to
    the recall of each class.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_names = [
        "World",
        "Sports",
        "Business",
        "Sci/Tech",
    ]

    classification_report = result["classification_report"]

    class_accuracies = [
        float(
            classification_report[class_name]["recall"]
        ) * 100
        for class_name in class_names
    ]

    fig, ax = plt.subplots(
        figsize=(8.5, 4.8)
    )

    bars = ax.bar(
        class_names,
        class_accuracies,
        width=0.62,
        color="#2f6fa7",
    )

    ax.set_title(
        "Per-Class Classification Accuracy",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    ax.set_xlabel(
        "AG News Class",
        fontsize=10,
    )

    ax.set_ylabel(
        "Correctly Classified (%)",
        fontsize=10,
    )

    ax.set_ylim(
        80,
        100,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.25,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, accuracy in zip(
        bars,
        class_accuracies,
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 0.4,
            f"{accuracy:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()

    fig.savefig(
        PER_CLASS_F1_FIGURE_PATH,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def ensure_baseline_per_class_plot(
    result: dict,
) -> str:
    """
    Ensure that the baseline figure exists.

    The plot is generated from the saved metrics, so retraining is
    not required when only the figure is missing.
    """

    if not PER_CLASS_F1_FIGURE_PATH.exists():
        create_baseline_per_class_plot(
            result
        )

    return PER_CLASS_F1_STATIC_PATH


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

    predictions = pipeline.predict(
        dataset.test["text"]
    )

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
        "classification_report": metrics[
            "classification_report"
        ],
        "confusion_matrix": metrics[
            "confusion_matrix"
        ],
        "class_names": metrics["class_names"],
        "configuration": {
            "ngram_range": [1, 2],
            "max_features": 100_000,
            "minimum_document_frequency": 2,
            "regularization_c": 4.0,
            "random_state": 42,
        },
    }

    save_baseline_artifacts(
        pipeline,
        result,
    )

    # Recreate the figure using the latest experiment results.
    create_baseline_per_class_plot(
        result
    )

    return result


def save_baseline_artifacts(
    pipeline: Pipeline,
    result: dict,
) -> None:
    """
    Save the trained model and evaluation results.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
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

    with METRICS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_baseline_model() -> Pipeline | None:
    """
    Load the previously trained baseline classifier.

    Returns None when the baseline experiment has not been run yet.
    """

    if not MODEL_PATH.exists():
        return None

    return joblib.load(
        MODEL_PATH
    )