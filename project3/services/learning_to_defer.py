"""
Learning-to-defer strategies for AG News.

Two strategies are evaluated:

1. Confidence-threshold deferral
   The classifier defers when its confidence is below a threshold.

2. Learned competence-aware deferral
   A binary model estimates whether querying the selected expert is likely
   to be beneficial for the current article.

The official AG News test set is used only for final evaluation. All
threshold selection and deferral-model training use the official training
split.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .baseline import build_baseline_pipeline
from .data_loader import load_ag_news
from .evaluation import evaluate_predictions
from .simulated_expert import (
    CLASS_NAMES,
    EXPERT_PROFILES,
    ExpertPrediction,
    ExpertProfile,
    simulate_expert_predictions,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT3_DIR = Path(__file__).resolve().parent.parent

DEFER_METRICS_DIR = PROJECT3_DIR / "artifacts" / "metrics"

DEFER_METRICS_PATH = (
    DEFER_METRICS_DIR / "learning_to_defer_metrics.json"
)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

META_POOL_SIZE = 0.30
META_VALIDATION_SIZE = 0.50


CONFIDENCE_THRESHOLDS = [
    round(value, 2)
    for value in np.arange(0.40, 0.96, 0.05)
]


LEARNED_THRESHOLDS = [
    round(value, 2)
    for value in np.arange(0.05, 0.96, 0.05)
]


NUMERIC_FEATURES = [
    "classifier_confidence",
    "classifier_entropy",
    "classifier_margin",
    "log_text_length",
    "expected_expert_accuracy",
]


CATEGORICAL_FEATURES = [
    "predicted_class",
    "expert_region",
]


# ---------------------------------------------------------------------------
# Classifier uncertainty
# ---------------------------------------------------------------------------

def calculate_confidence(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Maximum class probability for each article.
    """

    return np.max(
        probabilities,
        axis=1,
    )


def calculate_entropy(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Normalized predictive entropy.

    Values are approximately between 0 and 1:
        0 -> highly confident distribution
        1 -> maximally uncertain distribution
    """

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    entropy = -np.sum(
        probabilities * np.log(probabilities),
        axis=1,
    )

    maximum_entropy = np.log(
        probabilities.shape[1]
    )

    return entropy / maximum_entropy


def calculate_margin(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Difference between the largest and second-largest class probabilities.

    A large margin indicates greater classifier confidence.
    """

    sorted_probabilities = np.sort(
        probabilities,
        axis=1,
    )

    return (
        sorted_probabilities[:, -1]
        - sorted_probabilities[:, -2]
    )


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def create_deferral_development_splits(
    training_data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Split the official training set into:

        classifier training   70%
        deferral training     15%
        deferral validation   15%

    The official test set is not touched.
    """

    classifier_train, meta_pool = train_test_split(
        training_data,
        test_size=META_POOL_SIZE,
        random_state=RANDOM_STATE,
        stratify=training_data["label"],
    )

    meta_train, meta_validation = train_test_split(
        meta_pool,
        test_size=META_VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=meta_pool["label"],
    )

    return (
        classifier_train.reset_index(drop=True),
        meta_train.reset_index(drop=True),
        meta_validation.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Expert simulation helpers
# ---------------------------------------------------------------------------

def simulate_expert_for_split(
    texts: Iterable[str],
    true_labels: Iterable[int],
    profile: ExpertProfile,
    seed_offset: int,
) -> list[ExpertPrediction]:
    """
    Simulate one expert using a deterministic split-specific seed.

    Different offsets prevent the same random sequence from being reused
    across development and test datasets.
    """

    split_profile = replace(
        profile,
        random_state=(
            profile.random_state
            + seed_offset
        ),
    )

    return simulate_expert_predictions(
        texts=texts,
        true_labels=true_labels,
        profile=split_profile,
    )


# ---------------------------------------------------------------------------
# Deferral feature engineering
# ---------------------------------------------------------------------------

def build_deferral_features(
    texts: Iterable[str],
    classifier_predictions: Iterable[int],
    classifier_probabilities: np.ndarray,
    expert_outputs: list[ExpertPrediction],
) -> pd.DataFrame:
    """
    Build interpretable features for the learned deferral model.

    Importantly, the actual expert prediction is NOT included because the
    system must decide whether to query the expert before receiving that
    prediction.
    """

    text_list = list(texts)

    predictions = np.asarray(
        list(classifier_predictions),
        dtype=int,
    )

    if len(text_list) != len(expert_outputs):
        raise ValueError(
            "Texts and expert outputs must have equal length."
        )

    if len(text_list) != len(predictions):
        raise ValueError(
            "Texts and classifier predictions must have equal length."
        )

    if len(text_list) != len(classifier_probabilities):
        raise ValueError(
            "Texts and classifier probabilities must have equal length."
        )

    confidence = calculate_confidence(
        classifier_probabilities
    )

    entropy = calculate_entropy(
        classifier_probabilities
    )

    margin = calculate_margin(
        classifier_probabilities
    )

    text_lengths = np.array(
        [
            len(str(text).split())
            for text in text_list
        ],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "classifier_confidence": confidence,
            "classifier_entropy": entropy,
            "classifier_margin": margin,
            "log_text_length": np.log1p(
                text_lengths
            ),
            "expected_expert_accuracy": [
                output.expected_accuracy
                for output in expert_outputs
            ],
            "predicted_class": [
                CLASS_NAMES[int(prediction)]
                for prediction in predictions
            ],
            "expert_region": [
                output.region
                for output in expert_outputs
            ],
        }
    )


def build_beneficial_deferral_target(
    true_labels: Iterable[int],
    classifier_predictions: Iterable[int],
    expert_outputs: list[ExpertPrediction],
) -> np.ndarray:
    """
    Build the target for the learned deferral model.

    Target = 1 only when:

        classifier is wrong
        AND
        expert is correct

    In this situation, deferring improves final classification accuracy.
    """

    true_labels = np.asarray(
        list(true_labels),
        dtype=int,
    )

    classifier_predictions = np.asarray(
        list(classifier_predictions),
        dtype=int,
    )

    expert_predictions = np.asarray(
        [
            output.prediction
            for output in expert_outputs
        ],
        dtype=int,
    )

    classifier_correct = (
        classifier_predictions == true_labels
    )

    expert_correct = (
        expert_predictions == true_labels
    )

    return (
        (~classifier_correct)
        & expert_correct
    ).astype(int)


# ---------------------------------------------------------------------------
# Learned deferral model
# ---------------------------------------------------------------------------

def build_competence_model() -> Pipeline:
    """
    Build the learned competence-aware deferral model.
    """

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="liblinear",
    )

    return Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------

def calculate_deferral_metrics(
    true_labels: Iterable[int],
    classifier_predictions: Iterable[int],
    expert_predictions: Iterable[int],
    defer_mask: Iterable[bool],
) -> dict:
    """
    Calculate final team performance and deferral-quality metrics.
    """

    y_true = np.asarray(
        list(true_labels),
        dtype=int,
    )

    classifier_predictions = np.asarray(
        list(classifier_predictions),
        dtype=int,
    )

    expert_predictions = np.asarray(
        list(expert_predictions),
        dtype=int,
    )

    defer_mask = np.asarray(
        list(defer_mask),
        dtype=bool,
    )

    sample_count = len(y_true)

    if not (
        len(classifier_predictions)
        == len(expert_predictions)
        == len(defer_mask)
        == sample_count
    ):
        raise ValueError(
            "All deferral evaluation arrays must have equal length."
        )

    classifier_correct = (
        classifier_predictions == y_true
    )

    expert_correct = (
        expert_predictions == y_true
    )

    team_predictions = np.where(
        defer_mask,
        expert_predictions,
        classifier_predictions,
    )

    team_metrics = evaluate_predictions(
        y_true,
        team_predictions,
    )

    beneficial_available = (
        (~classifier_correct)
        & expert_correct
    )

    harmful_possible = (
        classifier_correct
        & (~expert_correct)
    )

    both_correct = (
        classifier_correct
        & expert_correct
    )

    both_wrong = (
        (~classifier_correct)
        & (~expert_correct)
    )

    beneficial_deferrals = (
        defer_mask
        & beneficial_available
    )

    harmful_deferrals = (
        defer_mask
        & harmful_possible
    )

    unnecessary_deferrals = (
        defer_mask
        & both_correct
    )

    hopeless_deferrals = (
        defer_mask
        & both_wrong
    )

    missed_beneficial = (
        (~defer_mask)
        & beneficial_available
    )

    deferred_count = int(
        defer_mask.sum()
    )

    non_deferred_count = (
        sample_count - deferred_count
    )

    deferred_correct = (
        team_predictions[defer_mask]
        == y_true[defer_mask]
    )

    non_deferred_correct = (
        team_predictions[~defer_mask]
        == y_true[~defer_mask]
    )

    beneficial_available_count = int(
        beneficial_available.sum()
    )

    beneficial_deferral_count = int(
        beneficial_deferrals.sum()
    )

    harmful_deferral_count = int(
        harmful_deferrals.sum()
    )

    unnecessary_deferral_count = int(
        unnecessary_deferrals.sum()
    )

    hopeless_deferral_count = int(
        hopeless_deferrals.sum()
    )

    missed_beneficial_count = int(
        missed_beneficial.sum()
    )

    oracle_correct = (
        classifier_correct
        | expert_correct
    )

    return {
        "sample_count": sample_count,

        "classifier_accuracy": float(
            classifier_correct.mean()
        ),

        "expert_accuracy": float(
            expert_correct.mean()
        ),

        "team_accuracy": team_metrics[
            "accuracy"
        ],

        "team_macro_f1": team_metrics[
            "macro_f1"
        ],

        "team_weighted_f1": team_metrics[
            "weighted_f1"
        ],

        "oracle_accuracy": float(
            oracle_correct.mean()
        ),

        "deferred_count": deferred_count,

        "deferral_rate": (
            deferred_count / sample_count
            if sample_count
            else 0.0
        ),

        "deferred_accuracy": (
            float(deferred_correct.mean())
            if deferred_count > 0
            else None
        ),

        "non_deferred_accuracy": (
            float(non_deferred_correct.mean())
            if non_deferred_count > 0
            else None
        ),

        "beneficial_opportunities": (
            beneficial_available_count
        ),

        "beneficial_deferrals": (
            beneficial_deferral_count
        ),

        "beneficial_deferral_precision": (
            beneficial_deferral_count
            / deferred_count
            if deferred_count > 0
            else 0.0
        ),

        "beneficial_deferral_recall": (
            beneficial_deferral_count
            / beneficial_available_count
            if beneficial_available_count > 0
            else 0.0
        ),

        "harmful_deferrals": (
            harmful_deferral_count
        ),

        "harmful_deferral_rate": (
            harmful_deferral_count
            / deferred_count
            if deferred_count > 0
            else 0.0
        ),

        "unnecessary_deferrals": (
            unnecessary_deferral_count
        ),

        "unnecessary_deferral_rate": (
            unnecessary_deferral_count
            / deferred_count
            if deferred_count > 0
            else 0.0
        ),

        "hopeless_deferrals": (
            hopeless_deferral_count
        ),

        "missed_beneficial_deferrals": (
            missed_beneficial_count
        ),

        "missed_beneficial_rate": (
            missed_beneficial_count
            / beneficial_available_count
            if beneficial_available_count > 0
            else 0.0
        ),

        "classification_report": team_metrics[
            "classification_report"
        ],

        "confusion_matrix": team_metrics[
            "confusion_matrix"
        ],
    }


# ---------------------------------------------------------------------------
# Confidence-threshold policy
# ---------------------------------------------------------------------------

def select_confidence_threshold(
    true_labels: Iterable[int],
    classifier_predictions: np.ndarray,
    classifier_probabilities: np.ndarray,
    expert_predictions: np.ndarray,
) -> tuple[float, list[dict]]:
    """
    Select a confidence threshold using development validation data.

    Highest team accuracy wins. If two thresholds have the same accuracy,
    the threshold with the lower human deferral rate is preferred.
    """

    confidence = calculate_confidence(
        classifier_probabilities
    )

    search_results = []

    for threshold in CONFIDENCE_THRESHOLDS:
        defer_mask = (
            confidence < threshold
        )

        team_predictions = np.where(
            defer_mask,
            expert_predictions,
            classifier_predictions,
        )

        team_accuracy = float(
            accuracy_score(
                true_labels,
                team_predictions,
            )
        )

        deferral_rate = float(
            defer_mask.mean()
        )

        search_results.append(
            {
                "threshold": float(threshold),
                "team_accuracy": team_accuracy,
                "deferral_rate": deferral_rate,
            }
        )

    best_result = max(
        search_results,
        key=lambda result: (
            result["team_accuracy"],
            -result["deferral_rate"],
        ),
    )

    return (
        best_result["threshold"],
        search_results,
    )


# ---------------------------------------------------------------------------
# Learned-policy threshold selection
# ---------------------------------------------------------------------------

def select_learned_threshold(
    true_labels: Iterable[int],
    classifier_predictions: np.ndarray,
    expert_predictions: np.ndarray,
    benefit_probabilities: np.ndarray,
) -> tuple[float, list[dict]]:
    """
    Select the deferral probability threshold on validation data.
    """

    search_results = []

    for threshold in LEARNED_THRESHOLDS:
        defer_mask = (
            benefit_probabilities
            >= threshold
        )

        team_predictions = np.where(
            defer_mask,
            expert_predictions,
            classifier_predictions,
        )

        team_accuracy = float(
            accuracy_score(
                true_labels,
                team_predictions,
            )
        )

        deferral_rate = float(
            defer_mask.mean()
        )

        search_results.append(
            {
                "threshold": float(threshold),
                "team_accuracy": team_accuracy,
                "deferral_rate": deferral_rate,
            }
        )

    best_result = max(
        search_results,
        key=lambda result: (
            result["team_accuracy"],
            -result["deferral_rate"],
        ),
    )

    return (
        best_result["threshold"],
        search_results,
    )


