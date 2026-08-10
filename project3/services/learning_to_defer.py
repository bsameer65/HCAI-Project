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
