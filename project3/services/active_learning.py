"""
Pool-based Active Learning for Expert Competence Discovery.

The classifier has access to all AG News training labels, as required by the
project. Expert predictions, however, are initially unknown.

The active learner selectively queries expert responses and trains a
competence model that estimates:

    P(expert is correct | article, classifier information)

Four query strategies are compared:

1. Random Sampling
2. Classifier Entropy
3. Expert-Competence Uncertainty
4. Hybrid Competence-Uncertainty + Diversity

The official AG News test set is used only for final evaluation.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .baseline import build_baseline_pipeline
from .data_loader import load_ag_news
from .learning_to_defer import (
    calculate_confidence,
    calculate_entropy,
    calculate_margin,
    calculate_deferral_metrics,
)
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

ACTIVE_LEARNING_METRICS_DIR = (
    PROJECT3_DIR / "artifacts" / "metrics"
)

ACTIVE_LEARNING_METRICS_PATH = (
    ACTIVE_LEARNING_METRICS_DIR
    / "active_learning_metrics.json"
)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

# Initial expert queries needed before a competence model can be trained.
INITIAL_QUERY_SIZE = 100

# Additional queries performed in each active-learning round.
BATCH_SIZE = 100

# Total expert budgets at which performance is recorded.
QUERY_BUDGETS = [
    100,
    200,
    400,
    600,
    800,
    1000,
]

STRATEGIES = {
    "random": "Random Sampling",
    "classifier_entropy": "Classifier Entropy",
    "competence_uncertainty": "Competence Uncertainty",
    "hybrid": "Hybrid Uncertainty + Diversity",
}


NUMERIC_FEATURES = [
    "classifier_confidence",
    "classifier_entropy",
    "classifier_margin",
    "log_text_length",
]

CATEGORICAL_FEATURES = [
    "predicted_class",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_competence_features(
    texts: Iterable[str],
    classifier_predictions: Iterable[int],
    classifier_probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Build features available without querying the expert.

    Expert predictions and configured expert reliability are deliberately
    excluded. Otherwise the active learner would already know the competence
    profile it is supposed to discover.
    """

    text_list = list(texts)

    classifier_predictions = np.asarray(
        list(classifier_predictions),
        dtype=int,
    )

    if len(text_list) != len(classifier_predictions):
        raise ValueError(
            "Texts and classifier predictions must have equal length."
        )

    if len(text_list) != len(classifier_probabilities):
        raise ValueError(
            "Texts and probability rows must have equal length."
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

    text_length = np.asarray(
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
                text_length
            ),
            "predicted_class": [
                CLASS_NAMES[int(label)]
                for label in classifier_predictions
            ],
        }
    )


# ---------------------------------------------------------------------------
# Expert oracle
# ---------------------------------------------------------------------------

def query_expert(
    texts: Iterable[str],
    true_labels: Iterable[int],
    profile: ExpertProfile,
    seed_offset: int,
) -> list[ExpertPrediction]:
    """
    Query the simulated expert.

    In the experiment, this function represents the oracle. The active
    learner may call it only for selected indices.
    """

    queried_profile = replace(
        profile,
        random_state=(
            profile.random_state
            + seed_offset
        ),
    )

    return simulate_expert_predictions(
        texts=texts,
        true_labels=true_labels,
        profile=queried_profile,
    )


def build_expert_correctness_target(
    true_labels: Iterable[int],
    expert_outputs: list[ExpertPrediction],
) -> np.ndarray:
    """
    Target for competence discovery.

    1 = expert prediction is correct
    0 = expert prediction is incorrect
    """

    true_labels = np.asarray(
        list(true_labels),
        dtype=int,
    )

    expert_predictions = np.asarray(
        [
            output.prediction
            for output in expert_outputs
        ],
        dtype=int,
    )

    return (
        expert_predictions
        == true_labels
    ).astype(int)


# ---------------------------------------------------------------------------
# Competence model
# ---------------------------------------------------------------------------

def build_competence_model() -> Pipeline:
    """
    Binary model estimating the probability that the expert is correct.
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


def fit_competence_model(
    features: pd.DataFrame,
    targets: np.ndarray,
    queried_indices: list[int],
) -> Pipeline:
    """
    Train a competence model using only queried expert examples.
    """

    queried_targets = targets[
        queried_indices
    ]

    if len(np.unique(queried_targets)) < 2:
        raise ValueError(
            "Queried expert labels contain only one correctness class. "
            "Increase the initial query size."
        )

    model = build_competence_model()

    model.fit(
        features.iloc[
            queried_indices
        ],
        queried_targets,
    )

    return model
