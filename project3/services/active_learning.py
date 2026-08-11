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


# ---------------------------------------------------------------------------
# Query utilities
# ---------------------------------------------------------------------------

def competence_uncertainty_score(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Binary uncertainty around P(expert correct)=0.5.

    Highest utility occurs at probability 0.5.
    """

    return 1.0 - (
        2.0
        * np.abs(
            probabilities - 0.5
        )
    )


def normalize_scores(
    values: np.ndarray,
) -> np.ndarray:
    """
    Scale numerical utilities to [0, 1].
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    minimum = values.min()
    maximum = values.max()

    if np.isclose(
        minimum,
        maximum,
    ):
        return np.zeros_like(values)

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


def calculate_diversity_score(
    pool_texts: list[str],
    queried_texts: list[str],
) -> np.ndarray:
    """
    Estimate diversity using TF-IDF distance from already queried samples.

    A candidate receives high diversity utility when it has low maximum
    cosine similarity with previously queried articles.
    """

    if not pool_texts:
        return np.asarray([])

    if not queried_texts:
        return np.ones(
            len(pool_texts),
            dtype=float,
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=5000,
        min_df=2,
        ngram_range=(1, 1),
    )

    all_texts = (
        queried_texts
        + pool_texts
    )

    matrix = vectorizer.fit_transform(
        all_texts
    )

    queried_matrix = matrix[
        :len(queried_texts)
    ]

    pool_matrix = matrix[
        len(queried_texts):
    ]

    similarities = (
        pool_matrix
        @ queried_matrix.T
    ).toarray()

    maximum_similarity = (
        similarities.max(axis=1)
    )

    diversity = (
        1.0 - maximum_similarity
    )

    return normalize_scores(
        diversity
    )


def select_query_indices(
    strategy_key: str,
    candidate_indices: np.ndarray,
    batch_size: int,
    classifier_entropy: np.ndarray,
    competence_probabilities: np.ndarray | None,
    texts: list[str],
    queried_indices: list[int],
    random_generator: random.Random,
) -> list[int]:
    """
    Select the next batch according to one query strategy.
    """

    if len(candidate_indices) <= batch_size:
        return candidate_indices.tolist()

    if strategy_key == "random":
        return random_generator.sample(
            candidate_indices.tolist(),
            batch_size,
        )

    if strategy_key == "classifier_entropy":
        utility = classifier_entropy[
            candidate_indices
        ]

    elif strategy_key == "competence_uncertainty":
        if competence_probabilities is None:
            raise ValueError(
                "Competence probabilities are required."
            )

        utility = competence_uncertainty_score(
            competence_probabilities[
                candidate_indices
            ]
        )

    elif strategy_key == "hybrid":
        if competence_probabilities is None:
            raise ValueError(
                "Competence probabilities are required."
            )

        uncertainty = (
            competence_uncertainty_score(
                competence_probabilities[
                    candidate_indices
                ]
            )
        )

        pool_texts = [
            texts[index]
            for index in candidate_indices
        ]

        queried_texts = [
            texts[index]
            for index in queried_indices
        ]

        diversity = calculate_diversity_score(
            pool_texts=pool_texts,
            queried_texts=queried_texts,
        )

        # Equal-weight combination of information and representation utility.
        utility = (
            0.5
            * normalize_scores(
                uncertainty
            )
            + 0.5
            * diversity
        )

    else:
        raise ValueError(
            f"Unknown active-learning strategy: {strategy_key}"
        )

    top_local_indices = np.argsort(
        utility
    )[-batch_size:]

    return candidate_indices[
        top_local_indices
    ].tolist()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_competence_predictions(
    true_correctness: np.ndarray,
    predicted_probabilities: np.ndarray,
) -> dict:
    """
    Evaluate how accurately expert competence has been learned.
    """

    predicted_labels = (
        predicted_probabilities
        >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        true_correctness,
        predicted_labels,
    )

    f1 = f1_score(
        true_correctness,
        predicted_labels,
        zero_division=0,
    )

    if len(
        np.unique(
            true_correctness
        )
    ) > 1:
        auroc = roc_auc_score(
            true_correctness,
            predicted_probabilities,
        )
    else:
        auroc = None

    brier = brier_score_loss(
        true_correctness,
        predicted_probabilities,
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "f1": float(
            f1
        ),
        "auroc": (
            float(auroc)
            if auroc is not None
            else None
        ),
        "brier_score": float(
            brier
        ),
    }


def evaluate_team_from_competence(
    true_labels: np.ndarray,
    classifier_predictions: np.ndarray,
    classifier_confidence: np.ndarray,
    expert_predictions: np.ndarray,
    estimated_expert_accuracy: np.ndarray,
) -> dict:
    """
    Defer when estimated expert competence exceeds classifier confidence.
    """

    defer_mask = (
        estimated_expert_accuracy
        > classifier_confidence
    )

    return calculate_deferral_metrics(
        true_labels=true_labels,
        classifier_predictions=(
            classifier_predictions
        ),
        expert_predictions=(
            expert_predictions
        ),
        defer_mask=defer_mask,
    )


# ---------------------------------------------------------------------------
# One active-learning strategy
# ---------------------------------------------------------------------------

def run_strategy_for_expert(
    strategy_key: str,
    profile: ExpertProfile,
    train_texts: list[str],
    train_labels: np.ndarray,
    train_features: pd.DataFrame,
    train_classifier_entropy: np.ndarray,
    test_features: pd.DataFrame,
    test_labels: np.ndarray,
    test_classifier_predictions: np.ndarray,
    test_classifier_confidence: np.ndarray,
    test_expert_outputs: list[ExpertPrediction],
    train_expert_outputs: list[ExpertPrediction],
) -> dict:
    """
    Run one complete pool-based active-learning experiment.
    """

    random_generator = random.Random(
        RANDOM_STATE
        + profile.random_state
        + sum(
            ord(character)
            for character in strategy_key
        )
    )

    train_expert_correctness = (
        build_expert_correctness_target(
            train_labels,
            train_expert_outputs,
        )
    )

    test_expert_correctness = (
        build_expert_correctness_target(
            test_labels,
            test_expert_outputs,
        )
    )

    test_expert_predictions = np.asarray(
        [
            output.prediction
            for output in test_expert_outputs
        ],
        dtype=int,
    )

    all_indices = np.arange(
        len(train_labels)
    )

    # Same-size random seed for every strategy.
    queried_indices = random_generator.sample(
        all_indices.tolist(),
        INITIAL_QUERY_SIZE,
    )

    learning_curve = []

    for query_budget in QUERY_BUDGETS:

        while (
            len(queried_indices)
            < query_budget
        ):

            model = fit_competence_model(
                features=train_features,
                targets=train_expert_correctness,
                queried_indices=queried_indices,
            )

            pool_mask = np.ones(
                len(train_labels),
                dtype=bool,
            )

            pool_mask[
                queried_indices
            ] = False

            candidate_indices = all_indices[
                pool_mask
            ]

            remaining_budget = (
                query_budget
                - len(queried_indices)
            )

            current_batch_size = min(
                BATCH_SIZE,
                remaining_budget,
            )

            competence_probabilities = (
                model.predict_proba(
                    train_features
                )[:, 1]
            )

            selected_indices = (
                select_query_indices(
                    strategy_key=strategy_key,
                    candidate_indices=(
                        candidate_indices
                    ),
                    batch_size=(
                        current_batch_size
                    ),
                    classifier_entropy=(
                        train_classifier_entropy
                    ),
                    competence_probabilities=(
                        competence_probabilities
                    ),
                    texts=train_texts,
                    queried_indices=(
                        queried_indices
                    ),
                    random_generator=(
                        random_generator
                    ),
                )
            )

            queried_indices.extend(
                selected_indices
            )

        # Retrain after reaching this budget.
        competence_model = (
            fit_competence_model(
                features=train_features,
                targets=train_expert_correctness,
                queried_indices=queried_indices,
            )
        )

        test_competence_probability = (
            competence_model.predict_proba(
                test_features
            )[:, 1]
        )

        competence_metrics = (
            evaluate_competence_predictions(
                true_correctness=(
                    test_expert_correctness
                ),
                predicted_probabilities=(
                    test_competence_probability
                ),
            )
        )

        team_metrics = (
            evaluate_team_from_competence(
                true_labels=test_labels,
                classifier_predictions=(
                    test_classifier_predictions
                ),
                classifier_confidence=(
                    test_classifier_confidence
                ),
                expert_predictions=(
                    test_expert_predictions
                ),
                estimated_expert_accuracy=(
                    test_competence_probability
                ),
            )
        )

        learning_curve.append(
            {
                "query_budget": int(
                    query_budget
                ),
                "query_fraction": float(
                    query_budget
                    / len(train_labels)
                ),
                "competence_accuracy": (
                    competence_metrics[
                        "accuracy"
                    ]
                ),
                "competence_f1": (
                    competence_metrics[
                        "f1"
                    ]
                ),
                "competence_auroc": (
                    competence_metrics[
                        "auroc"
                    ]
                ),
                "brier_score": (
                    competence_metrics[
                        "brier_score"
                    ]
                ),
                "team_accuracy": (
                    team_metrics[
                        "team_accuracy"
                    ]
                ),
                "deferral_rate": (
                    team_metrics[
                        "deferral_rate"
                    ]
                ),
                "beneficial_deferrals": (
                    team_metrics[
                        "beneficial_deferrals"
                    ]
                ),
                "harmful_deferrals": (
                    team_metrics[
                        "harmful_deferrals"
                    ]
                ),
            }
        )

    final_point = (
        learning_curve[-1]
    )

    return {
        "key": strategy_key,
        "name": STRATEGIES[
            strategy_key
        ],
        "learning_curve": learning_curve,
        "final": final_point,
    }
