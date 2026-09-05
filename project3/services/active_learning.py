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

from .baseline import load_baseline_model
from .data_loader import load_ag_news
from .learning_to_defer import (
    calculate_confidence,
    calculate_deferral_metrics,
    calculate_entropy,
    calculate_margin,
)
from .simulated_expert import (
    CLASS_NAMES,
    EXPERT_PROFILES,
    ExpertPrediction,
    ExpertProfile,
    simulate_expert_predictions,
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT3_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

ACTIVE_LEARNING_METRICS_DIR = (
    PROJECT3_DIR
    / "artifacts"
    / "metrics"
)

# Full benchmark artifact.
#
# This artifact is intentionally preserved for:
# - Compare Strategies
# - Advanced Analysis
# - final reporting
ACTIVE_LEARNING_METRICS_PATH = (
    ACTIVE_LEARNING_METRICS_DIR
    / "active_learning_metrics.json"
)

# Latest interactive/user-selected experiment.
#
# Keeping this separate prevents a single user experiment from overwriting
# the full benchmark required by Advanced Analysis.
SELECTED_ACTIVE_LEARNING_METRICS_PATH = (
    ACTIVE_LEARNING_METRICS_DIR
    / "active_learning_selected_metrics.json"
)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

# Initial expert queries required before a competence model can be trained.
INITIAL_QUERY_SIZE = 100

# Number of additional expert queries selected during one active-learning
# iteration.
BATCH_SIZE = 100

# Supported cumulative expert-query budgets.
QUERY_BUDGETS = [
    100,
    200,
    400,
    600,
    800,
    1000,
]

STRATEGIES = {
    "random": (
        "Random Sampling"
    ),
    "classifier_entropy": (
        "Classifier Entropy"
    ),
    "competence_uncertainty": (
        "Competence Uncertainty"
    ),
    "hybrid": (
        "Hybrid Uncertainty + Diversity"
    ),
}
PRIMARY_ACTIVE_LEARNING_STRATEGY = "classifier_entropy"


NUMERIC_FEATURES = [
    "classifier_confidence",
    "classifier_entropy",
    "classifier_margin",
    "log_text_length",
]

CATEGORICAL_FEATURES = [
    "predicted_class",
]

PROJECT3_DIR = Path(__file__).resolve().parent.parent

ACTIVE_LEARNING_FIGURE_DIR = (
    PROJECT3_DIR
    / "static"
    / "project3"
    / "figures"
    / "active_learning"
)

ACTIVE_LEARNING_FIGURE_PATH = (
    ACTIVE_LEARNING_FIGURE_DIR
    / "classifier_entropy_learning_curve.png"
)

ACTIVE_LEARNING_FIGURE_STATIC_PATH = (
    "project3/figures/active_learning/"
    "classifier_entropy_learning_curve.png"
)


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
    excluded. Otherwise the active learner would already know the
    competence profile it is supposed to discover.
    """

    text_list = list(
        texts
    )

    classifier_predictions = np.asarray(
        list(
            classifier_predictions
        ),
        dtype=int,
    )

    classifier_probabilities = np.asarray(
        classifier_probabilities,
        dtype=float,
    )

    if (
        len(text_list)
        != len(
            classifier_predictions
        )
    ):
        raise ValueError(
            "Texts and classifier predictions must have equal length."
        )

    if (
        len(text_list)
        != len(
            classifier_probabilities
        )
    ):
        raise ValueError(
            "Texts and probability rows must have equal length."
        )

    confidence = (
        calculate_confidence(
            classifier_probabilities
        )
    )

    entropy = (
        calculate_entropy(
            classifier_probabilities
        )
    )

    margin = (
        calculate_margin(
            classifier_probabilities
        )
    )

    text_length = np.asarray(
        [
            len(
                str(text).split()
            )
            for text
            in text_list
        ],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "classifier_confidence": (
                confidence
            ),
            "classifier_entropy": (
                entropy
            ),
            "classifier_margin": (
                margin
            ),
            "log_text_length": (
                np.log1p(
                    text_length
                )
            ),
            "predicted_class": [
                CLASS_NAMES[
                    int(label)
                ]
                for label
                in classifier_predictions
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

    Conceptually, this function represents the hidden expert oracle.
    Active learning may access expert outcomes only for selected examples
    when constructing the competence-training set.
    """

    queried_profile = replace(
        profile,
        random_state=(
            profile.random_state
            + seed_offset
        ),
    )

    return (
        simulate_expert_predictions(
            texts=texts,
            true_labels=true_labels,
            profile=queried_profile,
        )
    )


def build_expert_correctness_target(
    true_labels: Iterable[int],
    expert_outputs: list[ExpertPrediction],
) -> np.ndarray:
    """
    Build the binary target used for expert-competence discovery.

    1 = expert prediction is correct
    0 = expert prediction is incorrect
    """

    true_labels = np.asarray(
        list(
            true_labels
        ),
        dtype=int,
    )

    expert_predictions = np.asarray(
        [
            output.prediction
            for output
            in expert_outputs
        ],
        dtype=int,
    )

    if (
        len(true_labels)
        != len(
            expert_predictions
        )
    ):
        raise ValueError(
            "True labels and expert outputs must have equal length."
        )

    return (
        expert_predictions
        == true_labels
    ).astype(
        int
    )


# ---------------------------------------------------------------------------
# Competence model
# ---------------------------------------------------------------------------

def build_competence_model() -> Pipeline:
    """
    Build the binary model that estimates:

        P(expert is correct | observable information)
    """

    preprocessing = (
        ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    "passthrough",
                    NUMERIC_FEATURES,
                ),
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown=(
                            "ignore"
                        ),
                    ),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )
    )

    classifier = (
        LogisticRegression(
            max_iter=1000,
            class_weight=(
                "balanced"
            ),
            random_state=(
                RANDOM_STATE
            ),
            solver=(
                "liblinear"
            ),
        )
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
    Train the competence model using only queried expert examples.
    """

    if not queried_indices:
        raise ValueError(
            "At least one queried expert example is required."
        )

    queried_targets = (
        targets[
            queried_indices
        ]
    )

    if (
        len(
            np.unique(
                queried_targets
            )
        )
        < 2
    ):
        raise ValueError(
            "Queried expert labels contain only one correctness class. "
            "Increase the initial query size."
        )

    model = (
        build_competence_model()
    )

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
    Binary competence uncertainty around:

        P(expert correct) = 0.5

    Maximum utility occurs when estimated competence is closest to 0.5.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    return (
        1.0
        - (
            2.0
            * np.abs(
                probabilities
                - 0.5
            )
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

    if values.size == 0:
        return values

    minimum = (
        values.min()
    )

    maximum = (
        values.max()
    )

    if np.isclose(
        minimum,
        maximum,
    ):
        return np.zeros_like(
            values
        )

    return (
        values
        - minimum
    ) / (
        maximum
        - minimum
    )


def calculate_diversity_score(
    pool_texts: list[str],
    queried_texts: list[str],
) -> np.ndarray:
    """
    Estimate diversity using TF-IDF distance from previously queried samples.

    A candidate receives high diversity utility when it has low maximum
    cosine similarity with already queried articles.
    """

    if not pool_texts:
        return np.asarray(
            [],
            dtype=float,
        )

    if not queried_texts:
        return np.ones(
            len(
                pool_texts
            ),
            dtype=float,
        )

    vectorizer = (
        TfidfVectorizer(
            lowercase=True,
            max_features=5000,
            min_df=2,
            ngram_range=(
                1,
                1,
            ),
        )
    )

    all_texts = (
        queried_texts
        + pool_texts
    )

    try:
        matrix = (
            vectorizer.fit_transform(
                all_texts
            )
        )

    except ValueError:
        # Defensive fallback for an unusually small or empty vocabulary.
        return np.ones(
            len(
                pool_texts
            ),
            dtype=float,
        )

    queried_matrix = (
        matrix[
            :len(
                queried_texts
            )
        ]
    )

    pool_matrix = (
        matrix[
            len(
                queried_texts
            ):
        ]
    )

    similarities = (
        pool_matrix
        @ queried_matrix.T
    ).toarray()

    maximum_similarity = (
        similarities.max(
            axis=1
        )
    )

    diversity = (
        1.0
        - maximum_similarity
    )

    return (
        normalize_scores(
            diversity
        )
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
    Select the next batch according to one active-learning strategy.
    """

    if (
        strategy_key
        not in STRATEGIES
    ):
        raise ValueError(
            f"Unknown active-learning strategy: {strategy_key}"
        )

    if batch_size <= 0:
        return []

    if (
        len(
            candidate_indices
        )
        <= batch_size
    ):
        return (
            candidate_indices.tolist()
        )

    if (
        strategy_key
        == "random"
    ):
        return (
            random_generator.sample(
                candidate_indices.tolist(),
                batch_size,
            )
        )

    if (
        strategy_key
        == "classifier_entropy"
    ):

        utility = (
            classifier_entropy[
                candidate_indices
            ]
        )

    elif (
        strategy_key
        == "competence_uncertainty"
    ):

        if (
            competence_probabilities
            is None
        ):
            raise ValueError(
                "Competence probabilities are required."
            )

        utility = (
            competence_uncertainty_score(
                competence_probabilities[
                    candidate_indices
                ]
            )
        )

    elif (
        strategy_key
        == "hybrid"
    ):

        if (
            competence_probabilities
            is None
        ):
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
            texts[
                index
            ]
            for index
            in candidate_indices
        ]

        queried_texts = [
            texts[
                index
            ]
            for index
            in queried_indices
        ]

        diversity = (
            calculate_diversity_score(
                pool_texts=(
                    pool_texts
                ),
                queried_texts=(
                    queried_texts
                ),
            )
        )

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

    top_local_indices = (
        np.argsort(
            utility
        )[
            -batch_size:
        ]
    )

    return (
        candidate_indices[
            top_local_indices
        ].tolist()
    )


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

    true_correctness = np.asarray(
        true_correctness,
        dtype=int,
    )

    predicted_probabilities = np.asarray(
        predicted_probabilities,
        dtype=float,
    )

    predicted_labels = (
        predicted_probabilities
        >= 0.5
    ).astype(
        int
    )

    accuracy = (
        accuracy_score(
            true_correctness,
            predicted_labels,
        )
    )

    f1 = (
        f1_score(
            true_correctness,
            predicted_labels,
            zero_division=0,
        )
    )

    if (
        len(
            np.unique(
                true_correctness
            )
        )
        > 1
    ):
        auroc = (
            roc_auc_score(
                true_correctness,
                predicted_probabilities,
            )
        )
    else:
        auroc = None

    brier = (
        brier_score_loss(
            true_correctness,
            predicted_probabilities,
        )
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "f1": float(
            f1
        ),
        "auroc": (
            float(
                auroc
            )
            if auroc
            is not None
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

    return (
        calculate_deferral_metrics(
            true_labels=(
                true_labels
            ),
            classifier_predictions=(
                classifier_predictions
            ),
            expert_predictions=(
                expert_predictions
            ),
            defer_mask=(
                defer_mask
            ),
        )
    )


# ---------------------------------------------------------------------------
# Shared classifier/data preparation
# ---------------------------------------------------------------------------

def prepare_active_learning_data() -> dict:
    """
    Prepare all classifier information required by Active Learning.

    The saved baseline classifier is reused instead of being retrained here.

    This is preferable because:
    - the Baseline experiment is already responsible for training it;
    - the classifier uses the complete labeled AG News training set;
    - interactive Active Learning can therefore focus only on discovering
      expert competence.
    """

    dataset = (
        load_ag_news()
    )

    classifier = (
        load_baseline_model()
    )

    if classifier is None:
        raise RuntimeError(
            "The saved baseline classifier is unavailable. "
            "Run the Baseline experiment first."
        )

    train_predictions = (
        classifier.predict(
            dataset.train[
                "text"
            ]
        )
    )

    train_probabilities = (
        classifier.predict_proba(
            dataset.train[
                "text"
            ]
        )
    )

    test_predictions = (
        classifier.predict(
            dataset.test[
                "text"
            ]
        )
    )

    test_probabilities = (
        classifier.predict_proba(
            dataset.test[
                "text"
            ]
        )
    )

    train_entropy = (
        calculate_entropy(
            train_probabilities
        )
    )

    test_confidence = (
        calculate_confidence(
            test_probabilities
        )
    )

    train_features = (
        build_competence_features(
            texts=(
                dataset.train[
                    "text"
                ]
            ),
            classifier_predictions=(
                train_predictions
            ),
            classifier_probabilities=(
                train_probabilities
            ),
        )
    )

    test_features = (
        build_competence_features(
            texts=(
                dataset.test[
                    "text"
                ]
            ),
            classifier_predictions=(
                test_predictions
            ),
            classifier_probabilities=(
                test_probabilities
            ),
        )
    )

    train_texts = (
        dataset.train[
            "text"
        ].tolist()
    )

    train_labels = (
        dataset.train[
            "label"
        ].to_numpy()
    )

    test_labels = (
        dataset.test[
            "label"
        ].to_numpy()
    )

    classifier_accuracy = (
        accuracy_score(
            test_labels,
            test_predictions,
        )
    )

    return {
        "dataset": (
            dataset
        ),
        "train_texts": (
            train_texts
        ),
        "train_labels": (
            train_labels
        ),
        "test_labels": (
            test_labels
        ),
        "train_features": (
            train_features
        ),
        "test_features": (
            test_features
        ),
        "train_entropy": (
            train_entropy
        ),
        "test_predictions": (
            np.asarray(
                test_predictions,
                dtype=int,
            )
        ),
        "test_confidence": (
            np.asarray(
                test_confidence,
                dtype=float,
            )
        ),
        "classifier_accuracy": float(
            classifier_accuracy
        ),
    }


# ---------------------------------------------------------------------------
# One Active Learning strategy
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
    query_budgets: list[int] | None = None,
) -> dict:
    """
    Run one pool-based Active Learning strategy for one expert.

    query_budgets determines the checkpoints at which the competence model
    and Human-AI team are evaluated.

    When omitted, the complete benchmark budget sequence is used.
    """

    if (
        strategy_key
        not in STRATEGIES
    ):
        raise ValueError(
            f"Unknown active-learning strategy: {strategy_key}"
        )

    if query_budgets is None:
        query_budgets = list(
            QUERY_BUDGETS
        )

    query_budgets = sorted(
        {
            int(
                budget
            )
            for budget
            in query_budgets
        }
    )

    if not query_budgets:
        raise ValueError(
            "At least one query budget is required."
        )

    if (
        query_budgets[
            0
        ]
        < INITIAL_QUERY_SIZE
    ):
        raise ValueError(
            "Query budgets cannot be smaller than "
            f"{INITIAL_QUERY_SIZE}."
        )

    if (
        query_budgets[
            -1
        ]
        > len(
            train_labels
        )
    ):
        raise ValueError(
            "Query budget cannot exceed the size of the training pool."
        )

    random_generator = (
        random.Random(
            RANDOM_STATE
            + profile.random_state
            + sum(
                ord(
                    character
                )
                for character
                in strategy_key
            )
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

    test_expert_predictions = (
        np.asarray(
            [
                output.prediction
                for output
                in test_expert_outputs
            ],
            dtype=int,
        )
    )

    all_indices = (
        np.arange(
            len(
                train_labels
            )
        )
    )

    # Same-size random seed set for every strategy.
    queried_indices = (
        random_generator.sample(
            all_indices.tolist(),
            INITIAL_QUERY_SIZE,
        )
    )

    learning_curve = []

    for query_budget in query_budgets:

        # -----------------------------------------------------------
        # Actively acquire expert labels until this checkpoint.
        # -----------------------------------------------------------

        while (
            len(
                queried_indices
            )
            < query_budget
        ):

            model = (
                fit_competence_model(
                    features=(
                        train_features
                    ),
                    targets=(
                        train_expert_correctness
                    ),
                    queried_indices=(
                        queried_indices
                    ),
                )
            )

            pool_mask = np.ones(
                len(
                    train_labels
                ),
                dtype=bool,
            )

            pool_mask[
                queried_indices
            ] = False

            candidate_indices = (
                all_indices[
                    pool_mask
                ]
            )

            remaining_budget = (
                query_budget
                - len(
                    queried_indices
                )
            )

            current_batch_size = min(
                BATCH_SIZE,
                remaining_budget,
                len(
                    candidate_indices
                ),
            )

            competence_probabilities = (
                model.predict_proba(
                    train_features
                )[
                    :,
                    1
                ]
            )

            selected_indices = (
                select_query_indices(
                    strategy_key=(
                        strategy_key
                    ),
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
                    texts=(
                        train_texts
                    ),
                    queried_indices=(
                        queried_indices
                    ),
                    random_generator=(
                        random_generator
                    ),
                )
            )

            if not selected_indices:
                break

            queried_indices.extend(
                selected_indices
            )

        # -----------------------------------------------------------
        # Evaluate competence model at this budget.
        # -----------------------------------------------------------

        competence_model = (
            fit_competence_model(
                features=(
                    train_features
                ),
                targets=(
                    train_expert_correctness
                ),
                queried_indices=(
                    queried_indices
                ),
            )
        )

        test_competence_probability = (
            competence_model.predict_proba(
                test_features
            )[
                :,
                1
            ]
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
                true_labels=(
                    test_labels
                ),
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
                    / len(
                        train_labels
                    )
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

    if not learning_curve:
        raise RuntimeError(
            "Active Learning produced no evaluation checkpoints."
        )

    final_point = (
        learning_curve[
            -1
        ]
    )

    return {
        "key": (
            strategy_key
        ),
        "name": (
            STRATEGIES[
                strategy_key
            ]
        ),
        "learning_curve": (
            learning_curve
        ),
        "final": (
            final_point
        ),
    }


# ---------------------------------------------------------------------------
# Selected interactive experiment
# ---------------------------------------------------------------------------

def run_selected_active_learning(
    expert_key: str,
    query_budget: int,
) -> dict:
    """
    Run the primary Active Learning strategy for one selected expert
    and expert-query budget.

    Classifier Entropy is the project's chosen acquisition strategy.
    Alternative strategies are evaluated separately on the comparison page.
    """

    strategy_key = PRIMARY_ACTIVE_LEARNING_STRATEGY

    if expert_key not in EXPERT_PROFILES:
        raise ValueError(
            "Unknown simulated expert."
        )

    try:
        query_budget = int(
            query_budget
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Query budget must be an integer."
        ) from exc

    if query_budget not in QUERY_BUDGETS:
        raise ValueError(
            "Unsupported query budget. "
            f"Choose one of: {QUERY_BUDGETS}"
        )

    prepared = (
        prepare_active_learning_data()
    )

    dataset = prepared[
        "dataset"
    ]

    profile = EXPERT_PROFILES[
        expert_key
    ]

    train_expert_outputs = (
        query_expert(
            texts=dataset.train[
                "text"
            ],
            true_labels=dataset.train[
                "label"
            ],
            profile=profile,
            seed_offset=500,
        )
    )

    test_expert_outputs = (
        query_expert(
            texts=dataset.test[
                "text"
            ],
            true_labels=dataset.test[
                "label"
            ],
            profile=profile,
            seed_offset=600,
        )
    )

    selected_budgets = [
        budget
        for budget in QUERY_BUDGETS
        if budget <= query_budget
    ]

    strategy_result = (
        run_strategy_for_expert(
            strategy_key=strategy_key,
            profile=profile,
            train_texts=prepared[
                "train_texts"
            ],
            train_labels=prepared[
                "train_labels"
            ],
            train_features=prepared[
                "train_features"
            ],
            train_classifier_entropy=prepared[
                "train_entropy"
            ],
            test_features=prepared[
                "test_features"
            ],
            test_labels=prepared[
                "test_labels"
            ],
            test_classifier_predictions=prepared[
                "test_predictions"
            ],
            test_classifier_confidence=prepared[
                "test_confidence"
            ],
            test_expert_outputs=(
                test_expert_outputs
            ),
            train_expert_outputs=(
                train_expert_outputs
            ),
            query_budgets=selected_budgets,
        )
    )

    result = {
        "experiment": {
            "name": (
                "Selected Active Learning Experiment"
            ),
            "initial_query_size": (
                INITIAL_QUERY_SIZE
            ),
            "batch_size": (
                BATCH_SIZE
            ),
            "random_state": (
                RANDOM_STATE
            ),
        },

        "dataset": {
            "training_samples": int(
                len(dataset.train)
            ),
            "test_samples": int(
                len(dataset.test)
            ),
        },

        "classifier": {
            "test_accuracy": prepared[
                "classifier_accuracy"
            ],
        },

        "selection": {
            "expert_key": expert_key,
            "expert_name": profile.name,
            "expert_description": (
                profile.description
            ),
            "strategy_key": strategy_key,
            "strategy_name": STRATEGIES[
                strategy_key
            ],
            "query_budget": int(
                query_budget
            ),
        },

        "strategy": strategy_result,
    }

    save_selected_active_learning_results(
        result
    )

    return result

# ---------------------------------------------------------------------------
# Full comparison experiment
# ---------------------------------------------------------------------------

def run_active_learning_experiment() -> dict:
    """
    Compare all active-learning strategies for all simulated experts.

    This is the scientific benchmark used by:
    - Compare Strategies
    - Advanced Analysis
    - report generation

    It intentionally remains separate from the user-selected experiment.
    """

    prepared = (
        prepare_active_learning_data()
    )

    dataset = (
        prepared[
            "dataset"
        ]
    )

    expert_results = {}

    for (
        profile_key,
        profile,
    ) in EXPERT_PROFILES.items():

        train_expert_outputs = (
            query_expert(
                texts=(
                    dataset.train[
                        "text"
                    ]
                ),
                true_labels=(
                    dataset.train[
                        "label"
                    ]
                ),
                profile=(
                    profile
                ),
                seed_offset=500,
            )
        )

        test_expert_outputs = (
            query_expert(
                texts=(
                    dataset.test[
                        "text"
                    ]
                ),
                true_labels=(
                    dataset.test[
                        "label"
                    ]
                ),
                profile=(
                    profile
                ),
                seed_offset=600,
            )
        )

        strategy_results = {}

        for strategy_key in STRATEGIES:

            strategy_results[
                strategy_key
            ] = (
                run_strategy_for_expert(
                    strategy_key=(
                        strategy_key
                    ),
                    profile=(
                        profile
                    ),
                    train_texts=(
                        prepared[
                            "train_texts"
                        ]
                    ),
                    train_labels=(
                        prepared[
                            "train_labels"
                        ]
                    ),
                    train_features=(
                        prepared[
                            "train_features"
                        ]
                    ),
                    train_classifier_entropy=(
                        prepared[
                            "train_entropy"
                        ]
                    ),
                    test_features=(
                        prepared[
                            "test_features"
                        ]
                    ),
                    test_labels=(
                        prepared[
                            "test_labels"
                        ]
                    ),
                    test_classifier_predictions=(
                        prepared[
                            "test_predictions"
                        ]
                    ),
                    test_classifier_confidence=(
                        prepared[
                            "test_confidence"
                        ]
                    ),
                    test_expert_outputs=(
                        test_expert_outputs
                    ),
                    train_expert_outputs=(
                        train_expert_outputs
                    ),
                    query_budgets=(
                        QUERY_BUDGETS
                    ),
                )
            )

        # -----------------------------------------------------------
        # Determine best strategy by final Human-AI team accuracy.
        # -----------------------------------------------------------

        best_strategy_key = max(
            strategy_results,
            key=lambda key: (
                strategy_results[
                    key
                ][
                    "final"
                ][
                    "team_accuracy"
                ]
            ),
        )

        for (
            strategy_key,
            strategy,
        ) in strategy_results.items():

            strategy[
                "is_best"
            ] = (
                strategy_key
                == best_strategy_key
            )

        expert_results[
            profile_key
        ] = {
            "name": (
                profile.name
            ),
            "description": (
                profile.description
            ),
            "best_strategy_key": (
                best_strategy_key
            ),
            "best_strategy_name": (
                strategy_results[
                    best_strategy_key
                ][
                    "name"
                ]
            ),
            "strategies": (
                strategy_results
            ),
        }

    result = {
        "experiment": {
            "name": (
                "Active Learning for "
                "Expert Competence Discovery"
            ),
            "initial_query_size": (
                INITIAL_QUERY_SIZE
            ),
            "batch_size": (
                BATCH_SIZE
            ),
            "query_budgets": (
                QUERY_BUDGETS
            ),
            "random_state": (
                RANDOM_STATE
            ),
        },

        "dataset": {
            "training_samples": int(
                len(
                    dataset.train
                )
            ),
            "test_samples": int(
                len(
                    dataset.test
                )
            ),
        },

        "classifier": {
            "test_accuracy": (
                prepared[
                    "classifier_accuracy"
                ]
            ),
        },

        "experts": (
            expert_results
        ),
    }

    save_active_learning_results(
        result
    )

    return result


# ---------------------------------------------------------------------------
# Persistence - full benchmark
# ---------------------------------------------------------------------------

def save_active_learning_results(
    result: dict,
) -> None:
    """
    Save the complete Active Learning comparison artifact atomically.
    """

    ACTIVE_LEARNING_METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        ACTIVE_LEARNING_METRICS_PATH.with_suffix(
            ".json.tmp"
        )
    )

    try:

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                indent=2,
                ensure_ascii=False,
            )

        temporary_path.replace(
            ACTIVE_LEARNING_METRICS_PATH
        )

    except Exception:

        if (
            temporary_path.exists()
        ):
            temporary_path.unlink()

        raise


def load_active_learning_results() -> dict | None:
    """
    Load the previously generated full Active Learning benchmark.
    """

    if not (
        ACTIVE_LEARNING_METRICS_PATH.exists()
    ):
        return None

    with (
        ACTIVE_LEARNING_METRICS_PATH.open(
            "r",
            encoding="utf-8",
        )
    ) as file:

        return json.load(
            file
        )


# ---------------------------------------------------------------------------
# Persistence - selected interactive experiment
# ---------------------------------------------------------------------------

def save_selected_active_learning_results(
    result: dict,
) -> None:
    """
    Save the latest user-selected Active Learning experiment atomically.
    """

    ACTIVE_LEARNING_METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        SELECTED_ACTIVE_LEARNING_METRICS_PATH.with_suffix(
            ".json.tmp"
        )
    )

    try:

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                indent=2,
                ensure_ascii=False,
            )

        temporary_path.replace(
            SELECTED_ACTIVE_LEARNING_METRICS_PATH
        )

    except Exception:

        if (
            temporary_path.exists()
        ):
            temporary_path.unlink()

        raise


def load_selected_active_learning_results() -> dict | None:
    """
    Load the latest user-selected Active Learning experiment.
    """

    if not (
        SELECTED_ACTIVE_LEARNING_METRICS_PATH.exists()
    ):
        return None

    with (
        SELECTED_ACTIVE_LEARNING_METRICS_PATH.open(
            "r",
            encoding="utf-8",
        )
    ) as file:

        return json.load(
            file
        )
        

def create_classifier_entropy_learning_curve(result):
    """
    Create the learning curve for the selected Classifier Entropy
    active-learning experiment.

    The figure shows how Human-AI team accuracy changes as more
    expert responses are acquired.
    """

    if not result:
        return None

    strategy = result.get("strategy", {})
    learning_curve = strategy.get("learning_curve", [])

    if not learning_curve:
        return None

    selection = result.get("selection", {})

    expert_name = selection.get(
        "expert_name",
        "Selected Simulated Expert",
    )

    query_budgets = []
    team_accuracies = []

    for point in learning_curve:

        query_budget = point.get("query_budget")
        team_accuracy = point.get("team_accuracy")

        if query_budget is None or team_accuracy is None:
            continue

        query_budgets.append(
            int(query_budget)
        )

        team_accuracies.append(
            float(team_accuracy) * 100
        )

    if not query_budgets:
        return None

    ACTIVE_LEARNING_FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.2)
    )

    ax.plot(
        query_budgets,
        team_accuracies,
        marker="o",
        linewidth=2,
    )

    ax.set_title(
        "Classifier Entropy Learning Curve\n"
        f"{expert_name}",
        fontsize=13,
        pad=12,
    )

    ax.set_xlabel(
        "Expert Queries"
    )

    ax.set_ylabel(
        "Human-AI Team Accuracy (%)"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    # Keep some visual space around the observed values without
    # exaggerating small differences.
    minimum_accuracy = min(team_accuracies)
    maximum_accuracy = max(team_accuracies)

    lower_limit = max(
        0,
        minimum_accuracy - 0.5,
    )

    upper_limit = min(
        100,
        maximum_accuracy + 0.5,
    )

    if upper_limit - lower_limit < 1.0:
        midpoint = (
            minimum_accuracy + maximum_accuracy
        ) / 2

        lower_limit = max(
            0,
            midpoint - 0.5,
        )

        upper_limit = min(
            100,
            midpoint + 0.5,
        )

    ax.set_ylim(
        lower_limit,
        upper_limit,
    )

    for x_value, y_value in zip(
        query_budgets,
        team_accuracies,
    ):

        ax.annotate(
            f"{y_value:.2f}%",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()

    fig.savefig(
        ACTIVE_LEARNING_FIGURE_PATH,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    return ACTIVE_LEARNING_FIGURE_STATIC_PATH


def ensure_classifier_entropy_learning_curve(result):
    """
    Ensure that the Classifier Entropy learning-curve figure exists.

    The figure is recreated because the selected expert or query
    budget may have changed.
    """

    if not result:
        return None

    return create_classifier_entropy_learning_curve(
        result
    )