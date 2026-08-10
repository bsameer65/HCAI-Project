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


# ---------------------------------------------------------------------------
# Complete experiment
# ---------------------------------------------------------------------------

def run_learning_to_defer_experiment() -> dict:
    """
    Run learning-to-defer experiments for both simulated experts.

    The classifier is trained only once for the development stage and once
    for final test evaluation, making the experiment efficient across expert
    profiles.
    """

    dataset = load_ag_news()

    (
        classifier_train,
        meta_train,
        meta_validation,
    ) = create_deferral_development_splits(
        dataset.train
    )

    # ---------------------------------------------------------------
    # Development classifier
    # ---------------------------------------------------------------

    development_classifier = (
        build_baseline_pipeline()
    )

    development_classifier.fit(
        classifier_train["text"],
        classifier_train["label"],
    )

    meta_train_predictions = (
        development_classifier.predict(
            meta_train["text"]
        )
    )

    meta_train_probabilities = (
        development_classifier.predict_proba(
            meta_train["text"]
        )
    )

    meta_validation_predictions = (
        development_classifier.predict(
            meta_validation["text"]
        )
    )

    meta_validation_probabilities = (
        development_classifier.predict_proba(
            meta_validation["text"]
        )
    )

    development_validation_accuracy = float(
        accuracy_score(
            meta_validation["label"],
            meta_validation_predictions,
        )
    )

    # ---------------------------------------------------------------
    # Final classifier
    # ---------------------------------------------------------------

    final_classifier = build_baseline_pipeline()

    final_classifier.fit(
        dataset.train["text"],
        dataset.train["label"],
    )

    test_classifier_predictions = (
        final_classifier.predict(
            dataset.test["text"]
        )
    )

    test_classifier_probabilities = (
        final_classifier.predict_proba(
            dataset.test["text"]
        )
    )

    classifier_test_accuracy = float(
        accuracy_score(
            dataset.test["label"],
            test_classifier_predictions,
        )
    )

    # ---------------------------------------------------------------
    # Experts
    # ---------------------------------------------------------------

    expert_results = {}

    for profile_key, profile in EXPERT_PROFILES.items():

        meta_train_expert_outputs = (
            simulate_expert_for_split(
                texts=meta_train["text"],
                true_labels=meta_train["label"],
                profile=profile,
                seed_offset=101,
            )
        )

        meta_validation_expert_outputs = (
            simulate_expert_for_split(
                texts=meta_validation["text"],
                true_labels=meta_validation[
                    "label"
                ],
                profile=profile,
                seed_offset=202,
            )
        )

        test_expert_outputs = (
            simulate_expert_for_split(
                texts=dataset.test["text"],
                true_labels=dataset.test["label"],
                profile=profile,
                seed_offset=303,
            )
        )

        meta_validation_expert_predictions = (
            np.asarray(
                [
                    output.prediction
                    for output
                    in meta_validation_expert_outputs
                ],
                dtype=int,
            )
        )

        test_expert_predictions = np.asarray(
            [
                output.prediction
                for output in test_expert_outputs
            ],
            dtype=int,
        )

        # -----------------------------------------------------------
        # Strategy 1: confidence threshold
        # -----------------------------------------------------------

        (
            confidence_threshold,
            confidence_search,
        ) = select_confidence_threshold(
            true_labels=meta_validation[
                "label"
            ].to_numpy(),
            classifier_predictions=(
                meta_validation_predictions
            ),
            classifier_probabilities=(
                meta_validation_probabilities
            ),
            expert_predictions=(
                meta_validation_expert_predictions
            ),
        )

        test_confidence = calculate_confidence(
            test_classifier_probabilities
        )

        confidence_defer_mask = (
            test_confidence
            < confidence_threshold
        )

        confidence_metrics = (
            calculate_deferral_metrics(
                true_labels=dataset.test[
                    "label"
                ],
                classifier_predictions=(
                    test_classifier_predictions
                ),
                expert_predictions=(
                    test_expert_predictions
                ),
                defer_mask=(
                    confidence_defer_mask
                ),
            )
        )

        # -----------------------------------------------------------
        # Strategy 2: learned competence-aware deferral
        # -----------------------------------------------------------

        meta_train_features = (
            build_deferral_features(
                texts=meta_train["text"],
                classifier_predictions=(
                    meta_train_predictions
                ),
                classifier_probabilities=(
                    meta_train_probabilities
                ),
                expert_outputs=(
                    meta_train_expert_outputs
                ),
            )
        )

        meta_train_target = (
            build_beneficial_deferral_target(
                true_labels=meta_train[
                    "label"
                ],
                classifier_predictions=(
                    meta_train_predictions
                ),
                expert_outputs=(
                    meta_train_expert_outputs
                ),
            )
        )

        if len(np.unique(meta_train_target)) < 2:
            raise ValueError(
                f"Deferral target for '{profile.name}' "
                "contains only one class."
            )

        competence_model = (
            build_competence_model()
        )

        competence_model.fit(
            meta_train_features,
            meta_train_target,
        )

        validation_features = (
            build_deferral_features(
                texts=meta_validation["text"],
                classifier_predictions=(
                    meta_validation_predictions
                ),
                classifier_probabilities=(
                    meta_validation_probabilities
                ),
                expert_outputs=(
                    meta_validation_expert_outputs
                ),
            )
        )

        validation_benefit_probability = (
            competence_model.predict_proba(
                validation_features
            )[:, 1]
        )

        (
            learned_threshold,
            learned_search,
        ) = select_learned_threshold(
            true_labels=meta_validation[
                "label"
            ].to_numpy(),
            classifier_predictions=(
                meta_validation_predictions
            ),
            expert_predictions=(
                meta_validation_expert_predictions
            ),
            benefit_probabilities=(
                validation_benefit_probability
            ),
        )

        # Refit on all available development expert labels after the
        # decision threshold has been selected.

        validation_target = (
            build_beneficial_deferral_target(
                true_labels=meta_validation[
                    "label"
                ],
                classifier_predictions=(
                    meta_validation_predictions
                ),
                expert_outputs=(
                    meta_validation_expert_outputs
                ),
            )
        )

        all_meta_features = pd.concat(
            [
                meta_train_features,
                validation_features,
            ],
            ignore_index=True,
        )

        all_meta_targets = np.concatenate(
            [
                meta_train_target,
                validation_target,
            ]
        )

        final_competence_model = (
            build_competence_model()
        )

        final_competence_model.fit(
            all_meta_features,
            all_meta_targets,
        )

        test_features = build_deferral_features(
            texts=dataset.test["text"],
            classifier_predictions=(
                test_classifier_predictions
            ),
            classifier_probabilities=(
                test_classifier_probabilities
            ),
            expert_outputs=test_expert_outputs,
        )

        test_benefit_probability = (
            final_competence_model.predict_proba(
                test_features
            )[:, 1]
        )

        learned_defer_mask = (
            test_benefit_probability
            >= learned_threshold
        )

        learned_metrics = (
            calculate_deferral_metrics(
                true_labels=dataset.test[
                    "label"
                ],
                classifier_predictions=(
                    test_classifier_predictions
                ),
                expert_predictions=(
                    test_expert_predictions
                ),
                defer_mask=(
                    learned_defer_mask
                ),
            )
        )
        
        confidence_accuracy = (
            confidence_metrics["team_accuracy"]
        )

        learned_accuracy = (
            learned_metrics["team_accuracy"]
        )

        if learned_accuracy > confidence_accuracy:
            best_strategy_key = "learned"
            best_strategy_name = (
                "Competence-Aware Learned Deferral"
            )

        elif confidence_accuracy > learned_accuracy:
            best_strategy_key = "confidence"
            best_strategy_name = (
                "Confidence-Threshold Deferral"
            )

        else:
            # Tie-break:
            # prefer the strategy that uses less expert effort.
            if (
                learned_metrics["deferral_rate"]
                < confidence_metrics["deferral_rate"]
            ):
                best_strategy_key = "learned"
                best_strategy_name = (
                    "Competence-Aware Learned Deferral"
                )
            else:
                best_strategy_key = "confidence"
                best_strategy_name = (
                    "Confidence-Threshold Deferral"
                )

        expert_results[profile_key] = {
            "name": profile.name,
            "description": profile.description,

            "best_strategy_key": best_strategy_key,
            "best_strategy_name": best_strategy_name,

            "confidence_strategy": {
                "name": (
                    "Confidence-Threshold Deferral"
                ),
                "selected_threshold": (
                    confidence_threshold
                ),
                "is_best": (
                    best_strategy_key == "confidence"
                ),
                "metrics": confidence_metrics,
                "threshold_search": confidence_search,
            },

            "learned_strategy": {
                "name": (
                    "Competence-Aware "
                    "Learned Deferral"
                ),
                "selected_threshold": (
                    learned_threshold
                ),
                "is_best": (
                    best_strategy_key == "learned"
                ),
                "metrics": learned_metrics,
                "threshold_search": learned_search,
            },
        }

    result = {
        "experiment": {
            "name": "Learning to Defer",
            "random_state": RANDOM_STATE,
        },

        "development_split": {
            "classifier_train_samples": int(
                len(classifier_train)
            ),
            "deferral_train_samples": int(
                len(meta_train)
            ),
            "deferral_validation_samples": int(
                len(meta_validation)
            ),
        },

        "classifier": {
            "development_validation_accuracy": (
                development_validation_accuracy
            ),
            "final_test_accuracy": (
                classifier_test_accuracy
            ),
        },

        "experts": expert_results,
    }

    save_learning_to_defer_results(
        result
    )

    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_learning_to_defer_results(
    result: dict,
) -> None:
    """
    Persist the learning-to-defer experiment results atomically.
    """

    DEFER_METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        DEFER_METRICS_PATH.with_suffix(
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
            DEFER_METRICS_PATH
        )

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise


def load_learning_to_defer_results() -> dict | None:
    """
    Load saved learning-to-defer results.
    """

    if not DEFER_METRICS_PATH.exists():
        return None

    with DEFER_METRICS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)
