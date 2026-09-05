from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .baseline import build_baseline_pipeline
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
    ExpertProfile,
    get_region_accuracy,
    identify_expertise_region,
    simulate_expert_predictions,
)


RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------

META_POOL_SIZE = 0.30

# 30% of training data becomes the meta pool.
# Half is used to train deferral models and half to tune thresholds.
META_VALIDATION_SIZE = 0.50


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

ABLATION_THRESHOLDS = np.round(
    np.arange(
        0.05,
        1.00,
        0.05,
    ),
    2,
)


# ---------------------------------------------------------------------------
# Ablation variants
# ---------------------------------------------------------------------------

ABLATION_VARIANTS = {
    "full": {
        "name": "Full Model",
        "description": (
            "Uses classifier uncertainty, text length, predicted class, "
            "expert region, and configured expected expert accuracy."
        ),
        "numeric": [
            "classifier_confidence",
            "classifier_entropy",
            "classifier_margin",
            "log_text_length",
            "expected_expert_accuracy",
        ],
        "categorical": [
            "predicted_class",
            "expert_region",
        ],
    },

    "no_expected_accuracy": {
        "name": "No Expected Accuracy",
        "description": (
            "Removes the explicitly configured expected expert accuracy "
            "while retaining expert-region information."
        ),
        "numeric": [
            "classifier_confidence",
            "classifier_entropy",
            "classifier_margin",
            "log_text_length",
        ],
        "categorical": [
            "predicted_class",
            "expert_region",
        ],
    },

    "no_expert_profile": {
        "name": "No Expert Profile",
        "description": (
            "Uses only classifier-derived and article-level information. "
            "Both expert region and configured expert accuracy are removed."
        ),
        "numeric": [
            "classifier_confidence",
            "classifier_entropy",
            "classifier_margin",
            "log_text_length",
        ],
        "categorical": [
            "predicted_class",
        ],
    },

    "confidence_only": {
        "name": "AI Confidence Only",
        "description": (
            "Uses only classifier confidence as input to the learned "
            "binary deferral model."
        ),
        "numeric": [
            "classifier_confidence",
        ],
        "categorical": [],
    },
}


# ---------------------------------------------------------------------------
# Expert simulation
# ---------------------------------------------------------------------------

def _simulate_expert(
    texts,
    labels,
    profile: ExpertProfile,
    seed_offset: int,
):
    """
    Simulate one expert using a deterministic experiment-specific seed.
    """

    seeded_profile = replace(
        profile,
        random_state=(
            profile.random_state
            + seed_offset
        ),
    )

    return simulate_expert_predictions(
        texts=texts,
        true_labels=labels,
        profile=seeded_profile,
    )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_ablation_features(
    texts,
    classifier_predictions,
    classifier_probabilities,
    profile: ExpertProfile,
) -> pd.DataFrame:
    """
    Build the complete feature set once.

    Individual ablation models then select subsets of these columns.
    """

    text_list = [
        str(text)
        for text in texts
    ]

    predictions = np.asarray(
        classifier_predictions,
        dtype=int,
    )

    probabilities = np.asarray(
        classifier_probabilities,
        dtype=float,
    )

    confidence = calculate_confidence(
        probabilities
    )

    entropy = calculate_entropy(
        probabilities
    )

    margin = calculate_margin(
        probabilities
    )

    text_length = np.asarray(
        [
            len(text.split())
            for text in text_list
        ],
        dtype=float,
    )

    expert_regions = [
        identify_expertise_region(
            text=text,
            profile=profile,
        )
        for text in text_list
    ]

    expected_expert_accuracy = [
        get_region_accuracy(
            region=region,
            profile=profile,
        )
        for region in expert_regions
    ]

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
                in predictions
            ],

            "expert_region": (
                expert_regions
            ),

            "expected_expert_accuracy": (
                expected_expert_accuracy
            ),
        }
    )


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

def _build_beneficial_deferral_target(
    true_labels,
    classifier_predictions,
    expert_predictions,
):
    """
    Beneficial deferral:

    1 = classifier is wrong AND expert is correct
    0 = otherwise
    """

    true_labels = np.asarray(
        true_labels,
        dtype=int,
    )

    classifier_predictions = np.asarray(
        classifier_predictions,
        dtype=int,
    )

    expert_predictions = np.asarray(
        expert_predictions,
        dtype=int,
    )

    classifier_wrong = (
        classifier_predictions
        != true_labels
    )

    expert_correct = (
        expert_predictions
        == true_labels
    )

    return (
        classifier_wrong
        & expert_correct
    ).astype(int)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_ablation_model(
    numeric_features,
    categorical_features,
):
    """
    Create one binary Logistic Regression deferral model.
    """

    transformers = []

    if numeric_features:

        transformers.append(
            (
                "numeric",
                "passthrough",
                numeric_features,
            )
        )

    if categorical_features:

        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                categorical_features,
            )
        )

    preprocessing = ColumnTransformer(
        transformers=transformers
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
# Threshold selection
# ---------------------------------------------------------------------------

def _select_threshold(
    probabilities,
    true_labels,
    classifier_predictions,
    expert_predictions,
):
    """
    Select threshold using validation team accuracy.

    When two thresholds obtain the same team accuracy, prefer the one with
    lower expert workload.
    """

    best_result = None

    search = []

    for threshold in ABLATION_THRESHOLDS:

        defer_mask = (
            probabilities
            >= threshold
        )

        metrics = calculate_deferral_metrics(
            true_labels=true_labels,
            classifier_predictions=(
                classifier_predictions
            ),
            expert_predictions=(
                expert_predictions
            ),
            defer_mask=defer_mask,
        )

        candidate = {
            "threshold": float(
                threshold
            ),

            "team_accuracy": float(
                metrics[
                    "team_accuracy"
                ]
            ),

            "deferral_rate": float(
                metrics[
                    "deferral_rate"
                ]
            ),
        }

        search.append(
            candidate
        )

        if best_result is None:

            best_result = candidate
            continue

        better_accuracy = (
            candidate[
                "team_accuracy"
            ]
            >
            best_result[
                "team_accuracy"
            ]
        )

        same_accuracy_lower_workload = (
            np.isclose(
                candidate[
                    "team_accuracy"
                ],
                best_result[
                    "team_accuracy"
                ],
            )
            and
            candidate[
                "deferral_rate"
            ]
            <
            best_result[
                "deferral_rate"
            ]
        )

        if (
            better_accuracy
            or same_accuracy_lower_workload
        ):

            best_result = candidate

    return {
        "threshold": (
            best_result[
                "threshold"
            ]
        ),
        "threshold_search": search,
    }


# ---------------------------------------------------------------------------
# Single variant
# ---------------------------------------------------------------------------

def _run_variant(
    variant_key,
    train_features,
    validation_features,
    test_features,
    train_target,
    validation_labels,
    validation_classifier_predictions,
    validation_expert_predictions,
    test_labels,
    test_classifier_predictions,
    test_expert_predictions,
):
    """
    Train and evaluate one ablation configuration.
    """

    configuration = (
        ABLATION_VARIANTS[
            variant_key
        ]
    )

    model = _build_ablation_model(
        numeric_features=(
            configuration[
                "numeric"
            ]
        ),
        categorical_features=(
            configuration[
                "categorical"
            ]
        ),
    )

    model.fit(
        train_features,
        train_target,
    )

    validation_probabilities = (
        model.predict_proba(
            validation_features
        )[:, 1]
    )

    threshold_result = (
        _select_threshold(
            probabilities=(
                validation_probabilities
            ),
            true_labels=(
                validation_labels
            ),
            classifier_predictions=(
                validation_classifier_predictions
            ),
            expert_predictions=(
                validation_expert_predictions
            ),
        )
    )

    threshold = (
        threshold_result[
            "threshold"
        ]
    )

    test_probabilities = (
        model.predict_proba(
            test_features
        )[:, 1]
    )

    test_defer_mask = (
        test_probabilities
        >= threshold
    )

    metrics = (
        calculate_deferral_metrics(
            true_labels=test_labels,
            classifier_predictions=(
                test_classifier_predictions
            ),
            expert_predictions=(
                test_expert_predictions
            ),
            defer_mask=(
                test_defer_mask
            ),
        )
    )

    return {
        "key": variant_key,

        "name": configuration["name"],

        "description": configuration["description"],

        "numeric_features": configuration["numeric"],

        "categorical_features": configuration["categorical"],

        "selected_threshold": float(
            threshold
        ),

        "team_accuracy": float(
            metrics["team_accuracy"]
        ),

        "deferral_rate": float(
            metrics["deferral_rate"]
        ),

        "deferral_percent": float(
            metrics["deferral_rate"] * 100
        ),

        "beneficial_deferrals": int(
            metrics["beneficial_deferrals"]
        ),

        "harmful_deferrals": int(
            metrics["harmful_deferrals"]
        ),

        "unnecessary_deferrals": int(
            metrics.get(
                "unnecessary_deferrals",
                0,
            )
        ),

        "missed_beneficial_opportunities": int(
            metrics.get(
                "missed_beneficial_opportunities",
                0,
            )
        ),

        "threshold_search": (
            threshold_result[
                "threshold_search"
            ]
        ),
    }


# ---------------------------------------------------------------------------
# Full expert ablation
# ---------------------------------------------------------------------------

def _run_ablation_for_expert(
    profile: ExpertProfile,
    development_train,
    development_validation,
    test_data,
    classifier,
):
    """
    Run all ablation variants for one simulated expert.
    """

    # ---------------------------------------------------------------
    # Classifier outputs
    # ---------------------------------------------------------------

    train_predictions = (
        classifier.predict(
            development_train[
                "text"
            ]
        )
    )

    train_probabilities = (
        classifier.predict_proba(
            development_train[
                "text"
            ]
        )
    )

    validation_predictions = (
        classifier.predict(
            development_validation[
                "text"
            ]
        )
    )

    validation_probabilities = (
        classifier.predict_proba(
            development_validation[
                "text"
            ]
        )
    )

    test_predictions = (
        classifier.predict(
            test_data[
                "text"
            ]
        )
    )

    test_probabilities = (
        classifier.predict_proba(
            test_data[
                "text"
            ]
        )
    )


    # ---------------------------------------------------------------
    # Expert outputs
    # ---------------------------------------------------------------

    train_outputs = (
        _simulate_expert(
            texts=(
                development_train[
                    "text"
                ]
            ),
            labels=(
                development_train[
                    "label"
                ]
            ),
            profile=profile,
            seed_offset=101,
        )
    )

    validation_outputs = (
        _simulate_expert(
            texts=(
                development_validation[
                    "text"
                ]
            ),
            labels=(
                development_validation[
                    "label"
                ]
            ),
            profile=profile,
            seed_offset=202,
        )
    )

    test_outputs = (
        _simulate_expert(
            texts=test_data[
                "text"
            ],
            labels=test_data[
                "label"
            ],
            profile=profile,
            seed_offset=303,
        )
    )


    train_expert_predictions = np.asarray(
        [
            output.prediction
            for output
            in train_outputs
        ],
        dtype=int,
    )

    validation_expert_predictions = np.asarray(
        [
            output.prediction
            for output
            in validation_outputs
        ],
        dtype=int,
    )

    test_expert_predictions = np.asarray(
        [
            output.prediction
            for output
            in test_outputs
        ],
        dtype=int,
    )


    # ---------------------------------------------------------------
    # Features
    # ---------------------------------------------------------------

    train_features = (
        _build_ablation_features(
            texts=(
                development_train[
                    "text"
                ]
            ),
            classifier_predictions=(
                train_predictions
            ),
            classifier_probabilities=(
                train_probabilities
            ),
            profile=profile,
        )
    )

    validation_features = (
        _build_ablation_features(
            texts=(
                development_validation[
                    "text"
                ]
            ),
            classifier_predictions=(
                validation_predictions
            ),
            classifier_probabilities=(
                validation_probabilities
            ),
            profile=profile,
        )
    )

    test_features = (
        _build_ablation_features(
            texts=(
                test_data[
                    "text"
                ]
            ),
            classifier_predictions=(
                test_predictions
            ),
            classifier_probabilities=(
                test_probabilities
            ),
            profile=profile,
        )
    )


    # ---------------------------------------------------------------
    # Beneficial-deferral target
    # ---------------------------------------------------------------

    train_target = (
        _build_beneficial_deferral_target(
            true_labels=(
                development_train[
                    "label"
                ].to_numpy()
            ),
            classifier_predictions=(
                train_predictions
            ),
            expert_predictions=(
                train_expert_predictions
            ),
        )
    )


    # ---------------------------------------------------------------
    # Variants
    # ---------------------------------------------------------------

    variant_results = {}

    for variant_key in (
        ABLATION_VARIANTS
    ):

        variant_results[
            variant_key
        ] = _run_variant(
            variant_key=(
                variant_key
            ),

            train_features=(
                train_features
            ),

            validation_features=(
                validation_features
            ),

            test_features=(
                test_features
            ),

            train_target=(
                train_target
            ),

            validation_labels=(
                development_validation[
                    "label"
                ].to_numpy()
            ),

            validation_classifier_predictions=(
                validation_predictions
            ),

            validation_expert_predictions=(
                validation_expert_predictions
            ),

            test_labels=(
                test_data[
                    "label"
                ].to_numpy()
            ),

            test_classifier_predictions=(
                test_predictions
            ),

            test_expert_predictions=(
                test_expert_predictions
            ),
        )


    # ---------------------------------------------------------------
    # Compare against full model
    # ---------------------------------------------------------------

    full_accuracy = (
        variant_results[
            "full"
        ][
            "team_accuracy"
        ]
    )

    for result in (
        variant_results.values()
    ):

        result[
            "accuracy_difference_from_full"
        ] = float(
            result[
                "team_accuracy"
            ]
            - full_accuracy
        )


    best_variant_key = max(
        variant_results,
        key=lambda key: (
            variant_results[
                key
            ][
                "team_accuracy"
            ],
            -variant_results[
                key
            ][
                "deferral_rate"
            ],
        ),
    )

    return {
        "name": (
            profile.name
        ),

        "best_variant_key": (
            best_variant_key
        ),

        "best_variant_name": (
            variant_results[
                best_variant_key
            ][
                "name"
            ]
        ),

        "variants": (
            variant_results
        ),
    }


# ---------------------------------------------------------------------------
# Complete ablation study
# ---------------------------------------------------------------------------

def run_ablation_study():

    dataset = load_ag_news()

    # ---------------------------------------------------------------
    # Split official training data
    # ---------------------------------------------------------------

    (
        classifier_train,
        meta_pool,
    ) = train_test_split(
        dataset.train,
        test_size=META_POOL_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataset.train["label"],
    )

    (
        development_train,
        development_validation,
    ) = train_test_split(
        meta_pool,
        test_size=META_VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=meta_pool["label"],
    )

    # ---------------------------------------------------------------
    # Train development classifier once
    # ---------------------------------------------------------------

    classifier = build_baseline_pipeline()

    classifier.fit(
        classifier_train["text"],
        classifier_train["label"],
    )

    # ---------------------------------------------------------------
    # Run ablation for every simulated expert
    # ---------------------------------------------------------------

    expert_results = {}

    for (
        profile_key,
        profile,
    ) in EXPERT_PROFILES.items():

        expert_results[
            profile_key
        ] = _run_ablation_for_expert(
            profile=profile,
            development_train=development_train,
            development_validation=development_validation,
            test_data=dataset.test,
            classifier=classifier,
        )

    # ---------------------------------------------------------------
    # Final result
    # ---------------------------------------------------------------

    return {
        "description": (
            "Feature ablation for the learned competence-aware "
            "deferral model."
        ),

        "variants": {
            key: {
                "name": value["name"],
                "description": value["description"],
            }
            for (
                key,
                value,
            ) in ABLATION_VARIANTS.items()
        },

        "experts": expert_results,
    }