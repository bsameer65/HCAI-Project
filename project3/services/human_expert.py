
from __future__ import annotations

from functools import lru_cache
import random

import numpy as np

from .baseline import load_baseline_model
from .data_loader import load_ag_news


# ---------------------------------------------------------------------------
# AG News classes
# ---------------------------------------------------------------------------

CLASS_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


# ---------------------------------------------------------------------------
# Human query strategies
# ---------------------------------------------------------------------------

HUMAN_QUERY_STRATEGIES = {
    "random": {
        "name": "Random Sampling",
        "description": (
            "Selects articles randomly. This provides a transparent "
            "baseline for comparison with uncertainty-based querying."
        ),
    },

    "classifier_entropy": {
        "name": "Classifier Entropy",
        "description": (
            "Selects articles for which the classifier is most uncertain."
        ),
    },

    "balanced_entropy": {
        "name": "Balanced Entropy",
        "description": (
            "Combines classifier uncertainty with predicted-category "
            "coverage so that queries are not concentrated in only one class."
        ),
    },
}


QUERY_COUNT_OPTIONS = [
    8,
    12,
    20,
    40,
]


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------

def calculate_confidence(probabilities):
    """
    Maximum predicted class probability.
    """

    probabilities = np.asarray(
        probabilities
    )

    return np.max(
        probabilities,
        axis=1,
    )


def calculate_entropy(probabilities):
    """
    Predictive entropy.

    Larger values indicate greater classifier uncertainty.
    """

    probabilities = np.asarray(
        probabilities
    )

    safe_probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    return -(
        safe_probabilities
        * np.log(
            safe_probabilities
        )
    ).sum(
        axis=1
    )


# ---------------------------------------------------------------------------
# Expensive preparation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def prepare_human_expert_pool():
    """
    Prepare the AG News test pool for the Human Expert interface.

    The trained baseline model is reused rather than retrained.

    This function is cached once per Django process, so classifier
    predictions, confidence values and entropy are calculated only once.
    """

    # ---------------------------------------------------------------
    # Load locally cached AG News
    # ---------------------------------------------------------------

    dataset = load_ag_news()


    # ---------------------------------------------------------------
    # Reuse trained baseline classifier
    # ---------------------------------------------------------------

    classifier = load_baseline_model()

    if classifier is None:
        raise RuntimeError(
            "The baseline classifier has not been trained yet. "
            "Please run the Baseline experiment before starting "
            "a Human Expert session."
        )


    # ---------------------------------------------------------------
    # Run inference once on the test pool
    # ---------------------------------------------------------------

    probabilities = classifier.predict_proba(
        dataset.test["text"]
    )

    predictions = classifier.predict(
        dataset.test["text"]
    )


    # ---------------------------------------------------------------
    # AI uncertainty information
    # ---------------------------------------------------------------

    confidence = calculate_confidence(
        probabilities
    )

    entropy = calculate_entropy(
        probabilities
    )


    # ---------------------------------------------------------------
    # Build Human Expert query pool
    # ---------------------------------------------------------------

    dataframe = (
        dataset.test
        .copy()
        .reset_index(drop=True)
    )

    dataframe[
        "classifier_prediction"
    ] = predictions

    dataframe[
        "classifier_confidence"
    ] = confidence

    dataframe[
        "classifier_entropy"
    ] = entropy


    return dataframe


# ---------------------------------------------------------------------------
# Query selection
# ---------------------------------------------------------------------------

def select_human_query_indices(
    dataframe,
    strategy_key,
    query_count,
    random_state=42,
):
    """
    Select examples that the participant will annotate.
    """

    if strategy_key not in HUMAN_QUERY_STRATEGIES:
        raise ValueError(
            "Unknown Human Expert query strategy."
        )

    query_count = int(
        query_count
    )

    if query_count <= 0:
        raise ValueError(
            "Query count must be positive."
        )

    query_count = min(
        query_count,
        len(dataframe),
    )

    # ---------------------------------------------------------------
    # Random baseline
    # ---------------------------------------------------------------

    if strategy_key == "random":

        rng = random.Random(
            random_state
        )

        return rng.sample(
            range(
                len(dataframe)
            ),
            query_count,
        )

    # ---------------------------------------------------------------
    # Highest classifier entropy
    # ---------------------------------------------------------------

    if strategy_key == "classifier_entropy":

        ranked_indices = (
            dataframe[
                "classifier_entropy"
            ]
            .sort_values(
                ascending=False
            )
            .index
            .tolist()
        )

        return ranked_indices[
            :query_count
        ]

    # ---------------------------------------------------------------
    # Balanced entropy
    # ---------------------------------------------------------------

    if strategy_key == "balanced_entropy":

        return _select_balanced_entropy(
            dataframe=dataframe,
            query_count=query_count,
        )

    raise ValueError(
        f"Unknown query strategy: {strategy_key}"
    )


def _select_balanced_entropy(
    dataframe,
    query_count,
):
    """
    Select uncertain articles while encouraging representation of all
    predicted AG News classes.
    """

    class_ids = sorted(
        dataframe[
            "classifier_prediction"
        ].unique()
    )

    selected = []

    base_count = (
        query_count
        // len(
            class_ids
        )
    )

    remainder = (
        query_count
        % len(
            class_ids
        )
    )

    for position, class_id in enumerate(
        class_ids
    ):

        amount = base_count

        if position < remainder:
            amount += 1

        subset = dataframe[
            dataframe[
                "classifier_prediction"
            ]
            == class_id
        ]

        ranked = (
            subset[
                "classifier_entropy"
            ]
            .sort_values(
                ascending=False
            )
            .index
            .tolist()
        )

        selected.extend(
            ranked[
                :amount
            ]
        )

    # In case a predicted class had too few examples,
    # fill remaining places using highest entropy globally.
    if len(selected) < query_count:

        remaining = (
            dataframe[
                ~dataframe.index.isin(
                    selected
                )
            ][
                "classifier_entropy"
            ]
            .sort_values(
                ascending=False
            )
            .index
            .tolist()
        )

        required = (
            query_count
            - len(
                selected
            )
        )

        selected.extend(
            remaining[
                :required
            ]
        )

    return selected[
        :query_count
    ]


# ---------------------------------------------------------------------------
# Human competence summary
# ---------------------------------------------------------------------------

def calculate_human_competence(
    responses,
):
    """
    Calculate descriptive competence statistics for one human session.
    """

    total = responses.count()

    if total == 0:

        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "accuracy_percent": 0.0,
            "categories": [],
            "strongest_category": None,
            "weakest_category": None,
        }

    correct = responses.filter(
        is_correct=True
    ).count()

    accuracy = (
        correct
        / total
    )

    categories = []

    for class_id, class_name in CLASS_NAMES.items():

        category_responses = responses.filter(
            true_label=class_id
        )

        sample_count = (
            category_responses.count()
        )

        category_correct = (
            category_responses.filter(
                is_correct=True
            ).count()
        )

        if sample_count > 0:

            category_accuracy = (
                category_correct
                / sample_count
            )

            category_accuracy_percent = (
                category_accuracy
                * 100
            )

        else:

            category_accuracy = None

            category_accuracy_percent = None

        categories.append(
            {
                "class_id": class_id,
                "name": class_name,
                "samples": sample_count,
                "correct": category_correct,
                "accuracy": category_accuracy,
                "accuracy_percent": (
                    category_accuracy_percent
                ),
            }
        )

    observed_categories = [
        category
        for category in categories
        if category[
            "accuracy"
        ] is not None
    ]

    strongest_category = None
    weakest_category = None

    if observed_categories:

        strongest_category = max(
            observed_categories,
            key=lambda category: (
                category[
                    "accuracy"
                ],
                category[
                    "samples"
                ],
            ),
        )

        weakest_category = min(
            observed_categories,
            key=lambda category: (
                category[
                    "accuracy"
                ],
                -category[
                    "samples"
                ],
            ),
        )

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_percent": (
            accuracy
            * 100
        ),
        "categories": categories,
        "strongest_category": (
            strongest_category
        ),
        "weakest_category": (
            weakest_category
        ),
    }


# ---------------------------------------------------------------------------
# Retrospective query statistics
# ---------------------------------------------------------------------------

def calculate_query_statistics(
    responses,
):
    """
    AI information is analysed only after annotation has finished.

    This preserves participant independence while still providing
    transparency after the experiment.
    """

    response_list = list(
        responses
    )

    if not response_list:

        return {
            "average_confidence": 0.0,
            "average_confidence_percent": 0.0,
            "average_entropy": 0.0,
            "classifier_accuracy": 0.0,
            "classifier_accuracy_percent": 0.0,
        }

    confidence_values = [
        response.classifier_confidence
        for response in response_list
        if response.classifier_confidence
        is not None
    ]

    entropy_values = [
        response.classifier_entropy
        for response in response_list
        if response.classifier_entropy
        is not None
    ]

    classifier_correct = [
        int(
            response.classifier_prediction
            == response.true_label
        )
        for response in response_list
        if response.classifier_prediction
        is not None
    ]

    average_confidence = (
        float(
            np.mean(
                confidence_values
            )
        )
        if confidence_values
        else 0.0
    )

    average_entropy = (
        float(
            np.mean(
                entropy_values
            )
        )
        if entropy_values
        else 0.0
    )

    classifier_accuracy = (
        float(
            np.mean(
                classifier_correct
            )
        )
        if classifier_correct
        else 0.0
    )

    return {
        "average_confidence": (
            average_confidence
        ),

        "average_confidence_percent": (
            average_confidence
            * 100
        ),

        "average_entropy": (
            average_entropy
        ),

        "classifier_accuracy": (
            classifier_accuracy
        ),

        "classifier_accuracy_percent": (
            classifier_accuracy
            * 100
        ),
    }