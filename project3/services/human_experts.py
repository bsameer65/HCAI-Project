"""
Utilities for the optional interactive human-expert extension.

The goal is to select informative AG News articles and estimate the
participant's competence from their submitted labels.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .baseline import build_baseline_pipeline
from .data_loader import load_ag_news
from .learning_to_defer import (
    calculate_confidence,
    calculate_entropy,
)


CLASS_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


HUMAN_QUERY_STRATEGIES = {
    "random": "Random Sampling",
    "classifier_entropy": "Classifier Entropy",
    "balanced_entropy": "Balanced Entropy",
}


def prepare_human_expert_pool() -> dict:
    """
    Train the baseline classifier and prepare query metadata for the
    interactive human-expert interface.
    """

    dataset = load_ag_news()

    classifier = build_baseline_pipeline()

    classifier.fit(
        dataset.train["text"],
        dataset.train["label"],
    )

    probabilities = classifier.predict_proba(
        dataset.test["text"]
    )

    predictions = classifier.predict(
        dataset.test["text"]
    )

    confidence = calculate_confidence(
        probabilities
    )

    entropy = calculate_entropy(
        probabilities
    )

    dataframe = dataset.test.copy()

    dataframe["classifier_prediction"] = predictions
    dataframe["classifier_confidence"] = confidence
    dataframe["classifier_entropy"] = entropy

    return {
        "data": dataframe,
    }


def select_human_query_indices(
    dataframe: pd.DataFrame,
    strategy_key: str,
    query_count: int,
    random_state: int = 42,
) -> list[int]:
    """
    Select articles for the human annotation session.

    Supported strategies:
        random
        classifier_entropy
        balanced_entropy
    """

    if query_count <= 0:
        raise ValueError(
            "Query count must be positive."
        )

    query_count = min(
        query_count,
        len(dataframe),
    )

    if strategy_key == "random":
        random_generator = random.Random(
            random_state
        )

        return random_generator.sample(
            range(len(dataframe)),
            query_count,
        )

    if strategy_key == "classifier_entropy":
        ranked = (
            dataframe["classifier_entropy"]
            .sort_values(
                ascending=False
            )
            .index
            .tolist()
        )

        return ranked[:query_count]

    if strategy_key == "balanced_entropy":
        return _select_balanced_entropy(
            dataframe=dataframe,
            query_count=query_count,
        )

    raise ValueError(
        f"Unknown human query strategy: {strategy_key}"
    )


def _select_balanced_entropy(
    dataframe: pd.DataFrame,
    query_count: int,
) -> list[int]:
    """
    Select uncertain articles while approximately balancing the classifier's
    predicted classes.

    This gives the participant exposure to multiple regions of the input
    space instead of querying only one difficult category.
    """

    class_ids = sorted(
        dataframe[
            "classifier_prediction"
        ].unique()
    )

    per_class = max(
        1,
        query_count // len(class_ids),
    )

    selected = []

    for class_id in class_ids:
        subset = dataframe[
            dataframe[
                "classifier_prediction"
            ] == class_id
        ]

        ranked = (
            subset["classifier_entropy"]
            .sort_values(
                ascending=False
            )
            .index
            .tolist()
        )

        selected.extend(
            ranked[:per_class]
        )

    if len(selected) < query_count:
        remaining = (
            dataframe[
                ~dataframe.index.isin(
                    selected
                )
            ]["classifier_entropy"]
            .sort_values(
                ascending=False
            )
            .index
            .tolist()
        )

        selected.extend(
            remaining[
                :query_count - len(selected)
            ]
        )

    return selected[:query_count]


def calculate_human_competence(
    responses,
) -> dict:
    """
    Calculate an initial competence profile from stored human annotations.

    The result is descriptive only. Small sample sizes should not be treated
    as a statistically reliable estimate of human expertise.
    """

    total = responses.count()

    if total == 0:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "categories": [],
            "strongest_category": None,
            "weakest_category": None,
        }

    correct = responses.filter(
        is_correct=True
    ).count()

    accuracy = correct / total

    category_results = []

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

        category_accuracy = (
            category_correct / sample_count
            if sample_count > 0
            else None
        )

        category_results.append(
            {
                "class_id": class_id,
                "name": class_name,
                "samples": sample_count,
                "correct": category_correct,
                "accuracy": (
                    category_accuracy
                ),
                "accuracy_percent": (
                    category_accuracy * 100
                    if category_accuracy
                    is not None
                    else None
                ),
            }
        )

    observed_categories = [
        item
        for item in category_results
        if item["accuracy"] is not None
    ]

    strongest = None
    weakest = None

    if observed_categories:
        strongest = max(
            observed_categories,
            key=lambda item: (
                item["accuracy"],
                item["samples"],
            ),
        )

        weakest = min(
            observed_categories,
            key=lambda item: (
                item["accuracy"],
                -item["samples"],
            ),
        )

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_percent": (
            accuracy * 100
        ),
        "categories": category_results,
        "strongest_category": strongest,
        "weakest_category": weakest,
    }