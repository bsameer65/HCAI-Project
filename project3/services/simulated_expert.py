"""
Simulated experts for AG News.

This module defines two imperfect expert profiles:

1. Sports and Business Specialist
2. Technology and World Affairs Specialist

Each expert has high competence only in identifiable lexical regions of the
input space and lower competence elsewhere.

The true label is used only by the simulator to decide whether the expert
returns a correct or incorrect answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import re
from typing import Iterable

from .data_loader import load_ag_news
from .evaluation import evaluate_predictions


# ---------------------------------------------------------------------------
# Paths and class labels
# ---------------------------------------------------------------------------

PROJECT3_DIR = Path(__file__).resolve().parent.parent

EXPERT_METRICS_DIR = PROJECT3_DIR / "artifacts" / "metrics"
EXPERT_METRICS_PATH = (
    EXPERT_METRICS_DIR / "simulated_experts_metrics.json"
)

CLASS_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


# ---------------------------------------------------------------------------
# Expertise-region keywords
# ---------------------------------------------------------------------------

SPORTS_KEYWORDS = {
    "athlete",
    "baseball",
    "basketball",
    "champion",
    "championship",
    "coach",
    "cricket",
    "cup",
    "football",
    "game",
    "goal",
    "league",
    "match",
    "olympic",
    "player",
    "score",
    "season",
    "soccer",
    "team",
    "tennis",
    "tournament",
    "victory",
    "win",
}


BUSINESS_KEYWORDS = {
    "bank",
    "business",
    "company",
    "corporate",
    "currency",
    "earnings",
    "economic",
    "economy",
    "financial",
    "investment",
    "investor",
    "market",
    "merger",
    "profit",
    "revenue",
    "sales",
    "shares",
    "stock",
    "trade",
}


TECHNOLOGY_KEYWORDS = {
    "application",
    "computer",
    "digital",
    "internet",
    "microsoft",
    "mobile",
    "network",
    "online",
    "processor",
    "research",
    "robot",
    "science",
    "scientist",
    "security",
    "software",
    "space",
    "technology",
    "telecom",
    "web",
    "wireless",
}


WORLD_AFFAIRS_KEYWORDS = {
    "army",
    "attack",
    "country",
    "diplomatic",
    "election",
    "foreign",
    "government",
    "international",
    "military",
    "minister",
    "nation",
    "peace",
    "political",
    "president",
    "prime",
    "security",
    "state",
    "troops",
    "war",
    "world",
}


REGION_KEYWORDS = {
    "Sports lexical region": SPORTS_KEYWORDS,
    "Business lexical region": BUSINESS_KEYWORDS,
    "Technology lexical region": TECHNOLOGY_KEYWORDS,
    "World affairs lexical region": WORLD_AFFAIRS_KEYWORDS,
}


REGION_DISPLAY_NAMES = {
    "Sports lexical region": "Sports-focused articles",
    "Business lexical region": "Business-focused articles",
    "Technology lexical region": "Technology-focused articles",
    "World affairs lexical region": "World-affairs-focused articles",
    "General region": "General articles",
}


# ---------------------------------------------------------------------------
# Expert configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExpertProfile:
    """
    Configuration for one simulated expert.

    region_accuracies maps an input-space region to the probability that the
    expert returns the correct class in that region.
    """

    key: str
    name: str
    description: str
    region_accuracies: dict[str, float]
    general_accuracy: float
    random_state: int


@dataclass(frozen=True)
class ExpertPrediction:
    """
    Result of one simulated expert query.
    """

    prediction: int
    region: str
    expected_accuracy: float
    was_correct: bool


EXPERT_PROFILES = {
    "sports_business": ExpertProfile(
        key="sports_business",
        name="Sports and Business Specialist",
        description=(
            "Highly reliable for articles with clear Sports terminology and "
            "strong for Business-oriented articles, but substantially less "
            "reliable outside these areas."
        ),
        region_accuracies={
            "Sports lexical region": 0.96,
            "Business lexical region": 0.88,
        },
        general_accuracy=0.57,
        random_state=42,
    ),
    "technology_world": ExpertProfile(
        key="technology_world",
        name="Technology and World Affairs Specialist",
        description=(
            "Highly reliable for articles containing clear technology terms "
            "and strong for international or political news, but weaker for "
            "Sports and Business articles."
        ),
        region_accuracies={
            "Technology lexical region": 0.93,
            "World affairs lexical region": 0.85,
        },
        general_accuracy=0.56,
        random_state=84,
    ),
}


# ---------------------------------------------------------------------------
# Region identification
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """
    Convert text into a lowercase token set.

    Word-level token matching avoids accidental substring matches, such as
    finding 'war' inside another word.
    """

    return set(
        re.findall(
            r"[a-z0-9]+(?:'[a-z0-9]+)?",
            str(text).lower(),
        )
    )


def count_region_matches(text: str) -> dict[str, int]:
    """
    Count keyword matches for every possible lexical region.
    """

    tokens = _tokenize(text)

    return {
        region_name: len(
            tokens.intersection(
                keyword.lower()
                for keyword in keywords
            )
        )
        for region_name, keywords in REGION_KEYWORDS.items()
    }


def identify_expertise_region(
    text: str,
    profile: ExpertProfile,
    minimum_matches: int = 2,
) -> str:
    """
    Identify the most relevant competence region for one expert.

    Only regions defined for the selected expert profile are considered.
    If no specialist region has sufficient lexical evidence, the sample is
    assigned to the General region.
    """

    match_counts = count_region_matches(text)

    specialist_scores = {
        region_name: match_counts.get(region_name, 0)
        for region_name in profile.region_accuracies
    }

    if not specialist_scores:
        return "General region"

    best_region = max(
        specialist_scores,
        key=specialist_scores.get,
    )

    if specialist_scores[best_region] >= minimum_matches:
        return best_region

    return "General region"


def get_region_accuracy(
    region: str,
    profile: ExpertProfile,
) -> float:
    """
    Return the expert correctness probability for a given region.
    """

    return profile.region_accuracies.get(
        region,
        profile.general_accuracy,
    )

# ---------------------------------------------------------------------------
# Prediction simulation
# ---------------------------------------------------------------------------

def _choose_incorrect_label(
    true_label: int,
    random_generator: random.Random,
) -> int:
    """
    Return an incorrect label using a plausible confusion pattern.

    Duplicate labels in each candidate list give related classes a greater
    chance of being selected as the expert's mistake.
    """

    confusion_options = {
        # World is more often confused with Business or Sci/Tech.
        0: [2, 2, 3, 3, 1],

        # Sports is usually distinctive, but mistakes may map elsewhere.
        1: [0, 0, 2, 3],

        # Business is commonly confused with World and Sci/Tech.
        2: [0, 0, 3, 3, 1],

        # Sci/Tech is commonly confused with Business.
        3: [2, 2, 2, 0, 1],
    }

    return random_generator.choice(
        confusion_options[int(true_label)]
    )


def simulate_single_prediction(
    text: str,
    true_label: int,
    random_generator: random.Random,
    profile: ExpertProfile,
) -> ExpertPrediction:
    """
    Generate one simulated expert prediction.
    """

    true_label = int(true_label)

    if true_label not in CLASS_NAMES:
        raise ValueError(
            f"Unexpected class label: {true_label}"
        )

    region = identify_expertise_region(
        text=text,
        profile=profile,
    )

    expected_accuracy = get_region_accuracy(
        region=region,
        profile=profile,
    )

    expert_is_correct = (
        random_generator.random() < expected_accuracy
    )

    if expert_is_correct:
        prediction = true_label
    else:
        prediction = _choose_incorrect_label(
            true_label=true_label,
            random_generator=random_generator,
        )

    return ExpertPrediction(
        prediction=prediction,
        region=region,
        expected_accuracy=expected_accuracy,
        was_correct=prediction == true_label,
    )


def simulate_expert_predictions(
    texts: Iterable[str],
    true_labels: Iterable[int],
    profile: ExpertProfile,
) -> list[ExpertPrediction]:
    """
    Generate reproducible predictions for one expert profile.
    """

    text_list = list(texts)
    label_list = list(true_labels)

    if len(text_list) != len(label_list):
        raise ValueError(
            "Texts and labels must contain the same number of samples."
        )

    if not text_list:
        raise ValueError(
            "At least one sample is required."
        )

    random_generator = random.Random(
        profile.random_state
    )

    return [
        simulate_single_prediction(
            text=text,
            true_label=true_label,
            random_generator=random_generator,
            profile=profile,
        )
        for text, true_label in zip(
            text_list,
            label_list,
        )
    ]

# ---------------------------------------------------------------------------
# Template-friendly formatting
# ---------------------------------------------------------------------------

def build_confusion_matrix_rows(
    matrix: list[list[int]],
    class_names: list[str],
) -> list[dict]:
    """
    Convert a confusion matrix into template-friendly rows.

    Each row represents the true class. Each cell represents one predicted
    class and its count.
    """

    if len(matrix) != len(class_names):
        raise ValueError(
            "Confusion matrix and class names have incompatible sizes."
        )

    rows = []

    for true_class, matrix_row in zip(
        class_names,
        matrix,
    ):
        if len(matrix_row) != len(class_names):
            raise ValueError(
                "Confusion matrix must be square."
            )

        rows.append(
            {
                "true_class": true_class,
                "cells": [
                    {
                        "predicted_class": predicted_class,
                        "count": int(count),
                    }
                    for predicted_class, count
                    in zip(class_names, matrix_row)
                ],
            }
        )

    return rows

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_one_expert(
    profile: ExpertProfile,
    texts: Iterable[str],
    true_labels: Iterable[int],
) -> dict:
    """
    Evaluate one simulated expert on a supplied dataset.
    """

    text_list = list(texts)
    label_list = [
        int(label)
        for label in true_labels
    ]

    expert_outputs = simulate_expert_predictions(
        texts=text_list,
        true_labels=label_list,
        profile=profile,
    )

    predictions = [
        output.prediction
        for output in expert_outputs
    ]

    metrics = evaluate_predictions(
        label_list,
        predictions,
    )

    return {
        "key": profile.key,
        "name": profile.name,
        "description": profile.description,
        "test_samples": len(label_list),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "classification_report": metrics[
            "classification_report"
        ],
        "confusion_matrix": metrics["confusion_matrix"],
        "confusion_matrix_rows": build_confusion_matrix_rows(
            matrix=metrics["confusion_matrix"],
            class_names=metrics["class_names"],
        ),
        "class_names": metrics["class_names"],
        "configuration": {
            **asdict(profile),
            "region_accuracies": {
                region_name: float(accuracy)
                for region_name, accuracy
                in profile.region_accuracies.items()
            },
        },
        "region_analysis": calculate_region_analysis(
            true_labels=label_list,
            expert_outputs=expert_outputs,
            profile=profile,
        ),
        "strengths": get_profile_strengths(
            profile.key
        ),
        "weaknesses": get_profile_weaknesses(
            profile.key
        ),
    }


def evaluate_all_simulated_experts() -> dict:
    """
    Evaluate all configured experts on the official AG News test set.
    """

    dataset = load_ag_news()

    texts = dataset.test["text"].tolist()
    true_labels = dataset.test["label"].tolist()

    expert_results = {
        profile_key: evaluate_one_expert(
            profile=profile,
            texts=texts,
            true_labels=true_labels,
        )
        for profile_key, profile
        in EXPERT_PROFILES.items()
    }

    best_expert_key = max(
        expert_results,
        key=lambda key: expert_results[key]["accuracy"],
    )

    for expert_key, expert_result in expert_results.items():
        expert_result["is_best"] = (
            expert_key == best_expert_key
        )

    result = {
        "test_samples": len(true_labels),
        "expert_count": len(expert_results),
        "best_expert_key": best_expert_key,
        "best_expert_name": expert_results[
            best_expert_key
        ]["name"],
        "experts": expert_results,
    }

    save_expert_results(result)

    return result

def calculate_region_analysis(
    true_labels: list[int],
    expert_outputs: list[ExpertPrediction],
    profile: ExpertProfile,
) -> dict:
    """
    Calculate coverage and observed performance for each competence region.
    """

    if len(true_labels) != len(expert_outputs):
        raise ValueError(
            "True labels and expert outputs must have equal length."
        )

    region_names = [
        *profile.region_accuracies.keys(),
        "General region",
    ]

    total_samples = len(expert_outputs)
    analysis = {}

    for region_name in region_names:
        matching_indices = [
            index
            for index, output in enumerate(expert_outputs)
            if output.region == region_name
        ]

        sample_count = len(matching_indices)

        correct_count = sum(
            expert_outputs[index].was_correct
            for index in matching_indices
        )

        observed_accuracy = (
            correct_count / sample_count
            if sample_count > 0
            else 0.0
        )

        coverage = (
            sample_count / total_samples
            if total_samples > 0
            else 0.0
        )

        configured_accuracy = get_region_accuracy(
            region=region_name,
            profile=profile,
        )

        analysis[region_name] = {
            "internal_name": region_name,
            "display_name": REGION_DISPLAY_NAMES.get(
                region_name,
                region_name,
            ),
            "samples": sample_count,
            "correct_predictions": correct_count,
            "coverage": float(coverage),
            "coverage_percent": round(
                float(coverage * 100),
                2,
            ),
            "configured_accuracy": float(
                configured_accuracy
            ),
            "configured_percent": round(
                float(configured_accuracy * 100),
                2,
            ),
            "observed_accuracy": float(
                observed_accuracy
            ),
            "observed_percent": round(
                float(observed_accuracy * 100),
                2,
            ),
        }

    return analysis