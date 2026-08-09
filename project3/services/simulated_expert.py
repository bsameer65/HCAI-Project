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

