
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .active_learning import (
    load_active_learning_results,
)
from .baseline import MODEL_PATH
from .data_loader import load_ag_news
from .learning_to_defer import (
    calculate_confidence,
    calculate_entropy,
    calculate_margin,
)
from .simulated_expert import (
    CLASS_NAMES,
    EXPERT_PROFILES,
    ExpertProfile,
    simulate_expert_predictions,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT3_DIR = Path(__file__).resolve().parent.parent

ANALYSIS_DIR = (
    PROJECT3_DIR
    / "artifacts"
    / "analysis"
)

ANALYSIS_FIGURE_DIR = (
    PROJECT3_DIR
    / "static"
    / "project3"
    / "figures"
    / "advanced"
)

ANALYSIS_RESULT_PATH = (
    ANALYSIS_DIR
    / "advanced_analysis.json"
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

CONFIDENCE_THRESHOLDS = np.arange(
    0.30,
    0.96,
    0.025,
)

CALIBRATION_BINS = 10

STABILITY_SEEDS = [
    21,
    42,
    84,
    123,
    2026,
]


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _ensure_directories() -> None:
    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANALYSIS_FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def _load_baseline_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained baseline model could not be found. "
            "Run the Baseline experiment first."
        )

    return joblib.load(
        MODEL_PATH
    )


def _figure_url(
    filename: str,
) -> str:
    """
    Return path relative to Django static/project3.
    """

    return (
        "project3/figures/advanced/"
        + filename
    )


# ---------------------------------------------------------------------------
# Expert predictions
# ---------------------------------------------------------------------------

def _simulate_test_expert(
    profile: ExpertProfile,
    texts,
    true_labels,
    seed_offset: int = 900,
):
    """
    Generate deterministic expert outputs for advanced analysis.
    """

    analysis_profile = replace(
        profile,
        random_state=(
            profile.random_state
            + seed_offset
        ),
    )

    return simulate_expert_predictions(
        texts=texts,
        true_labels=true_labels,
        profile=analysis_profile,
    )


# ---------------------------------------------------------------------------
# 1. Complementarity
# ---------------------------------------------------------------------------

def calculate_complementarity(
    true_labels: np.ndarray,
    classifier_predictions: np.ndarray,
    expert_predictions: np.ndarray,
) -> dict:
    """
    Divide samples into four human-AI correctness combinations.
    """

    classifier_correct = (
        classifier_predictions
        == true_labels
    )

    expert_correct = (
        expert_predictions
        == true_labels
    )

    both_correct = int(
        np.sum(
            classifier_correct
            & expert_correct
        )
    )

    ai_only_correct = int(
        np.sum(
            classifier_correct
            & ~expert_correct
        )
    )

    expert_only_correct = int(
        np.sum(
            ~classifier_correct
            & expert_correct
        )
    )

    both_wrong = int(
        np.sum(
            ~classifier_correct
            & ~expert_correct
        )
    )

    total = len(
        true_labels
    )

    return {
        "both_correct": both_correct,
        "ai_only_correct": ai_only_correct,
        "expert_only_correct": expert_only_correct,
        "both_wrong": both_wrong,

        "both_correct_percent": (
            both_correct / total * 100
        ),

        "ai_only_correct_percent": (
            ai_only_correct / total * 100
        ),

        "expert_only_correct_percent": (
            expert_only_correct / total * 100
        ),

        "both_wrong_percent": (
            both_wrong / total * 100
        ),

        "complementary_opportunities": (
            expert_only_correct
        ),

        "sample_count": total,
    }


def plot_complementarity(
    expert_name: str,
    metrics: dict,
    filename: str,
) -> None:
    labels = [
        "Both correct",
        "AI only correct",
        "Expert only correct",
        "Both wrong",
    ]

    values = [
        metrics["both_correct"],
        metrics["ai_only_correct"],
        metrics["expert_only_correct"],
        metrics["both_wrong"],
    ]

    fig, ax = plt.subplots(
        figsize=(8, 4.8)
    )

    ax.bar(
        labels,
        values,
    )

    ax.set_ylabel(
        "Number of test articles"
    )

    ax.set_title(
        f"Human-AI Complementarity — {expert_name}"
    )

    ax.tick_params(
        axis="x",
        rotation=18,
    )

    fig.tight_layout()

    fig.savefig(
        ANALYSIS_FIGURE_DIR
        / filename,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Oracle gap / gain capture
# ---------------------------------------------------------------------------

def calculate_oracle_gap(
    true_labels: np.ndarray,
    classifier_predictions: np.ndarray,
    expert_predictions: np.ndarray,
    team_predictions: np.ndarray,
) -> dict:
    classifier_accuracy = accuracy_score(
        true_labels,
        classifier_predictions,
    )

    expert_accuracy = accuracy_score(
        true_labels,
        expert_predictions,
    )

    team_accuracy = accuracy_score(
        true_labels,
        team_predictions,
    )

    classifier_correct = (
        classifier_predictions
        == true_labels
    )

    expert_correct = (
        expert_predictions
        == true_labels
    )

    oracle_accuracy = float(
        np.mean(
            classifier_correct
            | expert_correct
        )
    )

    available_gain = (
        oracle_accuracy
        - classifier_accuracy
    )

    captured_gain = (
        team_accuracy
        - classifier_accuracy
    )

    if available_gain > 0:
        gain_capture_ratio = (
            captured_gain
            / available_gain
        )
    else:
        gain_capture_ratio = 0.0

    return {
        "classifier_accuracy": float(
            classifier_accuracy
        ),
        "expert_accuracy": float(
            expert_accuracy
        ),
        "team_accuracy": float(
            team_accuracy
        ),
        "oracle_accuracy": float(
            oracle_accuracy
        ),
        "available_gain": float(
            available_gain
        ),
        "captured_gain": float(
            captured_gain
        ),
        "gain_capture_ratio": float(
            gain_capture_ratio
        ),
    }


# ---------------------------------------------------------------------------
# 3. Coverage-accuracy trade-off
# ---------------------------------------------------------------------------

def calculate_coverage_accuracy_curve(
    true_labels: np.ndarray,
    classifier_predictions: np.ndarray,
    classifier_probabilities: np.ndarray,
    expert_predictions: np.ndarray,
) -> list[dict]:
    confidence = calculate_confidence(
        classifier_probabilities
    )

    points = []

    for threshold in CONFIDENCE_THRESHOLDS:
        defer_mask = (
            confidence
            < threshold
        )

        team_predictions = np.where(
            defer_mask,
            expert_predictions,
            classifier_predictions,
        )

        points.append(
            {
                "threshold": float(
                    threshold
                ),
                "deferral_rate": float(
                    defer_mask.mean()
                ),
                "deferral_percent": float(
                    defer_mask.mean()
                    * 100
                ),
                "team_accuracy": float(
                    accuracy_score(
                        true_labels,
                        team_predictions,
                    )
                ),
            }
        )

    return points


def plot_coverage_accuracy(
    expert_name: str,
    points: list[dict],
    classifier_accuracy: float,
    filename: str,
) -> None:
    x = [
        point[
            "deferral_percent"
        ]
        for point in points
    ]

    y = [
        point[
            "team_accuracy"
        ]
        for point in points
    ]

    fig, ax = plt.subplots(
        figsize=(7.5, 5)
    )

    ax.plot(
        x,
        y,
        marker="o",
        markersize=3,
    )

    ax.axhline(
        classifier_accuracy,
        linestyle="--",
        label="Classifier only",
    )

    ax.set_xlabel(
        "Articles deferred to expert (%)"
    )

    ax.set_ylabel(
        "Team accuracy"
    )

    ax.set_title(
        f"Coverage–Accuracy Trade-off — {expert_name}"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        ANALYSIS_FIGURE_DIR
        / filename,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Calibration
# ---------------------------------------------------------------------------

def calculate_calibration(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> dict:
    confidence = calculate_confidence(
        probabilities
    )

    correct = (
        predictions
        == true_labels
    ).astype(float)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    rows = []

    ece = 0.0

    for index in range(
        n_bins
    ):
        lower = bin_edges[
            index
        ]

        upper = bin_edges[
            index + 1
        ]

        if index == (
            n_bins - 1
        ):
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        mean_confidence = float(
            confidence[
                mask
            ].mean()
        )

        observed_accuracy = float(
            correct[
                mask
            ].mean()
        )

        proportion = (
            count
            / len(
                confidence
            )
        )

        ece += (
            proportion
            * abs(
                observed_accuracy
                - mean_confidence
            )
        )

        rows.append(
            {
                "lower": float(
                    lower
                ),
                "upper": float(
                    upper
                ),
                "samples": count,
                "mean_confidence": (
                    mean_confidence
                ),
                "observed_accuracy": (
                    observed_accuracy
                ),
                "gap": float(
                    observed_accuracy
                    - mean_confidence
                ),
            }
        )

    brier = brier_score_loss(
        correct,
        confidence,
    )

    return {
        "bins": rows,
        "ece": float(
            ece
        ),
        "brier_score": float(
            brier
        ),
    }


def plot_calibration(
    calibration: dict,
    filename: str,
) -> None:
    confidence = [
        row[
            "mean_confidence"
        ]
        for row in calibration[
            "bins"
        ]
    ]

    accuracy = [
        row[
            "observed_accuracy"
        ]
        for row in calibration[
            "bins"
        ]
    ]

    fig, ax = plt.subplots(
        figsize=(6.5, 6)
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    ax.plot(
        confidence,
        accuracy,
        marker="o",
        label="Classifier",
    )

    ax.set_xlabel(
        "Mean predicted confidence"
    )

    ax.set_ylabel(
        "Observed accuracy"
    )

    ax.set_title(
        "Classifier Reliability Diagram"
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        ANALYSIS_FIGURE_DIR
        / filename,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Active-learning learning curves
# ---------------------------------------------------------------------------

def plot_active_learning_curves(
    active_learning_results: dict,
) -> dict:
    """
    Generate two plots for each expert:
        - team accuracy vs query budget
        - competence AUROC vs query budget
    """

    figures = {}

    for (
        expert_key,
        expert,
    ) in active_learning_results[
        "experts"
    ].items():

        # -----------------------------------------------------------
        # Team accuracy
        # -----------------------------------------------------------

        team_filename = (
            f"active_learning_team_"
            f"{expert_key}.png"
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        for strategy in expert[
            "strategies"
        ].values():

            budgets = [
                point[
                    "query_budget"
                ]
                for point
                in strategy[
                    "learning_curve"
                ]
            ]

            accuracy = [
                point[
                    "team_accuracy"
                ]
                for point
                in strategy[
                    "learning_curve"
                ]
            ]

            ax.plot(
                budgets,
                accuracy,
                marker="o",
                label=strategy[
                    "name"
                ],
            )

        ax.set_xlabel(
            "Number of expert queries"
        )

        ax.set_ylabel(
            "Human-AI team accuracy"
        )

        ax.set_title(
            f"Active Learning — {expert['name']}"
        )

        ax.legend(
            fontsize=8
        )

        fig.tight_layout()

        fig.savefig(
            ANALYSIS_FIGURE_DIR
            / team_filename,
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)

        # -----------------------------------------------------------
        # AUROC
        # -----------------------------------------------------------

        auroc_filename = (
            f"active_learning_auroc_"
            f"{expert_key}.png"
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        for strategy in expert[
            "strategies"
        ].values():

            budgets = [
                point[
                    "query_budget"
                ]
                for point
                in strategy[
                    "learning_curve"
                ]
            ]

            auroc = [
                point[
                    "competence_auroc"
                ]
                for point
                in strategy[
                    "learning_curve"
                ]
            ]

            ax.plot(
                budgets,
                auroc,
                marker="o",
                label=strategy[
                    "name"
                ],
            )

        ax.axhline(
            0.5,
            linestyle="--",
            label="Random ranking",
        )

        ax.set_xlabel(
            "Number of expert queries"
        )

        ax.set_ylabel(
            "Competence AUROC"
        )

        ax.set_title(
            f"Expert Competence Discovery — {expert['name']}"
        )

        ax.legend(
            fontsize=8
        )

        fig.tight_layout()

        fig.savefig(
            ANALYSIS_FIGURE_DIR
            / auroc_filename,
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)

        figures[
            expert_key
        ] = {
            "team_accuracy": (
                _figure_url(
                    team_filename
                )
            ),
            "competence_auroc": (
                _figure_url(
                    auroc_filename
                )
            ),
        }

    return figures


# ---------------------------------------------------------------------------
# 6. Ablation study
# ---------------------------------------------------------------------------

ABLATION_FEATURE_SETS = {
    "full": {
        "numeric": [
            "confidence",
            "entropy",
            "margin",
            "log_text_length",
        ],
        "categorical": [
            "predicted_class",
            "expert_region",
        ],
    },

    "no_entropy": {
        "numeric": [
            "confidence",
            "margin",
            "log_text_length",
        ],
        "categorical": [
            "predicted_class",
            "expert_region",
        ],
    },

    "no_region": {
        "numeric": [
            "confidence",
            "entropy",
            "margin",
            "log_text_length",
        ],
        "categorical": [
            "predicted_class",
        ],
    },

    "uncertainty_only": {
        "numeric": [
            "confidence",
            "entropy",
            "margin",
        ],
        "categorical": [],
    },
}


def _build_ablation_features(
    texts,
    classifier_predictions,
    classifier_probabilities,
    expert_outputs,
) -> pd.DataFrame:

    confidence = calculate_confidence(
        classifier_probabilities
    )

    entropy = calculate_entropy(
        classifier_probabilities
    )

    margin = calculate_margin(
        classifier_probabilities
    )

    return pd.DataFrame(
        {
            "confidence": confidence,
            "entropy": entropy,
            "margin": margin,

            "log_text_length": np.log1p(
                [
                    len(
                        str(text).split()
                    )
                    for text in texts
                ]
            ),

            "predicted_class": [
                CLASS_NAMES[
                    int(value)
                ]
                for value
                in classifier_predictions
            ],

            "expert_region": [
                output.region
                for output
                in expert_outputs
            ],
        }
    )


def _build_ablation_model(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:

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

    return Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    solver="liblinear",
                ),
            ),
        ]
    )


def run_ablation_study(
    dataset,
    baseline_model,
    profile: ExpertProfile,
) -> list[dict]:
    """
    Evaluate which information is useful for predicting beneficial deferral.

    The official test set is used only for evaluation.
    """

    train_data, validation_data = (
        train_test_split(
            dataset.train,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=dataset.train[
                "label"
            ],
        )
    )

    train_probabilities = (
        baseline_model.predict_proba(
            train_data["text"]
        )
    )

    train_predictions = (
        baseline_model.predict(
            train_data["text"]
        )
    )

    validation_probabilities = (
        baseline_model.predict_proba(
            validation_data[
                "text"
            ]
        )
    )

    validation_predictions = (
        baseline_model.predict(
            validation_data[
                "text"
            ]
        )
    )

    train_expert = (
        _simulate_test_expert(
            profile,
            train_data[
                "text"
            ],
            train_data[
                "label"
            ],
            seed_offset=1100,
        )
    )

    validation_expert = (
        _simulate_test_expert(
            profile,
            validation_data[
                "text"
            ],
            validation_data[
                "label"
            ],
            seed_offset=1200,
        )
    )

    train_features = (
        _build_ablation_features(
            train_data["text"],
            train_predictions,
            train_probabilities,
            train_expert,
        )
    )

    validation_features = (
        _build_ablation_features(
            validation_data[
                "text"
            ],
            validation_predictions,
            validation_probabilities,
            validation_expert,
        )
    )

    train_true = train_data[
        "label"
    ].to_numpy()

    validation_true = (
        validation_data[
            "label"
        ].to_numpy()
    )

    train_expert_predictions = (
        np.asarray(
            [
                output.prediction
                for output
                in train_expert
            ]
        )
    )

    validation_expert_predictions = (
        np.asarray(
            [
                output.prediction
                for output
                in validation_expert
            ]
        )
    )

    train_target = (
        (
            train_predictions
            != train_true
        )
        & (
            train_expert_predictions
            == train_true
        )
    ).astype(int)

    validation_target = (
        (
            validation_predictions
            != validation_true
        )
        & (
            validation_expert_predictions
            == validation_true
        )
    ).astype(int)

    results = []

    for (
        key,
        configuration,
    ) in ABLATION_FEATURE_SETS.items():

        model = (
            _build_ablation_model(
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
        )

        model.fit(
            train_features,
            train_target,
        )

        probability = (
            model.predict_proba(
                validation_features
            )[:, 1]
        )

        # Simple 0.5 deferral probability threshold.
        defer_mask = (
            probability
            >= 0.5
        )

        team_predictions = np.where(
            defer_mask,
            validation_expert_predictions,
            validation_predictions,
        )

        team_accuracy = accuracy_score(
            validation_true,
            team_predictions,
        )

        results.append(
            {
                "key": key,
                "name": (
                    key
                    .replace(
                        "_",
                        " ",
                    )
                    .title()
                ),
                "features": (
                    configuration
                ),
                "team_accuracy": float(
                    team_accuracy
                ),
                "deferral_rate": float(
                    defer_mask.mean()
                ),
                "positive_target_rate": (
                    float(
                        validation_target.mean()
                    )
                ),
            }
        )

    return results


def plot_ablation(
    results: list[dict],
    filename: str,
) -> None:

    names = [
        item["name"]
        for item in results
    ]

    accuracy = [
        item[
            "team_accuracy"
        ]
        for item in results
    ]

    fig, ax = plt.subplots(
        figsize=(8, 4.8)
    )

    ax.bar(
        names,
        accuracy,
    )

    ax.set_ylabel(
        "Validation team accuracy"
    )

    ax.set_title(
        "Competence-Aware Deferral Ablation"
    )

    ax.tick_params(
        axis="x",
        rotation=18,
    )

    fig.tight_layout()

    fig.savefig(
        ANALYSIS_FIGURE_DIR
        / filename,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Multi-seed stability
# ---------------------------------------------------------------------------

def run_seed_stability(
    dataset,
    model,
    profile: ExpertProfile,
) -> dict:
    """
    Repeat confidence-threshold collaboration under multiple simulated
    expert seeds.

    This tests whether the observed collaboration gain depends strongly
    on one lucky expert simulation.
    """

    true_labels = (
        dataset.test[
            "label"
        ].to_numpy()
    )

    classifier_predictions = (
        model.predict(
            dataset.test[
                "text"
            ]
        )
    )

    classifier_probabilities = (
        model.predict_proba(
            dataset.test[
                "text"
            ]
        )
    )

    confidence = calculate_confidence(
        classifier_probabilities
    )

    # Use one fixed, interpretable threshold for stability comparison.
    threshold = 0.50

    defer_mask = (
        confidence
        < threshold
    )

    rows = []

    for seed in STABILITY_SEEDS:

        seeded_profile = replace(
            profile,
            random_state=seed,
        )

        expert_outputs = (
            simulate_expert_predictions(
                texts=dataset.test[
                    "text"
                ],
                true_labels=true_labels,
                profile=seeded_profile,
            )
        )

        expert_predictions = np.asarray(
            [
                output.prediction
                for output
                in expert_outputs
            ]
        )

        team_predictions = np.where(
            defer_mask,
            expert_predictions,
            classifier_predictions,
        )

        rows.append(
            {
                "seed": seed,
                "team_accuracy": float(
                    accuracy_score(
                        true_labels,
                        team_predictions,
                    )
                ),
            }
        )

    accuracy_values = np.asarray(
        [
            row[
                "team_accuracy"
            ]
            for row in rows
        ]
    )

    return {
        "threshold": threshold,
        "runs": rows,
        "mean_accuracy": float(
            accuracy_values.mean()
        ),
        "std_accuracy": float(
            accuracy_values.std(
                ddof=1
            )
        ),
        "min_accuracy": float(
            accuracy_values.min()
        ),
        "max_accuracy": float(
            accuracy_values.max()
        ),
    }


# ---------------------------------------------------------------------------
# Full advanced analysis
# ---------------------------------------------------------------------------

def run_advanced_analysis() -> dict:
    _ensure_directories()

    dataset = load_ag_news()

    baseline_model = (
        _load_baseline_model()
    )

    true_labels = (
        dataset.test[
            "label"
        ].to_numpy()
    )

    classifier_predictions = (
        baseline_model.predict(
            dataset.test[
                "text"
            ]
        )
    )

    classifier_probabilities = (
        baseline_model.predict_proba(
            dataset.test[
                "text"
            ]
        )
    )

    classifier_accuracy = float(
        accuracy_score(
            true_labels,
            classifier_predictions,
        )
    )

    # ---------------------------------------------------------------
    # Calibration
    # ---------------------------------------------------------------

    calibration = (
        calculate_calibration(
            true_labels=true_labels,
            predictions=(
                classifier_predictions
            ),
            probabilities=(
                classifier_probabilities
            ),
        )
    )

    calibration_filename = (
        "classifier_calibration.png"
    )

    plot_calibration(
        calibration,
        calibration_filename,
    )

    # ---------------------------------------------------------------
    # Expert-specific analyses
    # ---------------------------------------------------------------

    expert_results = {}

    for (
        expert_key,
        profile,
    ) in EXPERT_PROFILES.items():

        expert_outputs = (
            _simulate_test_expert(
                profile=profile,
                texts=dataset.test[
                    "text"
                ],
                true_labels=true_labels,
            )
        )

        expert_predictions = np.asarray(
            [
                output.prediction
                for output
                in expert_outputs
            ]
        )

        complementarity = (
            calculate_complementarity(
                true_labels=true_labels,
                classifier_predictions=(
                    classifier_predictions
                ),
                expert_predictions=(
                    expert_predictions
                ),
            )
        )

        complementarity_filename = (
            f"complementarity_"
            f"{expert_key}.png"
        )

        plot_complementarity(
            expert_name=profile.name,
            metrics=complementarity,
            filename=(
                complementarity_filename
            ),
        )

        curve = (
            calculate_coverage_accuracy_curve(
                true_labels=(
                    true_labels
                ),
                classifier_predictions=(
                    classifier_predictions
                ),
                classifier_probabilities=(
                    classifier_probabilities
                ),
                expert_predictions=(
                    expert_predictions
                ),
            )
        )

        curve_filename = (
            f"coverage_accuracy_"
            f"{expert_key}.png"
        )

        plot_coverage_accuracy(
            expert_name=profile.name,
            points=curve,
            classifier_accuracy=(
                classifier_accuracy
            ),
            filename=curve_filename,
        )

        best_curve_point = max(
            curve,
            key=lambda point: (
                point[
                    "team_accuracy"
                ],
                -point[
                    "deferral_rate"
                ],
            ),
        )

        confidence = (
            calculate_confidence(
                classifier_probabilities
            )
        )

        defer_mask = (
            confidence
            < best_curve_point[
                "threshold"
            ]
        )

        team_predictions = np.where(
            defer_mask,
            expert_predictions,
            classifier_predictions,
        )

        oracle_gap = (
            calculate_oracle_gap(
                true_labels=(
                    true_labels
                ),
                classifier_predictions=(
                    classifier_predictions
                ),
                expert_predictions=(
                    expert_predictions
                ),
                team_predictions=(
                    team_predictions
                ),
            )
        )

        stability = (
            run_seed_stability(
                dataset=dataset,
                model=baseline_model,
                profile=profile,
            )
        )

        expert_results[
            expert_key
        ] = {
            "name": profile.name,

            "complementarity": (
                complementarity
            ),

            "oracle_gap": (
                oracle_gap
            ),

            "coverage_accuracy": (
                curve
            ),

            "best_tradeoff": (
                best_curve_point
            ),

            "stability": (
                stability
            ),

            "figures": {
                "complementarity": (
                    _figure_url(
                        complementarity_filename
                    )
                ),

                "coverage_accuracy": (
                    _figure_url(
                        curve_filename
                    )
                ),
            },
        }

    # ---------------------------------------------------------------
    # Ablation
    # Use one expert for interpretability of feature contribution.
    # ---------------------------------------------------------------

    ablation = run_ablation_study(
        dataset=dataset,
        baseline_model=baseline_model,
        profile=EXPERT_PROFILES[
            "technology_world"
        ],
    )

    ablation_filename = (
        "deferral_ablation.png"
    )

    plot_ablation(
        results=ablation,
        filename=ablation_filename,
    )

    # ---------------------------------------------------------------
    # Existing active-learning curves
    # ---------------------------------------------------------------

    active_learning = (
        load_active_learning_results()
    )

    active_learning_figures = {}

    if active_learning is not None:
        active_learning_figures = (
            plot_active_learning_curves(
                active_learning
            )
        )

    result = {
        "classifier": {
            "accuracy": (
                classifier_accuracy
            ),

            "calibration": (
                calibration
            ),

            "calibration_figure": (
                _figure_url(
                    calibration_filename
                )
            ),
        },

        "experts": (
            expert_results
        ),

        "ablation": {
            "results": ablation,
            "figure": (
                _figure_url(
                    ablation_filename
                )
            ),
        },

        "active_learning_figures": (
            active_learning_figures
        ),
    }

    save_advanced_analysis(
        result
    )

    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_advanced_analysis(
    result: dict,
) -> None:

    _ensure_directories()

    temporary_path = (
        ANALYSIS_RESULT_PATH
        .with_suffix(
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
            ANALYSIS_RESULT_PATH
        )

    except Exception:

        if temporary_path.exists():
            temporary_path.unlink()

        raise


def load_advanced_analysis() -> dict | None:

    if not ANALYSIS_RESULT_PATH.exists():
        return None

    with ANALYSIS_RESULT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)