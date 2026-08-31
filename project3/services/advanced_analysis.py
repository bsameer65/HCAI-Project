"""
Advanced Human-AI analysis for Project 3.

This module reuses saved experiment artifacts wherever possible instead of
retraining the baseline, learning-to-defer, or active-learning experiments.

Analyses:
1. Human-AI complementarity
2. Collaboration headroom / oracle gap
3. Coverage-accuracy trade-off
4. Classifier confidence calibration
5. Active-learning efficiency curves
6. Multi-seed expert stability

The output is persisted as JSON and figures so the same results can later
be reused in the final report.

The module also records modification timestamps of all source experiment
artifacts. This allows the interface to detect when an already-generated
Advanced Analysis has become stale because an upstream experiment was rerun.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score

from .active_learning import (
    ACTIVE_LEARNING_METRICS_PATH,
    load_active_learning_results,
)
from .baseline import (
    METRICS_PATH as BASELINE_METRICS_PATH,
    load_baseline_model,
    load_baseline_results,
)
from .data_loader import load_ag_news
from .learning_to_defer import (
    DEFER_METRICS_PATH,
    calculate_confidence,
    load_learning_to_defer_results,
)
from .simulated_expert import (
    EXPERT_METRICS_PATH,
    EXPERT_PROFILES,
    load_expert_results,
    simulate_expert_predictions,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT3_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

METRICS_DIR = (
    PROJECT3_DIR
    / "artifacts"
    / "metrics"
)

ADVANCED_METRICS_PATH = (
    METRICS_DIR
    / "advanced_analysis_metrics.json"
)

FIGURE_DIR = (
    PROJECT3_DIR
    / "static"
    / "project3"
    / "figures"
    / "advanced"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

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

def _ensure_output_directories():
    """
    Create result and figure directories when missing.
    """

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def _figure_static_path(
    filename: str,
) -> str:
    """
    Return a path suitable for Django's {% static %} tag.
    """

    return (
        "project3/"
        "figures/"
        "advanced/"
        f"{filename}"
    )


# ---------------------------------------------------------------------------
# Source artifact freshness
# ---------------------------------------------------------------------------

def get_source_artifact_timestamps() -> dict:
    """
    Return modification timestamps of all artifacts consumed by the
    Advanced Analysis.

    These timestamps are saved inside the advanced-analysis artifact.
    On later page loads they can be compared with the current artifacts to
    determine whether the saved analysis is still current.
    """

    paths = {
        "baseline": (
            BASELINE_METRICS_PATH
        ),
        "experts": (
            EXPERT_METRICS_PATH
        ),
        "learning_to_defer": (
            DEFER_METRICS_PATH
        ),
        "active_learning": (
            ACTIVE_LEARNING_METRICS_PATH
        ),
    }

    timestamps = {}

    for key, path in paths.items():

        if path.exists():

            timestamps[key] = float(
                path.stat().st_mtime
            )

        else:

            timestamps[key] = None

    return timestamps


def advanced_analysis_is_stale(
    results: dict | None,
) -> bool:
    """
    Return True when an upstream experiment changed after the current
    Advanced Analysis was generated.

    Older advanced-analysis files that do not contain source_artifacts are
    automatically considered stale.
    """

    if results is None:
        return False

    saved_sources = results.get(
        "source_artifacts"
    )

    if not saved_sources:
        return True

    current_sources = (
        get_source_artifact_timestamps()
    )

    for (
        key,
        current_timestamp,
    ) in current_sources.items():

        saved_timestamp = (
            saved_sources.get(
                key
            )
        )

        if (
            saved_timestamp
            != current_timestamp
        ):
            return True

    return False


# ---------------------------------------------------------------------------
# Required artifact validation
# ---------------------------------------------------------------------------

def validate_required_results() -> None:
    """
    Ensure core experiments have already been executed.

    Advanced Analysis analyses existing experiment artifacts instead of
    silently retraining them.
    """

    missing = []

    if (
        load_baseline_results()
        is None
    ):
        missing.append(
            "Baseline"
        )

    if (
        load_expert_results()
        is None
    ):
        missing.append(
            "Simulated Experts"
        )

    if (
        load_learning_to_defer_results()
        is None
    ):
        missing.append(
            "Learning to Defer"
        )

    if (
        load_active_learning_results()
        is None
    ):
        missing.append(
            "Active Learning"
        )

    if missing:

        raise RuntimeError(
            "Run these experiments before Advanced Analysis: "
            + ", ".join(
                missing
            )
        )

    if (
        load_baseline_model()
        is None
    ):

        raise RuntimeError(
            "The saved baseline model could not be found. "
            "Run the Baseline experiment again."
        )


# ---------------------------------------------------------------------------
# Test predictions shared across analyses
# ---------------------------------------------------------------------------

def prepare_test_predictions():
    """
    Load the saved baseline and calculate test predictions once.
    """

    dataset = load_ag_news()

    classifier = (
        load_baseline_model()
    )

    if classifier is None:

        raise RuntimeError(
            "Baseline model is unavailable."
        )

    y_true = (
        dataset.test[
            "label"
        ].to_numpy(
            dtype=int
        )
    )

    predictions = (
        classifier.predict(
            dataset.test[
                "text"
            ]
        )
    )

    probabilities = (
        classifier.predict_proba(
            dataset.test[
                "text"
            ]
        )
    )

    return (
        dataset,
        y_true,
        np.asarray(
            predictions,
            dtype=int,
        ),
        np.asarray(
            probabilities,
            dtype=float,
        ),
    )


# ---------------------------------------------------------------------------
# Expert simulation
# ---------------------------------------------------------------------------

def get_test_expert_predictions(
    dataset,
    profile,
    seed_offset: int = 303,
):
    """
    Reproduce the expert simulation used by learning-to-defer test evaluation.

    learning_to_defer.py evaluates the test expert using:

        profile.random_state + 303

    Therefore the same seed convention is preserved here so that
    complementarity and headroom calculations are directly comparable.
    """

    test_profile = replace(
        profile,
        random_state=(
            profile.random_state
            + seed_offset
        ),
    )

    outputs = (
        simulate_expert_predictions(
            texts=dataset.test[
                "text"
            ],
            true_labels=dataset.test[
                "label"
            ],
            profile=test_profile,
        )
    )

    predictions = np.asarray(
        [
            output.prediction
            for output in outputs
        ],
        dtype=int,
    )

    return predictions


# ---------------------------------------------------------------------------
# Complementarity
# ---------------------------------------------------------------------------

def calculate_complementarity(
    y_true,
    ai_predictions,
    expert_predictions,
):
    """
    Partition examples according to AI and expert correctness.
    """

    ai_correct = (
        ai_predictions
        == y_true
    )

    expert_correct = (
        expert_predictions
        == y_true
    )

    both_correct = int(
        np.sum(
            ai_correct
            & expert_correct
        )
    )

    ai_only_correct = int(
        np.sum(
            ai_correct
            & ~expert_correct
        )
    )

    expert_only_correct = int(
        np.sum(
            ~ai_correct
            & expert_correct
        )
    )

    both_wrong = int(
        np.sum(
            ~ai_correct
            & ~expert_correct
        )
    )

    sample_count = int(
        len(y_true)
    )

    return {
        "sample_count": sample_count,

        "both_correct": (
            both_correct
        ),

        "ai_only_correct": (
            ai_only_correct
        ),

        "expert_only_correct": (
            expert_only_correct
        ),

        "both_wrong": (
            both_wrong
        ),

        "both_correct_percent": (
            both_correct
            / sample_count
            * 100
        ),

        "ai_only_correct_percent": (
            ai_only_correct
            / sample_count
            * 100
        ),

        "expert_only_correct_percent": (
            expert_only_correct
            / sample_count
            * 100
        ),

        "both_wrong_percent": (
            both_wrong
            / sample_count
            * 100
        ),
    }


def plot_complementarity(
    expert_name,
    result,
    filename,
):
    """
    Generate complementarity bar chart.
    """

    labels = [
        "Both correct",
        "AI only",
        "Expert only",
        "Both wrong",
    ]

    values = [
        result[
            "both_correct"
        ],
        result[
            "ai_only_correct"
        ],
        result[
            "expert_only_correct"
        ],
        result[
            "both_wrong"
        ],
    ]

    figure, axis = plt.subplots(
        figsize=(8, 4.8)
    )

    bars = axis.bar(
        labels,
        values,
    )

    axis.set_ylabel(
        "Test articles"
    )

    axis.set_title(
        "Human-AI Complementarity\n"
        f"{expert_name}"
    )

    axis.tick_params(
        axis="x",
        rotation=12,
    )

    for bar, value in zip(
        bars,
        values,
    ):

        axis.text(
            (
                bar.get_x()
                + bar.get_width()
                / 2
            ),
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIR
        / filename,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ---------------------------------------------------------------------------
# Collaboration headroom
# ---------------------------------------------------------------------------

def calculate_collaboration_headroom(
    y_true,
    ai_predictions,
    expert_predictions,
    best_team_metrics,
):
    """
    Compare achieved team gain with theoretical oracle gain.
    """

    ai_accuracy = float(
        accuracy_score(
            y_true,
            ai_predictions,
        )
    )

    expert_accuracy = float(
        accuracy_score(
            y_true,
            expert_predictions,
        )
    )

    ai_correct = (
        ai_predictions
        == y_true
    )

    expert_correct = (
        expert_predictions
        == y_true
    )

    oracle_accuracy = float(
        np.mean(
            ai_correct
            | expert_correct
        )
    )

    team_accuracy = float(
        best_team_metrics[
            "team_accuracy"
        ]
    )

    possible_gain = (
        oracle_accuracy
        - ai_accuracy
    )

    captured_gain = (
        team_accuracy
        - ai_accuracy
    )

    if possible_gain > 0:

        gain_capture_ratio = (
            captured_gain
            / possible_gain
        )

    else:

        gain_capture_ratio = 0.0

    return {
        "ai_accuracy": (
            ai_accuracy
        ),

        "expert_accuracy": (
            expert_accuracy
        ),

        "team_accuracy": (
            team_accuracy
        ),

        "oracle_accuracy": (
            oracle_accuracy
        ),

        "possible_gain": float(
            possible_gain
        ),

        "captured_gain": float(
            captured_gain
        ),

        "gain_capture_ratio": float(
            gain_capture_ratio
        ),

        "gain_capture_percent": float(
            gain_capture_ratio
            * 100
        ),
    }


# ---------------------------------------------------------------------------
# Coverage-accuracy trade-off
# ---------------------------------------------------------------------------

def prepare_coverage_accuracy(
    defer_results,
):
    """
    Reuse threshold searches stored by learning-to-defer.
    """

    output = {}

    for (
        expert_key,
        expert_result,
    ) in defer_results[
        "experts"
    ].items():

        output[
            expert_key
        ] = {
            "name": (
                expert_result[
                    "name"
                ]
            ),

            "confidence": (
                expert_result[
                    "confidence_strategy"
                ][
                    "threshold_search"
                ]
            ),

            "learned": (
                expert_result[
                    "learned_strategy"
                ][
                    "threshold_search"
                ]
            ),
        }

    return output


def plot_coverage_accuracy(
    expert_name,
    confidence_points,
    learned_points,
    classifier_accuracy,
    filename,
):
    """
    Plot team accuracy against expert deferral rate.
    """

    figure, axis = plt.subplots(
        figsize=(7.8, 5)
    )

    confidence_x = [
        point[
            "deferral_rate"
        ]
        * 100
        for point
        in confidence_points
    ]

    confidence_y = [
        point[
            "team_accuracy"
        ]
        for point
        in confidence_points
    ]

    learned_x = [
        point[
            "deferral_rate"
        ]
        * 100
        for point
        in learned_points
    ]

    learned_y = [
        point[
            "team_accuracy"
        ]
        for point
        in learned_points
    ]

    axis.plot(
        confidence_x,
        confidence_y,
        marker="o",
        label=(
            "Confidence threshold"
        ),
    )

    axis.plot(
        learned_x,
        learned_y,
        marker="o",
        label=(
            "Competence-aware"
        ),
    )

    axis.axhline(
        classifier_accuracy,
        linestyle="--",
        label="Classifier only",
    )

    axis.set_xlabel(
        "Deferred to expert (%)"
    )

    axis.set_ylabel(
        "Team accuracy"
    )

    axis.set_title(
        "Coverage-Accuracy Trade-off\n"
        f"{expert_name}"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIR
        / filename,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------

def calculate_calibration(
    y_true,
    predictions,
    probabilities,
):
    """
    Reliability analysis for top-class classifier confidence.
    """

    confidence = (
        calculate_confidence(
            probabilities
        )
    )

    correctness = (
        predictions
        == y_true
    ).astype(float)

    edges = np.linspace(
        0.0,
        1.0,
        CALIBRATION_BINS
        + 1,
    )

    rows = []

    ece = 0.0

    for index in range(
        CALIBRATION_BINS
    ):

        lower = edges[
            index
        ]

        upper = edges[
            index + 1
        ]

        if (
            index
            == CALIBRATION_BINS - 1
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
            correctness[
                mask
            ].mean()
        )

        weight = (
            count
            / len(confidence)
        )

        absolute_gap = abs(
            observed_accuracy
            - mean_confidence
        )

        ece += (
            weight
            * absolute_gap
        )

        rows.append(
            {
                "lower": float(
                    lower
                ),

                "upper": float(
                    upper
                ),

                "count": count,

                "mean_confidence": (
                    mean_confidence
                ),

                "observed_accuracy": (
                    observed_accuracy
                ),

                "absolute_gap": float(
                    absolute_gap
                ),
            }
        )

    return {
        "ece": float(
            ece
        ),
        "bins": rows,
    }


def plot_calibration(
    calibration,
    filename,
):
    """
    Generate reliability diagram.
    """

    confidence = [
        row[
            "mean_confidence"
        ]
        for row
        in calibration[
            "bins"
        ]
    ]

    accuracy = [
        row[
            "observed_accuracy"
        ]
        for row
        in calibration[
            "bins"
        ]
    ]

    figure, axis = plt.subplots(
        figsize=(6.2, 6)
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    axis.plot(
        confidence,
        accuracy,
        marker="o",
        label="Baseline classifier",
    )

    axis.set_xlim(
        0,
        1,
    )

    axis.set_ylim(
        0,
        1,
    )

    axis.set_xlabel(
        "Mean predicted confidence"
    )

    axis.set_ylabel(
        "Observed accuracy"
    )

    axis.set_title(
        "Classifier Reliability Diagram"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIR
        / filename,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ---------------------------------------------------------------------------
# Active-learning efficiency
# ---------------------------------------------------------------------------

def generate_active_learning_figures(
    active_results,
):
    """
    Generate learning-curve figures from saved active-learning checkpoints.
    """

    figures = {}

    for (
        expert_key,
        expert_result,
    ) in active_results[
        "experts"
    ].items():

        # -----------------------------------------------------------
        # Team accuracy
        # -----------------------------------------------------------

        team_filename = (
            "active_team_"
            f"{expert_key}.png"
        )

        figure, axis = plt.subplots(
            figsize=(8, 5)
        )

        for strategy in (
            expert_result[
                "strategies"
            ].values()
        ):

            points = (
                strategy[
                    "learning_curve"
                ]
            )

            budgets = [
                point[
                    "query_budget"
                ]
                for point
                in points
            ]

            team_accuracy = [
                point[
                    "team_accuracy"
                ]
                for point
                in points
            ]

            axis.plot(
                budgets,
                team_accuracy,
                marker="o",
                label=(
                    strategy[
                        "name"
                    ]
                ),
            )

        axis.set_xlabel(
            "Expert queries"
        )

        axis.set_ylabel(
            "Human-AI team accuracy"
        )

        axis.set_title(
            "Active-Learning Team Performance\n"
            f"{expert_result['name']}"
        )

        axis.legend(
            fontsize=8
        )

        figure.tight_layout()

        figure.savefig(
            FIGURE_DIR
            / team_filename,
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )


        # -----------------------------------------------------------
        # Competence AUROC
        # -----------------------------------------------------------

        auroc_filename = (
            "active_auroc_"
            f"{expert_key}.png"
        )

        figure, axis = plt.subplots(
            figsize=(8, 5)
        )

        for strategy in (
            expert_result[
                "strategies"
            ].values()
        ):

            points = (
                strategy[
                    "learning_curve"
                ]
            )

            usable_points = [
                point
                for point
                in points
                if point[
                    "competence_auroc"
                ] is not None
            ]

            budgets = [
                point[
                    "query_budget"
                ]
                for point
                in usable_points
            ]

            auroc = [
                point[
                    "competence_auroc"
                ]
                for point
                in usable_points
            ]

            axis.plot(
                budgets,
                auroc,
                marker="o",
                label=(
                    strategy[
                        "name"
                    ]
                ),
            )

        axis.axhline(
            0.5,
            linestyle="--",
            label="Random ranking",
        )

        axis.set_xlabel(
            "Expert queries"
        )

        axis.set_ylabel(
            "Competence AUROC"
        )

        axis.set_title(
            "Expert-Competence Discovery\n"
            f"{expert_result['name']}"
        )

        axis.legend(
            fontsize=8
        )

        figure.tight_layout()

        figure.savefig(
            FIGURE_DIR
            / auroc_filename,
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )

        figures[
            expert_key
        ] = {
            "name": (
                expert_result[
                    "name"
                ]
            ),

            "team_accuracy": (
                _figure_static_path(
                    team_filename
                )
            ),

            "competence_auroc": (
                _figure_static_path(
                    auroc_filename
                )
            ),
        }

    return figures


# ---------------------------------------------------------------------------
# Expert simulation stability
# ---------------------------------------------------------------------------

def calculate_expert_stability(
    dataset,
    profile,
):
    """
    Re-run only the lightweight expert simulator over several seeds.

    No classifier training occurs.
    """

    true_labels = (
        dataset.test[
            "label"
        ].to_numpy(
            dtype=int
        )
    )

    rows = []

    for seed in STABILITY_SEEDS:

        seeded_profile = replace(
            profile,
            random_state=seed,
        )

        outputs = (
            simulate_expert_predictions(
                texts=dataset.test[
                    "text"
                ],
                true_labels=dataset.test[
                    "label"
                ],
                profile=seeded_profile,
            )
        )

        predictions = np.asarray(
            [
                output.prediction
                for output
                in outputs
            ],
            dtype=int,
        )

        accuracy = float(
            accuracy_score(
                true_labels,
                predictions,
            )
        )

        rows.append(
            {
                "seed": int(
                    seed
                ),

                "accuracy": (
                    accuracy
                ),
            }
        )

    values = np.asarray(
        [
            row[
                "accuracy"
            ]
            for row
            in rows
        ],
        dtype=float,
    )

    return {
        "seeds": (
            STABILITY_SEEDS
        ),

        "runs": (
            rows
        ),

        "mean_accuracy": float(
            values.mean()
        ),

        "std_accuracy": float(
            values.std(
                ddof=1
            )
        ),

        "minimum_accuracy": float(
            values.min()
        ),

        "maximum_accuracy": float(
            values.max()
        ),
    }


# ---------------------------------------------------------------------------
# Complete advanced analysis
# ---------------------------------------------------------------------------

def run_advanced_analysis():
    """
    Generate the complete Advanced Analysis artifact.

    Expensive base experiments are not rerun.
    """

    validate_required_results()

    _ensure_output_directories()

    baseline_results = (
        load_baseline_results()
    )

    defer_results = (
        load_learning_to_defer_results()
    )

    active_results = (
        load_active_learning_results()
    )

    (
        dataset,
        y_true,
        ai_predictions,
        ai_probabilities,
    ) = prepare_test_predictions()


    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    calibration = (
        calculate_calibration(
            y_true=y_true,
            predictions=(
                ai_predictions
            ),
            probabilities=(
                ai_probabilities
            ),
        )
    )

    calibration_filename = (
        "classifier_calibration.png"
    )

    plot_calibration(
        calibration=calibration,
        filename=(
            calibration_filename
        ),
    )


    # ------------------------------------------------------------------
    # Coverage-accuracy
    # ------------------------------------------------------------------

    coverage_data = (
        prepare_coverage_accuracy(
            defer_results
        )
    )


    # ------------------------------------------------------------------
    # Expert-specific analyses
    # ------------------------------------------------------------------

    experts = {}

    for (
        expert_key,
        profile,
    ) in EXPERT_PROFILES.items():

        expert_predictions = (
            get_test_expert_predictions(
                dataset=dataset,
                profile=profile,
            )
        )


        # --------------------------------------------------------------
        # Complementarity
        # --------------------------------------------------------------

        complementarity = (
            calculate_complementarity(
                y_true=y_true,
                ai_predictions=(
                    ai_predictions
                ),
                expert_predictions=(
                    expert_predictions
                ),
            )
        )

        complementarity_filename = (
            "complementarity_"
            f"{expert_key}.png"
        )

        plot_complementarity(
            expert_name=(
                profile.name
            ),
            result=(
                complementarity
            ),
            filename=(
                complementarity_filename
            ),
        )


        # --------------------------------------------------------------
        # Best existing deferral policy
        # --------------------------------------------------------------

        defer_expert = (
            defer_results[
                "experts"
            ][
                expert_key
            ]
        )

        confidence_metrics = (
            defer_expert[
                "confidence_strategy"
            ][
                "metrics"
            ]
        )

        learned_metrics = (
            defer_expert[
                "learned_strategy"
            ][
                "metrics"
            ]
        )

        confidence_accuracy = (
            confidence_metrics[
                "team_accuracy"
            ]
        )

        learned_accuracy = (
            learned_metrics[
                "team_accuracy"
            ]
        )

        if (
            learned_accuracy
            > confidence_accuracy
        ):

            best_team_metrics = (
                learned_metrics
            )

            best_strategy_name = (
                defer_expert[
                    "learned_strategy"
                ][
                    "name"
                ]
            )

        elif (
            confidence_accuracy
            > learned_accuracy
        ):

            best_team_metrics = (
                confidence_metrics
            )

            best_strategy_name = (
                defer_expert[
                    "confidence_strategy"
                ][
                    "name"
                ]
            )

        else:

            # Same tie-break rule as learning_to_defer.py:
            # prefer lower expert workload.
            if (
                learned_metrics[
                    "deferral_rate"
                ]
                <
                confidence_metrics[
                    "deferral_rate"
                ]
            ):

                best_team_metrics = (
                    learned_metrics
                )

                best_strategy_name = (
                    defer_expert[
                        "learned_strategy"
                    ][
                        "name"
                    ]
                )

            else:

                best_team_metrics = (
                    confidence_metrics
                )

                best_strategy_name = (
                    defer_expert[
                        "confidence_strategy"
                    ][
                        "name"
                    ]
                )


        # --------------------------------------------------------------
        # Headroom
        # --------------------------------------------------------------

        headroom = (
            calculate_collaboration_headroom(
                y_true=y_true,
                ai_predictions=(
                    ai_predictions
                ),
                expert_predictions=(
                    expert_predictions
                ),
                best_team_metrics=(
                    best_team_metrics
                ),
            )
        )


        # --------------------------------------------------------------
        # Coverage curve
        # --------------------------------------------------------------

        coverage_filename = (
            "coverage_accuracy_"
            f"{expert_key}.png"
        )

        plot_coverage_accuracy(
            expert_name=(
                profile.name
            ),

            confidence_points=(
                coverage_data[
                    expert_key
                ][
                    "confidence"
                ]
            ),

            learned_points=(
                coverage_data[
                    expert_key
                ][
                    "learned"
                ]
            ),

            classifier_accuracy=(
                baseline_results[
                    "accuracy"
                ]
            ),

            filename=(
                coverage_filename
            ),
        )


        # --------------------------------------------------------------
        # Stability
        # --------------------------------------------------------------

        stability = (
            calculate_expert_stability(
                dataset=dataset,
                profile=profile,
            )
        )


        experts[
            expert_key
        ] = {
            "name": (
                profile.name
            ),

            "best_team_strategy": (
                best_strategy_name
            ),

            "complementarity": (
                complementarity
            ),

            "headroom": (
                headroom
            ),

            "stability": (
                stability
            ),

            "figures": {
                "complementarity": (
                    _figure_static_path(
                        complementarity_filename
                    )
                ),

                "coverage_accuracy": (
                    _figure_static_path(
                        coverage_filename
                    )
                ),
            },
        }


    # ------------------------------------------------------------------
    # Active learning
    # ------------------------------------------------------------------

    active_figures = (
        generate_active_learning_figures(
            active_results
        )
    )


    # ------------------------------------------------------------------
    # Final persisted output
    # ------------------------------------------------------------------

    result = {
        "experiment": {
            "name": (
                "Advanced Human-AI Analysis"
            ),

            "random_state": (
                RANDOM_STATE
            ),
        },

        # Store upstream artifact versions.
        "source_artifacts": (
            get_source_artifact_timestamps()
        ),

        "classifier": {
            "accuracy": float(
                baseline_results[
                    "accuracy"
                ]
            ),

            "calibration": (
                calibration
            ),

            "calibration_figure": (
                _figure_static_path(
                    calibration_filename
                )
            ),
        },

        "experts": (
            experts
        ),

        "active_learning_figures": (
            active_figures
        ),
    }

    save_advanced_analysis_results(
        result
    )

    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_advanced_analysis_results(
    result,
):
    """
    Save the Advanced Analysis artifact atomically.
    """

    _ensure_output_directories()

    temporary_path = (
        ADVANCED_METRICS_PATH
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
            ADVANCED_METRICS_PATH
        )

    except Exception:

        if temporary_path.exists():
            temporary_path.unlink()

        raise


def load_advanced_analysis_results():
    """
    Load saved Advanced Analysis results.
    """

    if not ADVANCED_METRICS_PATH.exists():
        return None

    with ADVANCED_METRICS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )