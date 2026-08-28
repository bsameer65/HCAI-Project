"""
ale_validation_utils.py

Utilities for comparing standard finite-difference ALE with
derivative-based ALE for Logistic Regression.

The actual ALE algorithms remain in effect_plot_utils.py.

This module only compares their outputs so that users can inspect
whether the model-agnostic ALE approximation agrees with the
analytical derivative-based calculation.
"""

import numpy as np


def _agreement_label(mean_absolute_difference):
    """
    Convert the numerical difference between the two ALE methods
    into a simple human-readable agreement level.

    The numerical difference should still be displayed to the user;
    this label is only a compact interpretation aid.
    """

    if mean_absolute_difference <= 0.01:
        return "High"

    if mean_absolute_difference <= 0.03:
        return "Moderate"

    return "Low"


def compare_ale_methods(
    standard_ale,
    derivative_ale,
    class_names,
):
    """
    Compare standard ALE and derivative-based ALE.

    Both ALE methods must have been computed using the same:

        - dataset
        - feature
        - number of bins
        - Logistic Regression model

    For every class, the function calculates:

        Mean Absolute Difference (MAD)

            mean(
                abs(
                    standard_ALE
                    -
                    derivative_ALE
                )
            )

    A smaller value means that the model-agnostic finite-difference
    approximation is closer to the derivative-based calculation.

    Returns
    -------
    dict
        Contains per-class comparison values and an overall difference.
    """

    standard_centres = np.asarray(
        standard_ale["bin_centres"],
        dtype=float,
    )

    derivative_centres = np.asarray(
        derivative_ale["bin_centres"],
        dtype=float,
    )

    # Both methods should use the same quantile bins.
    if len(standard_centres) != len(derivative_centres):
        raise ValueError(
            "The ALE methods produced different numbers of bins "
            "and therefore cannot be compared directly."
        )

    if len(standard_centres) == 0:
        raise ValueError(
            "No ALE bins are available for comparison."
        )

    if not np.allclose(
        standard_centres,
        derivative_centres,
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError(
            "The ALE methods used different bin centres "
            "and therefore cannot be compared directly."
        )

    class_comparisons = []

    all_absolute_differences = []

    for class_name in class_names:

        standard_values = np.asarray(
            standard_ale["ale_values"][class_name],
            dtype=float,
        )

        derivative_values = np.asarray(
            derivative_ale["ale_values"][class_name],
            dtype=float,
        )

        if len(standard_values) != len(derivative_values):
            raise ValueError(
                f"The ALE curves for {class_name} have "
                "different lengths."
            )

        absolute_differences = np.abs(
            standard_values
            -
            derivative_values
        )

        mean_absolute_difference = float(
            np.mean(
                absolute_differences
            )
        )

        max_absolute_difference = float(
            np.max(
                absolute_differences
            )
        )

        all_absolute_differences.extend(
            absolute_differences.tolist()
        )

        class_comparisons.append(
            {
                "class_name": class_name,

                "mean_absolute_difference": round(
                    mean_absolute_difference,
                    5,
                ),

                "max_absolute_difference": round(
                    max_absolute_difference,
                    5,
                ),

                "agreement": _agreement_label(
                    mean_absolute_difference
                ),
            }
        )

    overall_mean_difference = float(
        np.mean(
            all_absolute_differences
        )
    )

    overall_max_difference = float(
        np.max(
            all_absolute_differences
        )
    )

    overall_agreement = _agreement_label(
        overall_mean_difference
    )

    if overall_agreement == "High":

        interpretation = (
            "The two ALE methods show close agreement. "
            "This indicates that the model-agnostic finite-difference "
            "ALE approximation is consistent with the analytical "
            "probability derivatives of this Logistic Regression model."
        )

    elif overall_agreement == "Moderate":

        interpretation = (
            "The two ALE methods show broadly similar effects, "
            "but some numerical differences are visible. "
            "These differences can arise because standard ALE uses "
            "finite changes across each interval, while derivative-based "
            "ALE uses local analytical derivatives."
        )

    else:

        interpretation = (
            "The two ALE methods differ noticeably for this feature. "
            "The standard method estimates finite changes between bin "
            "boundaries, whereas the derivative-based method accumulates "
            "local analytical sensitivities. The disagreement should "
            "therefore be considered when interpreting the explanation."
        )

    return {
        "class_comparisons": class_comparisons,

        "overall_mean_difference": round(
            overall_mean_difference,
            5,
        ),

        "overall_max_difference": round(
            overall_max_difference,
            5,
        ),

        "overall_agreement": overall_agreement,

        "interpretation": interpretation,

        "n_bins": len(
            standard_centres
        ),
    }