import numpy as np


AGREEMENT_HIGH_THRESHOLD = 0.01
AGREEMENT_MODERATE_THRESHOLD = 0.03


def _agreement_label(mean_absolute_difference):
    """
    Convert the numerical difference between the two ALE methods
    into a human-readable agreement level.
    """

    if mean_absolute_difference <= AGREEMENT_HIGH_THRESHOLD:
        return "High"

    if mean_absolute_difference <= AGREEMENT_MODERATE_THRESHOLD:
        return "Moderate"

    return "Low"


def compare_ale_methods(
    standard_ale,
    derivative_ale,
    class_names,
):
    """
    Compare standard finite-difference ALE with derivative-based ALE.

    Both results must have been computed using identical ALE bins.

    Returns per-class:
        - mean absolute difference
        - maximum absolute difference
        - agreement label

    Also returns overall summary statistics.
    """

    standard_centres = np.asarray(
        standard_ale["bin_centres"],
        dtype=float,
    )

    derivative_centres = np.asarray(
        derivative_ale["bin_centres"],
        dtype=float,
    )

    if len(standard_centres) == 0:
        raise ValueError(
            "No ALE bins are available for comparison."
        )

    if len(standard_centres) != len(derivative_centres):
        raise ValueError(
            "The two ALE methods produced different numbers of bins."
        )

    if not np.allclose(
        standard_centres,
        derivative_centres,
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError(
            "The two ALE methods used different bin centres."
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

        if standard_values.shape != derivative_values.shape:
            raise ValueError(
                f"The ALE curves for {class_name} have different shapes."
            )

        residuals = (
            standard_values
            -
            derivative_values
        )

        absolute_differences = np.abs(
            residuals
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
            "The standard finite-difference ALE explanation is therefore "
            "consistent with the analytical local behaviour of the "
            "Logistic Regression model."
        )

    elif overall_agreement == "Moderate":

        interpretation = (
            "The methods show broadly similar effects, although some "
            "numerical differences are visible. Standard ALE estimates "
            "finite changes across intervals, while derivative-based ALE "
            "uses analytical local sensitivities."
        )

    else:

        interpretation = (
            "The two ALE calculations differ noticeably for this feature. "
            "The disagreement should be considered when interpreting "
            "the ALE explanation."
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


def build_bin_sensitivity_analysis(
    pipeline,
    X,
    feature_name,
    class_names,
    compute_ale_function,
    compute_derivative_ale_function,
    bin_options=(5, 10, 20),
):
    """
    Repeat ALE validation using several bin resolutions.

    This is deliberately described as bin sensitivity rather than
    convergence because increasing the number of bins does not
    necessarily reduce the discrepancy monotonically.
    """

    results = []

    for n_bins in bin_options:

        standard_ale = compute_ale_function(
            pipeline=pipeline,
            X=X,
            feature_name=feature_name,
            class_names=class_names,
            n_bins=n_bins,
        )

        derivative_ale = compute_derivative_ale_function(
            pipeline=pipeline,
            X=X,
            feature_name=feature_name,
            class_names=class_names,
            n_bins=n_bins,
        )

        comparison = compare_ale_methods(
            standard_ale=standard_ale,
            derivative_ale=derivative_ale,
            class_names=class_names,
        )

        results.append(
            {
                "n_bins": n_bins,

                "overall_mean_difference": (
                    comparison[
                        "overall_mean_difference"
                    ]
                ),

                "overall_max_difference": (
                    comparison[
                        "overall_max_difference"
                    ]
                ),

                "agreement": (
                    comparison[
                        "overall_agreement"
                    ]
                ),
            }
        )

    agreement_levels = {
        item["agreement"]
        for item in results
    }

    if agreement_levels == {"High"}:

        interpretation = (
            "Agreement remains high across all tested bin resolutions, "
            "indicating that the validation result is stable with respect "
            "to the selected number of ALE bins."
        )

    elif "Low" not in agreement_levels:

        interpretation = (
            "Agreement remains broadly stable across the tested bin "
            "resolutions, although the numerical difference changes "
            "slightly with the bin configuration."
        )

    else:

        interpretation = (
            "The agreement changes substantially across the tested bin "
            "resolutions. The number of ALE intervals therefore has a "
            "noticeable influence on this validation."
        )

    return {
        "rows": results,
        "interpretation": interpretation,
    }