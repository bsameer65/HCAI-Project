"""
plot_utils.py — Matplotlib plotting helpers for project2.

Each function saves a PNG to the media folder and returns a URL
that the Django template can embed directly in an <img> tag.
"""

import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from django.conf import settings


# Species colours kept consistent across all plots for easy comparison.
SPECIES_COLOURS = {
    "Adelie": "#4C72B0",
    "Chinstrap": "#DD8452",
    "Gentoo": "#55A868",
}


def _save_fig(
    fig,
    filename_prefix,
):
    """
    Save a matplotlib figure to MEDIA_ROOT and return its media URL.

    A timestamp suffix prevents stale browser caches from showing old plots.
    """

    media_dir = os.path.join(
        settings.MEDIA_ROOT,
        "project2_plots",
    )

    os.makedirs(
        media_dir,
        exist_ok=True,
    )

    filename = (
        f"{filename_prefix}_"
        f"{int(time.time())}.png"
    )

    filepath = os.path.join(
        media_dir,
        filename,
    )

    fig.savefig(
        filepath,
        dpi=120,
        bbox_inches="tight",
    )

    plt.close(fig)

    return (
        settings.MEDIA_URL
        +
        f"project2_plots/{filename}"
    )


# ============================================================================
# PDP
# ============================================================================

def plot_pdp(
    pdp_result,
    feature_name,
    model_label,
):
    """
    Plot Partial Dependence curves — one line per species.
    """

    grid = pdp_result[
        "grid_values"
    ]

    values = pdp_result[
        "pdp_values"
    ]

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    for cls, probabilities in values.items():

        colour = SPECIES_COLOURS.get(
            cls,
            None,
        )

        ax.plot(
            grid,
            probabilities,
            label=cls,
            linewidth=2.2,
            color=colour,
        )

    ax.set_xlabel(
        feature_name,
        fontsize=12,
    )

    ax.set_ylabel(
        "Average predicted probability",
        fontsize=12,
    )

    ax.text(
        0.5,
        1.02,
        f"Model: {model_label}",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#555",
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.legend(
        title="Species",
        fontsize=10,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    fig.tight_layout()

    return _save_fig(
        fig,
        f"pdp_{feature_name}",
    )


# ============================================================================
# STANDARD ALE
# ============================================================================

def plot_ale(
    ale_result,
    feature_name,
    model_label,
):
    """
    Plot standard bin-based ALE curves — one line per species.
    """

    centres = ale_result[
        "bin_centres"
    ]

    values = ale_result[
        "ale_values"
    ]

    if not centres:
        return None

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    for cls, ale_values in values.items():

        colour = SPECIES_COLOURS.get(
            cls,
            None,
        )

        ax.plot(
            centres,
            ale_values,
            label=cls,
            linewidth=2.2,
            color=colour,
            marker="o",
            markersize=4,
        )

    ax.axhline(
        0,
        color="#999",
        linewidth=1,
        linestyle="--",
    )

    ax.set_xlabel(
        feature_name,
        fontsize=12,
    )

    ax.set_ylabel(
        "ALE (centred local effect)",
        fontsize=12,
    )

    ax.text(
        0.5,
        1.02,
        (
            f"Model: {model_label}"
            f"  |  bins used: "
            f"{ale_result['n_bins_used']}"
        ),
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#555",
    )

    ax.legend(
        title="Species",
        fontsize=10,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    fig.tight_layout()

    return _save_fig(
        fig,
        f"ale_{feature_name}",
    )


# ============================================================================
# DERIVATIVE-BASED ALE
# ============================================================================

def plot_derivative_ale(
    derivative_ale_result,
    feature_name,
    model_label,
):
    """
    Plot derivative-based ALE for Logistic Regression.

    The class-probability derivatives are computed analytically from the
    Logistic Regression coefficients. Their accumulated effects are still
    estimated empirically using the observations inside the ALE intervals.
    """

    centres = derivative_ale_result[
        "bin_centres"
    ]

    values = derivative_ale_result[
        "ale_values"
    ]

    if not centres:
        return None

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    for cls, ale_values in values.items():

        colour = SPECIES_COLOURS.get(
            cls,
            None,
        )

        ax.plot(
            centres,
            ale_values,
            label=cls,
            linewidth=2.2,
            color=colour,
            marker="o",
            markersize=4,
        )

    ax.axhline(
        0,
        color="#999",
        linewidth=1,
        linestyle="--",
    )

    ax.set_xlabel(
        feature_name,
        fontsize=12,
    )

    ax.set_ylabel(
        "Derivative ALE (centred effect)",
        fontsize=12,
    )

    ax.text(
        0.5,
        1.02,
        (
            f"Logistic Regression: {model_label}"
            f"  |  bins used: "
            f"{derivative_ale_result['n_bins_used']}"
        ),
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#555",
    )

    ax.legend(
        title="Species",
        fontsize=10,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    fig.tight_layout()

    return _save_fig(
        fig,
        f"derivative_ale_{feature_name}",
    )

def plot_ale_comparison(
    standard_ale,
    derivative_ale,
    feature_name,
    model_label,
):
    """
    Overlay standard ALE and derivative-based ALE.

    Solid line:
        standard finite-difference ALE

    Dashed line:
        derivative-based ALE
    """

    centres = standard_ale["bin_centres"]

    if not centres:
        return None

    standard_values = standard_ale["ale_values"]
    derivative_values = derivative_ale["ale_values"]

    fig, ax = plt.subplots(
        figsize=(9, 5.2)
    )

    for cls in standard_values:

        colour = SPECIES_COLOURS.get(
            cls,
            None,
        )

        # Standard ALE
        ax.plot(
            centres,
            standard_values[cls],
            label=f"{cls} — Standard ALE",
            linewidth=2.2,
            color=colour,
            linestyle="-",
            marker="o",
            markersize=4,
        )

        # Derivative-based ALE
        ax.plot(
            centres,
            derivative_values[cls],
            label=f"{cls} — Derivative ALE",
            linewidth=2.0,
            color=colour,
            linestyle="--",
            marker="x",
            markersize=5,
        )

    ax.axhline(
        0,
        color="#999",
        linewidth=1,
        linestyle=":",
    )

    ax.set_xlabel(
        feature_name,
        fontsize=12,
    )

    ax.set_ylabel(
        "Centred accumulated effect",
        fontsize=12,
    )

    ax.text(
        0.5,
        1.02,
        (
            f"Model: {model_label}"
            f" | bins: {standard_ale['n_bins_used']}"
        ),
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#555",
    )

    ax.legend(
        title="Species and ALE method",
        fontsize=9,
        ncol=2,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.35,
    )

    fig.tight_layout()

    return _save_fig(
        fig,
        f"ale_validation_{feature_name}",
    )