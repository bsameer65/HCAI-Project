"""
plot_utils.py — Matplotlib plotting helpers for project2.

Each function saves a PNG to the media folder and returns a URL
that the Django template can embed directly in an <img> tag.
"""

import os
import time
import numpy as np

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
    Production-style ALE validation plot.

    Layout
    ------
    Top:
        Difference between ALE methods:

            Standard ALE - Derivative ALE

        Values near zero indicate strong agreement.

    Bottom:
        One small comparison plot per species.

        Standard ALE:
            solid line + filled circle

        Derivative-based ALE:
            dashed line + hollow triangle

    Colour represents species.
    Line style and marker represent the ALE method.
    """

    from matplotlib.lines import Line2D

    centres = np.asarray(
        standard_ale["bin_centres"],
        dtype=float,
    )

    if len(centres) == 0:
        return None

    standard_values = (
        standard_ale["ale_values"]
    )

    derivative_values = (
        derivative_ale["ale_values"]
    )

    species = list(
        standard_values.keys()
    )

    # ===============================================================
    # FIGURE LAYOUT
    # ===============================================================

    fig = plt.figure(
        figsize=(11.5, 8.0)
    )

    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[
            1.20,
            1.75,
        ],
        hspace=0.34,
        wspace=0.28,
    )

    # Top plot spans all three columns.
    ax_residual = fig.add_subplot(
        grid[0, :]
    )

    # Bottom:
    # one subplot per species.
    species_axes = [
        fig.add_subplot(
            grid[1, i]
        )
        for i in range(
            min(
                3,
                len(species),
            )
        )
    ]

    # ===============================================================
    # TOP:
    # DIFFERENCE BETWEEN ALE METHODS
    # ===============================================================

    all_residuals = []

    for cls in species:

        colour = SPECIES_COLOURS.get(
            cls,
            None,
        )

        standard_curve = np.asarray(
            standard_values[cls],
            dtype=float,
        )

        derivative_curve = np.asarray(
            derivative_values[cls],
            dtype=float,
        )

        residual = (
            standard_curve
            -
            derivative_curve
        )

        all_residuals.extend(
            residual.tolist()
        )

        ax_residual.plot(
            centres,
            residual,
            color=colour,
            linewidth=2.0,
            linestyle="-",
            marker="o",
            markersize=4,
            label=cls,
        )

    # ---------------------------------------------------------------
    # Perfect-agreement reference
    # ---------------------------------------------------------------

    ax_residual.axhline(
        0,
        color="#777777",
        linewidth=1,
        linestyle=":",
    )

    # ---------------------------------------------------------------
    # Residual-specific y-axis zoom
    # ---------------------------------------------------------------

    if all_residuals:

        max_abs_residual = max(
            abs(value)
            for value in all_residuals
        )

        if max_abs_residual == 0:
            max_abs_residual = 0.001

        residual_limit = (
            max_abs_residual
            * 1.30
        )

        ax_residual.set_ylim(
            -residual_limit,
            residual_limit,
        )

    # ---------------------------------------------------------------
    # Residual titles
    # ---------------------------------------------------------------

    ax_residual.set_title(
        "Difference Between ALE Methods",
        fontsize=11,
        fontweight="semibold",
        pad=23,
    )

    ax_residual.text(
        0.5,
        1.03,
        (
            "Standard ALE − Derivative ALE"
            " · Zero indicates perfect agreement"
        ),
        transform=ax_residual.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#666666",
    )

    ax_residual.set_ylabel(
        "ALE difference",
        fontsize=10,
    )

    ax_residual.set_xlabel(
        feature_name,
        fontsize=10,
    )

    ax_residual.grid(
        True,
        linestyle="--",
        alpha=0.25,
    )

    ax_residual.legend(
        title="Species",
        fontsize=9,
        title_fontsize=9,
        ncol=3,
        loc="upper right",
    )

    # ===============================================================
    # BOTTOM:
    # SPECIES-SPECIFIC ALE COMPARISONS
    # ===============================================================

    for index, cls in enumerate(
        species[:3]
    ):

        ax = species_axes[index]

        colour = SPECIES_COLOURS.get(
            cls,
            None,
        )

        standard_curve = np.asarray(
            standard_values[cls],
            dtype=float,
        )

        derivative_curve = np.asarray(
            derivative_values[cls],
            dtype=float,
        )

        # -----------------------------------------------------------
        # Standard ALE
        # -----------------------------------------------------------

        ax.plot(
            centres,
            standard_curve,
            color=colour,
            linewidth=2.4,
            linestyle="-",
            marker="o",
            markersize=5,
            markerfacecolor=colour,
            markeredgecolor=colour,
            label="Standard ALE",
            zorder=2,
        )

        # -----------------------------------------------------------
        # Derivative-based ALE
        # -----------------------------------------------------------

        ax.plot(
            centres,
            derivative_curve,
            color=colour,
            linewidth=1.5,
            linestyle="--",
            alpha=0.70,
            marker="^",
            markersize=6.5,
            markerfacecolor="white",
            markeredgecolor=colour,
            markeredgewidth=1.2,
            label="Derivative-based ALE",
            zorder=3,
        )

        # -----------------------------------------------------------
        # Zero reference
        # -----------------------------------------------------------

        ax.axhline(
            0,
            color="#999999",
            linewidth=0.8,
            linestyle=":",
        )

        # -----------------------------------------------------------
        # Species title
        # -----------------------------------------------------------

        ax.set_title(
            cls,
            fontsize=11,
            fontweight="semibold",
            pad=10,
        )

        ax.set_xlabel(
            feature_name,
            fontsize=9,
        )

        if index == 0:

            ax.set_ylabel(
                "Centred ALE effect",
                fontsize=9,
            )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.22,
        )

    # ===============================================================
    # SHARED METHOD LEGEND
    # ===============================================================

    method_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=2.4,
            linestyle="-",
            marker="o",
            markersize=5,
            markerfacecolor="#444444",
            markeredgecolor="#444444",
            label="Standard ALE",
        ),

        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=1.5,
            linestyle="--",
            marker="^",
            markersize=6.5,
            markerfacecolor="white",
            markeredgecolor="#444444",
            label="Derivative-based ALE",
        ),
    ]

    fig.legend(
        handles=method_handles,
        loc="lower center",
        ncol=2,
        fontsize=9,
        bbox_to_anchor=(
            0.5,
            0.015,
        ),
        frameon=False,
    )

    # ===============================================================
    # SMALL TECHNICAL HEADER
    # ===============================================================

    fig.suptitle(
        (
            f"{model_label}"
            f" · {standard_ale['n_bins_used']} ALE bins"
        ),
        fontsize=10,
        fontweight="normal",
        y=0.985,
        color="#444444",
    )

    # ===============================================================
    # FINAL SPACING
    # ===============================================================

    fig.subplots_adjust(
        top=0.88,
        bottom=0.13,
        left=0.08,
        right=0.97,
    )

    return _save_fig(
        fig,
        f"ale_validation_{feature_name}",
    )