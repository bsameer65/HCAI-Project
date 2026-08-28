"""
plot_utils.py — Matplotlib plotting helpers for project2.

Each function saves a PNG to the media folder and returns a URL
that the Django template can embed directly in an <img> tag.
"""

import os
import time
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for Django server threads
import matplotlib.pyplot as plt
from django.conf import settings

# Species colours kept consistent across all plots for easy comparison
SPECIES_COLOURS = {
    "Adelie":    "#4C72B0",
    "Chinstrap": "#DD8452",
    "Gentoo":    "#55A868",
}


def _save_fig(fig, filename_prefix):
    """
    Save a matplotlib figure to MEDIA_ROOT and return its media URL.
    A timestamp suffix prevents stale browser caches from showing old plots.
    """
    media_dir = os.path.join(settings.MEDIA_ROOT, "project2_plots")
    os.makedirs(media_dir, exist_ok=True)

    filename = f"{filename_prefix}_{int(time.time())}.png"
    filepath = os.path.join(media_dir, filename)
    fig.savefig(filepath, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return settings.MEDIA_URL + f"project2_plots/{filename}"


def plot_pdp(pdp_result, feature_name, model_label):
    """
    Plot Partial Dependence curves — one line per species.

    Parameters
    ----------
    pdp_result   : dict returned by compute_pdp()
    feature_name : str — used for x-axis label and title
    model_label  : str — e.g. "DT depth=2" shown in the subtitle

    Returns
    -------
    str — media URL of the saved PNG
    """
    grid   = pdp_result["grid_values"]
    values = pdp_result["pdp_values"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for cls, probs in values.items():
        colour = SPECIES_COLOURS.get(cls, None)
        ax.plot(grid, probs, label=cls, linewidth=2.2, color=colour)

    ax.set_xlabel(feature_name, fontsize=12)
    ax.set_ylabel("Average predicted probability", fontsize=12)

    ax.set_subtitle = lambda *a, **kw: None   # no-op guard
    ax.text(
        0.5, 1.02,
        f"Model: {model_label}",
        transform=ax.transAxes,
        ha="center", fontsize=9, color="#555"
    )
    ax.set_ylim(0, 1)
    ax.legend(title="Species", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    return _save_fig(fig, f"pdp_{feature_name}")


def plot_ale(ale_result, feature_name, model_label):
    """
    Plot Accumulated Local Effects curves — one line per species.

    ALE values are centred around zero, so the y-axis shows how much the model's
    prediction for each species is pushed up or down relative to the average.

    Parameters
    ----------
    ale_result   : dict returned by compute_ale()
    feature_name : str — used for x-axis label and title
    model_label  : str — e.g. "DT depth=2" shown in the subtitle

    Returns
    -------
    str — media URL of the saved PNG, or None if no bins had data
    """
    centres = ale_result["bin_centres"]
    values  = ale_result["ale_values"]

    if not centres:
        return None   # no data — caller should show a friendly message

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for cls, ale_vals in values.items():
        colour = SPECIES_COLOURS.get(cls, None)
        ax.plot(centres, ale_vals, label=cls, linewidth=2.2,
                color=colour, marker="o", markersize=4)

    # Zero reference line — makes it easy to see positive/negative effects
    ax.axhline(0, color="#999", linewidth=1, linestyle="--")

    ax.set_xlabel(feature_name, fontsize=12)
    ax.set_ylabel("ALE (centred local effect)", fontsize=12)

    ax.text(
        0.5, 1.02,
        f"Model: {model_label}  |  bins used: {ale_result['n_bins_used']}",
        transform=ax.transAxes,
        ha="center", fontsize=9, color="#555"
    )
    ax.legend(title="Species", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    return _save_fig(fig, f"ale_{feature_name}")
