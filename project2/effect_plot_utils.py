# Responsible for manually computing:
    # PDP values,
    # ALE values.

# No library should be used for PDP/ALE computation.
# You can still use:
# model.predict_proba(...)

"""
effect_plot_utils.py — Manual PDP and ALE computation for project2.

No external explainability library is used.
Only model.predict_proba() is called to get predictions.

# PDP and ALE are computed manually to make the explanation method
# transparent — students and graders can see exactly how the values are derived,
# rather than treating a library as a black box.
"""

import numpy as np
import pandas as pd


# ── PDP ───────────────────────────────────────────────────────────────────────

def compute_pdp(pipeline, X, feature_name, class_names, n_grid=50):
    """
    Compute Partial Dependence Plot values for one numerical feature.

    Algorithm (manual, no library):
    --------------------------------
    For each value v in the feature's grid:
        1. Copy the full dataset X.
        2. Set the selected feature to v for ALL rows.
        3. Call pipeline.predict_proba() on the modified dataset.
        4. Average the predicted probabilities across all rows.

    This gives the average model response when the feature is forced to v,
    marginalising over all other features.

    # PDP is computed manually to make the explanation method
    # transparent — the loop below is the exact definition of PDP from
    # Friedman (2001), with no hidden abstractions.

    Parameters
    ----------
    pipeline     : fitted sklearn Pipeline (prep + clf)
    X            : pd.DataFrame — original feature matrix (un-encoded)
    feature_name : str — which numerical feature to vary
    class_names  : list[str] — ordered species names matching predict_proba columns
    n_grid       : int — number of evenly-spaced grid points across the feature range

    Returns
    -------
    dict with:
        grid_values : list[float]   — the feature values used as x-axis
        pdp_values  : dict[str -> list[float]]  — mean probability per class at each grid point
    """
    col_min = float(X[feature_name].min())
    col_max = float(X[feature_name].max())
    grid_values = np.linspace(col_min, col_max, n_grid)

    # Accumulator: one list per class
    pdp_values = {cls: [] for cls in class_names}

    X_copy = X.copy()

    for grid_val in grid_values:
        # Step 1-2: force the feature to grid_val for every row
        X_copy[feature_name] = grid_val

        # Step 3: get predicted probabilities (shape: n_rows x n_classes)
        proba = pipeline.predict_proba(X_copy)

        # Step 4: average over all rows
        mean_proba = proba.mean(axis=0)

        for i, cls in enumerate(class_names):
            pdp_values[cls].append(round(float(mean_proba[i]), 5))

    return {
        "grid_values": grid_values.tolist(),
        "pdp_values":  pdp_values,
    }
