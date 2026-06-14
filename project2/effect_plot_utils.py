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


# ── ALE ───────────────────────────────────────────────────────────────────────

def compute_ale(pipeline, X, feature_name, class_names, n_bins=10):
    """
    Compute Accumulated Local Effects (ALE) for one numerical feature.

    ALE is more reliable than PDP when features are correlated, because it only
    looks at local changes within each bin — it never asks the model to predict
    on impossible (out-of-distribution) feature combinations.

    # bin-based ALE keeps the method model-agnostic and works
    # identically for both Decision Trees and Logistic Regression:
    #   - Logistic Regression is differentiable, so exact partial derivatives
    #     exist in theory, but bin-based ALE is simpler to implement and explain.
    #   - Decision Trees are piecewise constant and not smoothly differentiable,
    #     so a derivative-based approach would not work anyway.
    #   - Using bin-based ALE for both gives a consistent, model-agnostic interface.

    Algorithm (manual, no library):
    --------------------------------
    1. Compute quantile bin edges for the selected feature.
    2. For each bin [z_low, z_high]:
        a. Find all rows whose feature value falls inside this bin.
        b. If no rows → skip this bin.
        c. Create two copies of those rows:
               X_low  — feature set to z_low  (lower bin edge)
               X_high — feature set to z_high (upper bin edge)
        d. Compute predict_proba for both copies.
        e. Local effect = mean( proba_high - proba_low )  per class.
    3. Accumulate local effects across bins (running sum) → raw ALE.
    4. Centre the ALE by subtracting the weighted mean so the overall
       average effect is near zero — this makes curves comparable across features.

    Parameters
    ----------
    pipeline     : fitted sklearn Pipeline (prep + clf)
    X            : pd.DataFrame — original feature matrix (un-encoded)
    feature_name : str — which numerical feature to analyse
    class_names  : list[str] — species names matching predict_proba columns
    n_bins       : int — number of quantile bins

    Returns
    -------
    dict with:
        bin_centres : list[float]   — mid-point of each bin (x-axis)
        ale_values  : dict[str -> list[float]]  — centred ALE per class per bin
        n_bins_used : int — bins that actually contained data
    """
    # Step 1: compute quantile bin edges
    quantiles  = np.linspace(0, 100, n_bins + 1)
    bin_edges  = np.percentile(X[feature_name].dropna(), quantiles)
    # Remove duplicate edges that arise from low-variance columns
    bin_edges  = np.unique(bin_edges)
    n_intervals = len(bin_edges) - 1

    if n_intervals == 0:
        # Degenerate case: feature has no variance
        return {
            "bin_centres": [],
            "ale_values":  {cls: [] for cls in class_names},
            "n_bins_used": 0,
        }

    # Accumulators
    local_effects  = {cls: [] for cls in class_names}  # per-bin Δ
    bin_centres    = []
    bin_counts     = []   # number of samples in each bin (for centering)

    X_work = X.copy()

    for i in range(n_intervals):
        z_low  = bin_edges[i]
        z_high = bin_edges[i + 1]

        # Step 2a: find rows whose feature lies inside this bin
        if i == n_intervals - 1:
            # Include the right edge in the last bin
            mask = (X_work[feature_name] >= z_low) & (X_work[feature_name] <= z_high)
        else:
            mask = (X_work[feature_name] >= z_low) & (X_work[feature_name] < z_high)

        X_bin = X_work[mask].copy()

        if len(X_bin) == 0:
            # Step 2b: no data in this bin — record a zero effect so the
            # x-axis stays continuous, but don't let it distort centering
            bin_centres.append(round(float((z_low + z_high) / 2), 4))
            bin_counts.append(0)
            for cls in class_names:
                local_effects[cls].append(0.0)
            continue

        # Step 2c: two copies — feature forced to bin edges
        X_low  = X_bin.copy(); X_low[feature_name]  = z_low
        X_high = X_bin.copy(); X_high[feature_name] = z_high

        # Step 2d: predicted probabilities for both
        proba_low  = pipeline.predict_proba(X_low)   # shape (n_bin, n_classes)
        proba_high = pipeline.predict_proba(X_high)

        # Step 2e: mean local effect per class
        delta = (proba_high - proba_low).mean(axis=0)

        bin_centres.append(round(float((z_low + z_high) / 2), 4))
        bin_counts.append(len(X_bin))
        for j, cls in enumerate(class_names):
            local_effects[cls].append(round(float(delta[j]), 6))

    # Step 3: accumulate local effects (running sum) → raw ALE
    ale_raw = {}
    for cls in class_names:
        ale_raw[cls] = list(np.cumsum(local_effects[cls]))

    # Step 4: centre — subtract weighted mean so average ALE ≈ 0
    # Weighting by bin count makes the centering reflect the data distribution.
    counts = np.array(bin_counts, dtype=float)
    total  = counts.sum()

    ale_values = {}
    for cls in class_names:
        raw = np.array(ale_raw[cls])
        if total > 0:
            weighted_mean = float(np.dot(counts, raw) / total)
        else:
            weighted_mean = 0.0
        centred = [round(float(v - weighted_mean), 5) for v in raw]
        ale_values[cls] = centred

    return {
        "bin_centres": bin_centres,
        "ale_values":  ale_values,
        "n_bins_used": int(sum(1 for c in bin_counts if c > 0)),
    }
