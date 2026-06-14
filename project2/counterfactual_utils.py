# Responsible for:
    # selecting one penguin example,
    # selecting a desired target class,
    # generating random nearby examples,
    # checking which examples are predicted as the desired class,
    # ranking them by distance,
    # showing best counterfactuals.

# Important:
# Numerical features can be changed with random noise.

# Example:
# bill_length_mm + random noise

# Categorical features must be changed differently.
# Example:
# island can randomly become Biscoe, Dream, or Torgersen
# sex can become male or female

# Do not add decimal noise to categorical features.

"""
counterfactual_utils.py — Generate counterfactual explanations for Project 2.

A counterfactual is a nearby artificial example whose model prediction differs
from the original, specifically matching the user's desired target class.

Algorithm
---------
1. Take the original penguin row (original values, not encoded).
2. Sample N random neighbours:
   - Numerical: original value + Gaussian noise scaled by the feature's MAD.
   - Categorical: randomly resample from the real categories in the dataset.
3. Encode all neighbours with the pipeline's preprocessor.
4. Predict with the selected model's pipeline.
5. Keep only neighbours predicted as the desired target class.
6. Rank by MAD-weighted L1 distance + categorical mismatch penalty.
7. Return the top-k closest.
"""

import numpy as np
import pandas as pd

from .data_utils import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


# ── Distance helpers ──────────────────────────────────────────────────────────

def _compute_mads(df, numerical_features):
    """
    Compute the Median Absolute Deviation for each numerical feature.
    MAD is more robust than standard deviation for skewed distributions.

    # MAD-weighted distance avoids one large-scale feature
    # dominating the distance — e.g. body_mass_g (range ~3000) would
    # otherwise swamp bill_length_mm (range ~20) in a raw L1 distance.
    """
    mads = {}
    for feat in numerical_features:
        col = df[feat].dropna()
        mad = float(np.median(np.abs(col - col.median())))
        mads[feat] = mad if mad > 0 else float(col.std())  # fallback to std
    return mads


def _l1_distance(original_row, candidate_row, mads, numerical_features, categorical_features):
    """
    MAD-normalised L1 distance between the original and one candidate row.

    Numerical part   : sum |orig - cand| / MAD   (scale-independent)
    Categorical part : 1 per feature that changed (mismatch penalty)

    # categorical mismatch penalty ensures that changing island
    # or sex is counted as a real change, not silently ignored.
    """
    dist = 0.0

    for feat in numerical_features:
        mad = mads.get(feat, 1.0)
        dist += abs(float(original_row[feat]) - float(candidate_row[feat])) / mad

    for feat in categorical_features:
        if str(original_row[feat]) != str(candidate_row[feat]):
            dist += 1.0   # categorical mismatch penalty

    return round(dist, 4)


# ── Sampling helpers ──────────────────────────────────────────────────────────

def _sample_numerical(original_val, mad, col_min, col_max, rng, noise_scale=1.0):
    """
    Sample one numerical value by adding Gaussian noise scaled by MAD.
    Clipped to [col_min, col_max] to keep values realistic.
    """
    noise = rng.normal(0, noise_scale * mad)
    return float(np.clip(original_val + noise, col_min, col_max))


def _sample_categorical(feature, df, rng):
    """
    Pick a random valid category for a categorical feature from the real dataset.

    # Categorical features are sampled from real dataset categories
    # to keep counterfactuals realistic — we never invent impossible values
    # like a non-existent island or an invalid sex label.
    """
    choices = df[feature].dropna().unique().tolist()
    return rng.choice(choices)


# ── Main function ─────────────────────────────────────────────────────────────

def generate_counterfactuals(selected_model_info, row_index, target_class_name,
                             N=5000, k=5, noise_scale=1.0, max_retries=2):
    """
    Generate counterfactual explanations for one penguin.

    Parameters
    ----------
    selected_model_info : dict  returned by get_selected_model()
    row_index           : int   index into the cleaned dataframe
    target_class_name   : str   desired species (e.g. "Gentoo")
    N                   : int   number of random candidates to sample
    k                   : int   number of counterfactuals to return
    noise_scale         : float multiplier on MAD noise (increased on retry)
    max_retries         : int   how many times to retry with larger noise

    Returns
    -------
    dict with keys:
        original_row        — original feature values (dict)
        original_prediction — species predicted for the original
        target_class        — desired species
        counterfactuals     — list of dicts (feature values + distance)
        message             — user-facing status message
    """
    pipeline    = selected_model_info["pipeline"]
    df          = selected_model_info["df"]
    le          = selected_model_info["le"]

    # ── Get original row ──────────────────────────────────────────────────────
    original_row = df.iloc[row_index][NUMERICAL_FEATURES + CATEGORICAL_FEATURES].to_dict()

    # Predict the original
    original_df   = pd.DataFrame([original_row])
    orig_pred_enc = pipeline.predict(original_df)[0]
    original_pred = le.inverse_transform([orig_pred_enc])[0]

    # Target class integer index
    target_enc = le.transform([target_class_name])[0]

    # Pre-compute MADs and feature ranges from the full dataset
    mads = _compute_mads(df, NUMERICAL_FEATURES)
    col_ranges = {
        feat: (float(df[feat].min()), float(df[feat].max()))
        for feat in NUMERICAL_FEATURES
    }

    rng = np.random.default_rng(seed=42)

    counterfactuals = []
    current_noise   = noise_scale
    attempts        = 0

    while len(counterfactuals) == 0 and attempts <= max_retries:

        # ── Sample N neighbours ───────────────────────────────────────────────
        candidates = []
        for _ in range(N):
            candidate = {}

            for feat in NUMERICAL_FEATURES:
                col_min, col_max = col_ranges[feat]
                candidate[feat] = _sample_numerical(
                    original_row[feat], mads[feat], col_min, col_max, rng, current_noise
                )

            for feat in CATEGORICAL_FEATURES:
                # With 70% chance keep the original value, else resample.
                # This biases samples toward the original, so closer
                # counterfactuals are found first.
                if rng.random() < 0.7:
                    candidate[feat] = original_row[feat]
                else:
                    candidate[feat] = _sample_categorical(feat, df, rng)

            candidates.append(candidate)

        candidates_df = pd.DataFrame(candidates)

        # ── Predict all candidates ────────────────────────────────────────────
        predictions = pipeline.predict(candidates_df)

        # ── Filter: keep only those predicted as target class ─────────────────
        mask    = predictions == target_enc
        matched = candidates_df[mask].copy()

        if len(matched) == 0:
            attempts      += 1
            current_noise *= 2   # wider search on retry
            continue

        # ── Rank by distance ──────────────────────────────────────────────────
        matched["_distance"] = matched.apply(
            lambda r: _l1_distance(original_row, r, mads, NUMERICAL_FEATURES, CATEGORICAL_FEATURES),
            axis=1,
        )
        matched = matched.sort_values("_distance").head(k)

        # ── Build result list ─────────────────────────────────────────────────
        for _, row in matched.iterrows():
            entry = {feat: row[feat] for feat in NUMERICAL_FEATURES + CATEGORICAL_FEATURES}
            entry["distance"] = row["_distance"]

            # Mark which features changed vs the original
            changes = {}
            for feat in NUMERICAL_FEATURES:
                orig_val = float(original_row[feat])
                cand_val = float(row[feat])
                changes[feat] = abs(orig_val - cand_val) > 1e-6
            for feat in CATEGORICAL_FEATURES:
                changes[feat] = str(original_row[feat]) != str(row[feat])
            entry["changes"] = changes

            counterfactuals.append(entry)

        break  # found some — stop retrying

    # ── Build status message ──────────────────────────────────────────────────
    if counterfactuals:
        msg = (
            f"Found {len(counterfactuals)} counterfactual(s) after "
            f"{attempts + 1} attempt(s)."
        )
    else:
        msg = (
            "No counterfactuals found even after widening the search. "
            "The selected model may already predict this penguin as the desired class, "
            "or the classes are very hard to separate with small changes."
        )

    return {
        "original_row":        original_row,
        "original_prediction": original_pred,
        "target_class":        target_class_name,
        "counterfactuals":     counterfactuals,
        "message":             msg,
    }