# Responsible for:
    # training several decision trees,
    # training several logistic regression models,
    # calculating accuracy,
    # calculating complexity,
    # selecting best model for current λ.

# For decision tree:
    # complexity = number of leaves

# For logistic regression, use:
    # complexity = number of non-zero coefficients
    # or:
    # complexity = sum of absolute coefficient values
    # I recommend:
    # complexity = number of non-zero coefficients
    # because it is easier to explain.


"""
model_utils.py — Model training, complexity scoring, and lambda-based selection.

Central entry point: get_selected_model(model_type, lambda_value)

# central selected-model function avoids inconsistent explanations
# across UI sections — every view (train, counterfactual, PDP, ALE) calls this
# one function and gets the exact same pipeline, split, and metadata.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64
from sklearn.tree import plot_tree

from .data_utils import (
    load_penguin_data,
    get_preprocessor,
    encode_target,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)

# ── Candidate hyperparameter grids ───────────────────────────────────────────

DT_MAX_DEPTHS = [1, 2, 3, 4, 5, 7, 10, None]   # None = fully grown tree
LR_C_VALUES   = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

COEF_THRESHOLD = 1e-6   # abs(coef) > threshold → treated as non-zero


# ── Internal helpers ──────────────────────────────────────────────────────────

def _split(X, y_enc):
    """80/20 stratified split with a fixed seed for reproducibility."""
    return train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)


def _dt_complexity(clf):
    """Number of leaves — a simple, explainable proxy for tree complexity."""
    return int(clf.get_n_leaves())


def _lr_complexity(clf):
    """
    Number of non-zero coefficients across all classes.
    L1 regularisation drives many weights to zero, so this directly measures
    how many features the model is actually using.
    """
    return int(np.sum(np.abs(clf.coef_) > COEF_THRESHOLD))


def _train_candidates(model_type, X_train, X_test, y_train, y_test):
    """
    Train all candidate models for the given model_type and evaluate them.
    Returns a list of result dicts (one per candidate).
    Internal — called only by get_selected_model().
    """
    results = []

    if model_type == "dt":
        for depth in DT_MAX_DEPTHS:
            clf  = DecisionTreeClassifier(max_depth=depth, random_state=42)
            pipe = Pipeline([("prep", get_preprocessor()), ("clf", clf)])
            pipe.fit(X_train, y_train)

            acc        = accuracy_score(y_test, pipe.predict(X_test))
            complexity = _dt_complexity(pipe.named_steps["clf"])
            label      = f"DT depth={depth}" if depth is not None else "DT depth=unlimited"

            results.append({
                "model_type": "dt",
                "label":      label,
                "hyperparam": depth,          # the varied hyperparameter
                "pipeline":   pipe,
                "accuracy":   round(acc, 4),
                "complexity": complexity,
            })

    else:  # "lr"
        for C in LR_C_VALUES:
            # l1_ratio=1 ≡ pure L1 penalty; uses saga solver which supports L1.
            # Written this way for forward-compatibility with sklearn >= 1.8
            # which deprecated the 'penalty' keyword argument.
            clf  = LogisticRegression(solver="saga", l1_ratio=1.0, C=C,
                                      max_iter=5000, random_state=42)
            pipe = Pipeline([("prep", get_preprocessor()), ("clf", clf)])
            pipe.fit(X_train, y_train)

            acc        = accuracy_score(y_test, pipe.predict(X_test))
            complexity = _lr_complexity(pipe.named_steps["clf"])

            results.append({
                "model_type": "lr",
                "label":      f"LR C={C}",
                "hyperparam": C,
                "pipeline":   pipe,
                "accuracy":   round(acc, 4),
                "complexity": complexity,
            })

    return results


def _apply_objective(results, lambda_value):
    """
    Compute and attach objective + normalised complexity to every result dict.
    Mutates the list in-place and returns the best result.

    Objective (lower is better):
        objective = (1 - accuracy) + lambda * norm_complexity

    Normalising complexity to [0, 1] makes lambda interpretable regardless of
    how large the raw complexity numbers are.
    """
    max_c = max(r["complexity"] for r in results) or 1  # guard against all-zero

    best, best_obj = None, float("inf")
    for r in results:
        norm_c = r["complexity"] / max_c
        obj    = (1 - r["accuracy"]) + lambda_value * norm_c
        r["norm_complexity"] = round(norm_c, 4)
        r["objective"]       = round(obj, 4)
        if obj < best_obj:
            best_obj, best = obj, r

    return best


# ── Encoded feature name helper ───────────────────────────────────────────────

def get_encoded_feature_names(pipeline):
    """
    Return the feature names produced by the preprocessing step of a fitted
    pipeline: numerical names first, then OHE-expanded categorical names.

    # centralising feature-name extraction here means every view
    # (feature importance, PDP, ALE, coefficient table) labels axes consistently.
    """
    prep     = pipeline.named_steps["prep"]
    ohe      = prep.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return NUMERICAL_FEATURES + cat_names

def get_tree_image(pipeline, class_names):
    clf = pipeline.named_steps["clf"]

    feature_names = get_encoded_feature_names(pipeline)

    fig, ax = plt.subplots(figsize=(18, 10))

    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax
    )

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)

    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode()

# ── Coefficient table helper (Logistic Regression only) ──────────────────────

def get_lr_coef_table(pipeline, class_names):
    """
    Build a per-feature coefficient table for a fitted LR pipeline.

    Returns a list of dicts:
        [{"feature": str, "coef_<cls>": float, ..., "nonzero": bool}, ...]

    Showing per-class coefficients lets users see which features push the
    model towards each species — more informative than a single bar chart.
    """
    clf          = pipeline.named_steps["clf"]
    coef         = clf.coef_   # shape (n_classes, n_features_encoded)
    feature_names = get_encoded_feature_names(pipeline)

    table = []
    for i, feat in enumerate(feature_names):
        row         = {"feature": feat}
        any_nonzero = False
        for j, cls in enumerate(class_names):
            val = round(float(coef[j, i]), 4)
            row[f"coef_{cls}"] = val
            if abs(val) > COEF_THRESHOLD:
                any_nonzero = True
        row["nonzero"] = any_nonzero
        table.append(row)
    return table


# ── Public API ────────────────────────────────────────────────────────────────

def get_selected_model(model_type, lambda_value):
    """
    End-to-end function: load data → train candidates → select best → return
    everything that downstream views need.

    Parameters
    ----------
    model_type   : "dt" | "lr"
    lambda_value : float in [0, 1]

    Returns
    -------
    A single dict with keys:
        pipeline          — fitted sklearn Pipeline (prep + clf)
        model_type        — "dt" or "lr"
        lambda_value      — the lambda that was used
        hyperparam        — selected hyperparameter value (depth or C)
        label             — human-readable model name
        accuracy          — test-set accuracy (float)
        complexity        — raw complexity score (int)
        objective         — objective value (float)
        candidates        — list of all candidate result dicts
        X                 — full feature DataFrame (un-encoded, original values)
        y                 — full target Series (string labels)
        X_train           — training split (un-encoded)
        X_test            — test split (un-encoded)
        y_train           — training target (integer-encoded)
        y_test            — test target (integer-encoded)
        le                — fitted LabelEncoder (int → species name)
        df                — cleaned full DataFrame
        numerical_features
        categorical_features
        feature_names     — all original feature column names
        class_names       — sorted list of species strings
        coef_table        — coefficient table (LR only, else None)

    # central selected-model function avoids inconsistent explanations
    # across UI sections — every view (train, counterfactual, PDP, ALE) calls this
    # one function and gets the exact same pipeline, split, and metadata.
    """
    # 1. Load data
    df, X, y, num_feats, cat_feats, class_names = load_penguin_data()

    # 2. Encode target once; keep the same split for every candidate
    y_enc, le = encode_target(y)
    X_train, X_test, y_train, y_test = _split(X, y_enc)

    # 3. Train all candidates
    candidates = _train_candidates(model_type, X_train, X_test, y_train, y_test)

    # 4. Score with objective and pick best
    best = _apply_objective(candidates, lambda_value)

    # 5. Build coefficient table for LR (None for DT)
    coef_table = (
        get_lr_coef_table(best["pipeline"], class_names)
        if model_type == "lr"
        else None
    )
    tree_image = None

    if model_type == "dt":
        tree_image = get_tree_image(best["pipeline"], class_names)

    return {
        # Model identity
        "pipeline":           best["pipeline"],
        "model_type":         model_type,
        "lambda_value":       lambda_value,
        "hyperparam":         best["hyperparam"],
        "label":              best["label"],
        # Scores
        "accuracy":           best["accuracy"],
        "complexity":         best["complexity"],
        "objective":          best["objective"],
        # Candidate list (for the comparison table in the UI)
        "candidates":         candidates,
        # Data (needed by PDP, ALE, counterfactuals)
        "X":                  X,
        "y":                  y,
        "X_train":            X_train,
        "X_test":             X_test,
        "y_train":            y_train,
        "y_test":             y_test,
        "le":                 le,
        "df":                 df,
        # Feature / class metadata
        "numerical_features":   num_feats,
        "categorical_features": cat_feats,
        "feature_names":        num_feats + cat_feats,
        "class_names":          class_names,
        # LR-specific
        "coef_table":         coef_table,
        "tree_image": tree_image,
    }