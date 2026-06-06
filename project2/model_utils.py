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

Supports:
  - Decision Tree  (complexity = number of leaves)
  - Logistic Regression  (complexity = number of non-zero coefficients)
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .data_utils import get_preprocessor, encode_target

# ── Candidate hyperparameter grids ───────────────────────────────────────────

DT_MAX_DEPTHS = [1, 2, 3, 4, 5, 7, 10, None]   # None = fully grown tree

LR_C_VALUES = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

COEF_THRESHOLD = 1e-6   # abs(coef) > threshold → treated as non-zero


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split(X, y):
    """Standard 80/20 stratified split, fixed seed for reproducibility."""
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def _dt_complexity(model):
    """Number of leaves in the fitted Decision Tree."""
    return int(model.get_n_leaves())


def _lr_complexity(model):
    """
    Number of non-zero coefficients across all classes.
    L1 regularisation drives many weights to exactly zero, so this is a
    natural measure of model sparsity and is easy to explain to beginners.
    """
    return int(np.sum(np.abs(model.coef_) > COEF_THRESHOLD))


# ── Main training functions ───────────────────────────────────────────────────

def train_decision_trees(X, y):
    """
    Train one Decision Tree per max_depth candidate.

    Returns a list of result dicts, each containing:
        model_type, label, model, pipeline, accuracy, complexity, max_depth
    """
    y_enc, le = encode_target(y)
    X_train, X_test, y_train, y_test = _split(X, y_enc)

    preprocessor = get_preprocessor()

    results = []
    for depth in DT_MAX_DEPTHS:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        pipe = Pipeline([("prep", preprocessor), ("clf", dt)])
        pipe.fit(X_train, y_train)

        acc = accuracy_score(y_test, pipe.predict(X_test))
        complexity = _dt_complexity(pipe.named_steps["clf"])

        label = f"DT depth={depth}" if depth is not None else "DT depth=unlimited"
        results.append({
            "model_type": "dt",
            "label": label,
            "max_depth": depth,
            "pipeline": pipe,
            "accuracy": round(acc, 4),
            "complexity": complexity,
        })

    return results, le


def train_logistic_regressions(X, y):
    """
    Train one Logistic Regression per C candidate.

    L1 penalty with the saga solver is used because it produces sparse
    coefficient vectors — many coefficients become exactly zero, making
    the model easier to explain.

    Returns a list of result dicts, each containing:
        model_type, label, pipeline, accuracy, complexity, C
    """
    y_enc, le = encode_target(y)
    X_train, X_test, y_train, y_test = _split(X, y_enc)

    preprocessor = get_preprocessor()

    results = []
    for C in LR_C_VALUES:
        # l1_ratio=1 is equivalent to L1 penalty (sparse coefficients).
        # Using l1_ratio instead of penalty='l1' to be forward-compatible
        # with sklearn >= 1.8 which deprecated the penalty argument.
        lr = LogisticRegression(
            solver="saga",
            l1_ratio=1.0,
            C=C,
            max_iter=5000,
            random_state=42,
        )
        pipe = Pipeline([("prep", preprocessor), ("clf", lr)])
        pipe.fit(X_train, y_train)

        acc = accuracy_score(y_test, pipe.predict(X_test))
        complexity = _lr_complexity(pipe.named_steps["clf"])

        results.append({
            "model_type": "lr",
            "label": f"LR C={C}",
            "C": C,
            "pipeline": pipe,
            "accuracy": round(acc, 4),
            "complexity": complexity,
        })

    return results, le


def select_best(results, lambda_value):
    """
    Select the model with the lowest objective for the given lambda.

    Objective (lower is better):
        objective = (1 - accuracy) + lambda * complexity_normalised

    # normalising complexity to [0, 1] before combining with
    # accuracy error makes lambda interpretable across model types —
    # lambda=0 always means "ignore complexity" and lambda=1 means
    # "ignore accuracy error", regardless of the raw complexity scale.
    """
    complexities = [r["complexity"] for r in results]
    max_c = max(complexities) if max(complexities) > 0 else 1

    best = None
    best_obj = float("inf")

    for r in results:
        norm_complexity = r["complexity"] / max_c
        obj = (1 - r["accuracy"]) + lambda_value * norm_complexity
        r["objective"] = round(obj, 4)
        r["norm_complexity"] = round(norm_complexity, 4)
        if obj < best_obj:
            best_obj = obj
            best = r

    return best


def get_lr_coef_table(pipeline, feature_names_out, class_names):
    """
    Build a coefficient table for the selected Logistic Regression model.

    Returns a list of dicts:
        [{"feature": ..., "class_0_coef": ..., ..., "nonzero": True/False}, ...]

    # showing per-class coefficients lets users see which
    # features push the model towards each species — more informative
    # than a single importance bar.
    """
    lr = pipeline.named_steps["clf"]
    coef = lr.coef_          # shape (n_classes, n_features)
    table = []
    for i, feat in enumerate(feature_names_out):
        row = {"feature": feat}
        any_nonzero = False
        for j, cls in enumerate(class_names):
            val = round(float(coef[j, i]), 4)
            row[f"coef_{cls}"] = val
            if abs(val) > COEF_THRESHOLD:
                any_nonzero = True
        row["nonzero"] = any_nonzero
        table.append(row)
    return table