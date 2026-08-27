import numpy as np

from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


class ExplainabilityError(Exception):
    """Raised when model explainability cannot be calculated."""
    pass


def _get_final_estimator(model):
    """
    Return the final estimator when the model is a Pipeline.

    Example:
        StandardScaler -> LogisticRegression

    returns LogisticRegression.
    """

    if isinstance(model, Pipeline):
        return model.steps[-1][1]

    return model


def get_feature_importance(
    model,
    model_name,
    feature_columns,
    X_test,
    y_test,
):
    """
    Calculate feature importance for a trained classifier.

    Different algorithms require different explanation techniques:

    Decision Tree / Random Forest:
        native feature_importances_

    Logistic Regression:
        absolute model coefficients

    KNN:
        permutation importance

    Returns
    -------
    dict
        {
            "method": "...",
            "items": [
                {
                    "feature": "...",
                    "importance": ...
                }
            ]
        }
    """

    estimator = _get_final_estimator(model)

    # ==========================================================
    # Decision Tree / Random Forest
    # ==========================================================

    if model_name in {
        "decision_tree",
        "random_forest",
    }:

        if not hasattr(
            estimator,
            "feature_importances_",
        ):
            raise ExplainabilityError(
                "This model does not provide feature importance."
            )

        importances = (
            estimator.feature_importances_
        )

        method = "Feature Importance"

    # ==========================================================
    # Logistic Regression
    # ==========================================================

    elif model_name == "logistic_regression":

        if not hasattr(
            estimator,
            "coef_",
        ):
            raise ExplainabilityError(
                "Logistic Regression coefficients are unavailable."
            )

        coefficients = estimator.coef_

        # Multiclass Logistic Regression has one coefficient
        # vector per class. We calculate the average absolute
        # magnitude for each feature.
        importances = np.mean(
            np.abs(coefficients),
            axis=0,
        )

        method = "Coefficient Magnitude"

    # ==========================================================
    # KNN
    # ==========================================================

    elif model_name == "knn":

        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring="accuracy",
        )

        importances = (
            result.importances_mean
        )

        method = "Permutation Importance"

    else:

        raise ExplainabilityError(
            f"Explainability is not supported for '{model_name}'."
        )

    # ==========================================================
    # Convert to normalized values
    # ==========================================================

    importances = np.asarray(
        importances,
        dtype=float,
    )

    # Avoid negative permutation values making the visualization
    # confusing.
    importances = np.maximum(
        importances,
        0,
    )

    total = importances.sum()

    if total > 0:
        normalized = (
            importances / total
        )

    else:
        normalized = importances

    items = [
        {
            "feature": feature,
            "importance": float(importance),
            "percentage": round(
                float(importance) * 100,
                2,
            ),
        }
        for feature, importance
        in zip(
            feature_columns,
            normalized,
        )
    ]

    # Highest importance first
    items.sort(
        key=lambda item:
            item["importance"],
        reverse=True,
    )

    return {
        "method": method,
        "items": items,
    }


def get_regression_feature_importance(
    model,
    model_name,
    X_test,
    y_test,
):
    """Explain a fitted regression pipeline at raw or transformed feature level."""
    estimator = _get_final_estimator(model)
    preprocessor = model.named_steps.get("preprocessor")

    if model_name in {
        "decision_tree_regressor",
        "random_forest_regressor",
    }:
        if not hasattr(estimator, "feature_importances_"):
            raise ExplainabilityError("This regressor has no native feature importance.")
        feature_names = preprocessor.get_feature_names_out().tolist()
        importances = estimator.feature_importances_
        method = "Native Feature Importance (after preprocessing)"
    elif model_name == "linear_regression":
        if not hasattr(estimator, "coef_"):
            raise ExplainabilityError("Linear Regression coefficients are unavailable.")
        feature_names = preprocessor.get_feature_names_out().tolist()
        importances = np.abs(np.asarray(estimator.coef_).reshape(-1))
        method = "Absolute Coefficient Magnitude (after preprocessing)"
    elif model_name == "knn_regressor":
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring="neg_mean_absolute_error",
        )
        feature_names = X_test.columns.tolist()
        importances = result.importances_mean
        method = "Permutation Importance (MAE decrease)"
    else:
        raise ExplainabilityError(
            f"Regression explainability is not supported for '{model_name}'."
        )

    importances = np.maximum(np.asarray(importances, dtype=float), 0)
    if len(feature_names) != len(importances):
        raise ExplainabilityError("Feature names could not be aligned with importance values.")
    total = importances.sum()
    normalized = importances / total if total > 0 else importances
    items = [
        {
            "feature": str(feature),
            "importance": float(importance),
            "percentage": round(float(importance) * 100, 2),
        }
        for feature, importance in zip(feature_names, normalized)
    ]
    items.sort(key=lambda item: item["importance"], reverse=True)
    return {"method": method, "items": items}
