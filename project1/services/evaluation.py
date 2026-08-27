from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import math


CLASSIFICATION_METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
}

REGRESSION_METRICS = {
    "mae": "Mean Absolute Error (MAE)",
    "mse": "Mean Squared Error (MSE)",
    "rmse": "Root Mean Squared Error (RMSE)",
    "r2": "R² Score",
}


class UnsupportedMetricError(ValueError):
    """Raised when an unsupported evaluation metric is requested."""
    pass


def evaluate_classifier(y_true, y_pred):
    """
    Calculate the main classification metrics.

    Weighted averaging is used for precision, recall and F1 so that
    multi-class classification datasets are supported.
    """

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }


def get_selected_metric(metrics, metric_name):
    """
    Return the score corresponding to the user's selected metric.
    """

    if metric_name not in CLASSIFICATION_METRICS:
        raise UnsupportedMetricError(
            f"Unsupported classification metric: {metric_name}"
        )

    return metrics[metric_name]


def calculate_confusion_matrix(y_true, y_pred):
    """
    Return the confusion matrix for additional model evaluation.
    """

    return confusion_matrix(
        y_true,
        y_pred,
    )


def evaluate_regressor(y_true, y_pred):
    """Calculate the four regression metrics used throughout the UI."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": r2_score(y_true, y_pred),
    }


def get_selected_regression_metric(metrics, metric_name):
    if metric_name not in REGRESSION_METRICS:
        raise UnsupportedMetricError(
            f"Unsupported regression metric: {metric_name}"
        )
    return metrics[metric_name]
