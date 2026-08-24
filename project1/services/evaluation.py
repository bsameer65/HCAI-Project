from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


CLASSIFICATION_METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
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