from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


CLASS_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def evaluate_predictions(y_true, y_pred) -> dict[str, Any]:
    labels = list(CLASS_NAMES.keys())
    target_names = list(CLASS_NAMES.values())

    raw_report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    normalized_report = {}

    for section_name, values in raw_report.items():
        if isinstance(values, dict):
            normalized_report[section_name] = {
                key.replace("-", "_"): float(value)
                for key, value in values.items()
            }
        else:
            normalized_report[section_name] = float(values)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": normalized_report,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ).tolist(),
        "class_names": target_names,
    }