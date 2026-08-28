import numpy as np
import pandas as pd

from .model_utils import (
    get_selected_model,
)


def _get_prediction_details(
    selected_model,
    row_index,
):
    """
    Generate prediction information for one selected model
    and one dataset observation.

    Returns:
        prediction
        confidence
        probability distribution
        model accuracy
        model complexity
        selected hyperparameter label
    """

    pipeline = (
        selected_model[
            "pipeline"
        ]
    )

    df = (
        selected_model[
            "df"
        ]
    )

    label_encoder = (
        selected_model[
            "le"
        ]
    )

    numerical_features = (
        selected_model[
            "numerical_features"
        ]
    )

    categorical_features = (
        selected_model[
            "categorical_features"
        ]
    )

    class_names = list(
        selected_model[
            "class_names"
        ]
    )

    feature_columns = (
        numerical_features
        + categorical_features
    )

    # ------------------------------------------------------------
    # Validate row
    # ------------------------------------------------------------

    if (
        row_index < 0
        or row_index >= len(df)
    ):

        raise ValueError(
            f"Invalid row index: "
            f"{row_index}"
        )

    # ------------------------------------------------------------
    # Selected observation
    # ------------------------------------------------------------

    row = (
        df.iloc[
            row_index
        ][
            feature_columns
        ]
        .copy()
    )

    row_frame = pd.DataFrame(
        [row],
        columns=feature_columns,
    )

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    prediction_encoded = int(
        pipeline.predict(
            row_frame
        )[0]
    )

    prediction_name = (
        label_encoder
        .inverse_transform(
            [
                prediction_encoded
            ]
        )[0]
    )

    # ------------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------------

    probabilities = (
        pipeline.predict_proba(
            row_frame
        )[0]
    )

    confidence = float(
        np.max(
            probabilities
        )
    )

    probability_distribution = []

    for (
        class_name,
        probability,
    ) in zip(
        class_names,
        probabilities,
    ):

        probability_distribution.append(
            {
                "class_name": (
                    class_name
                ),
                "probability": round(
                    float(
                        probability
                    )
                    * 100,
                    2,
                ),
            }
        )

    # ------------------------------------------------------------
    # Prepare observation for display
    # ------------------------------------------------------------

    display_row = {}

    for feature in numerical_features:

        display_row[
            feature
        ] = round(
            float(
                row[
                    feature
                ]
            ),
            2,
        )

    for feature in categorical_features:

        display_row[
            feature
        ] = (
            row[
                feature
            ]
        )

    return {
        "prediction": (
            prediction_name
        ),
        "confidence": round(
            confidence
            * 100,
            2,
        ),
        "probabilities": (
            probability_distribution
        ),
        "accuracy": round(
            float(
                selected_model[
                    "accuracy"
                ]
            )
            * 100,
            2,
        ),
        "complexity": (
            selected_model[
                "complexity"
            ]
        ),
        "model_label": (
            selected_model[
                "label"
            ]
        ),
        "hyperparam": (
            selected_model[
                "hyperparam"
            ]
        ),
        "objective": (
            selected_model[
                "objective"
            ]
        ),
        "observation": (
            display_row
        ),
    }


def compare_models(
    row_index,
    lambda_value,
):
    """
    Compare the selected Decision Tree and Logistic Regression
    models for the same observation and lambda value.

    The function uses the existing model-selection logic:

        accuracy - lambda * complexity

    for both model families.
    """

    # ------------------------------------------------------------
    # Select Decision Tree
    # ------------------------------------------------------------

    dt_selected = (
        get_selected_model(
            "dt",
            lambda_value,
        )
    )

    # ------------------------------------------------------------
    # Select Logistic Regression
    # ------------------------------------------------------------

    lr_selected = (
        get_selected_model(
            "lr",
            lambda_value,
        )
    )

    # ------------------------------------------------------------
    # Prediction information
    # ------------------------------------------------------------

    decision_tree = (
        _get_prediction_details(
            dt_selected,
            row_index,
        )
    )

    logistic_regression = (
        _get_prediction_details(
            lr_selected,
            row_index,
        )
    )

    # ------------------------------------------------------------
    # Agreement
    # ------------------------------------------------------------

    agree = (
        decision_tree[
            "prediction"
        ]
        ==
        logistic_regression[
            "prediction"
        ]
    )

    # ------------------------------------------------------------
    # Confidence difference
    # ------------------------------------------------------------

    confidence_difference = abs(
        decision_tree[
            "confidence"
        ]
        -
        logistic_regression[
            "confidence"
        ]
    )

    # ------------------------------------------------------------
    # Human-readable interpretation
    # ------------------------------------------------------------

    if agree:

        interpretation = (
            "Both models make the same prediction for this "
            "observation. However, their confidence and internal "
            "reasoning may still differ, so explanations remain "
            "model-specific."
        )

    else:

        interpretation = (
            "The models disagree on this observation. This shows "
            "that the predicted class and its explanation depend "
            "on the selected model. Counterfactuals and feature "
            "effect explanations should therefore be interpreted "
            "as model-dependent rather than as properties of the "
            "penguin itself."
        )

    return {
        "row_index": (
            row_index
        ),
        "lambda_value": (
            lambda_value
        ),
        "agree": (
            agree
        ),
        "decision_tree": (
            decision_tree
        ),
        "logistic_regression": (
            logistic_regression
        ),
        "confidence_difference": round(
            confidence_difference,
            2,
        ),
        "interpretation": (
            interpretation
        ),
    }