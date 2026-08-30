import numpy as np
import pandas as pd


def _classify_correlation_strength(correlation):
    """
    Convert the absolute Pearson correlation into a simple,
    human-readable strength category.

    Thresholds:
        |r| < 0.40        -> Weak
        0.40 <= |r| < .70 -> Moderate
        |r| >= 0.70       -> Strong
    """

    absolute_correlation = abs(
        float(correlation)
    )

    if absolute_correlation >= 0.70:
        return "Strong"

    if absolute_correlation >= 0.40:
        return "Moderate"

    return "Weak"


def _get_correlation_direction(correlation):
    """
    Return a human-readable direction for the correlation.
    """

    correlation = float(
        correlation
    )

    if correlation > 0:
        return "Positive"

    if correlation < 0:
        return "Negative"

    return "None"


def get_feature_relationship_analysis(
    X,
    feature_name,
    numerical_features,
):
    """
    Analyse the relationship between one selected numerical feature
    and all other numerical features.

    The function:

        1. calculates Pearson correlations,
        2. identifies the strongest related feature,
        3. classifies the correlation strength,
        4. returns information suitable for the UI.

    This analysis is descriptive. Correlation does not imply
    causation.
    """

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    if feature_name not in numerical_features:

        raise ValueError(
            f"'{feature_name}' is not a valid numerical feature."
        )

    missing_features = [
        feature
        for feature in numerical_features
        if feature not in X.columns
    ]

    if missing_features:

        raise ValueError(
            "The following numerical features are missing "
            f"from the dataset: {missing_features}"
        )

    # --------------------------------------------------------------
    # Numerical data only
    # --------------------------------------------------------------

    numerical_data = (
        X[
            numerical_features
        ]
        .copy()
        .astype(float)
    )

    # --------------------------------------------------------------
    # Correlation matrix
    # --------------------------------------------------------------

    correlation_matrix = (
        numerical_data.corr(
            method="pearson"
        )
    )

    # --------------------------------------------------------------
    # Correlations for selected feature
    # --------------------------------------------------------------

    selected_correlations = (
        correlation_matrix[
            feature_name
        ]
        .drop(
            labels=[
                feature_name
            ]
        )
    )

    if selected_correlations.empty:

        raise ValueError(
            "No other numerical features are available "
            "for correlation analysis."
        )

    # --------------------------------------------------------------
    # Strongest relationship
    # --------------------------------------------------------------

    strongest_feature = (
        selected_correlations
        .abs()
        .idxmax()
    )

    strongest_correlation = float(
        selected_correlations[
            strongest_feature
        ]
    )

    correlation_strength = (
        _classify_correlation_strength(
            strongest_correlation
        )
    )

    correlation_direction = (
        _get_correlation_direction(
            strongest_correlation
        )
    )

    # --------------------------------------------------------------
    # All relationships
    # --------------------------------------------------------------

    relationships = []

    for other_feature in numerical_features:

        if other_feature == feature_name:
            continue

        correlation = float(
            correlation_matrix.loc[
                feature_name,
                other_feature,
            ]
        )

        relationships.append(
            {
                "feature": (
                    other_feature
                ),
                "correlation": round(
                    correlation,
                    3,
                ),
                "absolute_correlation": round(
                    abs(correlation),
                    3,
                ),
                "strength": (
                    _classify_correlation_strength(
                        correlation
                    )
                ),
                "direction": (
                    _get_correlation_direction(
                        correlation
                    )
                ),
            }
        )

    # Strongest relationships first
    relationships.sort(
        key=lambda item: (
            item[
                "absolute_correlation"
            ]
        ),
        reverse=True,
    )

    # --------------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------------

    if correlation_strength == "Strong":

        warning_level = "high"

        interpretation = (
            f"{feature_name} has a strong "
            f"{correlation_direction.lower()} relationship with "
            f"{strongest_feature}. PDP changes the selected feature "
            f"while leaving the remaining features unchanged. With "
            f"strongly related features, this may evaluate combinations "
            f"that are uncommon in the observed dataset. ALE focuses on "
            f"local changes and is generally less affected by this issue."
        )

    elif correlation_strength == "Moderate":

        warning_level = "medium"

        interpretation = (
            f"{feature_name} has a moderate "
            f"{correlation_direction.lower()} relationship with "
            f"{strongest_feature}. Some caution is useful when "
            f"interpreting PDP because independently changing the "
            f"selected feature may create less common feature "
            f"combinations. ALE uses more local comparisons and can "
            f"provide a useful complementary view."
        )

    else:

        warning_level = "low"

        interpretation = (
            f"No strong numerical relationship was detected for "
            f"{feature_name}. Its strongest relationship is with "
            f"{strongest_feature}. Correlation-related concerns for "
            f"PDP are therefore less pronounced for this feature, "
            f"although PDP and ALE still describe model behaviour "
            f"in different ways."
        )

    # --------------------------------------------------------------
    # Correlation matrix for display
    # --------------------------------------------------------------

    matrix_rows = []

    for row_feature in numerical_features:

        values = []

        for column_feature in numerical_features:

            value = float(
                correlation_matrix.loc[
                    row_feature,
                    column_feature,
                ]
            )

            values.append(
                {
                    "feature": (
                        column_feature
                    ),
                    "value": round(
                        value,
                        3,
                    ),
                }
            )

        matrix_rows.append(
            {
                "feature": (
                    row_feature
                ),
                "values": (
                    values
                ),
            }
        )

    # --------------------------------------------------------------
    # Result
    # --------------------------------------------------------------

    return {
        "selected_feature": (
            feature_name
        ),
        "strongest_feature": (
            strongest_feature
        ),
        "strongest_correlation": round(
            strongest_correlation,
            3,
        ),
        "absolute_correlation": round(
            abs(
                strongest_correlation
            ),
            3,
        ),
        "strength": (
            correlation_strength
        ),
        "direction": (
            correlation_direction
        ),
        "warning_level": (
            warning_level
        ),
        "interpretation": (
            interpretation
        ),
        "relationships": (
            relationships
        ),
        "matrix_rows": (
            matrix_rows
        ),
        "matrix_features": list(
            numerical_features
        ),
    }