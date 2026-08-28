import numpy as np
import pandas as pd

from .data_utils import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


def _compute_mads(df):
    """
    Compute Median Absolute Deviation (MAD) for numerical features.

    MAD is used to normalise numerical differences when calculating
    counterfactual distance.
    """

    mads = {}

    for feature in NUMERICAL_FEATURES:

        values = df[feature].astype(float)

        median = np.median(values)

        mad = np.median(
            np.abs(values - median)
        )

        # Fallback if MAD is zero
        if mad == 0:
            mad = np.std(values)

        # Final safety fallback
        if mad == 0:
            mad = 1.0

        mads[feature] = float(mad)

    return mads


def _l1_distance(
    original,
    candidate,
    mads,
):
    """
    Compute MAD-weighted L1 distance.

    Numerical features:
        |x - x'| / MAD

    Categorical features:
        0 if unchanged
        1 if changed
    """

    distance = 0.0

    # Numerical features
    for feature in NUMERICAL_FEATURES:

        original_value = float(
            original[feature]
        )

        candidate_value = float(
            candidate[feature]
        )

        distance += (
            abs(
                original_value
                - candidate_value
            )
            / mads[feature]
        )

    # Categorical features
    for feature in CATEGORICAL_FEATURES:

        if (
            original[feature]
            != candidate[feature]
        ):
            distance += 1.0

    return float(distance)


def _sample_numerical(
    original_value,
    feature,
    mads,
    feature_ranges,
    rng,
    noise_scale,
):
    """
    Generate a nearby numerical value using Gaussian noise.

    Noise is scaled using MAD and clipped to the observed
    overall data range.
    """

    noise = rng.normal(
        loc=0.0,
        scale=(
            mads[feature]
            * noise_scale
        ),
    )

    candidate_value = (
        float(original_value)
        + noise
    )

    minimum, maximum = (
        feature_ranges[feature]
    )

    candidate_value = np.clip(
        candidate_value,
        minimum,
        maximum,
    )

    return float(
        candidate_value
    )


def _sample_categorical(
    original_value,
    feature,
    df,
    rng,
):
    """
    Generate a categorical value.

    Keep the original category 70% of the time.
    Otherwise randomly select a valid observed category.
    """

    if rng.random() < 0.70:
        return original_value

    possible_values = (
        df[feature]
        .dropna()
        .unique()
        .tolist()
    )

    if not possible_values:
        return original_value

    return rng.choice(
        possible_values
    )


def _check_target_class_plausibility(
    candidate,
    df,
    target_class_name,
):
    """
    Check whether the numerical values of a counterfactual fall
    within the observed min/max ranges of the desired species.

    This is only a simple range-based plausibility heuristic.

    It does NOT prove that the complete combination of features
    represents a biologically realistic penguin.
    """

    target_rows = df[
        df["species"]
        == target_class_name
    ]

    if target_rows.empty:

        return {
            "is_plausible": False,
            "label": "Unknown",
        }

    outside_features = []

    for feature in NUMERICAL_FEATURES:

        minimum = float(
            target_rows[
                feature
            ].min()
        )

        maximum = float(
            target_rows[
                feature
            ].max()
        )

        value = float(
            candidate[feature]
        )

        if (
            value < minimum
            or value > maximum
        ):

            outside_features.append(
                feature
            )

    if outside_features:

        return {
            "is_plausible": False,
            "label": (
                "Outside target-class range"
            ),
            "outside_features": (
                outside_features
            ),
        }

    return {
        "is_plausible": True,
        "label": (
            "Within target-class ranges"
        ),
        "outside_features": [],
    }


def _compute_changes(
    original_row,
    candidate,
):
    """
    Determine which features changed between the original
    observation and a candidate counterfactual.
    """

    changes = {}

    # Numerical features
    for feature in NUMERICAL_FEATURES:

        changes[feature] = (
            not np.isclose(
                float(
                    original_row[feature]
                ),
                float(
                    candidate[feature]
                ),
            )
        )

    # Categorical features
    for feature in CATEGORICAL_FEATURES:

        changes[feature] = (
            original_row[feature]
            != candidate[feature]
        )

    return changes


def _count_changes(
    changes,
):
    """
    Count the number of changed features.

    This is used as a simple sparsity measure:
    fewer changed features = sparser counterfactual.
    """

    return sum(
        1
        for changed
        in changes.values()
        if changed
    )


def _prepare_original_display(
    original_row,
):
    """
    Prepare the original observation for display in the UI.

    Numerical values are rounded only for presentation.
    """

    original_display = {}

    for feature in NUMERICAL_FEATURES:

        original_display[feature] = round(
            float(
                original_row[feature]
            ),
            2,
        )

    for feature in CATEGORICAL_FEATURES:

        original_display[feature] = (
            original_row[feature]
        )

    return original_display


def generate_counterfactuals(
    selected_model_info,
    row_index,
    target_class_name,
    N=5000,
    k=5,
    noise_scale=1.0,
    max_retries=2,
    sort_by="distance",
):
    """
    Generate counterfactual explanations.

    Main procedure:

    1. Select an original observation.
    2. Generate N nearby candidate observations.
    3. Predict each candidate.
    4. Keep candidates predicted as the desired class.
    5. Compute MAD-weighted L1 distance.
    6. Compute sparsity as number of changed features.
    7. Estimate simple target-class range plausibility.
    8. Rank candidates.
    9. Return the best k counterfactuals.

    Ranking options:

    distance:
        Primary = smallest MAD-weighted distance.
        Secondary = fewer changed features.

    sparsity:
        Primary = fewest changed features.
        Secondary = smallest MAD-weighted distance.
    """

    # ============================================================
    # Validate ranking method
    # ============================================================

    if sort_by not in {
        "distance",
        "sparsity",
    }:

        sort_by = "distance"

    # ============================================================
    # Selected model information
    # ============================================================

    pipeline = (
        selected_model_info[
            "pipeline"
        ]
    )

    df = (
        selected_model_info[
            "df"
        ]
    )

    class_names = list(
        selected_model_info[
            "class_names"
        ]
    )

    # ============================================================
    # Validate row
    # ============================================================

    if (
        row_index < 0
        or row_index >= len(df)
    ):

        raise ValueError(
            f"Invalid row index: "
            f"{row_index}"
        )

    # ============================================================
    # Validate desired target
    # ============================================================

    if (
        target_class_name
        not in class_names
    ):

        raise ValueError(
            f"Unknown target class: "
            f"{target_class_name}"
        )

    # ============================================================
    # Feature columns
    # ============================================================

    feature_columns = (
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
    )

    # ============================================================
    # Original observation
    # ============================================================

    original_row = (
        df.iloc[row_index][
            feature_columns
        ]
        .copy()
    )

    original_frame = pd.DataFrame(
        [original_row],
        columns=feature_columns,
    )

    # ============================================================
    # Determine classifier class mapping
    # ============================================================

    clf = (
        pipeline.named_steps[
            "clf"
        ]
    )

    model_classes = list(
        clf.classes_
    )

    # The classifier was trained using encoded integer labels.
    # class_names corresponds to the same ordering.
    class_to_model_value = {
        class_name: model_value
        for class_name, model_value
        in zip(
            class_names,
            model_classes,
        )
    }

    model_value_to_class = {
        model_value: class_name
        for class_name, model_value
        in zip(
            class_names,
            model_classes,
        )
    }

    # ============================================================
    # Original prediction
    # ============================================================

    original_prediction_value = (
        pipeline.predict(
            original_frame
        )[0]
    )

    original_prediction = (
        model_value_to_class[
            original_prediction_value
        ]
    )

    # ============================================================
    # Desired target model value
    # ============================================================

    target_model_value = (
        class_to_model_value[
            target_class_name
        ]
    )

    # ============================================================
    # Prepare original row for display
    # ============================================================

    original_display = (
        _prepare_original_display(
            original_row
        )
    )

    # ============================================================
    # If model already predicts desired target
    # ============================================================

    if (
        original_prediction
        == target_class_name
    ):

        return {
            "status": (
                "already_target"
            ),
            "original_row": (
                original_display
            ),
            "original_prediction": (
                original_prediction
            ),
            "target_class": (
                target_class_name
            ),
            "counterfactuals": [],
            "sort_by": sort_by,
            "message": (
                "The selected model already predicts "
                f"'{target_class_name}' "
                "for this penguin."
            ),
        }

    # ============================================================
    # Compute MAD values
    # ============================================================

    mads = _compute_mads(
        df
    )

    # ============================================================
    # Overall numerical feature ranges
    # ============================================================

    feature_ranges = {}

    for feature in NUMERICAL_FEATURES:

        feature_ranges[feature] = (
            float(
                df[
                    feature
                ].min()
            ),
            float(
                df[
                    feature
                ].max()
            ),
        )

    # ============================================================
    # Reproducible random generator
    # ============================================================

    rng = (
        np.random.default_rng(
            42
        )
    )

    found_candidates = []

    current_noise_scale = (
        noise_scale
    )

    # ============================================================
    # Counterfactual search
    # ============================================================

    for attempt in range(
        max_retries + 1
    ):

        candidates = []

        # --------------------------------------------------------
        # Generate local candidates
        # --------------------------------------------------------

        for _ in range(N):

            candidate = {}

            # Numerical features
            for feature in (
                NUMERICAL_FEATURES
            ):

                candidate[feature] = (
                    _sample_numerical(
                        original_value=(
                            original_row[
                                feature
                            ]
                        ),
                        feature=feature,
                        mads=mads,
                        feature_ranges=(
                            feature_ranges
                        ),
                        rng=rng,
                        noise_scale=(
                            current_noise_scale
                        ),
                    )
                )

            # Categorical features
            for feature in (
                CATEGORICAL_FEATURES
            ):

                candidate[feature] = (
                    _sample_categorical(
                        original_value=(
                            original_row[
                                feature
                            ]
                        ),
                        feature=feature,
                        df=df,
                        rng=rng,
                    )
                )

            candidates.append(
                candidate
            )

        # --------------------------------------------------------
        # Candidate DataFrame
        # --------------------------------------------------------

        candidate_df = pd.DataFrame(
            candidates,
            columns=feature_columns,
        )

        # --------------------------------------------------------
        # Predict candidates
        # --------------------------------------------------------

        predictions = (
            pipeline.predict(
                candidate_df
            )
        )

        # --------------------------------------------------------
        # Keep candidates predicted as target class
        # --------------------------------------------------------

        matching_indices = np.where(
            predictions
            == target_model_value
        )[0]

        if (
            len(
                matching_indices
            )
            > 0
        ):

            for index in (
                matching_indices
            ):

                candidate = (
                    candidates[
                        index
                    ]
                )

                # -----------------------------------------------
                # Distance
                # -----------------------------------------------

                distance = (
                    _l1_distance(
                        original=(
                            original_row
                        ),
                        candidate=(
                            candidate
                        ),
                        mads=mads,
                    )
                )

                # -----------------------------------------------
                # Changed features
                # -----------------------------------------------

                changes = (
                    _compute_changes(
                        original_row=(
                            original_row
                        ),
                        candidate=(
                            candidate
                        ),
                    )
                )

                # -----------------------------------------------
                # Sparsity
                # -----------------------------------------------

                num_changes = (
                    _count_changes(
                        changes
                    )
                )

                # -----------------------------------------------
                # Range plausibility
                # -----------------------------------------------

                plausibility = (
                    _check_target_class_plausibility(
                        candidate=(
                            candidate
                        ),
                        df=df,
                        target_class_name=(
                            target_class_name
                        ),
                    )
                )

                # -----------------------------------------------
                # Store enriched candidate
                # -----------------------------------------------

                found_candidates.append(
                    {
                        "distance": (
                            float(
                                distance
                            )
                        ),
                        "candidate": (
                            candidate
                        ),
                        "changes": (
                            changes
                        ),
                        "num_changes": (
                            num_changes
                        ),
                        "plausibility": (
                            plausibility
                        ),
                    }
                )

            # Stop once at least one target candidate
            # has been found.
            break

        # --------------------------------------------------------
        # Increase search radius if nothing was found
        # --------------------------------------------------------

        current_noise_scale *= (
            2.0
        )

    # ============================================================
    # No candidates found
    # ============================================================

    if not found_candidates:

        return {
            "status": (
                "not_found"
            ),
            "original_row": (
                original_display
            ),
            "original_prediction": (
                original_prediction
            ),
            "target_class": (
                target_class_name
            ),
            "counterfactuals": [],
            "sort_by": sort_by,
            "message": (
                "No counterfactuals were found. "
                "Try another target species, model, "
                "λ value, or increase the search size."
            ),
        }

    # ============================================================
    # Rank candidates
    # ============================================================

    if sort_by == "sparsity":

        found_candidates.sort(
            key=lambda item: (
                item[
                    "num_changes"
                ],
                item[
                    "distance"
                ],
            )
        )

    else:

        found_candidates.sort(
            key=lambda item: (
                item[
                    "distance"
                ],
                item[
                    "num_changes"
                ],
            )
        )

    best_candidates = (
        found_candidates[
            :k
        ]
    )

    # ============================================================
    # Prepare counterfactuals for UI
    # ============================================================

    counterfactuals = []

    for item in best_candidates:

        candidate = (
            item[
                "candidate"
            ]
        )

        entry = {}

        # --------------------------------------------------------
        # Numerical features
        # --------------------------------------------------------

        for feature in (
            NUMERICAL_FEATURES
        ):

            full_value = float(
                candidate[
                    feature
                ]
            )

            # Only round for display.
            # Prediction and distance use full precision.
            entry[feature] = round(
                full_value,
                2,
            )

        # --------------------------------------------------------
        # Categorical features
        # --------------------------------------------------------

        for feature in (
            CATEGORICAL_FEATURES
        ):

            entry[feature] = (
                candidate[
                    feature
                ]
            )

        # --------------------------------------------------------
        # Explanation metadata
        # --------------------------------------------------------

        entry["distance"] = round(
            float(
                item[
                    "distance"
                ]
            ),
            4,
        )

        entry["changes"] = (
            item[
                "changes"
            ]
        )

        entry["num_changes"] = (
            item[
                "num_changes"
            ]
        )

        entry["plausibility"] = (
            item[
                "plausibility"
            ]
        )

        counterfactuals.append(
            entry
        )

    # ============================================================
    # Final result
    # ============================================================

    return {
        "status": (
            "success"
        ),
        "original_row": (
            original_display
        ),
        "original_prediction": (
            original_prediction
        ),
        "target_class": (
            target_class_name
        ),
        "counterfactuals": (
            counterfactuals
        ),
        "sort_by": sort_by,
        "message": (
            f"Found "
            f"{len(counterfactuals)} "
            f"counterfactual"
            f"{'s' if len(counterfactuals) != 1 else ''} "
            f"for target class "
            f"'{target_class_name}'."
        ),
    }