"""
effect_plot_utils.py — Manual PDP, ALE, and derivative-based ALE
computation.

No external explainability library is used.


Derivative-based ALE additionally uses the fitted Logistic Regression
coefficients and the StandardScaler parameters so that the analytical
probability derivative can be computed with respect to the original
numerical feature.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================================
# PDP
# ============================================================================

def compute_pdp(
    pipeline,
    X,
    feature_name,
    class_names,
    n_grid=50,
):
    """
    Compute Partial Dependence Plot values for one numerical feature.

    Algorithm
    ---------
    For each value v in the feature grid:

        1. Copy X.
        2. Set the selected feature to v for every row.
        3. Compute predicted class probabilities.
        4. Average probabilities over all observations.

    No external PDP library is used.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
        Preprocessing + classifier.

    X : pd.DataFrame
        Original unencoded feature matrix.

    feature_name : str
        Numerical feature to analyse.

    class_names : list[str]
        Class names in the same order as predict_proba columns.

    n_grid : int
        Number of feature values at which PDP is evaluated.

    Returns
    -------
    dict
        {
            "grid_values": [...],
            "pdp_values": {
                class_name: [...]
            }
        }
    """

    col_min = float(X[feature_name].min())
    col_max = float(X[feature_name].max())

    grid_values = np.linspace(
        col_min,
        col_max,
        n_grid,
    )

    pdp_values = {
        cls: []
        for cls in class_names
    }

    for grid_val in grid_values:

        X_modified = X.copy()

        X_modified[feature_name] = grid_val

        probabilities = pipeline.predict_proba(
            X_modified
        )

        mean_probabilities = probabilities.mean(
            axis=0
        )

        for class_index, cls in enumerate(class_names):

            pdp_values[cls].append(
                round(
                    float(
                        mean_probabilities[class_index]
                    ),
                    5,
                )
            )

    return {
        "grid_values": grid_values.tolist(),
        "pdp_values": pdp_values,
    }


# ============================================================================
# STANDARD BIN-BASED ALE
# ============================================================================

def compute_ale(
    pipeline,
    X,
    feature_name,
    class_names,
    n_bins=10,
):
    """
    Compute standard bin-based Accumulated Local Effects (ALE).

    This implementation is model-agnostic and therefore works for both:

        - Decision Trees
        - Logistic Regression

    Algorithm
    ---------
    1. Divide the selected feature into quantile intervals.

    2. For each interval [z_low, z_high]:

       a. Find observations naturally occurring inside the interval.

       b. Replace the feature by z_low.

       c. Replace the feature by z_high.

       d. Compute the difference between the two predictions.

       e. Average the difference.

    3. Accumulate the local effects.

    4. Centre the accumulated curve around zero.

    No external ALE library is used.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline

    X : pd.DataFrame

    feature_name : str

    class_names : list[str]

    n_bins : int

    Returns
    -------
    dict
        {
            "bin_centres": [...],
            "ale_values": {...},
            "n_bins_used": int
        }
    """

    quantiles = np.linspace(
        0,
        100,
        n_bins + 1,
    )

    bin_edges = np.percentile(
        X[feature_name].dropna(),
        quantiles,
    )

    # Duplicate edges can occur for low-variance variables.
    bin_edges = np.unique(bin_edges)

    n_intervals = len(bin_edges) - 1

    if n_intervals == 0:

        return {
            "bin_centres": [],
            "ale_values": {
                cls: []
                for cls in class_names
            },
            "n_bins_used": 0,
        }

    local_effects = {
        cls: []
        for cls in class_names
    }

    bin_centres = []
    bin_counts = []

    X_work = X.copy()

    for interval_index in range(n_intervals):

        z_low = bin_edges[interval_index]
        z_high = bin_edges[interval_index + 1]

        # Include the maximum value in the final interval.
        if interval_index == n_intervals - 1:

            mask = (
                (X_work[feature_name] >= z_low)
                &
                (X_work[feature_name] <= z_high)
            )

        else:

            mask = (
                (X_work[feature_name] >= z_low)
                &
                (X_work[feature_name] < z_high)
            )

        X_bin = X_work.loc[mask].copy()

        bin_centre = float(
            (z_low + z_high) / 2
        )

        bin_centres.append(
            round(bin_centre, 4)
        )

        # ------------------------------------------------------------------
        # Empty interval
        # ------------------------------------------------------------------

        if len(X_bin) == 0:

            bin_counts.append(0)

            for cls in class_names:
                local_effects[cls].append(0.0)

            continue

        # ------------------------------------------------------------------
        # Evaluate lower and upper boundaries
        # ------------------------------------------------------------------

        X_low = X_bin.copy()
        X_high = X_bin.copy()

        X_low[feature_name] = z_low
        X_high[feature_name] = z_high

        proba_low = pipeline.predict_proba(
            X_low
        )

        proba_high = pipeline.predict_proba(
            X_high
        )

        local_delta = (
            proba_high - proba_low
        ).mean(axis=0)

        bin_counts.append(len(X_bin))

        for class_index, cls in enumerate(class_names):

            local_effects[cls].append(
                round(
                    float(
                        local_delta[class_index]
                    ),
                    6,
                )
            )

    # ----------------------------------------------------------------------
    # Accumulate local effects
    # ----------------------------------------------------------------------

    ale_raw = {}

    for cls in class_names:

        ale_raw[cls] = list(
            np.cumsum(
                local_effects[cls]
            )
        )

    # ----------------------------------------------------------------------
    # Centre ALE around zero
    # ----------------------------------------------------------------------

    counts = np.asarray(
        bin_counts,
        dtype=float,
    )

    total_count = counts.sum()

    ale_values = {}

    for cls in class_names:

        raw_values = np.asarray(
            ale_raw[cls],
            dtype=float,
        )

        if total_count > 0:

            weighted_mean = float(
                np.dot(
                    counts,
                    raw_values,
                )
                / total_count
            )

        else:
            weighted_mean = 0.0

        centred_values = (
            raw_values - weighted_mean
        )

        ale_values[cls] = [
            round(float(value), 5)
            for value in centred_values
        ]

    return {
        "bin_centres": bin_centres,
        "ale_values": ale_values,
        "n_bins_used": int(
            sum(
                count > 0
                for count in bin_counts
            )
        ),
    }


# ============================================================================
# DERIVATIVE-BASED ALE — LOGISTIC REGRESSION ONLY
# ============================================================================

def _get_classifier_from_pipeline(pipeline):
    """
    Return the final estimator from a fitted sklearn Pipeline.

    The project pipeline consists of preprocessing followed by the
    classifier, so the final step is expected to be LogisticRegression.
    """

    if not isinstance(pipeline, Pipeline):
        raise ValueError(
            "Derivative-based ALE requires a fitted sklearn Pipeline."
        )

    if not pipeline.steps:
        raise ValueError(
            "The supplied pipeline contains no steps."
        )

    return pipeline.steps[-1][1]


def _get_preprocessor_from_pipeline(pipeline):
    """
    Find the fitted ColumnTransformer inside the pipeline.
    """

    for _, step in pipeline.steps:

        if isinstance(step, ColumnTransformer):
            return step

        # Some projects wrap preprocessing inside another Pipeline.
        if isinstance(step, Pipeline):

            for _, nested_step in step.steps:

                if isinstance(
                    nested_step,
                    ColumnTransformer,
                ):
                    return nested_step

    raise ValueError(
        "Could not find the fitted ColumnTransformer "
        "required for derivative-based ALE."
    )


def _find_standard_scaler(transformer):
    """
    Find a StandardScaler inside either:

        StandardScaler

    or:

        Pipeline(... StandardScaler ...)
    """

    if isinstance(
        transformer,
        StandardScaler,
    ):
        return transformer

    if isinstance(
        transformer,
        Pipeline,
    ):

        for _, step in transformer.steps:

            if isinstance(
                step,
                StandardScaler,
            ):
                return step

    return None


def _get_feature_scale(
    preprocessor,
    feature_name,
):
    """
    Return the StandardScaler scale for one original numerical feature.

    The derivative of Logistic Regression is naturally expressed with
    respect to the transformed feature:

        z = (x - mean) / scale

    To obtain the derivative with respect to the original feature x,
    the derivative must therefore be divided by `scale`.
    """

    for (
        _,
        transformer,
        columns,
    ) in preprocessor.transformers_:

        if transformer == "drop":
            continue

        if isinstance(
            columns,
            (str, int),
        ):
            columns = [columns]

        else:
            try:
                columns = list(columns)
            except TypeError:
                continue

        if feature_name not in columns:
            continue

        scaler = _find_standard_scaler(
            transformer
        )

        if scaler is None:

            # If the feature is not scaled, dz/dx = 1.
            return 1.0

        feature_position = columns.index(
            feature_name
        )

        scale = float(
            scaler.scale_[feature_position]
        )

        if scale == 0:
            return 1.0

        return scale

    raise ValueError(
        f"Could not determine preprocessing scale "
        f"for feature '{feature_name}'."
    )


def _get_transformed_feature_index(
    preprocessor,
    feature_name,
):
    """
    Find the selected original feature inside the transformed design matrix.

    ColumnTransformer.get_feature_names_out() typically returns names such as:

        num__bill_length_mm
        cat__island_Biscoe

    The selected numerical feature is identified by matching either the
    complete name or the suffix after '__'.
    """

    try:
        transformed_names = list(
            preprocessor.get_feature_names_out()
        )

    except Exception as exc:

        raise ValueError(
            "Could not obtain transformed feature names "
            "from the preprocessing pipeline."
        ) from exc

    matching_indices = []

    for index, transformed_name in enumerate(
        transformed_names
    ):

        transformed_name = str(
            transformed_name
        )

        if (
            transformed_name == feature_name
            or
            transformed_name.endswith(
                f"__{feature_name}"
            )
        ):
            matching_indices.append(index)

    if len(matching_indices) != 1:

        raise ValueError(
            f"Could not uniquely locate transformed feature "
            f"'{feature_name}'."
        )

    return matching_indices[0]


def _prepare_logistic_coefficients(
    classifier,
    n_probability_classes,
):
    """
    Return one coefficient row per probability class.

    Multiclass Logistic Regression
    ------------------------------
    sklearn provides:

        coef_.shape == (n_classes, n_features)

    which can directly be used in the softmax derivative.

    Binary Logistic Regression
    --------------------------
    sklearn may provide only one coefficient vector:

        coef_.shape == (1, n_features)

    In that case we represent the two logits as:

        class 0: 0
        class 1: beta

    so the same derivative formula remains valid.
    """

    coefficients = np.asarray(
        classifier.coef_,
        dtype=float,
    )

    if (
        coefficients.shape[0]
        == n_probability_classes
    ):
        return coefficients

    if (
        n_probability_classes == 2
        and coefficients.shape[0] == 1
    ):

        zero_row = np.zeros_like(
            coefficients[0]
        )

        return np.vstack(
            [
                zero_row,
                coefficients[0],
            ]
        )

    raise ValueError(
        "Unexpected Logistic Regression coefficient shape."
    )


def compute_derivative_ale(
    pipeline,
    X,
    feature_name,
    class_names,
    n_bins=10,
):
    """
    Compute derivative-based ALE for Logistic Regression probabilities.

    This is an additional model-specific explanation.

    It does NOT replace the standard bin-based ALE implementation.

    """

    classifier = _get_classifier_from_pipeline(
        pipeline
    )

    if not hasattr(
        classifier,
        "coef_",
    ):
        raise ValueError(
            "Derivative-based ALE is only available "
            "for Logistic Regression."
        )

    preprocessor = _get_preprocessor_from_pipeline(
        pipeline
    )

    transformed_feature_index = (
        _get_transformed_feature_index(
            preprocessor,
            feature_name,
        )
    )

    feature_scale = _get_feature_scale(
        preprocessor,
        feature_name,
    )

    # ----------------------------------------------------------------------
    # Quantile ALE intervals
    # ----------------------------------------------------------------------

    feature_values = (
        X[feature_name]
        .dropna()
        .to_numpy()
    )

    if len(feature_values) == 0:

        return {
            "bin_centres": [],
            "ale_values": {
                cls: []
                for cls in class_names
            },
            "n_bins_used": 0,
            "method": "derivative",
        }

    quantiles = np.linspace(
        0,
        100,
        n_bins + 1,
    )

    bin_edges = np.percentile(
        feature_values,
        quantiles,
    )

    bin_edges = np.unique(
        bin_edges
    )

    n_intervals = (
        len(bin_edges) - 1
    )

    if n_intervals == 0:

        return {
            "bin_centres": [],
            "ale_values": {
                cls: []
                for cls in class_names
            },
            "n_bins_used": 0,
            "method": "derivative",
        }

    # ----------------------------------------------------------------------
    # Determine classifier coefficient matrix
    # ----------------------------------------------------------------------

    sample_probabilities = (
        pipeline.predict_proba(
            X.iloc[:1]
        )
    )

    n_probability_classes = (
        sample_probabilities.shape[1]
    )

    if len(class_names) != n_probability_classes:

        raise ValueError(
            "class_names does not match the number "
            "of predict_proba classes."
        )

    coefficient_matrix = (
        _prepare_logistic_coefficients(
            classifier,
            n_probability_classes,
        )
    )

    beta_feature = coefficient_matrix[
        :,
        transformed_feature_index,
    ]

    # Convert derivative from standardized units
    # back to original feature units.
    beta_feature_original = (
        beta_feature / feature_scale
    )

    # ----------------------------------------------------------------------
    # Compute derivative local effects
    # ----------------------------------------------------------------------

    local_effects = {
        cls: []
        for cls in class_names
    }

    bin_centres = []
    bin_counts = []

    for interval_index in range(
        n_intervals
    ):

        z_low = float(
            bin_edges[interval_index]
        )

        z_high = float(
            bin_edges[interval_index + 1]
        )

        if interval_index == n_intervals - 1:

            mask = (
                (X[feature_name] >= z_low)
                &
                (X[feature_name] <= z_high)
            )

        else:

            mask = (
                (X[feature_name] >= z_low)
                &
                (X[feature_name] < z_high)
            )

        X_bin = X.loc[mask].copy()

        bin_centre = (
            z_low + z_high
        ) / 2

        bin_centres.append(
            round(
                float(bin_centre),
                4,
            )
        )

        # ------------------------------------------------------------------
        # Empty interval
        # ------------------------------------------------------------------

        if len(X_bin) == 0:

            bin_counts.append(0)

            for cls in class_names:
                local_effects[cls].append(
                    0.0
                )

            continue

        # ------------------------------------------------------------------
        # Probabilities at actual observed rows
        # ------------------------------------------------------------------

        probabilities = (
            pipeline.predict_proba(
                X_bin
            )
        )

        # Shape:
        #
        # probabilities:
        #     n_samples x n_classes
        #
        # beta_feature_original:
        #     n_classes
        #
        # For each row:
        #
        # sum_l p_l * beta_lj
        #
        weighted_beta = (
            probabilities
            @ beta_feature_original
        )

        # Softmax probability derivative:
        #
        # dp_k/dx_j =
        #
        # p_k *
        # (
        #   beta_kj
        #   -
        #   sum_l p_l beta_lj
        # )
        derivatives = (
            probabilities
            *
            (
                beta_feature_original[
                    np.newaxis,
                    :
                ]
                -
                weighted_beta[
                    :,
                    np.newaxis,
                ]
            )
        )

        mean_derivative = (
            derivatives.mean(
                axis=0
            )
        )

        interval_width = (
            z_high - z_low
        )

        # Numerical approximation of integral over this interval.
        local_delta = (
            mean_derivative
            * interval_width
        )

        bin_counts.append(
            len(X_bin)
        )

        for class_index, cls in enumerate(
            class_names
        ):

            local_effects[cls].append(
                round(
                    float(
                        local_delta[
                            class_index
                        ]
                    ),
                    6,
                )
            )

    # ----------------------------------------------------------------------
    # Accumulate local derivative effects
    # ----------------------------------------------------------------------

    derivative_ale_raw = {}

    for cls in class_names:

        derivative_ale_raw[cls] = (
            np.cumsum(
                local_effects[cls]
            )
        )

    # ----------------------------------------------------------------------
    # Centre the curves
    # ----------------------------------------------------------------------

    counts = np.asarray(
        bin_counts,
        dtype=float,
    )

    total_count = counts.sum()

    derivative_ale_values = {}

    for cls in class_names:

        raw_values = np.asarray(
            derivative_ale_raw[cls],
            dtype=float,
        )

        if total_count > 0:

            weighted_mean = float(
                np.dot(
                    counts,
                    raw_values,
                )
                / total_count
            )

        else:
            weighted_mean = 0.0

        centred_values = (
            raw_values
            - weighted_mean
        )

        derivative_ale_values[cls] = [
            round(float(value), 5)
            for value in centred_values
        ]

    return {
        "bin_centres": bin_centres,
        "ale_values": derivative_ale_values,
        "n_bins_used": int(
            sum(
                count > 0
                for count in bin_counts
            )
        ),
        "method": "derivative",
    }