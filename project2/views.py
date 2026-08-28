from django.shortcuts import render

from .services.data_utils import (
    load_penguin_data,
    NUMERICAL_FEATURES,
)

from .services.model_utils import (
    get_selected_model,
)

from .services.counterfactual_utils import (
    generate_counterfactuals,
)

from .services.effect_plot_utils import (
    compute_pdp,
    compute_ale,
)

from .services.plot_utils import (
    plot_pdp,
    plot_ale,
)

from .services.model_comparison_utils import (
    compare_models,
)

from .services.feature_relationship_utils import (
    get_feature_relationship_analysis,
)


# ---------------------------------------------------------------------------
# Shared model-selection defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_TYPE = "dt"
DEFAULT_LAMBDA = 0.02
MIN_LAMBDA = 0.0
MAX_LAMBDA = 0.10


def _get_model_settings(request):
    """
    Read model_type and lambda_value.

    On GET:
        Use the most recent values stored in the Django session.

    On POST:
        Use the submitted values and save them in the session.

    This keeps the Project 2 explanation pages linked together.
    """

    model_type = request.session.get(
        "p2_model_type",
        DEFAULT_MODEL_TYPE,
    )

    lambda_value = request.session.get(
        "p2_lambda_value",
        DEFAULT_LAMBDA,
    )

    if request.method == "POST":

        model_type = request.POST.get(
            "model_type",
            model_type,
        )

        if model_type not in {"dt", "lr"}:
            model_type = DEFAULT_MODEL_TYPE

        try:
            lambda_value = float(
                request.POST.get(
                    "lambda_value",
                    lambda_value,
                )
            )

        except (TypeError, ValueError):
            lambda_value = DEFAULT_LAMBDA

        lambda_value = max(
            MIN_LAMBDA,
            min(
                MAX_LAMBDA,
                lambda_value,
            ),
        )

        request.session[
            "p2_model_type"
        ] = model_type

        request.session[
            "p2_lambda_value"
        ] = lambda_value

    return model_type, lambda_value


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

def index(request):
    """
    Landing page — shows dataset overview.
    """

    (
        df,
        X,
        y,
        num_feats,
        cat_feats,
        class_names,
    ) = load_penguin_data()

    context = {
        "page_title": "Project 2: Explainability",
        "num_rows": len(df),
        "numerical_features": num_feats,
        "categorical_features": cat_feats,
        "all_features": (
            num_feats
            + cat_feats
        ),
        "target_name": "species",
        "class_names": class_names,
        "preview_table": df.head().to_html(
            classes="p2-table",
            index=False,
            border=0,
        ),
    }

    return render(
        request,
        "project2/index.html",
        context,
    )


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(request):
    """
    Train several candidate models and select the model that maximises:

        test_accuracy - lambda * complexity

    Decision Tree complexity:
        number of leaves

    Logistic Regression complexity:
        number of non-zero coefficients
    """

    (
        model_type,
        lambda_value,
    ) = _get_model_settings(
        request
    )

    result = None

    if request.method == "POST":

        result = get_selected_model(
            model_type,
            lambda_value,
        )

    context = {
        "page_title": (
            "Train Model — Project 2"
        ),
        "model_type": model_type,
        "lambda_value": lambda_value,
        "result": result,
    }

    return render(
        request,
        "project2/train.html",
        context,
    )


# ---------------------------------------------------------------------------
# Counterfactuals
# ---------------------------------------------------------------------------

def counterfactual(request):
    """
    Generate counterfactual explanations using the same model class
    and lambda selection shared with the other explanation pages.

    Counterfactuals can be ranked either by:

        distance:
            smallest MAD-weighted L1 distance first

        sparsity:
            fewest changed features first
    """

    (
        df,
        X,
        y,
        num_feats,
        cat_feats,
        class_names,
    ) = load_penguin_data()

    (
        model_type,
        lambda_value,
    ) = _get_model_settings(
        request
    )

    row_index = 0
    target_class = class_names[0]
    sort_by = "distance"

    cf_result = None
    error = None

    # ------------------------------------------------------------------
    # Penguin dropdown
    # ------------------------------------------------------------------

    sample_options = [
        {
            "index": i,
            "label": (
                f"Row {i} — "
                f"{row['species']}"
            ),
        }
        for i, row in df.iterrows()
    ]

    # ------------------------------------------------------------------
    # Generate counterfactual
    # ------------------------------------------------------------------

    if request.method == "POST":

        try:

            row_index = int(
                request.POST.get(
                    "row_index",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            row_index = 0

        if (
            row_index < 0
            or row_index >= len(df)
        ):

            row_index = 0

        target_class = request.POST.get(
            "target_class",
            class_names[0],
        )

        if target_class not in class_names:

            target_class = (
                class_names[0]
            )

        sort_by = request.POST.get(
            "sort_by",
            "distance",
        )

        if sort_by not in {
            "distance",
            "sparsity",
        }:

            sort_by = "distance"

        try:

            selected = (
                get_selected_model(
                    model_type,
                    lambda_value,
                )
            )

            cf_result = (
                generate_counterfactuals(
                    selected_model_info=selected,
                    row_index=row_index,
                    target_class_name=target_class,
                    sort_by=sort_by,
                )
            )

        except Exception as exc:

            error = str(
                exc
            )

    context = {
        "page_title": (
            "Counterfactuals — Project 2"
        ),
        "model_type": model_type,
        "lambda_value": lambda_value,
        "class_names": class_names,
        "sample_options": sample_options,
        "row_index": row_index,
        "target_class": target_class,
        "sort_by": sort_by,
        "cf_result": cf_result,
        "error": error,
        "num_features": num_feats,
        "cat_features": cat_feats,
        "all_features": (
            num_feats
            + cat_feats
        ),
    }

    return render(
        request,
        "project2/counterfactual.html",
        context,
    )


# ---------------------------------------------------------------------------
# PDP / ALE
# ---------------------------------------------------------------------------

def pdp_ale(request):
    """
    Generate PDP and ALE plots for the currently selected model.

    The selected feature is also stored in the session so that the
    Feature Relationships page can analyse the same feature without
    asking the user to select it again.
    """

    (
        model_type,
        lambda_value,
    ) = _get_model_settings(
        request
    )

    # --------------------------------------------------------------
    # Remember the most recently analysed feature
    # --------------------------------------------------------------

    feature_name = request.session.get(
        "p2_feature_name",
        NUMERICAL_FEATURES[0],
    )

    # Safety check in case the session contains an old/invalid value
    if feature_name not in NUMERICAL_FEATURES:

        feature_name = (
            NUMERICAL_FEATURES[0]
        )

    pdp_plot_url = None
    ale_plot_url = None
    error = None

    if request.method == "POST":

        feature_name = request.POST.get(
            "feature_name",
            NUMERICAL_FEATURES[0],
        )

        if feature_name not in NUMERICAL_FEATURES:

            feature_name = (
                NUMERICAL_FEATURES[0]
            )

        # ----------------------------------------------------------
        # Save selected feature for Feature Relationships
        # ----------------------------------------------------------

        request.session[
            "p2_feature_name"
        ] = feature_name

        try:

            selected = (
                get_selected_model(
                    model_type,
                    lambda_value,
                )
            )

            # ----------------------------------------------------------
            # PDP
            # ----------------------------------------------------------

            pdp_result = (
                compute_pdp(
                    pipeline=selected[
                        "pipeline"
                    ],
                    X=selected[
                        "X"
                    ],
                    feature_name=feature_name,
                    class_names=selected[
                        "class_names"
                    ],
                )
            )

            pdp_plot_url = (
                plot_pdp(
                    pdp_result,
                    feature_name,
                    selected[
                        "label"
                    ],
                )
            )

            # ----------------------------------------------------------
            # ALE
            # ----------------------------------------------------------

            ale_result = (
                compute_ale(
                    pipeline=selected[
                        "pipeline"
                    ],
                    X=selected[
                        "X"
                    ],
                    feature_name=feature_name,
                    class_names=selected[
                        "class_names"
                    ],
                )
            )

            ale_plot_url = (
                plot_ale(
                    ale_result,
                    feature_name,
                    selected[
                        "label"
                    ],
                )
            )

        except Exception as exc:

            error = str(
                exc
            )

    context = {
        "page_title": (
            "Feature Effect Plots — Project 2"
        ),
        "model_type": model_type,
        "lambda_value": lambda_value,
        "feature_name": feature_name,
        "numerical_features": (
            NUMERICAL_FEATURES
        ),
        "pdp_plot_url": pdp_plot_url,
        "ale_plot_url": ale_plot_url,
        "error": error,
    }

    return render(
        request,
        "project2/pdp_ale.html",
        context,
    )


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------

def model_comparison(request):
    """
    Compare the selected Decision Tree and Logistic Regression models
    for the same penguin observation.

    Both model families use the same lambda value.
    """

    (
        df,
        X,
        y,
        num_feats,
        cat_feats,
        class_names,
    ) = load_penguin_data()

    (
        model_type,
        lambda_value,
    ) = _get_model_settings(
        request
    )

    row_index = 0
    comparison = None
    error = None

    sample_options = [
        {
            "index": i,
            "label": (
                f"Row {i} — "
                f"{row['species']}"
            ),
        }
        for i, row in df.iterrows()
    ]

    if request.method == "POST":

        try:

            row_index = int(
                request.POST.get(
                    "row_index",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            row_index = 0

        if (
            row_index < 0
            or row_index >= len(df)
        ):

            row_index = 0

        try:

            comparison = (
                compare_models(
                    row_index=row_index,
                    lambda_value=lambda_value,
                )
            )

        except Exception as exc:

            error = str(
                exc
            )

    context = {
        "page_title": (
            "Model Comparison — Project 2"
        ),
        "lambda_value": lambda_value,
        "row_index": row_index,
        "sample_options": sample_options,
        "comparison": comparison,
        "error": error,
        "all_features": (
            num_feats
            + cat_feats
        ),
        "class_names": class_names,
    }

    return render(
        request,
        "project2/model_comparison.html",
        context,
    )


# ---------------------------------------------------------------------------
# Feature Relationships
# ---------------------------------------------------------------------------

def feature_relationships(request):
    """
    Analyse relationships between the numerical feature selected on
    the PDP/ALE page and the other numerical features.

    The user does not need to select the feature again.

    The page also regenerates PDP and ALE for the same:
        - feature
        - model type
        - lambda value

    This allows the user to compare the feature-effect plots while
    considering feature correlation.
    """

    (
        model_type,
        lambda_value,
    ) = _get_model_settings(
        request
    )

    # --------------------------------------------------------------
    # Get the feature selected on PDP/ALE
    # --------------------------------------------------------------

    feature_name = request.session.get(
        "p2_feature_name",
        NUMERICAL_FEATURES[0],
    )

    if feature_name not in NUMERICAL_FEATURES:

        feature_name = (
            NUMERICAL_FEATURES[0]
        )

    relationship_analysis = None

    pdp_plot_url = None
    ale_plot_url = None

    selected_model_label = None

    error = None

    try:

        # ----------------------------------------------------------
        # Use same selected model
        # ----------------------------------------------------------

        selected = (
            get_selected_model(
                model_type,
                lambda_value,
            )
        )

        selected_model_label = (
            selected[
                "label"
            ]
        )

        # ----------------------------------------------------------
        # Relationship analysis
        # ----------------------------------------------------------

        relationship_analysis = (
            get_feature_relationship_analysis(
                X=selected[
                    "X"
                ],
                feature_name=feature_name,
                numerical_features=(
                    NUMERICAL_FEATURES
                ),
            )
        )

        # ----------------------------------------------------------
        # PDP
        # ----------------------------------------------------------

        pdp_result = (
            compute_pdp(
                pipeline=selected[
                    "pipeline"
                ],
                X=selected[
                    "X"
                ],
                feature_name=feature_name,
                class_names=selected[
                    "class_names"
                ],
            )
        )

        pdp_plot_url = (
            plot_pdp(
                pdp_result,
                feature_name,
                selected[
                    "label"
                ],
            )
        )

        # ----------------------------------------------------------
        # ALE
        # ----------------------------------------------------------

        ale_result = (
            compute_ale(
                pipeline=selected[
                    "pipeline"
                ],
                X=selected[
                    "X"
                ],
                feature_name=feature_name,
                class_names=selected[
                    "class_names"
                ],
            )
        )

        ale_plot_url = (
            plot_ale(
                ale_result,
                feature_name,
                selected[
                    "label"
                ],
            )
        )

    except Exception as exc:

        error = str(
            exc
        )

    # --------------------------------------------------------------
    # Template context
    # --------------------------------------------------------------

    context = {
        "page_title": (
            "Feature Relationships — Project 2"
        ),
        "model_type": (
            model_type
        ),
        "lambda_value": (
            lambda_value
        ),
        "feature_name": (
            feature_name
        ),
        "selected_model_label": (
            selected_model_label
        ),
        "relationship_analysis": (
            relationship_analysis
        ),
        "pdp_plot_url": (
            pdp_plot_url
        ),
        "ale_plot_url": (
            ale_plot_url
        ),
        "error": (
            error
        ),
    }

    return render(
        request,
        "project2/feature_relationships.html",
        context,
    )