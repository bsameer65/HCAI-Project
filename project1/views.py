import os

import pandas as pd

from django.conf import settings
from django.shortcuts import render, redirect

from sklearn.model_selection import train_test_split


# ==============================================================
# Forms
# ==============================================================

from .forms import (
    ClassificationTrainingForm,
    get_classifier_parameters,
)


# ==============================================================
# Dataset
# ==============================================================

from .services.dataset import (
    load_dataset,
    DatasetValidationError,
)


# ==============================================================
# Algorithms
# ==============================================================

from .services.algorithms import (
    create_classifier,
    CLASSIFIER_CHOICES,
    UnsupportedAlgorithmError,
)


# ==============================================================
# Evaluation
# ==============================================================

from .services.evaluation import (
    evaluate_classifier,
    get_selected_metric,
    CLASSIFICATION_METRICS,
    UnsupportedMetricError,
)


# ==============================================================
# Visualization
# ==============================================================

from .services.visualization import (
    create_classification_scatter,
    create_class_distribution,
    create_accuracy_comparison_chart,
)


# ==============================================================
# Persistence
# ==============================================================

from .services.persistence import (
    save_classification_model,
    load_classification_model,
    ModelNotFoundError,
)


# ==============================================================
# Constants
# ==============================================================

CLASSIFICATION_DATASET_FILENAME = (
    "current_classification_dataset.csv"
)


# ==============================================================
# Helper functions
# ==============================================================

def _get_upload_directory():
    """
    Return the directory used to store uploaded datasets.
    """

    upload_directory = os.path.join(
        settings.MEDIA_ROOT,
        "uploads",
    )

    os.makedirs(
        upload_directory,
        exist_ok=True,
    )

    return upload_directory


def _get_classification_dataset_path():
    """
    Return the path of the currently uploaded
    classification dataset.
    """

    return os.path.join(
        _get_upload_directory(),
        CLASSIFICATION_DATASET_FILENAME,
    )


def _validate_numeric_features(
    df,
    feature_columns,
):
    """
    Return non-numeric input feature names.

    The current classification algorithms expect
    numerical description features.
    """

    return [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]


# ==============================================================
# Project landing page
# ==============================================================

def index(request):
    """
    Project 1 landing page.

    User explicitly selects classification
    or regression.
    """

    return render(
        request,
        "project1/index.html",
    )


# ==============================================================
# CLASSIFICATION — Upload
# ==============================================================

def classification(request):
    """
    Upload and validate a classification dataset.
    """

    context = {}

    if request.method == "POST":

        uploaded_file = request.FILES.get(
            "csv_file"
        )

        # ------------------------------------------------------
        # No file selected
        # ------------------------------------------------------

        if uploaded_file is None:

            context["error"] = (
                "Please choose a CSV file "
                "before continuing."
            )

            return render(
                request,
                "project1/classification.html",
                context,
            )

        # ------------------------------------------------------
        # Validate extension
        # ------------------------------------------------------

        if not uploaded_file.name.lower().endswith(
            ".csv"
        ):

            context["error"] = (
                "Only CSV files are supported."
            )

            return render(
                request,
                "project1/classification.html",
                context,
            )

        file_path = (
            _get_classification_dataset_path()
        )

        try:

            # --------------------------------------------------
            # Save / replace current dataset
            # --------------------------------------------------

            with open(
                file_path,
                "wb+",
            ) as destination:

                for chunk in uploaded_file.chunks():
                    destination.write(
                        chunk
                    )

            # --------------------------------------------------
            # Validate dataset
            # --------------------------------------------------

            (
                df,
                feature_columns,
                target_column,
                removed_identifier_columns,
            ) = load_dataset(
                file_path
            )

            # --------------------------------------------------
            # Check features are numeric
            # --------------------------------------------------

            non_numeric_features = (
                _validate_numeric_features(
                    df,
                    feature_columns,
                )
            )

            if non_numeric_features:

                raise DatasetValidationError(
                    "The following feature columns "
                    "are not numeric: "
                    + ", ".join(
                        non_numeric_features
                    )
                    + "."
                )

        except DatasetValidationError as error:

            if os.path.exists(
                file_path
            ):
                os.remove(
                    file_path
                )

            context["error"] = str(
                error
            )

            return render(
                request,
                "project1/classification.html",
                context,
            )

        # ------------------------------------------------------
        # IMPORTANT:
        # After upload → Analysis page
        # ------------------------------------------------------

        return redirect(
            "project1:classification_analyze"
        )

    return render(
        request,
        "project1/classification.html",
        context,
    )


# ==============================================================
# CLASSIFICATION — Analysis
# ==============================================================

def classification_analyze(request):
    """
    Visualize the uploaded classification dataset.
    """

    context = {}

    file_path = (
        _get_classification_dataset_path()
    )

    try:

        (
            df,
            feature_columns,
            target_column,
            removed_identifier_columns,
        ) = load_dataset(
            file_path
        )

    except DatasetValidationError as error:

        context["error"] = str(
            error
        )

        return render(
            request,
            "project1/classification.html",
            context,
        )

    # ----------------------------------------------------------
    # Selected visualization columns
    # ----------------------------------------------------------

    x_column = (
        request.POST.get(
            "x_col"
        )
        or feature_columns[0]
    )

    if len(feature_columns) > 1:

        default_y = (
            feature_columns[1]
        )

    else:

        default_y = (
            feature_columns[0]
        )

    y_column = (
        request.POST.get(
            "y_col"
        )
        or default_y
    )

    # ----------------------------------------------------------
    # Validate POSTed column names
    # ----------------------------------------------------------

    if x_column not in feature_columns:

        x_column = (
            feature_columns[0]
        )

    if y_column not in feature_columns:

        y_column = (
            default_y
        )

    # ----------------------------------------------------------
    # Scatter plot
    # ----------------------------------------------------------

    scatter_plot_url = (
        create_classification_scatter(
            df=df,
            x_column=x_column,
            y_column=y_column,
            target_column=target_column,
            media_root=settings.MEDIA_ROOT,
            media_url=settings.MEDIA_URL,
        )
    )

    # ----------------------------------------------------------
    # Class distribution
    # ----------------------------------------------------------

    class_distribution_url = (
        create_class_distribution(
            df=df,
            target_column=target_column,
            media_root=settings.MEDIA_ROOT,
            media_url=settings.MEDIA_URL,
        )
    )

    context.update({

        "columns":
            feature_columns,

        "target_column":
            target_column,

        "selected_x":
            x_column,

        "selected_y":
            y_column,

        "scatter_plot_url":
            scatter_plot_url,

        "bar_plot_url":
            class_distribution_url,

        "tables":
            df.head().to_html(
                classes="data-table",
                index=False,
            ),

        "removed_identifier_columns":
            removed_identifier_columns,

        "classifier_choices":
            CLASSIFIER_CHOICES,

        "metric_choices":
            CLASSIFICATION_METRICS,

        "training_form":
            ClassificationTrainingForm(),
    })

    return render(
        request,
        "project1/classification_analyze.html",
        context,
    )


# ==============================================================
# CLASSIFICATION — Training
# ==============================================================

def classification_train(request):
    """
    Train a classification model.

    The training configuration is submitted
    from the popup on the analysis page.
    """

    # ----------------------------------------------------------
    # IMPORTANT:
    # Direct GET should NOT show a second configuration form.
    # Return to analysis instead.
    # ----------------------------------------------------------

    if request.method != "POST":

        return redirect(
            "project1:classification_analyze"
        )

    context = {}

    file_path = (
        _get_classification_dataset_path()
    )

    try:

        (
            df,
            feature_columns,
            target_column,
            removed_identifier_columns,
        ) = load_dataset(
            file_path
        )

    except DatasetValidationError as error:

        context["error"] = str(
            error
        )

        return render(
            request,
            "project1/classification.html",
            context,
        )

    # ----------------------------------------------------------
    # Django form validation
    # ----------------------------------------------------------

    form = ClassificationTrainingForm(
        request.POST
    )

    if not form.is_valid():

        # Return to analysis rather than displaying
        # an empty training configuration page.

        context = {
            "error": (
                "Please check the selected "
                "training settings."
            ),

            "form_errors":
                form.errors,
        }

        # We could redirect, but render analysis again
        # so the user gets an error.
        return redirect(
            "project1:classification_analyze"
        )

    cleaned_data = (
        form.cleaned_data
    )

    selected_model = (
        cleaned_data["model"]
    )

    selected_metric = (
        cleaned_data["metric"]
    )

    test_size = float(
        cleaned_data[
            "test_size"
        ]
    )

    parameters = (
        get_classifier_parameters(
            cleaned_data
        )
    )

    # ----------------------------------------------------------
    # Features and target
    # ----------------------------------------------------------

    X = df[
        feature_columns
    ]

    y = df[
        target_column
    ]

    # ----------------------------------------------------------
    # Split
    # ----------------------------------------------------------

    try:

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(

            X,
            y,

            test_size=test_size,

            random_state=42,

            # Preserve class proportions
            stratify=y,
        )

    except ValueError as error:

        context["error"] = (
            "The dataset could not be split "
            "using the selected test size. "
            f"{error}"
        )

        return render(
            request,
            "project1/classification_train.html",
            context,
        )

    # ----------------------------------------------------------
    # Create classifier
    # ----------------------------------------------------------

    try:

        model = (
            create_classifier(
                model_name=
                    selected_model,

                parameters=
                    parameters,
            )
        )

    except UnsupportedAlgorithmError as error:

        context["error"] = str(
            error
        )

        return render(
            request,
            "project1/classification_train.html",
            context,
        )

    # ----------------------------------------------------------
    # Train model
    # ----------------------------------------------------------

    try:

        model.fit(
            X_train,
            y_train,
        )

    except ValueError as error:

        context["error"] = (
            "The model could not be trained "
            "with the selected settings. "
            f"{error}"
        )

        return render(
            request,
            "project1/classification_train.html",
            context,
        )

    # ----------------------------------------------------------
    # Predictions
    # ----------------------------------------------------------

    train_predictions = (
        model.predict(
            X_train
        )
    )

    test_predictions = (
        model.predict(
            X_test
        )
    )

    # ----------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------

    train_metrics = (
        evaluate_classifier(
            y_train,
            train_predictions,
        )
    )

    test_metrics = (
        evaluate_classifier(
            y_test,
            test_predictions,
        )
    )

    try:

        selected_train_score = (
            get_selected_metric(
                train_metrics,
                selected_metric,
            )
        )

        selected_test_score = (
            get_selected_metric(
                test_metrics,
                selected_metric,
            )
        )

    except UnsupportedMetricError as error:

        context["error"] = str(
            error
        )

        return render(
            request,
            "project1/classification_train.html",
            context,
        )

    # ----------------------------------------------------------
    # Performance chart
    # ----------------------------------------------------------

    score_chart_url = (
        create_accuracy_comparison_chart(

            train_score=
                selected_train_score,

            test_score=
                selected_test_score,

            media_root=
                settings.MEDIA_ROOT,

            media_url=
                settings.MEDIA_URL,
        )
    )

    # ----------------------------------------------------------
    # Human-readable names
    # ----------------------------------------------------------

    model_display_name = (
        CLASSIFIER_CHOICES[
            selected_model
        ]
    )

    metric_display_name = (
        CLASSIFICATION_METRICS[
            selected_metric
        ]
    )

    # ----------------------------------------------------------
    # Save trained model
    # ----------------------------------------------------------

    metadata = {

        "problem_type":
            "classification",

        "model_key":
            selected_model,

        "model_name":
            model_display_name,

        "feature_columns":
            feature_columns,

        "target_column":
            target_column,

        "test_size":
            test_size,

        "metric":
            selected_metric,

        "metric_name":
            metric_display_name,

        "parameters":
            parameters,
    }

    save_classification_model(

        model=model,

        metadata=metadata,

        media_root=
            settings.MEDIA_ROOT,
    )

    # ----------------------------------------------------------
    # Results context
    # ----------------------------------------------------------

    context.update({

        "training_complete":
            True,

        "model_name":
            model_display_name,

        "metric_name":
            metric_display_name,

        "selected_model":
            selected_model,

        "selected_metric":
            selected_metric,

        "parameters":
            parameters,

        "train_size":
            round(
                (1 - test_size)
                * 100
            ),

        "test_size":
            round(
                test_size
                * 100
            ),

        # -----------------------------------------
        # User-selected primary score
        # -----------------------------------------

        "train_score":
            round(
                selected_train_score
                * 100,
                2,
            ),

        "test_score":
            round(
                selected_test_score
                * 100,
                2,
            ),

        # -----------------------------------------
        # Accuracy
        # -----------------------------------------

        "train_accuracy":
            round(
                train_metrics[
                    "accuracy"
                ]
                * 100,
                2,
            ),

        "test_accuracy":
            round(
                test_metrics[
                    "accuracy"
                ]
                * 100,
                2,
            ),

        # -----------------------------------------
        # Precision
        # -----------------------------------------

        "train_precision":
            round(
                train_metrics[
                    "precision"
                ]
                * 100,
                2,
            ),

        "test_precision":
            round(
                test_metrics[
                    "precision"
                ]
                * 100,
                2,
            ),

        # -----------------------------------------
        # Recall
        # -----------------------------------------

        "train_recall":
            round(
                train_metrics[
                    "recall"
                ]
                * 100,
                2,
            ),

        "test_recall":
            round(
                test_metrics[
                    "recall"
                ]
                * 100,
                2,
            ),

        # -----------------------------------------
        # F1
        # -----------------------------------------

        "train_f1":
            round(
                train_metrics[
                    "f1"
                ]
                * 100,
                2,
            ),

        "test_f1":
            round(
                test_metrics[
                    "f1"
                ]
                * 100,
                2,
            ),

        "accuracy_chart_url":
            score_chart_url,

        "target_column":
            target_column,

        "feature_columns":
            feature_columns,
    })

    return render(
        request,
        "project1/classification_train.html",
        context,
    )


# ==============================================================
# CLASSIFICATION — Test / prediction
# ==============================================================

def classification_test(request):
    """
    Use the most recently trained model
    for prediction.
    """

    context = {}

    try:

        (
            model,
            metadata,
        ) = load_classification_model(
            settings.MEDIA_ROOT
        )

    except ModelNotFoundError:

        context["error"] = (
            "No trained classification "
            "model is available. "
            "Please train a model first."
        )

        return render(
            request,
            "project1/classification_test.html",
            context,
        )

    feature_columns = metadata[
        "feature_columns"
    ]

    target_column = metadata[
        "target_column"
    ]

    context.update({

        "feature_columns":
            feature_columns,

        "target_column":
            target_column,

        "model_name":
            metadata[
                "model_name"
            ],

        "model_parameters":
            metadata.get(
                "parameters",
                {},
            ),
    })

    # ----------------------------------------------------------
    # First page load
    # ----------------------------------------------------------

    if request.method != "POST":

        return render(
            request,
            "project1/classification_test.html",
            context,
        )

    # ----------------------------------------------------------
    # Read feature values
    # ----------------------------------------------------------

    input_data = {}

    try:

        for feature in feature_columns:

            raw_value = (
                request.POST.get(
                    feature
                )
            )

            if (
                raw_value is None
                or raw_value.strip() == ""
            ):

                raise ValueError(
                    f"A value is required "
                    f"for '{feature}'."
                )

            input_data[
                feature
            ] = float(
                raw_value
            )

    except ValueError as error:

        context["error"] = str(
            error
        )

        context[
            "entered_values"
        ] = request.POST

        return render(
            request,
            "project1/classification_test.html",
            context,
        )

    # ----------------------------------------------------------
    # Preserve training feature order
    # ----------------------------------------------------------

    prediction_frame = (
        pd.DataFrame(
            [input_data],
            columns=
                feature_columns,
        )
    )

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------

    prediction = model.predict(
        prediction_frame
    )[0]

    context[
        "prediction"
    ] = prediction

    # ----------------------------------------------------------
    # Confidence / probabilities
    # ----------------------------------------------------------

    if hasattr(
        model,
        "predict_proba",
    ):

        probabilities = (
            model.predict_proba(
                prediction_frame
            )[0]
        )

        classes = (
            model.classes_
        )

        class_probabilities = [

            {
                "class":
                    class_value,

                "probability":
                    round(
                        probability
                        * 100,
                        2,
                    ),
            }

            for (
                class_value,
                probability,
            )

            in zip(
                classes,
                probabilities,
            )
        ]

        context.update({

            "confidence":
                round(
                    max(
                        probabilities
                    )
                    * 100,
                    2,
                ),

            "class_probabilities":
                class_probabilities,
        })

    context[
        "entered_values"
    ] = input_data

    return render(
        request,
        "project1/classification_test.html",
        context,
    )