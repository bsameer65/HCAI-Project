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
    RegressionDatasetSetupForm,
    RegressionTrainingForm,
    get_classifier_parameters,
    get_regressor_parameters,
)


# ==============================================================
# Dataset
# ==============================================================

from .services.dataset import (
    load_dataset,
    inspect_dataset,
    get_regression_target_choices,
    prepare_regression_dataset,
    DatasetValidationError,
)


# ==============================================================
# Algorithms
# ==============================================================

from .services.algorithms import (
    create_classifier,
    create_regressor,
    CLASSIFIER_CHOICES,
    REGRESSOR_CHOICES,
    UnsupportedAlgorithmError,
)


# ==============================================================
# Evaluation
# ==============================================================

from .services.evaluation import (
    evaluate_classifier,
    get_selected_metric,
    calculate_confusion_matrix,
    CLASSIFICATION_METRICS,
    UnsupportedMetricError,
    evaluate_regressor,
    get_selected_regression_metric,
    REGRESSION_METRICS,
)


# ==============================================================
# Visualization
# ==============================================================

from .services.visualization import (
    create_classification_scatter,
    create_class_distribution,
    create_score_comparison_chart,
    create_feature_importance_chart,
    create_model_comparison_chart,
    create_regression_scatter,
    create_target_distribution,
    create_actual_vs_predicted_plot,
    create_residual_plot,
    create_regression_comparison_chart,
    create_hyperparameter_experiment_chart,
)


# ==============================================================
# Explainability
# ==============================================================

from .services.explainability import (
    get_feature_importance,
    get_regression_feature_importance,
    ExplainabilityError,
)


# ==============================================================
# Persistence
# ==============================================================

from .services.persistence import (
    save_classification_model,
    load_classification_model,
    ModelNotFoundError,
    clear_regression_artifacts,
    save_regression_model,
    load_regression_model,
    save_regression_results,
    load_regression_results,
)

# ==============================================================
# Comparison
# ==============================================================

from .services.comparison import (
    compare_classifiers,
    compare_regressors,
    run_classifier_parameter_experiment,
    run_regressor_parameter_experiment,
    CLASSIFIER_EXPERIMENT_CHOICES,
    REGRESSOR_EXPERIMENT_CHOICES,
)

# ==============================================================
# Constants
# ==============================================================

CLASSIFICATION_DATASET_FILENAME = (
    "current_classification_dataset.csv"
)

REGRESSION_DATASET_FILENAME = "current_regression_dataset.csv"



# ==============================================================
# Helper functions
# ==============================================================

def _get_upload_directory():
    """
    Return the directory used for uploaded datasets.
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


def _get_regression_dataset_path():
    return os.path.join(
        _get_upload_directory(),
        REGRESSION_DATASET_FILENAME,
    )


def _validate_numeric_features(
    df,
    feature_columns,
):
    """
    Return names of non-numeric input features.

    The current ML implementation expects scalar/numerical
    description features.
    """

    return [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]


# ==============================================================
# Project 1 landing page
# ==============================================================

def index(request):
    """
    Project 1 landing page.

    The human explicitly selects Classification or Regression.
    """

    return render(
        request,
        "project1/index.html",
    )


def regression(request):
    """Upload and immediately validate a regression CSV dataset."""
    context = {}
    if request.method == "POST":
        uploaded_file = request.FILES.get("csv_file")
        if uploaded_file is None:
            context["error"] = "Please choose a CSV file before continuing."
            return render(request, "project1/regression.html", context)
        if not uploaded_file.name.lower().endswith(".csv"):
            context["error"] = "Only CSV files are supported."
            return render(request, "project1/regression.html", context)

        file_path = _get_regression_dataset_path()
        try:
            with open(file_path, "wb+") as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            get_regression_target_choices(file_path)
        except DatasetValidationError as error:
            if os.path.exists(file_path):
                os.remove(file_path)
            context["error"] = str(error)
            return render(request, "project1/regression.html", context)

        request.session.pop("regression_target", None)
        clear_regression_artifacts(settings.MEDIA_ROOT)
        return redirect("project1:regression_setup")

    return render(request, "project1/regression.html", context)


def regression_setup(request):
    """Let the user confirm a numeric target after inspecting the upload."""
    file_path = _get_regression_dataset_path()
    try:
        df, column_information, removed_identifiers = inspect_dataset(file_path)
        target_choices = get_regression_target_choices(file_path)
    except DatasetValidationError as error:
        return render(request, "project1/regression.html", {"error": str(error)})

    suggested_target = (
        df.columns[-1]
        if df.columns[-1] in target_choices
        else target_choices[-1]
    )
    form = RegressionDatasetSetupForm(
        request.POST or None,
        target_choices=target_choices,
        initial={"target_column": suggested_target},
    )
    if request.method == "POST" and form.is_valid():
        target_column = form.cleaned_data["target_column"]
        try:
            prepare_regression_dataset(file_path, target_column)
        except DatasetValidationError as error:
            form.add_error("target_column", str(error))
        else:
            request.session["regression_target"] = target_column
            return redirect("project1:regression_analyze")

    return render(request, "project1/regression_setup.html", {
        "form": form,
        "suggested_target": suggested_target,
        "column_information": column_information,
        "removed_identifier_columns": removed_identifiers,
        "tables": df.head().to_html(classes="data-table", index=False),
    })


def regression_analyze(request):
    """Explore numerical relationships without discarding categorical features."""
    target_column = request.session.get("regression_target")
    if not target_column:
        return redirect("project1:regression_setup")

    try:
        dataset = prepare_regression_dataset(
            _get_regression_dataset_path(), target_column
        )
    except DatasetValidationError as error:
        return render(request, "project1/regression.html", {"error": str(error)})

    df = dataset["dataframe"]
    numeric_features = dataset["numeric_features"]
    plot_type = request.POST.get("plot_type", "feature_target")
    if plot_type not in {"feature_target", "feature_feature", "target_distribution"}:
        plot_type = "feature_target"

    selected_x = request.POST.get("x_col")
    if selected_x not in numeric_features:
        selected_x = numeric_features[0] if numeric_features else None
    second_default = numeric_features[1] if len(numeric_features) > 1 else selected_x
    selected_y = request.POST.get("y_col")
    if selected_y not in numeric_features:
        selected_y = second_default

    plot_url = None
    plot_heading = "Target Distribution"
    if plot_type == "target_distribution":
        plot_url = create_target_distribution(
            df, target_column, settings.MEDIA_ROOT, settings.MEDIA_URL
        )
    elif selected_x is not None:
        y_column = target_column if plot_type == "feature_target" else selected_y
        plot_heading = (
            "Feature vs Target" if plot_type == "feature_target"
            else "Feature vs Feature"
        )
        plot_url = create_regression_scatter(
            df, selected_x, y_column, settings.MEDIA_ROOT, settings.MEDIA_URL,
            title=f"{selected_x} vs {y_column}",
        )

    target_distribution_url = create_target_distribution(
        df, target_column, settings.MEDIA_ROOT, settings.MEDIA_URL
    )
    return render(request, "project1/regression_analyze.html", {
        **dataset,
        "numeric_columns": numeric_features,
        "selected_x": selected_x,
        "selected_y": selected_y,
        "plot_type": plot_type,
        "plot_heading": plot_heading,
        "plot_url": plot_url,
        "target_distribution_url": target_distribution_url,
        "training_form": RegressionTrainingForm(),
        "regressor_choices": REGRESSOR_CHOICES,
        "metric_choices": REGRESSION_METRICS,
        "tables": df.head().to_html(classes="data-table", index=False),
    })


def regression_train(request):
    """Train the selected regression pipeline and persist reusable results."""
    if request.method != "POST":
        return redirect("project1:regression_analyze")

    target_column = request.session.get("regression_target")
    if not target_column:
        return redirect("project1:regression_setup")
    try:
        dataset = prepare_regression_dataset(
            _get_regression_dataset_path(), target_column
        )
    except DatasetValidationError as error:
        return render(request, "project1/regression.html", {"error": str(error)})

    form = RegressionTrainingForm(request.POST)
    if not form.is_valid():
        return render(request, "project1/regression_train.html", {
            "error": (
                "Some training settings are invalid. Please return to the "
                "analysis page and check the selected configuration."
            ),
            "form_errors": form.errors,
        })

    cleaned = form.cleaned_data
    model_key = cleaned["model"]
    metric = cleaned["metric"]
    test_size = float(cleaned["test_size"])
    parameters = get_regressor_parameters(cleaned)
    X = dataset["dataframe"][dataset["feature_columns"]]
    y = dataset["dataframe"][target_column]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        model = create_regressor(
            model_key,
            dataset["numeric_features"],
            dataset["categorical_features"],
            parameters,
        )
        model.fit(X_train, y_train)
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
    except (ValueError, UnsupportedAlgorithmError) as error:
        return render(request, "project1/regression_train.html", {
            "error": f"The model could not be trained with this configuration. Details: {error}"
        })

    train_metrics = evaluate_regressor(y_train, train_predictions)
    test_metrics = evaluate_regressor(y_test, test_predictions)
    try:
        train_score = get_selected_regression_metric(train_metrics, metric)
        test_score = get_selected_regression_metric(test_metrics, metric)
    except UnsupportedMetricError as error:
        return render(request, "project1/regression_train.html", {"error": str(error)})

    categorical_options = {
        feature: sorted(
            dataset["dataframe"][feature].dropna().astype(str).unique().tolist()
        )
        for feature in dataset["categorical_features"]
    }
    metadata = {
        "problem_type": "regression",
        "model_key": model_key,
        "model_name": REGRESSOR_CHOICES[model_key],
        "feature_columns": dataset["feature_columns"],
        "numeric_features": dataset["numeric_features"],
        "categorical_features": dataset["categorical_features"],
        "target_column": target_column,
        "test_size": test_size,
        "metric": metric,
        "metric_name": REGRESSION_METRICS[metric],
        "parameters": parameters,
        "categorical_options": categorical_options,
    }
    results = {
        "training_complete": True,
        "model_name": metadata["model_name"],
        "metric_name": metadata["metric_name"],
        "selected_metric": metric,
        "parameters": parameters,
        "target_column": target_column,
        "feature_columns": dataset["feature_columns"],
        "train_size": round((1 - test_size) * 100),
        "test_size": round(test_size * 100),
        "train_score": round(float(train_score), 4),
        "test_score": round(float(test_score), 4),
        "train_metrics": {key: round(float(value), 4) for key, value in train_metrics.items()},
        "test_metrics": {key: round(float(value), 4) for key, value in test_metrics.items()},
        "actual_test": y_test.tolist(),
        "predicted_test": test_predictions.tolist(),
    }
    results["actual_vs_predicted_url"] = create_actual_vs_predicted_plot(
        results["actual_test"],
        results["predicted_test"],
        target_column,
        settings.MEDIA_ROOT,
        settings.MEDIA_URL,
    )
    results["residual_plot_url"] = create_residual_plot(
        results["actual_test"],
        results["predicted_test"],
        settings.MEDIA_ROOT,
        settings.MEDIA_URL,
    )
    save_regression_model(model, metadata, settings.MEDIA_ROOT)
    save_regression_results(results, settings.MEDIA_ROOT)
    return redirect("project1:regression_result")


def regression_result(request):
    """Display the latest regression result without retraining."""
    try:
        _, metadata = load_regression_model(settings.MEDIA_ROOT)
        results = load_regression_results(settings.MEDIA_ROOT)
    except ModelNotFoundError:
        return render(request, "project1/regression_train.html", {
            "error": "No trained regression model is available. Please train a model first."
        })
    return render(request, "project1/regression_train.html", {
        **results,
        "metadata": metadata,
    })


def regression_test(request):
    """Predict from raw numeric and categorical values using the saved pipeline."""
    try:
        model, metadata = load_regression_model(settings.MEDIA_ROOT)
        results = load_regression_results(settings.MEDIA_ROOT)
    except ModelNotFoundError:
        return render(request, "project1/regression_test.html", {
            "error": "No trained regression model is available. Please train a model first."
        })

    feature_fields = []
    for feature in metadata["feature_columns"]:
        is_numeric = feature in metadata["numeric_features"]
        feature_fields.append({
            "name": feature,
            "type": "numeric" if is_numeric else "categorical",
            "options": metadata.get("categorical_options", {}).get(feature, []),
        })
    context = {
        "model_name": metadata["model_name"],
        "target_column": metadata["target_column"],
        "feature_fields": feature_fields,
        "test_mae": results["test_metrics"]["mae"],
        "test_rmse": results["test_metrics"]["rmse"],
    }
    if request.method == "POST":
        input_data = {}
        try:
            for field in feature_fields:
                raw_value = request.POST.get(field["name"], "").strip()
                if not raw_value:
                    raise ValueError(f"A value is required for '{field['name']}'.")
                input_data[field["name"]] = (
                    float(raw_value) if field["type"] == "numeric" else raw_value
                )
            prediction_frame = pd.DataFrame(
                [input_data], columns=metadata["feature_columns"]
            )
            context["prediction"] = float(model.predict(prediction_frame)[0])
            context["has_prediction"] = True
            context["entered_values"] = input_data
        except ValueError as error:
            context["error"] = f"The prediction could not be made. {error}"
            context["entered_values"] = request.POST

    return render(request, "project1/regression_test.html", context)


def regression_explain(request):
    """Explain the latest fitted regression pipeline without retraining it."""
    try:
        model, metadata = load_regression_model(settings.MEDIA_ROOT)
        dataset = prepare_regression_dataset(
            _get_regression_dataset_path(), metadata["target_column"]
        )
        X = dataset["dataframe"][metadata["feature_columns"]]
        y = dataset["dataframe"][metadata["target_column"]]
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=metadata["test_size"], random_state=42
        )
        explanation = get_regression_feature_importance(
            model, metadata["model_key"], X_test, y_test
        )
        chart_url = create_feature_importance_chart(
            explanation["items"],
            explanation["method"],
            settings.MEDIA_ROOT,
            settings.MEDIA_URL,
        )
    except (ModelNotFoundError, DatasetValidationError) as error:
        return render(request, "project1/regression_explain.html", {"error": str(error)})
    except (ValueError, ExplainabilityError) as error:
        return render(request, "project1/regression_explain.html", {
            "error": f"The model explanation could not be calculated. {error}"
        })

    return render(request, "project1/regression_explain.html", {
        "model_name": metadata["model_name"],
        "target_column": metadata["target_column"],
        "parameters": metadata["parameters"],
        "explanation_method": explanation["method"],
        "explanation_items": explanation["items"],
        "explanation_chart_url": chart_url,
    })


def regression_compare(request):
    """Compare all regressors fairly without replacing the user's saved model."""
    target_column = request.session.get("regression_target")
    if not target_column:
        return redirect("project1:regression_setup")
    try:
        dataset = prepare_regression_dataset(
            _get_regression_dataset_path(), target_column
        )
    except DatasetValidationError as error:
        return render(request, "project1/regression_compare.html", {"error": str(error)})

    selected_metric = request.GET.get("metric", "mae")
    if selected_metric not in REGRESSION_METRICS:
        selected_metric = "mae"
    try:
        test_size = float(request.GET.get("test_size", 0.2))
    except ValueError:
        test_size = 0.2
    if test_size not in {0.2, 0.3, 0.4}:
        test_size = 0.2
    experiment_model = request.GET.get(
        "experiment_model", "decision_tree_regressor"
    )
    if experiment_model not in REGRESSOR_EXPERIMENT_CHOICES:
        experiment_model = "decision_tree_regressor"

    X = dataset["dataframe"][dataset["feature_columns"]]
    y = dataset["dataframe"][target_column]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        results = compare_regressors(
            X_train, X_test, y_train, y_test, selected_metric,
            dataset["numeric_features"], dataset["categorical_features"],
        )
        experiment = run_regressor_parameter_experiment(
            experiment_model,
            X_train,
            X_test,
            y_train,
            y_test,
            selected_metric,
            dataset["numeric_features"],
            dataset["categorical_features"],
        )
    except ValueError as error:
        return render(request, "project1/regression_compare.html", {
            "error": f"The comparison could not be completed. {error}"
        })

    chart_url = create_regression_comparison_chart(
        results,
        REGRESSION_METRICS[selected_metric],
        settings.MEDIA_ROOT,
        settings.MEDIA_URL,
    )
    experiment_chart_url = create_hyperparameter_experiment_chart(
        experiment["results"],
        experiment["parameter_name"],
        REGRESSION_METRICS[selected_metric],
        settings.MEDIA_ROOT,
        settings.MEDIA_URL,
    )
    for index, result in enumerate(results):
        result["is_best"] = index == 0
        for metric in ("primary_score", "mae", "mse", "rmse", "r2"):
            result[metric] = round(float(result[metric]), 4)
    for item in experiment["results"]:
        item["display_score"] = round(float(item["score"]), 4)

    return render(request, "project1/regression_compare.html", {
        "results": results,
        "selected_metric": selected_metric,
        "metric_name": REGRESSION_METRICS[selected_metric],
        "metric_choices": REGRESSION_METRICS,
        "test_size": test_size,
        "train_size_percent": round((1 - test_size) * 100),
        "test_size_percent": round(test_size * 100),
        "comparison_chart_url": chart_url,
        "target_column": target_column,
        "lower_is_better": selected_metric != "r2",
        "experiment_choices": REGRESSOR_EXPERIMENT_CHOICES,
        "experiment_model": experiment_model,
        "experiment_model_name": REGRESSOR_EXPERIMENT_CHOICES[experiment_model],
        "experiment_parameter": experiment["parameter_name"],
        "experiment_results": experiment["results"],
        "experiment_chart_url": experiment_chart_url,
    })


# ==============================================================
# CLASSIFICATION — Dataset upload
# ==============================================================

def classification(request):
    """
    Upload and validate a classification CSV dataset.
    """

    context = {}

    if request.method == "POST":

        uploaded_file = request.FILES.get(
            "csv_file"
        )

        # ------------------------------------------------------
        # File required
        # ------------------------------------------------------

        if uploaded_file is None:

            context["error"] = (
                "Please choose a CSV file before continuing."
            )

            return render(
                request,
                "project1/classification.html",
                context,
            )

        # ------------------------------------------------------
        # File extension
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
            # Save uploaded dataset
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
            # Features must currently be numeric
            # --------------------------------------------------

            non_numeric_features = (
                _validate_numeric_features(
                    df,
                    feature_columns,
                )
            )

            if non_numeric_features:

                raise DatasetValidationError(
                    "The following feature columns are not numeric: "
                    + ", ".join(
                        non_numeric_features
                    )
                    + ". Please upload a dataset with numeric features."
                )

        except DatasetValidationError as error:

            # Do not retain invalid uploaded datasets
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
        # Successful upload → analysis
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
# CLASSIFICATION — Dataset analysis
# ==============================================================

def classification_analyze(request):
    """
    Visualize the classification dataset.

    User chooses two description features for the scatter plot.
    Target class is represented by color.
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
    # Selected X-axis
    # ----------------------------------------------------------

    x_column = (
        request.POST.get(
            "x_col"
        )
        or feature_columns[0]
    )

    # ----------------------------------------------------------
    # Default Y-axis
    # ----------------------------------------------------------

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
    # Class-distribution plot
    # ----------------------------------------------------------

    class_distribution_url = (
        create_class_distribution(
            df=df,
            target_column=target_column,
            media_root=settings.MEDIA_ROOT,
            media_url=settings.MEDIA_URL,
        )
    )

    # ----------------------------------------------------------
    # Template context
    # ----------------------------------------------------------

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
# CLASSIFICATION — Model training
# ==============================================================

def classification_train(request):
    """
    Train a classification model using human-selected:

        - ML algorithm
        - hyperparameters
        - train/test split
        - evaluation metric

    Configuration is submitted from the analysis popup.
    """

    # ----------------------------------------------------------
    # Direct GET → back to analysis
    # ----------------------------------------------------------

    if request.method != "POST":

        return redirect(
            "project1:classification_analyze"
        )

    context = {}

    file_path = (
        _get_classification_dataset_path()
    )

    # ----------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------

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
    # Validate user training configuration
    # ----------------------------------------------------------

    form = ClassificationTrainingForm(
        request.POST
    )

    if not form.is_valid():

        print(
            "TRAINING FORM ERRORS:"
        )

        print(
            form.errors
        )

        context["error"] = (
            "Some training settings are invalid. "
            "Please return to the analysis page and "
            "check your configuration."
        )

        context["form_errors"] = (
            form.errors
        )

        return render(
            request,
            "project1/classification_train.html",
            context,
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

    # ----------------------------------------------------------
    # Convert form controls → sklearn parameters
    # ----------------------------------------------------------

    parameters = (
        get_classifier_parameters(
            cleaned_data
        )
    )

    # ----------------------------------------------------------
    # Feature matrix and target
    # ----------------------------------------------------------

    X = df[
        feature_columns
    ]

    y = df[
        target_column
    ]

    # ----------------------------------------------------------
    # Train / test split
    #
    # stratify keeps class proportions approximately equal.
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

            stratify=y,
        )

    except ValueError as error:

        context["error"] = (
            "The dataset could not be split using the "
            "selected test size. "
            f"Details: {error}"
        )

        return render(
            request,
            "project1/classification_train.html",
            context,
        )

    # ----------------------------------------------------------
    # Create selected classifier
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
    # Train classifier
    # ----------------------------------------------------------

    try:

        model.fit(
            X_train,
            y_train,
        )

    except ValueError as error:

        context["error"] = (
            "The selected model could not be trained "
            "with the current hyperparameters. "
            f"Details: {error}"
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
    # Evaluation metrics
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

    # ----------------------------------------------------------
    # Selected primary evaluation metric
    # ----------------------------------------------------------

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
    # Human-readable labels
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
    # Training vs testing chart
    # ----------------------------------------------------------

    score_chart_url = (
        create_score_comparison_chart(

            train_score=
                selected_train_score,

            test_score=
                selected_test_score,

            metric_name=
                metric_display_name,

            media_root=
                settings.MEDIA_ROOT,

            media_url=
                settings.MEDIA_URL,
        )
    )

    # ==========================================================
    # CONFUSION MATRIX
    # ==========================================================

    confusion = (
        calculate_confusion_matrix(
            y_test,
            test_predictions,
        )
    )

    # Use the actual labels understood by the trained model.
    if hasattr(
        model,
        "classes_",
    ):

        class_labels = (
            model.classes_.tolist()
        )

    else:

        class_labels = sorted(
            y.unique().tolist()
        )

    confusion_rows = []

    for row_index, (
        actual_label,
        row,
    ) in enumerate(
        zip(
            class_labels,
            confusion,
        )
    ):

        cells = []

        for column_index, value in enumerate(
            row.tolist()
        ):

            cells.append({

                "value":
                    value,

                "is_correct":
                    (
                        row_index
                        == column_index
                    ),
            })

        confusion_rows.append({

            "actual":
                actual_label,

            "cells":
                cells,
        })

    # ==========================================================
    # EXPLAINABILITY
    # ==========================================================

    try:

        explanation = (
            get_feature_importance(

                model=model,

                model_name=
                    selected_model,

                feature_columns=
                    feature_columns,

                X_test=
                    X_test,

                y_test=
                    y_test,
            )
        )

        explanation_method = (
            explanation[
                "method"
            ]
        )

        explanation_items = (
            explanation[
                "items"
            ]
        )

        explanation_chart_url = (
            create_feature_importance_chart(

                explanation_items=
                    explanation_items,

                method_name=
                    explanation_method,

                media_root=
                    settings.MEDIA_ROOT,

                media_url=
                    settings.MEDIA_URL,
            )
        )

    except ExplainabilityError as error:

        print(
            "EXPLAINABILITY ERROR:",
            error,
        )

        explanation_method = None
        explanation_items = []
        explanation_chart_url = None

    # ----------------------------------------------------------
    # Save model + metadata
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

    # ==========================================================
    # RESULT PAGE CONTEXT
    # ==========================================================

    context.update({

        "training_complete":
            True,

        # ------------------------------------------------------
        # Model information
        # ------------------------------------------------------

        "model_name":
            model_display_name,

        "selected_model":
            selected_model,

        "metric_name":
            metric_display_name,

        "selected_metric":
            selected_metric,

        "parameters":
            parameters,

        "target_column":
            target_column,

        "feature_columns":
            feature_columns,

        # ------------------------------------------------------
        # Split
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # Primary score
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # All metrics
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # Visualization
        # ------------------------------------------------------

        "accuracy_chart_url":
            score_chart_url,

        # ------------------------------------------------------
        # Confusion matrix
        # ------------------------------------------------------

        "confusion_rows":
            confusion_rows,

        "class_labels":
            class_labels,

        # ------------------------------------------------------
        # Explainability
        # ------------------------------------------------------

        "explanation_method":
            explanation_method,

        "explanation_items":
            explanation_items,

        "explanation_chart_url":
            explanation_chart_url,
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
    Use the most recently trained classification model
    for prediction on manually entered values.

    If available, also display class probabilities and
    prediction confidence.
    """

    context = {}

    # ----------------------------------------------------------
    # Load trained model
    # ----------------------------------------------------------

    try:

        (
            model,
            metadata,
        ) = load_classification_model(
            settings.MEDIA_ROOT
        )

    except ModelNotFoundError:

        context["error"] = (
            "No trained classification model is available. "
            "Please train a model first."
        )

        return render(
            request,
            "project1/classification_test.html",
            context,
        )

    feature_columns = (
        metadata[
            "feature_columns"
        ]
    )

    target_column = (
        metadata[
            "target_column"
        ]
    )

    # ----------------------------------------------------------
    # Basic page information
    # ----------------------------------------------------------

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

        "metric_name":
            metadata.get(
                "metric_name",
            ),
    })

    # ----------------------------------------------------------
    # GET → only show prediction form
    # ----------------------------------------------------------

    if request.method != "POST":

        return render(
            request,
            "project1/classification_test.html",
            context,
        )

    # ----------------------------------------------------------
    # Read manually entered feature values
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
                    f"A value is required for '{feature}'."
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
    # Build prediction DataFrame
    #
    # Columns are explicitly ordered exactly as during training.
    # ----------------------------------------------------------

    prediction_frame = (
        pd.DataFrame(
            [
                input_data
            ],
            columns=
                feature_columns,
        )
    )

    # ----------------------------------------------------------
    # Predict class
    # ----------------------------------------------------------

    prediction = (
        model.predict(
            prediction_frame
        )[0]
    )

    context[
        "prediction"
    ] = prediction

    # ==========================================================
    # PREDICTION CONFIDENCE
    # ==========================================================

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

        confidence = max(
            probabilities
        )

        context.update({

            "confidence":
                round(
                    confidence
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
    
def classification_explain(request):
    """
    Explain the most recently trained classification model.
    """

    context = {}

    try:
        model, metadata = load_classification_model(
            settings.MEDIA_ROOT
        )

    except ModelNotFoundError:
        context["error"] = (
            "No trained classification model is available. "
            "Please train a model first."
        )

        return render(
            request,
            "project1/classification_explain.html",
            context,
        )

    file_path = _get_classification_dataset_path()

    try:
        (
            df,
            feature_columns,
            target_column,
            removed_identifier_columns,
        ) = load_dataset(file_path)

    except DatasetValidationError as error:
        context["error"] = str(error)

        return render(
            request,
            "project1/classification_explain.html",
            context,
        )

    X = df[feature_columns]
    y = df[target_column]

    test_size = metadata.get(
        "test_size",
        0.2,
    )

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
            stratify=y,
        )

        explanation = get_feature_importance(
            model=model,
            model_name=metadata["model_key"],
            feature_columns=feature_columns,
            X_test=X_test,
            y_test=y_test,
        )

        explanation_items = explanation["items"]
        explanation_method = explanation["method"]

        explanation_chart_url = (
            create_feature_importance_chart(
                explanation_items=explanation_items,
                method_name=explanation_method,
                media_root=settings.MEDIA_ROOT,
                media_url=settings.MEDIA_URL,
            )
        )

    except (ValueError, ExplainabilityError) as error:
        context["error"] = str(error)

        return render(
            request,
            "project1/classification_explain.html",
            context,
        )

    context.update({
        "model_name":
            metadata["model_name"],

        "model_key":
            metadata["model_key"],

        "target_column":
            target_column,

        "metric_name":
            metadata.get("metric_name"),

        "parameters":
            metadata.get("parameters", {}),

        "explanation_method":
            explanation_method,

        "explanation_items":
            explanation_items,

        "explanation_chart_url":
            explanation_chart_url,
    })

    return render(
        request,
        "project1/classification_explain.html",
        context,
    )

def classification_compare(request):
    """
    Compare all supported classifiers using the same
    train/test split and evaluation metric.
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

        context["error"] = str(error)

        return render(
            request,
            "project1/classification_compare.html",
            context,
        )

    selected_metric = (
        request.GET.get(
            "metric",
            "accuracy",
        )
    )

    if selected_metric not in CLASSIFICATION_METRICS:
        selected_metric = "accuracy"

    test_size = float(
        request.GET.get(
            "test_size",
            0.2,
        )
    )

    if test_size not in {
        0.2,
        0.3,
        0.4,
    }:
        test_size = 0.2

    experiment_model = request.GET.get(
        "experiment_model",
        "decision_tree",
    )
    if experiment_model not in CLASSIFIER_EXPERIMENT_CHOICES:
        experiment_model = "decision_tree"

    X = df[
        feature_columns
    ]

    y = df[
        target_column
    ]

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
            stratify=y,
        )

        results = compare_classifiers(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            primary_metric=selected_metric,
        )

        experiment = run_classifier_parameter_experiment(
            experiment_model,
            X_train,
            X_test,
            y_train,
            y_test,
            selected_metric,
        )

    except ValueError as error:

        context["error"] = str(error)

        return render(
            request,
            "project1/classification_compare.html",
            context,
        )

    metric_name = (
        CLASSIFICATION_METRICS[
            selected_metric
        ]
    )

    chart_url = (
        create_model_comparison_chart(
            comparison_results=results,
            metric_name=metric_name,
            media_root=settings.MEDIA_ROOT,
            media_url=settings.MEDIA_URL,
        )
    )

    experiment_chart_url = create_hyperparameter_experiment_chart(
        experiment["results"],
        experiment["parameter_name"],
        metric_name,
        settings.MEDIA_ROOT,
        settings.MEDIA_URL,
        percentage=True,
    )
    for item in experiment["results"]:
        item["score_percent"] = round(float(item["score"]) * 100, 2)

    formatted_results = []

    for index, result in enumerate(
        results
    ):

        formatted_results.append({
            **result,

            "is_best":
                index == 0,

            "primary_score_percent":
                round(
                    result["primary_score"]
                    * 100,
                    2,
                ),

            "accuracy_percent":
                round(
                    result["accuracy"]
                    * 100,
                    2,
                ),

            "precision_percent":
                round(
                    result["precision"]
                    * 100,
                    2,
                ),

            "recall_percent":
                round(
                    result["recall"]
                    * 100,
                    2,
                ),

            "f1_percent":
                round(
                    result["f1"]
                    * 100,
                    2,
                ),
        })

    context.update({
        "results":
            formatted_results,

        "metric_name":
            metric_name,

        "selected_metric":
            selected_metric,

        "test_size":
            test_size,

        "train_size":
            round(
                (1 - test_size)
                * 100
            ),

        "test_size_percent":
            round(
                test_size * 100
            ),

        "comparison_chart_url":
            chart_url,

        "target_column":
            target_column,

        "experiment_choices":
            CLASSIFIER_EXPERIMENT_CHOICES,

        "experiment_model":
            experiment_model,

        "experiment_model_name":
            CLASSIFIER_EXPERIMENT_CHOICES[experiment_model],

        "experiment_parameter":
            experiment["parameter_name"],

        "experiment_results":
            experiment["results"],

        "experiment_chart_url":
            experiment_chart_url,
    })

    return render(
        request,
        "project1/classification_compare.html",
        context,
    )
