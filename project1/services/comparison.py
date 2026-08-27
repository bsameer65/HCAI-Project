from .algorithms import (
    create_classifier,
    CLASSIFIER_CHOICES,
    create_regressor,
    REGRESSOR_CHOICES,
)

from .evaluation import (
    evaluate_classifier,
    evaluate_regressor,
)


DEFAULT_CLASSIFIER_PARAMETERS = {
    "decision_tree": {
        "max_depth": 5,
        "min_samples_split": 2,
    },

    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
    },

    "knn": {
        "n_neighbors": 5,
        "weights": "uniform",
        "metric": "euclidean",
    },

    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
    },
}

DEFAULT_REGRESSOR_PARAMETERS = {
    "linear_regression": {},
    "decision_tree_regressor": {"max_depth": 5, "min_samples_split": 2},
    "random_forest_regressor": {
        "n_estimators": 100, "max_depth": 10, "min_samples_split": 2,
    },
    "knn_regressor": {
        "n_neighbors": 5, "weights": "uniform", "metric": "euclidean",
    },
}

CLASSIFIER_EXPERIMENT_CHOICES = {
    "decision_tree": "Decision Tree — maximum depth",
    "random_forest": "Random Forest — number of trees",
    "knn": "KNN — number of neighbors",
    "logistic_regression": "Logistic Regression — C",
}

REGRESSOR_EXPERIMENT_CHOICES = {
    "decision_tree_regressor": "Decision Tree — maximum depth",
    "random_forest_regressor": "Random Forest — number of trees",
    "knn_regressor": "KNN — number of neighbors",
}


def compare_classifiers(
    X_train,
    X_test,
    y_train,
    y_test,
    primary_metric,
):
    """
    Train all supported classifiers using the same data split
    and compare their performance fairly.
    """

    results = []

    for model_key, model_name in CLASSIFIER_CHOICES.items():

        parameters = (
            DEFAULT_CLASSIFIER_PARAMETERS[
                model_key
            ]
        )

        model = create_classifier(
            model_name=model_key,
            parameters=parameters,
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        metrics = evaluate_classifier(
            y_test,
            predictions,
        )

        results.append({
            "model_key":
                model_key,

            "model_name":
                model_name,

            "primary_score":
                metrics[primary_metric],

            "accuracy":
                metrics["accuracy"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "f1":
                metrics["f1"],
        })

    results.sort(
        key=lambda result:
            result["primary_score"],
        reverse=True,
    )

    return results


def compare_regressors(
    X_train,
    X_test,
    y_train,
    y_test,
    primary_metric,
    numeric_features,
    categorical_features,
):
    """Compare all regressors on one shared split and metric."""
    results = []
    for model_key, model_name in REGRESSOR_CHOICES.items():
        parameters = DEFAULT_REGRESSOR_PARAMETERS[model_key]
        model = create_regressor(
            model_key, numeric_features, categorical_features, parameters
        )
        model.fit(X_train, y_train)
        metrics = evaluate_regressor(y_test, model.predict(X_test))
        results.append({
            "model_key": model_key,
            "model_name": model_name,
            "primary_score": metrics[primary_metric],
            **metrics,
        })

    results.sort(
        key=lambda result: result["primary_score"],
        reverse=primary_metric == "r2",
    )
    return results


def run_classifier_parameter_experiment(
    model_key,
    X_train,
    X_test,
    y_train,
    y_test,
    primary_metric,
):
    """Evaluate one understandable classifier parameter over a visible grid."""
    grids = {
        "decision_tree": ("max_depth", [2, 4, 6, 8, 10]),
        "random_forest": ("n_estimators", [25, 50, 100, 150]),
        "knn": ("n_neighbors", [1, 3, 5, 7, 9]),
        "logistic_regression": ("C", [0.01, 0.1, 1.0, 10.0, 100.0]),
    }
    parameter_name, candidates = grids[model_key]
    if parameter_name == "n_neighbors":
        candidates = [value for value in candidates if value <= len(X_train)]
    results = []
    for value in candidates:
        parameters = dict(DEFAULT_CLASSIFIER_PARAMETERS[model_key])
        parameters[parameter_name] = value
        model = create_classifier(model_key, parameters)
        model.fit(X_train, y_train)
        metrics = evaluate_classifier(y_test, model.predict(X_test))
        results.append({
            "parameter_value": value,
            "score": metrics[primary_metric],
        })
    best_score = max(item["score"] for item in results)
    for item in results:
        item["is_best"] = item["score"] == best_score
    return {"parameter_name": parameter_name, "results": results}


def run_regressor_parameter_experiment(
    model_key,
    X_train,
    X_test,
    y_train,
    y_test,
    primary_metric,
    numeric_features,
    categorical_features,
):
    """Evaluate one understandable regressor parameter over a visible grid."""
    grids = {
        "decision_tree_regressor": ("max_depth", [2, 4, 6, 8, 10]),
        "random_forest_regressor": ("n_estimators", [25, 50, 100, 150]),
        "knn_regressor": ("n_neighbors", [1, 3, 5, 7, 9]),
    }
    parameter_name, candidates = grids[model_key]
    if parameter_name == "n_neighbors":
        candidates = [value for value in candidates if value <= len(X_train)]
    results = []
    for value in candidates:
        parameters = dict(DEFAULT_REGRESSOR_PARAMETERS[model_key])
        parameters[parameter_name] = value
        model = create_regressor(
            model_key, numeric_features, categorical_features, parameters
        )
        model.fit(X_train, y_train)
        metrics = evaluate_regressor(y_test, model.predict(X_test))
        results.append({
            "parameter_value": value,
            "score": metrics[primary_metric],
        })
    best_score = (
        max(item["score"] for item in results)
        if primary_metric == "r2"
        else min(item["score"] for item in results)
    )
    for item in results:
        item["is_best"] = item["score"] == best_score
    return {"parameter_name": parameter_name, "results": results}
