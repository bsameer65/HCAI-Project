from .algorithms import (
    create_classifier,
    CLASSIFIER_CHOICES,
)

from .evaluation import (
    evaluate_classifier,
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