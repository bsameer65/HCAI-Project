from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------
# Supported classification algorithms
# ------------------------------------------------------------------

CLASSIFIER_CHOICES = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "knn": "K-Nearest Neighbors",
    "logistic_regression": "Logistic Regression",
}


class UnsupportedAlgorithmError(ValueError):
    """Raised when an unsupported machine learning algorithm is requested."""
    pass


def create_classifier(model_name, parameters=None):
    """
    Create a classification model using the selected algorithm
    and user-defined hyperparameters.

    Parameters
    ----------
    model_name : str
        Identifier of the selected classification algorithm.

    parameters : dict, optional
        Hyperparameters selected by the user.

    Returns
    -------
    sklearn estimator
        Configured classification model.
    """

    parameters = parameters or {}

    # --------------------------------------------------------------
    # Decision Tree
    # --------------------------------------------------------------
    if model_name == "decision_tree":

        max_depth = parameters.get("max_depth")
        min_samples_split = parameters.get(
            "min_samples_split",
            2
        )

        return DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
        )

    # --------------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------------
    if model_name == "random_forest":

        n_estimators = parameters.get(
            "n_estimators",
            100
        )

        max_depth = parameters.get("max_depth")

        min_samples_split = parameters.get(
            "min_samples_split",
            2
        )

        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1,
        )

    # --------------------------------------------------------------
    # K-Nearest Neighbors
    # --------------------------------------------------------------
    if model_name == "knn":

        n_neighbors = parameters.get(
            "n_neighbors",
            5
        )

        weights = parameters.get(
            "weights",
            "uniform"
        )

        distance_metric = parameters.get(
            "metric",
            "minkowski"
        )

        # KNN is distance-based, therefore scaling is important.
        return Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric=distance_metric,
                )
            ),
        ])

    # --------------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------------
    if model_name == "logistic_regression":

        c_value = parameters.get(
            "C",
            1.0
        )

        max_iter = parameters.get(
            "max_iter",
            1000
        )

        # Logistic Regression also benefits from standardized features.
        return Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    max_iter=max_iter,
                    random_state=42,
                )
            ),
        ])

    # --------------------------------------------------------------
    # Unsupported model
    # --------------------------------------------------------------
    raise UnsupportedAlgorithmError(
        f"Unsupported classification algorithm: {model_name}"
    )