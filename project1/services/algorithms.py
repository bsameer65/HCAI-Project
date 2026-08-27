from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ------------------------------------------------------------------
# Supported classification algorithms
# ------------------------------------------------------------------

CLASSIFIER_CHOICES = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "knn": "K-Nearest Neighbors",
    "logistic_regression": "Logistic Regression",
}

REGRESSOR_CHOICES = {
    "linear_regression": "Linear Regression",
    "decision_tree_regressor": "Decision Tree Regressor",
    "random_forest_regressor": "Random Forest Regressor",
    "knn_regressor": "K-Nearest Neighbors Regressor",
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


def _create_regression_preprocessor(
    numeric_features,
    categorical_features,
    scale_numeric,
):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = []
    if numeric_features:
        transformers.append((
            "numeric",
            Pipeline(numeric_steps),
            list(numeric_features),
        ))
    if categorical_features:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            list(categorical_features),
        ))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def create_regressor(
    model_name,
    numeric_features,
    categorical_features,
    parameters=None,
):
    """Return a complete preprocessing and regression pipeline."""
    parameters = parameters or {}

    if model_name == "linear_regression":
        estimator = LinearRegression()
        scale_numeric = True
    elif model_name == "decision_tree_regressor":
        estimator = DecisionTreeRegressor(
            max_depth=parameters.get("max_depth"),
            min_samples_split=parameters.get("min_samples_split", 2),
            random_state=42,
        )
        scale_numeric = False
    elif model_name == "random_forest_regressor":
        estimator = RandomForestRegressor(
            n_estimators=parameters.get("n_estimators", 100),
            max_depth=parameters.get("max_depth"),
            min_samples_split=parameters.get("min_samples_split", 2),
            random_state=42,
            n_jobs=-1,
        )
        scale_numeric = False
    elif model_name == "knn_regressor":
        estimator = KNeighborsRegressor(
            n_neighbors=parameters.get("n_neighbors", 5),
            weights=parameters.get("weights", "uniform"),
            metric=parameters.get("metric", "minkowski"),
        )
        scale_numeric = True
    else:
        raise UnsupportedAlgorithmError(
            f"Unsupported regression algorithm: {model_name}"
        )

    preprocessor = _create_regression_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", estimator),
    ])
