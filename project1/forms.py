from django import forms


class RegressionDatasetSetupForm(forms.Form):
    target_column = forms.ChoiceField(label="Target variable")

    def __init__(self, *args, target_choices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["target_column"].choices = [
            (column, column) for column in (target_choices or [])
        ]


class ClassificationTrainingForm(forms.Form):

    MODEL_CHOICES = [
        (
            "decision_tree",
            "Decision Tree",
        ),
        (
            "random_forest",
            "Random Forest",
        ),
        (
            "knn",
            "K-Nearest Neighbors",
        ),
        (
            "logistic_regression",
            "Logistic Regression",
        ),
    ]

    TEST_SIZE_CHOICES = [
        (
            "0.2",
            "20% test / 80% train",
        ),
        (
            "0.3",
            "30% test / 70% train",
        ),
        (
            "0.4",
            "40% test / 60% train",
        ),
    ]

    METRIC_CHOICES = [
        (
            "accuracy",
            "Accuracy",
        ),
        (
            "precision",
            "Precision",
        ),
        (
            "recall",
            "Recall",
        ),
        (
            "f1",
            "F1 Score",
        ),
    ]

    model = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label="Model",
    )

    test_size = forms.ChoiceField(
        choices=TEST_SIZE_CHOICES,
        label="Test size",
    )

    metric = forms.ChoiceField(
        choices=METRIC_CHOICES,
        label="Evaluation metric",
    )

    # ==========================================================
    # Decision Tree
    # ==========================================================

    tree_max_depth = forms.IntegerField(
        required=False,
        min_value=1,
        max_value=50,
        initial=5,
        label="Maximum depth",
    )

    tree_min_samples_split = forms.IntegerField(
        required=False,
        min_value=2,
        max_value=50,
        initial=2,
        label="Minimum samples split",
    )

    # ==========================================================
    # Random Forest
    # ==========================================================

    rf_n_estimators = forms.IntegerField(
        required=False,
        min_value=10,
        max_value=500,
        initial=100,
        label="Number of trees",
    )

    rf_max_depth = forms.IntegerField(
        required=False,
        min_value=1,
        max_value=50,
        initial=10,
        label="Maximum depth",
    )

    rf_min_samples_split = forms.IntegerField(
        required=False,
        min_value=2,
        max_value=50,
        initial=2,
        label="Minimum samples split",
    )

    # ==========================================================
    # KNN
    # ==========================================================

    knn_neighbors = forms.IntegerField(
        required=False,
        min_value=1,
        max_value=50,
        initial=5,
        label="Number of neighbors",
    )

    knn_weights = forms.ChoiceField(
        required=False,
        choices=[
            (
                "uniform",
                "Uniform",
            ),
            (
                "distance",
                "Distance",
            ),
        ],
        initial="uniform",
        label="Neighbor weighting",
    )

    knn_metric = forms.ChoiceField(
        required=False,
        choices=[
            (
                "euclidean",
                "Euclidean",
            ),
            (
                "manhattan",
                "Manhattan",
            ),
        ],
        initial="euclidean",
        label="Distance metric",
    )

    # ==========================================================
    # Logistic Regression
    # ==========================================================

    logistic_c = forms.FloatField(
        required=False,
        min_value=0.001,
        max_value=1000,
        initial=1.0,
        label="Regularization strength (C)",
    )

    logistic_max_iter = forms.IntegerField(
        required=False,
        min_value=100,
        max_value=5000,
        initial=1000,
        label="Maximum iterations",
    )
def get_classifier_parameters(cleaned_data):
    """
    Build the hyperparameter dictionary for the selected model.
    """

    model_name = cleaned_data["model"]

    if model_name == "decision_tree":

        return {
            "max_depth":
                cleaned_data.get(
                    "tree_max_depth"
                ),

            "min_samples_split":
                cleaned_data.get(
                    "tree_min_samples_split"
                )
                or 2,
        }

    if model_name == "random_forest":

        return {
            "n_estimators":
                cleaned_data.get(
                    "rf_n_estimators"
                )
                or 100,

            "max_depth":
                cleaned_data.get(
                    "rf_max_depth"
                ),

            "min_samples_split":
                cleaned_data.get(
                    "rf_min_samples_split"
                )
                or 2,
        }

    if model_name == "knn":

        return {
            "n_neighbors":
                cleaned_data.get(
                    "knn_neighbors"
                )
                or 5,

            "weights":
                cleaned_data.get(
                    "knn_weights"
                )
                or "uniform",

            "metric":
                cleaned_data.get(
                    "knn_metric"
                )
                or "euclidean",
        }

    if model_name == "logistic_regression":

        return {
            "C":
                cleaned_data.get(
                    "logistic_c"
                )
                or 1.0,

            "max_iter":
                cleaned_data.get(
                    "logistic_max_iter"
                )
                or 1000,
        }

    return {}
