import os
import joblib
from django.shortcuts import render
from django.conf import settings

from .data_utils import load_penguin_data, get_preprocessor, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from .model_utils import (
    train_decision_trees,
    train_logistic_regressions,
    select_best,
    get_lr_coef_table,
)


# Path where the selected model pipeline is saved so later views can reuse it
MODEL_SAVE_DIR = os.path.join(settings.MEDIA_ROOT, "project2_models")


def _save_model(pipeline, le, model_type, metadata):
    """Persist the selected pipeline + metadata to disk."""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "le": le, "model_type": model_type, **metadata},
        os.path.join(MODEL_SAVE_DIR, "selected_model.pkl"),
    )


def index(request):
    """Landing page — loads dataset stats."""
    df, X, y, numerical_features, categorical_features, class_names = load_penguin_data()

    context = {
        "page_title": "Project 2: Explainability",
        "num_rows": len(df),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "all_features": numerical_features + categorical_features,
        "target_name": "species",
        "class_names": class_names,
        "preview_table": df.head().to_html(
            classes="p2-table", index=False, border=0
        ),
    }
    return render(request, "project2/index.html", context)


def train(request):
    """
    Train page — lets the user choose a model type and lambda value,
    then shows a candidate table and highlights the best model.
    """
    df, X, y, numerical_features, categorical_features, class_names = load_penguin_data()

    # Defaults
    model_type = "dt"
    lambda_value = 0.5

    results = []
    best = None
    coef_table = None
    coef_headers = []

    if request.method == "POST":
        model_type = request.POST.get("model_type", "dt")
        try:
            lambda_value = float(request.POST.get("lambda_value", 0.5))
            lambda_value = max(0.0, min(1.0, lambda_value))
        except ValueError:
            lambda_value = 0.5

        if model_type == "dt":
            results, le = train_decision_trees(X, y)
        else:
            results, le = train_logistic_regressions(X, y)

        best = select_best(results, lambda_value)

        # Save the selected model so later views (counterfactual, PDP/ALE) can load it
        _save_model(
            best["pipeline"], le, model_type,
            {"class_names": class_names, "lambda_value": lambda_value},
        )

        # Build coefficient table for Logistic Regression
        if model_type == "lr":
            # Get feature names after one-hot encoding
            prep = best["pipeline"].named_steps["prep"]
            ohe_names = prep.named_transformers_["cat"] \
                           .named_steps["onehot"] \
                           .get_feature_names_out(CATEGORICAL_FEATURES).tolist()
            feature_names_out = NUMERICAL_FEATURES + ohe_names
            coef_table = get_lr_coef_table(best["pipeline"], feature_names_out, class_names)
            coef_headers = class_names

    context = {
        "page_title": "Train Model — Project 2",
        "model_type": model_type,
        "lambda_value": lambda_value,
        "results": results,
        "best": best,
        "class_names": class_names,
        "coef_table": coef_table,
        "coef_headers": coef_headers,
    }
    return render(request, "project2/train.html", context)
