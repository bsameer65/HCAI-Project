import os
import joblib
from django.shortcuts import render
from django.conf import settings

from .data_utils import load_penguin_data, NUMERICAL_FEATURES
from .model_utils import get_selected_model
from .counterfactual_utils import generate_counterfactuals
from .effect_plot_utils import compute_pdp, compute_ale
from .plot_utils import plot_pdp, plot_ale

# Path where the selected model result is cached so other views can reload it
MODEL_SAVE_DIR  = os.path.join(settings.MEDIA_ROOT, "project2_models")
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "selected_model.pkl")


def _save_selected(result):
    """Persist the full get_selected_model() result dict to disk."""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    joblib.dump(result, MODEL_SAVE_PATH)


def _load_selected():
    """Return the last saved result dict, or None if nothing trained yet."""
    if os.path.exists(MODEL_SAVE_PATH):
        return joblib.load(MODEL_SAVE_PATH)
    return None


# ── Views ─────────────────────────────────────────────────────────────────────

def index(request):
    """Landing page — shows dataset overview."""
    df, X, y, num_feats, cat_feats, class_names = load_penguin_data()

    context = {
        "page_title":           "Project 2: Explainability",
        "num_rows":             len(df),
        "numerical_features":   num_feats,
        "categorical_features": cat_feats,
        "all_features":         num_feats + cat_feats,
        "target_name":          "species",
        "class_names":          class_names,
        "preview_table":        df.head().to_html(
                                    classes="p2-table", index=False, border=0
                                ),
    }
    return render(request, "project2/index.html", context)


def train(request):
    """
    Train page — user picks model type + lambda, then sees the best model and
    a full candidate comparison table.

    All training logic lives in get_selected_model(); this view only handles
    HTTP and template rendering.
    """
    model_type   = "dt"
    lambda_value = 0.5
    result       = None

    if request.method == "POST":
        model_type = request.POST.get("model_type", "dt")
        try:
            lambda_value = float(request.POST.get("lambda_value", 0.5))
            lambda_value = max(0.0, min(1.0, lambda_value))
        except ValueError:
            lambda_value = 0.5

        # Single call — trains, evaluates, selects, returns everything
        result = get_selected_model(model_type, lambda_value)

        # Persist so counterfactual / PDP / ALE views can reload without re-training
        _save_selected(result)

    context = {
        "page_title":   "Train Model — Project 2",
        "model_type":   model_type,
        "lambda_value": lambda_value,
        "result":       result,
    }
    return render(request, "project2/train.html", context)


def counterfactual(request):
    """
    Counterfactual page — user picks model settings, a penguin row, and a
    desired target species, then sees the closest counterfactual examples.
    """
    from .data_utils import load_penguin_data

    df, X, y, num_feats, cat_feats, class_names = load_penguin_data()

    # Form state defaults
    model_type   = "dt"
    lambda_value = 0.5
    row_index    = 0
    target_class = class_names[0]
    cf_result    = None
    error        = None

    # Build a sample list for the dropdown: "Row 0 — Adelie"
    sample_options = [
        {"index": i, "label": f"Row {i} — {row['species']}"}
        for i, row in df.iterrows()
    ]

    if request.method == "POST":
        model_type = request.POST.get("model_type", "dt")
        try:
            lambda_value = float(request.POST.get("lambda_value", 0.5))
            lambda_value = max(0.0, min(1.0, lambda_value))
        except ValueError:
            lambda_value = 0.5

        try:
            row_index = int(request.POST.get("row_index", 0))
        except ValueError:
            row_index = 0

        target_class = request.POST.get("target_class", class_names[0])

        # Train / select model fresh for the chosen settings
        selected = get_selected_model(model_type, lambda_value)
        _save_selected(selected)

        try:
            cf_result = generate_counterfactuals(
                selected_model_info=selected,
                row_index=row_index,
                target_class_name=target_class,
            )
        except Exception as exc:
            error = f"Counterfactual generation failed: {exc}"

    context = {
        "page_title":     "Counterfactuals — Project 2",
        "model_type":     model_type,
        "lambda_value":   lambda_value,
        "class_names":    class_names,
        "sample_options": sample_options,
        "row_index":      row_index,
        "target_class":   target_class,
        "cf_result":      cf_result,
        "error":          error,
        "num_features":   num_feats,
        "cat_features":   cat_feats,
        "all_features":   num_feats + cat_feats,
    }
    return render(request, "project2/counterfactual.html", context)


def pdp_ale(request):
    """
    Feature Effect Plots page — shows a PDP for the selected numerical feature
    and the selected model. ALE will be added in the next step.
    """
    model_type    = "dt"
    lambda_value  = 0.5
    feature_name  = NUMERICAL_FEATURES[0]
    pdp_plot_url  = None
    ale_plot_url  = None
    error         = None

    if request.method == "POST":
        model_type = request.POST.get("model_type", "dt")
        try:
            lambda_value = float(request.POST.get("lambda_value", 0.5))
            lambda_value = max(0.0, min(1.0, lambda_value))
        except ValueError:
            lambda_value = 0.5

        feature_name = request.POST.get("feature_name", NUMERICAL_FEATURES[0])
        if feature_name not in NUMERICAL_FEATURES:
            feature_name = NUMERICAL_FEATURES[0]

        try:
            selected = get_selected_model(model_type, lambda_value)
            _save_selected(selected)

            pdp_result  = compute_pdp(
                pipeline     = selected["pipeline"],
                X            = selected["X"],
                feature_name = feature_name,
                class_names  = selected["class_names"],
            )
            pdp_plot_url = plot_pdp(pdp_result, feature_name, selected["label"])

            ale_result  = compute_ale(
                pipeline     = selected["pipeline"],
                X            = selected["X"],
                feature_name = feature_name,
                class_names  = selected["class_names"],
            )
            ale_plot_url = plot_ale(ale_result, feature_name, selected["label"])

        except Exception as exc:
            error = f"Plot generation failed: {exc}"

    context = {
        "page_title":         "Feature Effect Plots — Project 2",
        "model_type":         model_type,
        "lambda_value":       lambda_value,
        "feature_name":       feature_name,
        "numerical_features": NUMERICAL_FEATURES,
        "pdp_plot_url":       pdp_plot_url,
        "ale_plot_url":       ale_plot_url,
        "error":              error,
    }
    return render(request, "project2/pdp_ale.html", context)
