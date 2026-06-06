import os
import joblib
from django.shortcuts import render
from django.conf import settings

from .data_utils import load_penguin_data
from .model_utils import get_selected_model

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
