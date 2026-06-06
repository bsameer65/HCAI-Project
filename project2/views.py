from django.shortcuts import render
from .data_utils import load_penguin_data


def index(request):
    """
    Landing page — loads dataset stats to give users an immediate overview
    of what they are working with before any model is trained.
    """
    df, X, y, numerical_features, categorical_features, class_names = load_penguin_data()

    context = {
        "page_title": "Project 2: Explainability",
        "num_rows": len(df),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "all_features": numerical_features + categorical_features,
        "target_name": "species",
        "class_names": class_names,
        # First 5 rows as an HTML table for quick data preview
        "preview_table": df.head().to_html(
            classes="p2-table", index=False, border=0
        ),
    }
    return render(request, "project2/index.html", context)
