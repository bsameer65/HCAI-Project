from django.shortcuts import render


def index(request):
    """
    Landing page for Project 2.

    # HCAI extra: keeping the view thin — all data/model logic will live in
    # dedicated utility modules (data_utils, model_utils, etc.) so this file
    # stays readable and easy to grade.
    """
    context = {
        "page_title": "Project 2: Explainability",
    }
    return render(request, "project2/index.html", context)
