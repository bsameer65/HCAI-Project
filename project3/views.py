from django.contrib import messages
from django.shortcuts import redirect, render

from .services.baseline import (
    load_baseline_results,
    train_and_evaluate_baseline,
)


def index(request):
    return render(request, "project3/index.html")


def baseline(request):
    if request.method == "POST":
        try:
            train_and_evaluate_baseline()

            messages.success(
                request,
                "Baseline experiment completed successfully.",
            )

        except Exception as exc:
            messages.error(
                request,
                f"Baseline experiment failed: {exc}",
            )

        return redirect("project3:baseline")

    result = load_baseline_results()

    return render(
        request,
        "project3/baseline.html",
        {
            "result": result,
            "results_available": result is not None,
        },
    )


def expert(request):
    return render(request, "project3/expert.html")


def learning_to_defer(request):
    return render(request, "project3/defer.html")


def active_learning(request):
    return render(request, "project3/active_learning.html")


def compare_results(request):
    return render(request, "project3/compare.html")