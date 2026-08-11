from django.contrib import messages
from django.shortcuts import redirect, render

from .services.baseline import (
    load_baseline_results,
    train_and_evaluate_baseline,
)

from .services.learning_to_defer import (
    load_learning_to_defer_results,
    run_learning_to_defer_experiment,
)

from .services.simulated_expert import (
    evaluate_all_simulated_experts,
    load_expert_results,
)

from .services.active_learning import (
    load_active_learning_results,
    run_active_learning_experiment,
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
    if request.method == "POST":
        try:
            evaluate_all_simulated_experts()

            messages.success(
                request,
                "Both simulated experts were evaluated successfully.",
            )

        except Exception as exc:
            messages.error(
                request,
                f"Simulated expert evaluation failed: {exc}",
            )

        return redirect("project3:expert")

    result = load_expert_results()

    return render(
        request,
        "project3/expert.html",
        {
            "result": result,
            "results_available": result is not None,
        },
    )


def learning_to_defer(request):
    if request.method == "POST":
        try:
            run_learning_to_defer_experiment()

            messages.success(
                request,
                "Learning-to-defer experiment completed successfully.",
            )

        except Exception as exc:
            messages.error(
                request,
                f"Learning-to-defer experiment failed: {exc}",
            )

        return redirect("project3:learning_to_defer")

    result = load_learning_to_defer_results()

    return render(
        request,
        "project3/defer.html",
        {
            "result": result,
            "results_available": result is not None,
        },
    )


def active_learning(request):

    if request.method == "POST":

        try:
            run_active_learning_experiment()

            messages.success(
                request,
                (
                    "Active-learning experiment "
                    "completed successfully."
                ),
            )

        except Exception as exc:

            messages.error(
                request,
                (
                    "Active-learning experiment "
                    f"failed: {exc}"
                ),
            )

        return redirect(
            "project3:active_learning"
        )

    result = (
        load_active_learning_results()
    )

    return render(
        request,
        "project3/active_learning.html",
        {
            "result": result,
            "results_available": (
                result is not None
            ),
        },
    )



def compare_results(request):
    return render(request, "project3/compare.html")
