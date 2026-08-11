from django.contrib import messages
from django.shortcuts import redirect, render

from .services.active_learning import load_active_learning_results
from .services.baseline import load_baseline_results
from .services.learning_to_defer import load_learning_to_defer_results
from .services.simulated_expert import load_expert_results

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
    baseline_result = load_baseline_results()
    expert_result = load_expert_results()
    defer_result = load_learning_to_defer_results()
    active_learning_result = load_active_learning_results()

    results_available = all(
        [
            baseline_result is not None,
            expert_result is not None,
            defer_result is not None,
            active_learning_result is not None,
        ]
    )

    expert_summary = []

    if expert_result is not None:
        for expert_key, expert in expert_result["experts"].items():

            best_region = max(
                expert["region_analysis"].values(),
                key=lambda region: region["observed_accuracy"],
            )

            expert_summary.append(
                {
                    "key": expert_key,
                    "name": expert["name"],
                    "accuracy": expert["accuracy"],
                    "macro_f1": expert["macro_f1"],
                    "best_region_name": best_region["display_name"],
                    "best_region_accuracy": best_region[
                        "observed_accuracy"
                    ],
                }
            )

    context = {
        "baseline": baseline_result,
        "experts": expert_result,
        "expert_summary": expert_summary,
        "defer": defer_result,
        "active_learning": active_learning_result,
        "results_available": results_available,
    }

    return render(
        request,
        "project3/compare.html",
        context,
    )
