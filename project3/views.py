from django.contrib import messages
from django.shortcuts import redirect, render
import random
from .models import HumanExpertResponse
from .services.data_loader import load_ag_news

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

from .services.human_expert import (
    CLASS_NAMES,
    HUMAN_QUERY_STRATEGIES,
    calculate_human_competence,
    prepare_human_expert_pool,
    select_human_query_indices,
)

from .services.advanced_analysis import (
    load_advanced_analysis,
    run_advanced_analysis,
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


def human_expert(request):
    """
    Optional interactive human-in-the-loop extension.

    A user selects a query strategy, labels a sequence of selected AG News
    articles, and receives an initial competence profile after completion.
    """

    if not request.session.session_key:
        request.session.create()

    session_key = request.session.session_key

    responses = HumanExpertResponse.objects.filter(
        session_key=session_key,
    )

    # ---------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------

    if (
        request.method == "POST"
        and request.POST.get("action") == "reset"
    ):
        responses.delete()

        for key in [
            "human_expert_strategy",
            "human_expert_indices",
            "human_expert_position",
            "human_expert_query_count",
        ]:
            request.session.pop(
                key,
                None,
            )

        messages.success(
            request,
            "Human expert session was reset.",
        )

        return redirect(
            "project3:human_expert"
        )

    # ---------------------------------------------------------------
    # Start session
    # ---------------------------------------------------------------

    if (
        request.method == "POST"
        and request.POST.get("action") == "start"
    ):
        strategy_key = request.POST.get(
            "strategy"
        )

        try:
            query_count = int(
                request.POST.get(
                    "query_count",
                    12,
                )
            )

        except ValueError:
            query_count = 12

        query_count = max(
            4,
            min(
                query_count,
                40,
            ),
        )

        if (
            strategy_key
            not in HUMAN_QUERY_STRATEGIES
        ):
            messages.error(
                request,
                "Please select a valid query strategy.",
            )

            return redirect(
                "project3:human_expert"
            )

        prepared = (
            prepare_human_expert_pool()
        )

        dataframe = prepared["data"]

        indices = (
            select_human_query_indices(
                dataframe=dataframe,
                strategy_key=strategy_key,
                query_count=query_count,
            )
        )

        responses.delete()

        request.session[
            "human_expert_strategy"
        ] = strategy_key

        request.session[
            "human_expert_indices"
        ] = indices

        request.session[
            "human_expert_position"
        ] = 0

        request.session[
            "human_expert_query_count"
        ] = query_count

        return redirect(
            "project3:human_expert"
        )

    strategy_key = request.session.get(
        "human_expert_strategy"
    )

    # ---------------------------------------------------------------
    # No active session
    # ---------------------------------------------------------------

    if strategy_key is None:
        return render(
            request,
            "project3/human_expert.html",
            {
                "session_started": False,
                "strategies": (
                    HUMAN_QUERY_STRATEGIES
                ),
            },
        )

    prepared = prepare_human_expert_pool()

    dataframe = prepared["data"]

    indices = request.session.get(
        "human_expert_indices",
        [],
    )

    position = request.session.get(
        "human_expert_position",
        0,
    )

    query_count = request.session.get(
        "human_expert_query_count",
        len(indices),
    )

    # ---------------------------------------------------------------
    # Submit annotation
    # ---------------------------------------------------------------

    if (
        request.method == "POST"
        and request.POST.get(
            "selected_label"
        )
        is not None
    ):
        selected_label = int(
            request.POST[
                "selected_label"
            ]
        )

        article_index = int(
            request.POST[
                "article_index"
            ]
        )

        article = dataframe.iloc[
            article_index
        ]

        true_label = int(
            article["label"]
        )

        HumanExpertResponse.objects.create(
            article_index=article_index,
            article_text=article["text"],
            selected_label=selected_label,
            true_label=true_label,
            is_correct=(
                selected_label
                == true_label
            ),
            query_strategy=(
                HUMAN_QUERY_STRATEGIES[
                    strategy_key
                ]
            ),
            classifier_prediction=int(
                article[
                    "classifier_prediction"
                ]
            ),
            classifier_confidence=float(
                article[
                    "classifier_confidence"
                ]
            ),
            classifier_entropy=float(
                article[
                    "classifier_entropy"
                ]
            ),
            session_key=session_key,
        )

        position += 1

        request.session[
            "human_expert_position"
        ] = position

        return redirect(
            "project3:human_expert"
        )

    # ---------------------------------------------------------------
    # Completed
    # ---------------------------------------------------------------

    if position >= len(indices):
        responses = (
            HumanExpertResponse.objects
            .filter(
                session_key=session_key
            )
        )

        competence = (
            calculate_human_competence(
                responses
            )
        )

        return render(
            request,
            "project3/human_expert.html",
            {
                "session_started": True,
                "completed": True,
                "responses": responses,
                "competence": competence,
                "strategy_name": (
                    HUMAN_QUERY_STRATEGIES[
                        strategy_key
                    ]
                ),
            },
        )

    # ---------------------------------------------------------------
    # Current query
    # ---------------------------------------------------------------

    article_index = indices[
        position
    ]

    article = dataframe.iloc[
        article_index
    ]

    progress_percent = (
        position / query_count
    ) * 100

    return render(
        request,
        "project3/human_expert.html",
        {
            "session_started": True,
            "completed": False,
            "article_index": (
                article_index
            ),
            "article_text": (
                article["text"]
            ),
            "classifier_prediction": (
                CLASS_NAMES[
                    int(
                        article[
                            "classifier_prediction"
                        ]
                    )
                ]
            ),
            "classifier_confidence": (
                float(
                    article[
                        "classifier_confidence"
                    ]
                )
            ),
            "classifier_entropy": (
                float(
                    article[
                        "classifier_entropy"
                    ]
                )
            ),
            "strategy_name": (
                HUMAN_QUERY_STRATEGIES[
                    strategy_key
                ]
            ),
            "position": position + 1,
            "target_annotations": (
                query_count
            ),
            "progress_percent": (
                progress_percent
            ),
            "class_choices": [
                (0, "World"),
                (1, "Sports"),
                (2, "Business"),
                (3, "Sci/Tech"),
            ],
        },
    )

def advanced_analysis(request):

    if request.method == "POST":

        try:
            run_advanced_analysis()

            messages.success(
                request,
                (
                    "Advanced human-AI analysis "
                    "completed successfully."
                ),
            )

        except Exception as exc:

            messages.error(
                request,
                (
                    "Advanced analysis failed: "
                    f"{exc}"
                ),
            )

        return redirect(
            "project3:advanced_analysis"
        )

    result = (
        load_advanced_analysis()
    )

    return render(
        request,
        "project3/advanced_analysis.html",
        {
            "result": result,
            "results_available": (
                result is not None
            ),
        },
    )