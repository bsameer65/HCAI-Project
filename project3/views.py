from django.contrib import messages
from django.shortcuts import redirect, render

import random
import uuid

from .models import HumanExpertResponse

from .services.data_loader import load_ag_news

from .services.baseline import (
    load_baseline_results,
    train_and_evaluate_baseline,
)

from .services.simulated_expert import (
    evaluate_all_simulated_experts,
    load_expert_results,
)

from .services.learning_to_defer import (
    load_learning_to_defer_results,
    run_learning_to_defer_experiment,
)

from .services.active_learning import (
    EXPERT_PROFILES,
    QUERY_BUDGETS,
    load_active_learning_results,
    load_selected_active_learning_results,
    run_active_learning_experiment,
    run_selected_active_learning,
)

from .services.human_expert import (
    CLASS_NAMES,
    HUMAN_QUERY_STRATEGIES,
    QUERY_COUNT_OPTIONS,
    calculate_human_competence,
    calculate_query_statistics,
    prepare_human_expert_pool,
    select_human_query_indices,
)

from .services.advanced_analysis import (
    advanced_analysis_is_stale,
    load_advanced_analysis_results,
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
    """
    Run the project's selected Active Learning strategy.

    Classifier Entropy is fixed as the primary acquisition strategy.
    The user controls the simulated expert and expert-query budget.
    """

    results = (
        load_selected_active_learning_results()
    )

    selected_expert = (
        request.POST.get("expert")
        or (
            results.get(
                "selection",
                {},
            ).get(
                "expert_key"
            )
            if results
            else None
        )
        or next(
            iter(EXPERT_PROFILES)
        )
    )

    selected_budget_raw = (
        request.POST.get(
            "query_budget"
        )
    )

    if selected_budget_raw is None:

        if results:
            selected_budget = (
                results.get(
                    "selection",
                    {},
                ).get(
                    "query_budget",
                    QUERY_BUDGETS[-1],
                )
            )
        else:
            selected_budget = (
                QUERY_BUDGETS[-1]
            )

    else:

        try:
            selected_budget = int(
                selected_budget_raw
            )
        except ValueError:
            selected_budget = (
                QUERY_BUDGETS[-1]
            )

    if request.method == "POST":

        try:

            results = (
                run_selected_active_learning(
                    expert_key=(
                        selected_expert
                    ),
                    query_budget=(
                        selected_budget
                    ),
                )
            )

            messages.success(
                request,
                "Active Learning experiment completed.",
            )

        except Exception as exc:

            messages.error(
                request,
                str(exc),
            )

    expert_options = [
        {
            "key": key,
            "name": profile.name,
            "description": (
                profile.description
            ),
        }
        for key, profile
        in EXPERT_PROFILES.items()
    ]

    return render(
        request,
        "project3/active_learning.html",
        {
            "results": results,
            "results_available": (
                results is not None
            ),
            "expert_options": (
                expert_options
            ),
            "query_budgets": (
                QUERY_BUDGETS
            ),
            "selected_expert": (
                selected_expert
            ),
            "selected_budget": (
                selected_budget
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
    Optional Human Expert extension.

    Flow:
        1. Choose query strategy and annotation budget.
        2. System selects articles.
        3. Human labels them without seeing AI predictions.
        4. Human competence and AI metadata are revealed afterwards.
    """

    # ------------------------------------------------------------------
    # Human Expert session keys
    # ------------------------------------------------------------------

    strategy_session_key = (
        "human_expert_strategy"
    )

    indices_session_key = (
        "human_expert_indices"
    )

    position_session_key = (
        "human_expert_position"
    )

    count_session_key = (
        "human_expert_query_count"
    )

    run_id_session_key = (
        "human_expert_run_id"
    )


    # ------------------------------------------------------------------
    # Fresh visit from navigation
    # ------------------------------------------------------------------

    if request.GET.get(
        "new"
    ) == "1":

        keys_to_clear = [
            strategy_session_key,
            indices_session_key,
            position_session_key,
            count_session_key,
            run_id_session_key,
        ]

        for key in keys_to_clear:
            request.session.pop(
                key,
                None,
            )

        request.session.modified = True

        return redirect(
            "project3:human_expert"
        )


    # ------------------------------------------------------------------
    # Explicit reset button
    # ------------------------------------------------------------------

    if (
        request.method == "POST"
        and request.POST.get(
            "action"
        ) == "reset"
    ):

        keys_to_clear = [
            strategy_session_key,
            indices_session_key,
            position_session_key,
            count_session_key,
            run_id_session_key,
        ]

        for key in keys_to_clear:
            request.session.pop(
                key,
                None,
            )

        request.session.modified = True

        return redirect(
            "project3:human_expert"
        )


    # ------------------------------------------------------------------
    # Start a new annotation session
    # ------------------------------------------------------------------

    if (
        request.method == "POST"
        and request.POST.get(
            "action"
        ) == "start"
    ):

        strategy_key = (
            request.POST.get(
                "strategy"
            )
        )

        query_count_raw = (
            request.POST.get(
                "query_count"
            )
        )

        # Validate strategy
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

        # Validate query count
        try:

            query_count = int(
                query_count_raw
            )

        except (
            TypeError,
            ValueError,
        ):

            messages.error(
                request,
                "Please select a valid number of queries.",
            )

            return redirect(
                "project3:human_expert"
            )

        if (
            query_count
            not in QUERY_COUNT_OPTIONS
        ):

            messages.error(
                request,
                "Please select a supported query budget.",
            )

            return redirect(
                "project3:human_expert"
            )


        # --------------------------------------------------------------
        # Expensive preparation happens here once.
        # Cached afterwards.
        # --------------------------------------------------------------

        try:

            dataframe = (
                prepare_human_expert_pool()
            )

            selected_indices = (
                select_human_query_indices(
                    dataframe=dataframe,
                    strategy_key=(
                        strategy_key
                    ),
                    query_count=(
                        query_count
                    ),
                )
            )

        except Exception as exc:

            messages.error(
                request,
                (
                    "Could not prepare the Human Expert session: "
                    f"{exc}"
                ),
            )

            return redirect(
                "project3:human_expert"
            )


        # Every annotation run gets its own ID.
        run_id = uuid.uuid4().hex


        request.session[
            strategy_session_key
        ] = strategy_key

        request.session[
            indices_session_key
        ] = [
            int(index)
            for index in selected_indices
        ]

        request.session[
            position_session_key
        ] = 0

        request.session[
            count_session_key
        ] = query_count

        request.session[
            run_id_session_key
        ] = run_id

        request.session.modified = True


        return redirect(
            "project3:human_expert"
        )


    # ------------------------------------------------------------------
    # Check whether a session currently exists
    # ------------------------------------------------------------------

    strategy_key = request.session.get(
        strategy_session_key
    )

    selected_indices = request.session.get(
        indices_session_key
    )

    position = request.session.get(
        position_session_key
    )

    query_count = request.session.get(
        count_session_key
    )

    run_id = request.session.get(
        run_id_session_key
    )


    session_active = all(
        [
            strategy_key is not None,
            selected_indices is not None,
            position is not None,
            query_count is not None,
            run_id is not None,
        ]
    )


    # ------------------------------------------------------------------
    # No active session → show strategy selection
    # ------------------------------------------------------------------

    if not session_active:

        return render(
            request,
            "project3/human_expert.html",
            {
                "session_active": False,

                "strategies": (
                    HUMAN_QUERY_STRATEGIES
                ),

                "query_count_options": (
                    QUERY_COUNT_OPTIONS
                ),
            },
        )


    # ------------------------------------------------------------------
    # Load cached pool.
    # This is instant after first preparation.
    # ------------------------------------------------------------------

    dataframe = (
        prepare_human_expert_pool()
    )


    # ------------------------------------------------------------------
    # Handle one annotation
    # ------------------------------------------------------------------

    if (
        request.method == "POST"
        and request.POST.get(
            "action"
        ) == "annotate"
    ):

        selected_label_raw = (
            request.POST.get(
                "selected_label"
            )
        )

        try:

            selected_label = int(
                selected_label_raw
            )

        except (
            TypeError,
            ValueError,
        ):

            messages.error(
                request,
                "Please select one of the four news categories.",
            )

            return redirect(
                "project3:human_expert"
            )


        if selected_label not in CLASS_NAMES:

            messages.error(
                request,
                "Invalid category selection.",
            )

            return redirect(
                "project3:human_expert"
            )


        # Protect against duplicate browser submissions.
        if position >= len(
            selected_indices
        ):

            return redirect(
                "project3:human_expert"
            )


        article_index = int(
            selected_indices[
                position
            ]
        )

        article = dataframe.iloc[
            article_index
        ]

        true_label = int(
            article[
                "label"
            ]
        )

        classifier_prediction = int(
            article[
                "classifier_prediction"
            ]
        )

        classifier_confidence = float(
            article[
                "classifier_confidence"
            ]
        )

        classifier_entropy = float(
            article[
                "classifier_entropy"
            ]
        )


        # --------------------------------------------------------------
        # Store human decision + AI metadata.
        #
        # AI metadata is NOT displayed while annotation is in progress.
        # --------------------------------------------------------------

        HumanExpertResponse.objects.create(
            article_index=(
                article_index
            ),

            article_text=str(
                article[
                    "text"
                ]
            ),

            selected_label=(
                selected_label
            ),

            true_label=(
                true_label
            ),

            is_correct=(
                selected_label
                == true_label
            ),

            query_strategy=(
                strategy_key
            ),

            classifier_prediction=(
                classifier_prediction
            ),

            classifier_confidence=(
                classifier_confidence
            ),

            classifier_entropy=(
                classifier_entropy
            ),

            session_key=(
                run_id
            ),
        )


        # --------------------------------------------------------------
        # Advance one position
        # --------------------------------------------------------------

        position += 1

        request.session[
            position_session_key
        ] = position

        request.session.modified = True


        return redirect(
            "project3:human_expert"
        )


    # ------------------------------------------------------------------
    # Current session responses
    # ------------------------------------------------------------------

    responses = (
        HumanExpertResponse.objects
        .filter(
            session_key=run_id
        )
        .order_by(
            "created_at"
        )
    )


    # ------------------------------------------------------------------
    # Completed session
    # ------------------------------------------------------------------

    if position >= query_count:

        competence = (
            calculate_human_competence(
                responses
            )
        )

        query_statistics = (
            calculate_query_statistics(
                responses
            )
        )


        # --------------------------------------------------------------
        # Prepare retrospective table.
        #
        # AI predictions are revealed only now.
        # --------------------------------------------------------------

        response_rows = []

        for response in responses:

            response_rows.append(
                {
                    "article_index": (
                        response.article_index
                    ),

                    "article_text": (
                        response.article_text
                    ),

                    "human_label": (
                        CLASS_NAMES[
                            response.selected_label
                        ]
                    ),

                    "true_label": (
                        CLASS_NAMES[
                            response.true_label
                        ]
                    ),

                    "classifier_label": (
                        CLASS_NAMES[
                            response.classifier_prediction
                        ]
                        if response.classifier_prediction
                        is not None
                        else "—"
                    ),

                    "classifier_confidence_percent": (
                        response.classifier_confidence
                        * 100
                        if response.classifier_confidence
                        is not None
                        else None
                    ),

                    "classifier_entropy": (
                        response.classifier_entropy
                    ),

                    "is_correct": (
                        response.is_correct
                    ),
                }
            )


        return render(
            request,
            "project3/human_expert.html",
            {
                "session_active": True,
                "completed": True,

                "strategy_key": (
                    strategy_key
                ),

                "strategy": (
                    HUMAN_QUERY_STRATEGIES[
                        strategy_key
                    ]
                ),

                "query_count": (
                    query_count
                ),

                "competence": (
                    competence
                ),

                "query_statistics": (
                    query_statistics
                ),

                "response_rows": (
                    response_rows
                ),
            },
        )


    # ------------------------------------------------------------------
    # Current article
    # ------------------------------------------------------------------

    article_index = int(
        selected_indices[
            position
        ]
    )

    article = dataframe.iloc[
        article_index
    ]


    # Progress counts completed annotations,
    # not the number of the currently visible article.
    completed_count = position

    progress_percent = int(
        (
            completed_count
            / query_count
        )
        * 100
    )


    return render(
        request,
        "project3/human_expert.html",
        {
            "session_active": True,
            "completed": False,

            "strategy_key": (
                strategy_key
            ),

            "strategy": (
                HUMAN_QUERY_STRATEGIES[
                    strategy_key
                ]
            ),

            "query_count": (
                query_count
            ),

            "position": (
                position
            ),

            "annotation_number": (
                position + 1
            ),

            "completed_count": (
                completed_count
            ),

            "progress_percent": (
                progress_percent
            ),

            # IMPORTANT:
            # only article text is exposed during annotation.
            "article": {
                "text": str(
                    article[
                        "text"
                    ]
                ),
            },

            "class_names": (
                CLASS_NAMES
            ),
        },
    )
    
    
def advanced_analysis(request):
    """
    Display or regenerate the Advanced Human-AI Analysis.

    Existing results are displayed immediately when available. If one of the
    source experiments has changed, the previous analysis remains visible but
    is clearly marked as stale until the user regenerates it.
    """

    # ------------------------------------------------------------------
    # Check prerequisite experiment artifacts
    # ------------------------------------------------------------------

    prerequisite_status = {
        "baseline": (
            load_baseline_results()
            is not None
        ),

        "experts": (
            load_expert_results()
            is not None
        ),

        "learning_to_defer": (
            load_learning_to_defer_results()
            is not None
        ),

        "active_learning": (
            load_active_learning_results()
            is not None
        ),
    }


    missing_prerequisites = []

    if not prerequisite_status[
        "baseline"
    ]:

        missing_prerequisites.append(
            "Baseline"
        )


    if not prerequisite_status[
        "experts"
    ]:

        missing_prerequisites.append(
            "Simulated Experts"
        )


    if not prerequisite_status[
        "learning_to_defer"
    ]:

        missing_prerequisites.append(
            "Learning to Defer"
        )


    if not prerequisite_status[
        "active_learning"
    ]:

        missing_prerequisites.append(
            "Active Learning"
        )


    prerequisites_ready = (
        len(
            missing_prerequisites
        )
        == 0
    )


    # ------------------------------------------------------------------
    # Generate / regenerate analysis
    # ------------------------------------------------------------------

    if request.method == "POST":

        if not prerequisites_ready:

            messages.error(
                request,
                (
                    "Advanced Analysis cannot be run yet. "
                    "Please complete: "
                    + ", ".join(
                        missing_prerequisites
                    )
                    + "."
                ),
            )

            return redirect(
                "project3:advanced_analysis"
            )


        try:

            run_advanced_analysis()

            messages.success(
                request,
                (
                    "Advanced Analysis completed "
                    "successfully."
                ),
            )

        except Exception as exc:

            messages.error(
                request,
                (
                    "Advanced Analysis could not "
                    "be completed: "
                    f"{exc}"
                ),
            )

        return redirect(
            "project3:advanced_analysis"
        )


    # ------------------------------------------------------------------
    # Load previous saved analysis
    # ------------------------------------------------------------------

    results = (
        load_advanced_analysis_results()
    )


    # ------------------------------------------------------------------
    # Check whether those saved results are still current
    # ------------------------------------------------------------------

    results_stale = (
        advanced_analysis_is_stale(
            results
        )
    )


    return render(
        request,
        "project3/advanced_analysis.html",
        {
            "results": (
                results
            ),

            "results_available": (
                results
                is not None
            ),

            "results_stale": (
                results_stale
            ),

            "prerequisites_ready": (
                prerequisites_ready
            ),

            "missing_prerequisites": (
                missing_prerequisites
            ),

            "prerequisite_status": (
                prerequisite_status
            ),
        },
    )

def active_learning_compare(request):
    """
    Compare all Active Learning strategies.

    The complete benchmark is intentionally separated from the
    user-controlled single-strategy workflow.
    """

    results = (
        load_active_learning_results()
    )

    selected_expert = (
        request.GET.get("expert")
        or request.POST.get("expert")
        or next(iter(EXPERT_PROFILES))
    )

    if selected_expert not in EXPERT_PROFILES:
        selected_expert = next(
            iter(EXPERT_PROFILES)
        )

    if request.method == "POST":

        try:
            results = (
                run_active_learning_experiment()
            )

            messages.success(
                request,
                "Active Learning strategy comparison completed.",
            )

        except Exception as exc:

            messages.error(
                request,
                str(exc),
            )

    selected_expert_result = None

    if results:
        selected_expert_result = (
            results.get(
                "experts",
                {},
            ).get(
                selected_expert
            )
        )

    expert_options = [
        {
            "key": key,
            "name": profile.name,
        }
        for key, profile
        in EXPERT_PROFILES.items()
    ]

    return render(
        request,
        "project3/active_learning_compare.html",
        {
            "results": results,
            "results_available": (
                results is not None
            ),
            "selected_expert": (
                selected_expert
            ),
            "selected_expert_result": (
                selected_expert_result
            ),
            "expert_options": (
                expert_options
            ),
        },
    )