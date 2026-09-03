import secrets
import uuid

import pandas as pd
from django.http import HttpResponseBadRequest
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods, require_POST

from .services.movie_data import load_movie_catalog
from .services.study_flow import CONDITION_LABELS, create_study_plan
from .services.study_results import fit_study_models, summarize_validation


STUDY_SESSION_KEY = "project4_study"
MAX_RESPONSE_TIME_MS = 60 * 60 * 1000
QUESTIONNAIRE_ITEMS = (
    {
        "key": "ease_of_use",
        "prompt": "This activity was easy to use.",
        "low_label": "Strongly disagree",
        "high_label": "Strongly agree",
    },
    {
        "key": "preference_expression",
        "prompt": "This activity let me express my movie preferences accurately.",
        "low_label": "Strongly disagree",
        "high_label": "Strongly agree",
    },
    {
        "key": "response_confidence",
        "prompt": "I feel confident about the responses I gave.",
        "low_label": "Strongly disagree",
        "high_label": "Strongly agree",
    },
    {
        "key": "mental_demand",
        "prompt": "How mentally demanding was this activity?",
        "low_label": "Very low",
        "high_label": "Very high",
    },
)
LIKERT_VALUES = tuple(range(1, 8))
MAX_COMMENT_LENGTH = 1000


def _format_movie_for_display(movie_row, position):
    """Return only participant-facing metadata for one movie card."""
    year = movie_row["year"]
    duration = movie_row["duration"]
    cast_members = [
        movie_row[column]
        for column in ("actor_1", "actor_2", "actor_3")
        if movie_row[column] != "Unknown"
    ]

    return {
        "movie_id": movie_row["movie_id"],
        "position": position,
        "title": movie_row["title"],
        "year": "Year unavailable" if pd.isna(year) else str(int(year)),
        "genres": ", ".join(movie_row["genres"].split("|")),
        "duration": (
            "Duration unavailable"
            if pd.isna(duration)
            else f"{int(round(duration))} min"
        ),
        "language": movie_row["language"],
        "content_rating": movie_row["content_rating"],
        "director": movie_row["director"],
        "cast": ", ".join(cast_members) if cast_members else "Cast unavailable",
    }


def _load_display_movies(movie_ids, positions=None):
    catalog = load_movie_catalog().set_index("movie_id", drop=False)
    positions = positions or [str(index + 1) for index in range(len(movie_ids))]
    if len(positions) != len(movie_ids):
        raise ValueError("Every displayed movie needs one position label.")
    try:
        return [
            _format_movie_for_display(catalog.loc[movie_id], positions[index])
            for index, movie_id in enumerate(movie_ids)
        ]
    except KeyError as error:
        raise ValueError("A study movie is missing from the catalogue.") from error


def _response_time_ms(raw_value):
    """Accept a plausible optional client-side response time."""
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return None
    return value if 0 <= value <= MAX_RESPONSE_TIME_MS else None


def _finish_condition(study_state, condition):
    """Advance the study but require feedback before starting the next stage."""
    study_state["condition_index"] += 1
    study_state["task_index"] = 0
    study_state["pending_questionnaire"] = condition
    study_state["status"] = "questionnaire"


def _finish_validation(study_state):
    study_state["validation_summary"] = summarize_validation(
        study_state["model_results"],
        study_state["responses"]["validation"],
    )
    study_state["status"] = "complete"
    study_state["completed_at"] = timezone.now().isoformat()


def index(request):
    """Show the Project 4 landing page."""
    return render(request, "project4/index.html")


def study_intro(request):
    """Show the participant information page before the study begins."""
    return render(
        request,
        "project4/study_intro.html",
        {"has_active_study": STUDY_SESSION_KEY in request.session},
    )


@require_POST
def start_study(request):
    """Create a randomized study plan after the participant consents."""
    if request.POST.get("consent") != "yes":
        return render(
            request,
            "project4/study_intro.html",
            {
                "consent_error": "Please confirm the consent statement to continue.",
                "has_active_study": STUDY_SESSION_KEY in request.session,
            },
        )

    movie_ids = load_movie_catalog()["movie_id"].tolist()
    seed = secrets.randbits(63)
    study_state = create_study_plan(movie_ids, seed=seed)
    study_state.update(
        {
            "study_id": str(uuid.uuid4()),
            "started_at": timezone.now().isoformat(),
            "status": "ready",
            "condition_index": 0,
            "task_index": 0,
            "responses": {
                "pairwise": [],
                "ranking": [],
                "validation": [],
                "questionnaires": [],
            },
        }
    )
    request.session[STUDY_SESSION_KEY] = study_state
    return redirect("project4:study_session")


def study_session(request):
    """Show the assigned study order and the participant's next activity."""
    study_state = request.session.get(STUDY_SESSION_KEY)
    if study_state is None:
        return redirect("project4:study")
    if (
        study_state.get("status") == "questionnaire"
        and study_state.get("pending_questionnaire") in CONDITION_LABELS
    ):
        return redirect("project4:condition_questionnaire")
    if study_state.get("status") == "complete":
        return redirect("project4:study_complete")

    condition_index = study_state["condition_index"]
    condition_order = [
        {
            "key": condition,
            "label": CONDITION_LABELS[condition],
            "task_count": (
                len(study_state["pairwise_tasks"])
                if condition == "pairwise"
                else len(study_state["ranking_tasks"])
            ),
            "is_complete": index < condition_index,
            "is_current": index == condition_index,
        }
        for index, condition in enumerate(study_state["condition_order"])
    ]

    if condition_index < len(condition_order):
        current_condition = condition_order[condition_index]
    else:
        current_condition = {
            "key": "validation",
            "label": "Validation choices",
            "task_count": len(study_state["validation_tasks"]),
        }

    return render(
        request,
        "project4/study_ready.html",
        {
            "condition_order": condition_order,
            "current_condition": current_condition,
            "has_completed_condition": condition_index > 0,
            "resume_pairwise": study_state["status"] == "pairwise",
            "resume_ranking": study_state["status"] == "ranking",
            "resume_validation": study_state["status"] == "validation",
            "validation_task_count": len(study_state["validation_tasks"]),
            "validation_is_current": condition_index >= len(condition_order),
        },
    )


@require_http_methods(["GET", "POST"])
def pairwise_task(request):
    """Show one movie pair and record exactly one choice for that task."""
    study_state = request.session.get(STUDY_SESSION_KEY)
    if study_state is None:
        return redirect("project4:study")
    if study_state.get("status") == "questionnaire":
        return redirect("project4:condition_questionnaire")

    condition_index = study_state["condition_index"]
    condition_order = study_state["condition_order"]
    if (
        condition_index >= len(condition_order)
        or condition_order[condition_index] != "pairwise"
    ):
        return redirect("project4:study_session")

    task_index = study_state["task_index"]
    tasks = study_state["pairwise_tasks"]
    if task_index >= len(tasks):
        _finish_condition(study_state, "pairwise")
        request.session[STUDY_SESSION_KEY] = study_state
        return redirect("project4:condition_questionnaire")

    current_pair = tasks[task_index]
    if request.method == "POST":
        try:
            submitted_task_index = int(request.POST.get("task_index", ""))
        except (TypeError, ValueError):
            return HttpResponseBadRequest("Invalid pairwise task number.")

        # A browser-back resubmission should not create a duplicate response.
        if submitted_task_index != task_index:
            return redirect("project4:pairwise_task")

        chosen_movie_id = request.POST.get("chosen_movie_id")
        if chosen_movie_id not in current_pair:
            return HttpResponseBadRequest("Choose one of the two displayed movies.")

        study_state["responses"]["pairwise"].append(
            {
                "task_index": task_index,
                "left_movie_id": current_pair[0],
                "right_movie_id": current_pair[1],
                "chosen_movie_id": chosen_movie_id,
                "chosen_position": (
                    "left" if chosen_movie_id == current_pair[0] else "right"
                ),
                "response_time_ms": _response_time_ms(
                    request.POST.get("response_time_ms")
                ),
                "submitted_at": timezone.now().isoformat(),
            }
        )
        study_state["task_index"] += 1

        if study_state["task_index"] == len(tasks):
            _finish_condition(study_state, "pairwise")
            next_page = "project4:condition_questionnaire"
        else:
            study_state["status"] = "pairwise"
            next_page = "project4:pairwise_task"

        request.session[STUDY_SESSION_KEY] = study_state
        return redirect(next_page)

    if study_state["status"] != "pairwise":
        study_state["status"] = "pairwise"
        request.session[STUDY_SESSION_KEY] = study_state

    try:
        left_movie, right_movie = _load_display_movies(
            current_pair,
            positions=("A", "B"),
        )
    except ValueError as error:
        return HttpResponseBadRequest(str(error))

    total_tasks = len(tasks)
    return render(
        request,
        "project4/pairwise_task.html",
        {
            "left_movie": left_movie,
            "right_movie": right_movie,
            "task_index": task_index,
            "task_number": task_index + 1,
            "total_tasks": total_tasks,
            "progress_percent": round((task_index + 1) / total_tasks * 100),
        },
    )


@require_http_methods(["GET", "POST"])
def ranking_task(request):
    """Show ten movies and record a validated most-to-least ranking."""
    study_state = request.session.get(STUDY_SESSION_KEY)
    if study_state is None:
        return redirect("project4:study")
    if study_state.get("status") == "questionnaire":
        return redirect("project4:condition_questionnaire")

    condition_index = study_state["condition_index"]
    condition_order = study_state["condition_order"]
    if (
        condition_index >= len(condition_order)
        or condition_order[condition_index] != "ranking"
    ):
        return redirect("project4:study_session")

    task_index = study_state["task_index"]
    tasks = study_state["ranking_tasks"]
    if task_index >= len(tasks):
        _finish_condition(study_state, "ranking")
        request.session[STUDY_SESSION_KEY] = study_state
        return redirect("project4:condition_questionnaire")

    presented_movie_ids = tasks[task_index]
    if request.method == "POST":
        try:
            submitted_task_index = int(request.POST.get("task_index", ""))
        except (TypeError, ValueError):
            return HttpResponseBadRequest("Invalid ranking task number.")

        if submitted_task_index != task_index:
            return redirect("project4:ranking_task")

        ranked_movie_ids = request.POST.getlist("movie_order")
        submitted_is_complete_ranking = (
            len(ranked_movie_ids) == len(presented_movie_ids)
            and len(set(ranked_movie_ids)) == len(presented_movie_ids)
            and set(ranked_movie_ids) == set(presented_movie_ids)
        )
        if not submitted_is_complete_ranking:
            return HttpResponseBadRequest(
                "Rank every displayed movie exactly once before continuing."
            )

        study_state["responses"]["ranking"].append(
            {
                "task_index": task_index,
                "presented_movie_ids": list(presented_movie_ids),
                "ranked_movie_ids": ranked_movie_ids,
                "response_time_ms": _response_time_ms(
                    request.POST.get("response_time_ms")
                ),
                "submitted_at": timezone.now().isoformat(),
            }
        )
        study_state["task_index"] += 1

        if study_state["task_index"] == len(tasks):
            _finish_condition(study_state, "ranking")
            next_page = "project4:condition_questionnaire"
        else:
            study_state["status"] = "ranking"
            next_page = "project4:ranking_task"

        request.session[STUDY_SESSION_KEY] = study_state
        return redirect(next_page)

    if study_state["status"] != "ranking":
        study_state["status"] = "ranking"
        request.session[STUDY_SESSION_KEY] = study_state

    try:
        movies = _load_display_movies(presented_movie_ids)
    except ValueError as error:
        return HttpResponseBadRequest(str(error))

    total_tasks = len(tasks)
    return render(
        request,
        "project4/ranking_task.html",
        {
            "movies": movies,
            "task_index": task_index,
            "task_number": task_index + 1,
            "total_tasks": total_tasks,
            "progress_percent": round((task_index + 1) / total_tasks * 100),
        },
    )


@require_http_methods(["GET", "POST"])
def condition_questionnaire(request):
    """Collect the same short subjective measures after each method."""
    study_state = request.session.get(STUDY_SESSION_KEY)
    if study_state is None:
        return redirect("project4:study")

    condition = study_state.get("pending_questionnaire")
    if (
        study_state.get("status") != "questionnaire"
        or condition not in CONDITION_LABELS
    ):
        return redirect("project4:study_session")

    if request.method == "POST":
        if request.POST.get("condition") != condition:
            return HttpResponseBadRequest("This questionnaire is no longer active.")

        scores = {}
        for item in QUESTIONNAIRE_ITEMS:
            try:
                score = int(request.POST.get(item["key"], ""))
            except (TypeError, ValueError):
                return HttpResponseBadRequest("Answer every questionnaire item.")
            if score not in LIKERT_VALUES:
                return HttpResponseBadRequest(
                    "Questionnaire answers must use the scale from 1 to 7."
                )
            scores[item["key"]] = score

        comment = request.POST.get("comment", "").strip()
        if len(comment) > MAX_COMMENT_LENGTH:
            return HttpResponseBadRequest(
                f"The optional comment must be {MAX_COMMENT_LENGTH} characters or fewer."
            )

        study_state["responses"]["questionnaires"].append(
            {
                "condition": condition,
                "condition_order_position": study_state["condition_index"],
                "scores": scores,
                "comment": comment,
                "response_time_ms": _response_time_ms(
                    request.POST.get("response_time_ms")
                ),
                "submitted_at": timezone.now().isoformat(),
            }
        )
        study_state.pop("pending_questionnaire", None)
        study_state["status"] = "ready"
        request.session[STUDY_SESSION_KEY] = study_state
        return redirect("project4:study_session")

    return render(
        request,
        "project4/condition_questionnaire.html",
        {
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "questionnaire_items": QUESTIONNAIRE_ITEMS,
            "likert_values": LIKERT_VALUES,
            "is_final_condition": (
                study_state["condition_index"]
                >= len(study_state["condition_order"])
            ),
        },
    )


@require_http_methods(["GET", "POST"])
def validation_task(request):
    """Collect blind choices on held-out movies for both fitted models."""
    study_state = request.session.get(STUDY_SESSION_KEY)
    if study_state is None:
        return redirect("project4:study")
    if study_state.get("status") == "questionnaire":
        return redirect("project4:condition_questionnaire")
    if study_state.get("status") == "complete":
        return redirect("project4:study_complete")
    if study_state["condition_index"] < len(study_state["condition_order"]):
        return redirect("project4:study_session")

    if "model_results" not in study_state:
        try:
            study_state["model_results"] = fit_study_models(study_state)
        except (RuntimeError, ValueError) as error:
            return HttpResponseBadRequest(f"Could not prepare validation: {error}")
        request.session[STUDY_SESSION_KEY] = study_state

    task_index = study_state["task_index"]
    tasks = study_state["validation_tasks"]
    if task_index >= len(tasks):
        try:
            _finish_validation(study_state)
        except ValueError as error:
            return HttpResponseBadRequest(str(error))
        request.session[STUDY_SESSION_KEY] = study_state
        return redirect("project4:study_complete")

    current_pair = tasks[task_index]
    if request.method == "POST":
        try:
            submitted_task_index = int(request.POST.get("task_index", ""))
        except (TypeError, ValueError):
            return HttpResponseBadRequest("Invalid validation task number.")

        if submitted_task_index != task_index:
            return redirect("project4:validation_task")

        chosen_movie_id = request.POST.get("chosen_movie_id")
        if chosen_movie_id not in current_pair:
            return HttpResponseBadRequest("Choose one of the two displayed movies.")

        study_state["responses"]["validation"].append(
            {
                "task_index": task_index,
                "left_movie_id": current_pair[0],
                "right_movie_id": current_pair[1],
                "chosen_movie_id": chosen_movie_id,
                "chosen_position": (
                    "left" if chosen_movie_id == current_pair[0] else "right"
                ),
                "response_time_ms": _response_time_ms(
                    request.POST.get("response_time_ms")
                ),
                "submitted_at": timezone.now().isoformat(),
            }
        )
        study_state["task_index"] += 1

        if study_state["task_index"] == len(tasks):
            _finish_validation(study_state)
            next_page = "project4:study_complete"
        else:
            study_state["status"] = "validation"
            next_page = "project4:validation_task"

        request.session[STUDY_SESSION_KEY] = study_state
        return redirect(next_page)

    if study_state["status"] != "validation":
        study_state["status"] = "validation"
        request.session[STUDY_SESSION_KEY] = study_state

    try:
        left_movie, right_movie = _load_display_movies(
            current_pair,
            positions=("A", "B"),
        )
    except ValueError as error:
        return HttpResponseBadRequest(str(error))

    total_tasks = len(tasks)
    return render(
        request,
        "project4/validation_task.html",
        {
            "left_movie": left_movie,
            "right_movie": right_movie,
            "task_index": task_index,
            "task_number": task_index + 1,
            "total_tasks": total_tasks,
            "progress_percent": round((task_index + 1) / total_tasks * 100),
        },
    )


def study_complete(request):
    """Debrief the participant and show descriptive personal match rates."""
    study_state = request.session.get(STUDY_SESSION_KEY)
    if study_state is None:
        return redirect("project4:study")
    if study_state.get("status") != "complete":
        return redirect("project4:study_session")

    summary = study_state["validation_summary"]
    return render(
        request,
        "project4/study_complete.html",
        {
            "study_code": study_state["study_id"].split("-")[0].upper(),
            "task_count": summary["task_count"],
            "pairwise_matches": summary["pairwise_matches"],
            "ranking_matches": summary["ranking_matches"],
            "pairwise_percent": round(summary["pairwise_accuracy"] * 100),
            "ranking_percent": round(summary["ranking_accuracy"] * 100),
            "model_agreements": summary["model_agreements"],
        },
    )
