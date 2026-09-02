import secrets
import uuid

import pandas as pd
from django.http import HttpResponseBadRequest
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods, require_POST

from .services.movie_data import load_movie_catalog
from .services.study_flow import CONDITION_LABELS, create_study_plan


STUDY_SESSION_KEY = "project4_study"
MAX_RESPONSE_TIME_MS = 60 * 60 * 1000


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


def _load_display_movies(movie_ids):
    catalog = load_movie_catalog().set_index("movie_id", drop=False)
    positions = ("A", "B")
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
        study_state["condition_index"] += 1
        study_state["task_index"] = 0
        study_state["status"] = "ready"
        request.session[STUDY_SESSION_KEY] = study_state
        return redirect("project4:study_session")

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
            study_state["condition_index"] += 1
            study_state["task_index"] = 0
            study_state["status"] = "ready"
            next_page = "project4:study_session"
        else:
            study_state["status"] = "pairwise"
            next_page = "project4:pairwise_task"

        request.session[STUDY_SESSION_KEY] = study_state
        return redirect(next_page)

    if study_state["status"] != "pairwise":
        study_state["status"] = "pairwise"
        request.session[STUDY_SESSION_KEY] = study_state

    try:
        left_movie, right_movie = _load_display_movies(current_pair)
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
