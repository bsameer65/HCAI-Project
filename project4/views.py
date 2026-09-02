import secrets
import uuid

from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from .services.movie_data import load_movie_catalog
from .services.study_flow import CONDITION_LABELS, create_study_plan


STUDY_SESSION_KEY = "project4_study"


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
    """Show the assigned study order before the first elicitation task."""
    study_state = request.session.get(STUDY_SESSION_KEY)
    if study_state is None:
        return redirect("project4:study")

    condition_order = [
        {
            "key": condition,
            "label": CONDITION_LABELS[condition],
            "task_count": (
                len(study_state["pairwise_tasks"])
                if condition == "pairwise"
                else len(study_state["ranking_tasks"])
            ),
        }
        for condition in study_state["condition_order"]
    ]

    return render(
        request,
        "project4/study_ready.html",
        {
            "condition_order": condition_order,
            "first_condition": condition_order[0],
            "validation_task_count": len(study_state["validation_tasks"]),
        },
    )
