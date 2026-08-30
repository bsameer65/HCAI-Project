from django.urls import path

from . import views

app_name = "project3"

urlpatterns = [
    path("", views.index, name="index"),
    path("baseline/", views.baseline, name="baseline"),
    path("expert/", views.expert, name="expert"),

    path(
        "learning-to-defer/",
        views.learning_to_defer,
        name="learning_to_defer",
    ),

    path(
        "active-learning/",
        views.active_learning,
        name="active_learning",
    ),

    path(
        "human-expert/",
        views.human_expert,
        name="human_expert",
    ),

    path(
        "analysis/",
        views.advanced_analysis,
        name="advanced_analysis",
    ),

    path(
        "compare/",
        views.compare_results,
        name="compare_results",
    ),
]