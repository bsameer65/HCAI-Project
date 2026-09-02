from django.urls import path

from . import views


app_name = "project4"

urlpatterns = [
    path("", views.index, name="index"),
    path("study/", views.study_intro, name="study"),
    path("study/start/", views.start_study, name="start_study"),
    path("study/session/", views.study_session, name="study_session"),
    path("study/pairwise/", views.pairwise_task, name="pairwise_task"),
]
