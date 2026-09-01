from django.urls import path

from . import views


app_name = "project4"

urlpatterns = [
    path("", views.index, name="index"),
    path("study/", views.study_intro, name="study"),
]

