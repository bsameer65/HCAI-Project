from django.http import HttpResponse
from django.urls import include, path


home_patterns = ([
    path("", lambda request: HttpResponse("Home"), name="index"),
], "home")

urlpatterns = [
    path("project1/", include("project1.urls")),
    path("home/", include(home_patterns, namespace="home")),
]
