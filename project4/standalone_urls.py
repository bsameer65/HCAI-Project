"""URL configuration for the independently runnable Project 4 application."""

from django.urls import include, path
from django.views.generic import RedirectView


home_urls = (
    [
        path(
            "",
            RedirectView.as_view(pattern_name="project4:index", permanent=False),
            name="index",
        )
    ],
    "home",
)

urlpatterns = [
    path(
        "",
        RedirectView.as_view(pattern_name="project4:index", permanent=False),
        name="root",
    ),
    path("home/", include(home_urls, namespace="home")),
    path("project4/", include("project4.urls")),
]
