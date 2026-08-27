from django.urls import path
from . import views

app_name = "project1"

urlpatterns = [
    path("", views.index, name="index"),

    path("regression/", views.regression, name="regression"),
    path("regression/setup/", views.regression_setup, name="regression_setup"),
    path("regression/analyze/", views.regression_analyze, name="regression_analyze"),
    path("regression/train/", views.regression_train, name="regression_train"),
    path("regression/result/", views.regression_result, name="regression_result"),
    path("regression/test/", views.regression_test, name="regression_test"),

    path(
        "classification/",
        views.classification,
        name="classification",
    ),

    path(
        "classification/analyze/",
        views.classification_analyze,
        name="classification_analyze",
    ),

    path(
        "classification/train/",
        views.classification_train,
        name="classification_train",
    ),

    path(
        "classification/test/",
        views.classification_test,
        name="classification_test",
    ),

    path(
        "classification/explain/",
        views.classification_explain,
        name="classification_explain",
    ),

    path(
        "classification/compare/",
        views.classification_compare,
        name="classification_compare",
    ),
]
