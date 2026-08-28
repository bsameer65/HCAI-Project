from django.urls import path

from . import views


app_name = "project2"


urlpatterns = [

    path(
        "",
        views.index,
        name="index",
    ),

    path(
        "train/",
        views.train,
        name="train",
    ),

    path(
        "counterfactual/",
        views.counterfactual,
        name="counterfactual",
    ),

    path(
        "pdp-ale/",
        views.pdp_ale,
        name="pdp_ale",
    ),

    path(
        "model-comparison/",
        views.model_comparison,
        name="model_comparison",
    ),

]