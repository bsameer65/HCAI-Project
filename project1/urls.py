from django.urls import path
from . import views

app_name = 'project1'

urlpatterns = [
    path('', views.index, name='index'),
    path('classification/', views.classification, name='classification'),
    path('classification/analyze/', views.classification_analyze, name='classification_analyze'),
    path('classification/train/', views.classification_train, name='classification_train'),
    path('classification/test/', views.classification_test, name='classification_test'),
]