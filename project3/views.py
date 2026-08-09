from django.shortcuts import render


def index(request):
    return render(request, "project3/index.html")


def baseline(request):
    return render(request, "project3/baseline.html")


def expert(request):
    return render(request, "project3/expert.html")


def learning_to_defer(request):
    return render(request, "project3/defer.html")


def active_learning(request):
    return render(request, "project3/active_learning.html")


def compare_results(request):
    return render(request, "project3/compare.html")