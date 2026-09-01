from django.shortcuts import render


def index(request):
    """Show the Project 4 landing page."""
    return render(request, "project4/index.html")


def study_intro(request):
    """Show the participant information page before the study begins."""
    return render(request, "project4/study_intro.html")

