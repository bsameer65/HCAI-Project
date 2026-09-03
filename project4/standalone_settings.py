"""Minimal Django settings for running Project 4 independently.

The shared coursework project also contains earlier applications with their own
optional dependencies.  Keeping this settings module focused on Project 4 makes
the assignment reproducible from ``requirements-project4.txt`` alone.
"""

from pbl.settings import *  # noqa: F403


INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "project4.apps.Project4Config",
]

ROOT_URLCONF = "project4.standalone_urls"
