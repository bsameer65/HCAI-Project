from unittest import mock

from django.test import SimpleTestCase, override_settings
from django.urls import reverse


@override_settings(
    SESSION_ENGINE="django.contrib.sessions.backends.signed_cookies"
)
class Project4PageTests(SimpleTestCase):
    def test_landing_page_loads(self):
        response = self.client.get(reverse("project4:index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Preference Elicitation")
        self.assertContains(response, "Start the study")

    def test_study_intro_page_loads(self):
        response = self.client.get(reverse("project4:study"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Before you begin")
        self.assertContains(response, "Create my study session")

    def test_study_start_requires_consent(self):
        response = self.client.post(reverse("project4:start_study"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Please confirm the consent statement")
        self.assertNotIn("project4_study", self.client.session)

    @mock.patch("project4.views.secrets.randbits", return_value=12345)
    def test_study_start_creates_session_and_redirects(self, _mock_seed):
        response = self.client.post(
            reverse("project4:start_study"),
            {"consent": "yes"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("project4:study_session"))

        study_state = self.client.session["project4_study"]
        self.assertEqual(study_state["seed"], 12345)
        self.assertEqual(study_state["status"], "ready")
        self.assertEqual(len(study_state["pairwise_tasks"]), 27)
        self.assertEqual(len(study_state["ranking_tasks"]), 3)
        self.assertEqual(len(study_state["validation_tasks"]), 20)

    def test_study_session_requires_an_active_plan(self):
        response = self.client.get(reverse("project4:study_session"))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("project4:study"))

    @mock.patch("project4.views.secrets.randbits", return_value=12345)
    def test_ready_page_shows_assigned_first_condition(self, _mock_seed):
        self.client.post(
            reverse("project4:start_study"),
            {"consent": "yes"},
        )

        response = self.client.get(reverse("project4:study_session"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Study session ready")
        self.assertContains(response, "Pairwise choices")
        self.assertContains(response, "Ten-movie rankings")
