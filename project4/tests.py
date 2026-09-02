from unittest import mock

from django.test import SimpleTestCase, override_settings
from django.urls import reverse


@override_settings(
    SESSION_ENGINE="django.contrib.sessions.backends.signed_cookies"
)
class Project4PageTests(SimpleTestCase):
    def _start_pairwise_first_study(self):
        # This seed maps to counterbalancing cell 1: pairwise, then ranking.
        with mock.patch("project4.views.secrets.randbits", return_value=12344):
            return self.client.post(
                reverse("project4:start_study"),
                {"consent": "yes"},
            )

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

    def test_pairwise_first_session_can_begin_from_ready_page(self):
        self._start_pairwise_first_study()

        response = self.client.get(reverse("project4:study_session"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse("project4:pairwise_task"))
        self.assertContains(response, "Begin")
        self.assertContains(response, "pairwise choices")

    def test_pairwise_page_shows_task_movies_and_progress(self):
        self._start_pairwise_first_study()
        first_pair = self.client.session["project4_study"]["pairwise_tasks"][0]

        response = self.client.get(reverse("project4:pairwise_task"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Which movie would you rather watch?")
        self.assertContains(response, "1 of 27")
        for movie_id in first_pair:
            self.assertContains(response, f'value="{movie_id}"')

        displayed_movie = response.context["left_movie"]
        self.assertEqual(displayed_movie["movie_id"], first_pair[0])
        self.assertIn("genres", displayed_movie)
        self.assertNotIn("imdb_score", displayed_movie)
        self.assertNotIn("num_voted_users", displayed_movie)
        self.assertNotIn("gross", displayed_movie)

    def test_pairwise_choice_is_recorded_and_advances(self):
        self._start_pairwise_first_study()
        first_pair = self.client.session["project4_study"]["pairwise_tasks"][0]

        response = self.client.post(
            reverse("project4:pairwise_task"),
            {
                "task_index": 0,
                "chosen_movie_id": first_pair[0],
                "response_time_ms": 1420,
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("project4:pairwise_task"))
        study_state = self.client.session["project4_study"]
        self.assertEqual(study_state["task_index"], 1)
        self.assertEqual(study_state["status"], "pairwise")
        self.assertEqual(len(study_state["responses"]["pairwise"]), 1)
        saved_response = study_state["responses"]["pairwise"][0]
        self.assertEqual(saved_response["chosen_movie_id"], first_pair[0])
        self.assertEqual(saved_response["chosen_position"], "left")
        self.assertEqual(saved_response["response_time_ms"], 1420)

    def test_pairwise_choice_must_match_the_displayed_pair(self):
        self._start_pairwise_first_study()

        response = self.client.post(
            reverse("project4:pairwise_task"),
            {"task_index": 0, "chosen_movie_id": "tt-not-displayed"},
        )

        self.assertEqual(response.status_code, 400)
        study_state = self.client.session["project4_study"]
        self.assertEqual(study_state["task_index"], 0)
        self.assertEqual(study_state["responses"]["pairwise"], [])

    def test_stale_pairwise_submission_is_not_recorded_twice(self):
        self._start_pairwise_first_study()
        first_pair = self.client.session["project4_study"]["pairwise_tasks"][0]
        submission = {
            "task_index": 0,
            "chosen_movie_id": first_pair[0],
        }

        self.client.post(reverse("project4:pairwise_task"), submission)
        response = self.client.post(reverse("project4:pairwise_task"), submission)

        self.assertEqual(response.status_code, 302)
        study_state = self.client.session["project4_study"]
        self.assertEqual(study_state["task_index"], 1)
        self.assertEqual(len(study_state["responses"]["pairwise"]), 1)

    def test_finishing_pairwise_condition_opens_the_next_condition(self):
        self._start_pairwise_first_study()
        tasks = self.client.session["project4_study"]["pairwise_tasks"]

        for task_index, pair in enumerate(tasks):
            response = self.client.post(
                reverse("project4:pairwise_task"),
                {"task_index": task_index, "chosen_movie_id": pair[0]},
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("project4:study_session"))
        study_state = self.client.session["project4_study"]
        self.assertEqual(study_state["condition_index"], 1)
        self.assertEqual(study_state["task_index"], 0)
        self.assertEqual(study_state["status"], "ready")
        self.assertEqual(len(study_state["responses"]["pairwise"]), 27)

    @mock.patch("project4.views.secrets.randbits", return_value=12345)
    def test_pairwise_page_does_not_bypass_assigned_ranking_condition(
        self, _mock_seed
    ):
        self.client.post(
            reverse("project4:start_study"),
            {"consent": "yes"},
        )

        response = self.client.get(reverse("project4:pairwise_task"))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("project4:study_session"))
