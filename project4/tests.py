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

    def _start_ranking_first_study(self):
        # This seed maps to counterbalancing cell 2: ranking, then pairwise.
        with mock.patch("project4.views.secrets.randbits", return_value=12345):
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

        next_response = self.client.get(reverse("project4:ranking_task"))
        self.assertEqual(next_response.status_code, 200)

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

    def test_ranking_first_session_can_begin_from_ready_page(self):
        self._start_ranking_first_study()

        response = self.client.get(reverse("project4:study_session"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse("project4:ranking_task"))
        self.assertContains(response, "Begin")
        self.assertContains(response, "ten-movie rankings")

    def test_ranking_page_shows_all_ten_movies_and_progress(self):
        self._start_ranking_first_study()
        first_ranking = self.client.session["project4_study"]["ranking_tasks"][0]

        response = self.client.get(reverse("project4:ranking_task"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "Order the movies from most to least preferred",
        )
        self.assertContains(response, "1 of 3")
        self.assertEqual(len(response.context["movies"]), 10)
        for movie_id in first_ranking:
            self.assertContains(response, f'value="{movie_id}"')

        displayed_movie = response.context["movies"][0]
        self.assertNotIn("imdb_score", displayed_movie)
        self.assertNotIn("num_voted_users", displayed_movie)
        self.assertNotIn("gross", displayed_movie)

    def test_complete_ranking_is_recorded_in_submitted_order(self):
        self._start_ranking_first_study()
        first_ranking = self.client.session["project4_study"]["ranking_tasks"][0]
        submitted_order = list(reversed(first_ranking))

        response = self.client.post(
            reverse("project4:ranking_task"),
            {
                "task_index": 0,
                "movie_order": submitted_order,
                "response_time_ms": 18750,
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("project4:ranking_task"))
        study_state = self.client.session["project4_study"]
        self.assertEqual(study_state["task_index"], 1)
        self.assertEqual(study_state["status"], "ranking")
        saved_response = study_state["responses"]["ranking"][0]
        self.assertEqual(saved_response["presented_movie_ids"], first_ranking)
        self.assertEqual(saved_response["ranked_movie_ids"], submitted_order)
        self.assertEqual(saved_response["response_time_ms"], 18750)

    def test_incomplete_or_duplicate_ranking_is_rejected(self):
        self._start_ranking_first_study()
        first_ranking = self.client.session["project4_study"]["ranking_tasks"][0]
        invalid_orders = (
            first_ranking[:-1],
            first_ranking[:-1] + [first_ranking[0]],
        )

        for invalid_order in invalid_orders:
            with self.subTest(invalid_order=invalid_order):
                response = self.client.post(
                    reverse("project4:ranking_task"),
                    {"task_index": 0, "movie_order": invalid_order},
                )
                self.assertEqual(response.status_code, 400)

        study_state = self.client.session["project4_study"]
        self.assertEqual(study_state["task_index"], 0)
        self.assertEqual(study_state["responses"]["ranking"], [])

    def test_stale_ranking_submission_is_not_recorded_twice(self):
        self._start_ranking_first_study()
        first_ranking = self.client.session["project4_study"]["ranking_tasks"][0]
        submission = {"task_index": 0, "movie_order": first_ranking}

        self.client.post(reverse("project4:ranking_task"), submission)
        response = self.client.post(reverse("project4:ranking_task"), submission)

        self.assertEqual(response.status_code, 302)
        study_state = self.client.session["project4_study"]
        self.assertEqual(study_state["task_index"], 1)
        self.assertEqual(len(study_state["responses"]["ranking"]), 1)

    def test_finishing_ranking_condition_opens_pairwise_condition(self):
        self._start_ranking_first_study()
        tasks = self.client.session["project4_study"]["ranking_tasks"]

        for task_index, ranking in enumerate(tasks):
            response = self.client.post(
                reverse("project4:ranking_task"),
                {"task_index": task_index, "movie_order": ranking},
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("project4:study_session"))
        study_state = self.client.session["project4_study"]
        self.assertEqual(study_state["condition_index"], 1)
        self.assertEqual(study_state["task_index"], 0)
        self.assertEqual(study_state["status"], "ready")
        self.assertEqual(len(study_state["responses"]["ranking"]), 3)

        next_response = self.client.get(reverse("project4:pairwise_task"))
        self.assertEqual(next_response.status_code, 200)

    def test_ranking_page_does_not_bypass_assigned_pairwise_condition(self):
        self._start_pairwise_first_study()

        response = self.client.get(reverse("project4:ranking_task"))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("project4:study_session"))
