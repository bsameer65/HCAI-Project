from pathlib import Path
from unittest import mock

from django.test import SimpleTestCase, override_settings
from django.urls import reverse

from project4.services.movie_data import load_movie_catalog
from project4.services.study_flow import create_study_plan


REPORT_PDF_PATH = (
    Path(__file__).resolve().parent
    / "static"
    / "project4"
    / "report"
    / "project4_report.pdf"
)


@override_settings(
    ROOT_URLCONF="project4.standalone_urls",
    SESSION_ENGINE="django.contrib.sessions.backends.signed_cookies",
)
class Project4StandaloneConfigTests(SimpleTestCase):
    def test_standalone_root_redirects_to_project_landing_page(self):
        response = self.client.get("/")

        self.assertRedirects(
            response,
            reverse("project4:index"),
            fetch_redirect_response=False,
        )


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

    def _questionnaire_answers(self, condition, **overrides):
        answers = {
            "condition": condition,
            "ease_of_use": 6,
            "preference_expression": 5,
            "response_confidence": 6,
            "mental_demand": 3,
            "comment": "",
            "response_time_ms": 4200,
        }
        answers.update(overrides)
        return answers

    def _start_compact_study(self):
        """Use one task per method and two validation pairs in flow tests."""
        movie_ids = load_movie_catalog()["movie_id"].tolist()
        plan = create_study_plan(movie_ids, seed=12344, cell=1)
        plan["pairwise_tasks"] = plan["pairwise_tasks"][:1]
        plan["ranking_tasks"] = plan["ranking_tasks"][:1]
        plan["validation_tasks"] = plan["validation_tasks"][:2]

        with mock.patch("project4.views.create_study_plan", return_value=plan):
            return self.client.post(
                reverse("project4:start_study"),
                {"consent": "yes"},
            )

    def _complete_compact_study_conditions(self):
        self._start_compact_study()
        state = self.client.session["project4_study"]
        pair = state["pairwise_tasks"][0]
        self.client.post(
            reverse("project4:pairwise_task"),
            {"task_index": 0, "chosen_movie_id": pair[0]},
        )
        self.client.post(
            reverse("project4:condition_questionnaire"),
            self._questionnaire_answers("pairwise"),
        )

        ranking = self.client.session["project4_study"]["ranking_tasks"][0]
        self.client.post(
            reverse("project4:ranking_task"),
            {"task_index": 0, "movie_order": ranking},
        )
        self.client.post(
            reverse("project4:condition_questionnaire"),
            self._questionnaire_answers("ranking"),
        )

    def test_landing_page_loads(self):
        response = self.client.get(reverse("project4:index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Preference Elicitation")
        self.assertContains(response, "Start the study")

    def test_landing_page_links_a_valid_pdf_report(self):
        response = self.client.get(reverse("project4:index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "/static/project4/report/project4_report.pdf",
        )
        self.assertContains(response, "Download report")
        self.assertContains(response, "download")
        self.assertTrue(REPORT_PDF_PATH.is_file())
        self.assertEqual(REPORT_PDF_PATH.read_bytes()[:5], b"%PDF-")

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
        self.assertEqual(
            response.url,
            reverse("project4:condition_questionnaire"),
        )
        study_state = self.client.session["project4_study"]
        self.assertEqual(study_state["condition_index"], 1)
        self.assertEqual(study_state["task_index"], 0)
        self.assertEqual(study_state["status"], "questionnaire")
        self.assertEqual(study_state["pending_questionnaire"], "pairwise")
        self.assertEqual(len(study_state["responses"]["pairwise"]), 27)

        questionnaire_page = self.client.get(
            reverse("project4:condition_questionnaire")
        )
        self.assertEqual(questionnaire_page.status_code, 200)
        self.assertContains(questionnaire_page, "pairwise choices activity feel")

        self.client.post(
            reverse("project4:condition_questionnaire"),
            self._questionnaire_answers("pairwise"),
        )
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
        self.assertEqual(
            response.url,
            reverse("project4:condition_questionnaire"),
        )
        study_state = self.client.session["project4_study"]
        self.assertEqual(study_state["condition_index"], 1)
        self.assertEqual(study_state["task_index"], 0)
        self.assertEqual(study_state["status"], "questionnaire")
        self.assertEqual(study_state["pending_questionnaire"], "ranking")
        self.assertEqual(len(study_state["responses"]["ranking"]), 3)

        questionnaire_response = self.client.post(
            reverse("project4:condition_questionnaire"),
            self._questionnaire_answers(
                "ranking",
                comment="  Moving rows was clear.  ",
            ),
        )
        self.assertEqual(questionnaire_response.status_code, 302)
        self.assertEqual(
            questionnaire_response.url,
            reverse("project4:study_session"),
        )

        study_state = self.client.session["project4_study"]
        self.assertEqual(study_state["status"], "ready")
        self.assertNotIn("pending_questionnaire", study_state)
        saved_feedback = study_state["responses"]["questionnaires"][0]
        self.assertEqual(saved_feedback["condition"], "ranking")
        self.assertEqual(saved_feedback["condition_order_position"], 1)
        self.assertEqual(saved_feedback["scores"]["mental_demand"], 3)
        self.assertEqual(saved_feedback["comment"], "Moving rows was clear.")
        self.assertEqual(saved_feedback["response_time_ms"], 4200)

        next_response = self.client.get(reverse("project4:pairwise_task"))
        self.assertEqual(next_response.status_code, 200)

    def test_ranking_page_does_not_bypass_assigned_pairwise_condition(self):
        self._start_pairwise_first_study()

        response = self.client.get(reverse("project4:ranking_task"))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("project4:study_session"))

    def test_questionnaire_cannot_be_opened_before_a_condition_finishes(self):
        self._start_ranking_first_study()

        response = self.client.get(reverse("project4:condition_questionnaire"))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("project4:study_session"))

    def test_questionnaire_requires_all_four_valid_scale_answers(self):
        self._start_ranking_first_study()
        tasks = self.client.session["project4_study"]["ranking_tasks"]
        for task_index, ranking in enumerate(tasks):
            self.client.post(
                reverse("project4:ranking_task"),
                {"task_index": task_index, "movie_order": ranking},
            )

        missing_answer = self._questionnaire_answers("ranking")
        missing_answer.pop("response_confidence")
        missing_response = self.client.post(
            reverse("project4:condition_questionnaire"),
            missing_answer,
        )
        invalid_response = self.client.post(
            reverse("project4:condition_questionnaire"),
            self._questionnaire_answers("ranking", mental_demand=8),
        )

        self.assertEqual(missing_response.status_code, 400)
        self.assertEqual(invalid_response.status_code, 400)
        study_state = self.client.session["project4_study"]
        self.assertEqual(study_state["status"], "questionnaire")
        self.assertEqual(study_state["responses"]["questionnaires"], [])

    def test_stale_questionnaire_is_not_recorded_twice(self):
        self._start_ranking_first_study()
        tasks = self.client.session["project4_study"]["ranking_tasks"]
        for task_index, ranking in enumerate(tasks):
            self.client.post(
                reverse("project4:ranking_task"),
                {"task_index": task_index, "movie_order": ranking},
            )

        answers = self._questionnaire_answers("ranking")
        self.client.post(reverse("project4:condition_questionnaire"), answers)
        response = self.client.post(
            reverse("project4:condition_questionnaire"),
            answers,
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("project4:study_session"))
        study_state = self.client.session["project4_study"]
        self.assertEqual(len(study_state["responses"]["questionnaires"]), 1)

    def test_feedback_after_both_conditions_opens_validation_stage(self):
        self._start_ranking_first_study()
        ranking_tasks = self.client.session["project4_study"]["ranking_tasks"]
        for task_index, ranking in enumerate(ranking_tasks):
            self.client.post(
                reverse("project4:ranking_task"),
                {"task_index": task_index, "movie_order": ranking},
            )
        self.client.post(
            reverse("project4:condition_questionnaire"),
            self._questionnaire_answers("ranking"),
        )

        pairwise_tasks = self.client.session["project4_study"]["pairwise_tasks"]
        for task_index, pair in enumerate(pairwise_tasks):
            self.client.post(
                reverse("project4:pairwise_task"),
                {"task_index": task_index, "chosen_movie_id": pair[0]},
            )
        self.client.post(
            reverse("project4:condition_questionnaire"),
            self._questionnaire_answers("pairwise"),
        )

        study_state = self.client.session["project4_study"]
        self.assertEqual(study_state["condition_index"], 2)
        self.assertEqual(study_state["status"], "ready")
        self.assertEqual(
            [
                response["condition"]
                for response in study_state["responses"]["questionnaires"]
            ],
            ["ranking", "pairwise"],
        )

        response = self.client.get(reverse("project4:study_session"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Validation choices")
        self.assertContains(response, reverse("project4:validation_task"))

    def test_validation_cannot_start_before_both_conditions(self):
        self._start_pairwise_first_study()

        response = self.client.get(reverse("project4:validation_task"))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("project4:study_session"))

    def test_validation_page_fits_models_and_keeps_predictions_hidden(self):
        self._complete_compact_study_conditions()
        first_pair = self.client.session["project4_study"]["validation_tasks"][0]

        response = self.client.get(reverse("project4:validation_task"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "These are new movie pairs")
        self.assertContains(response, "1 of 2")
        self.assertNotContains(response, "pairwise_predicted_movie_id")
        self.assertNotContains(response, "ranking_predicted_movie_id")
        for movie_id in first_pair:
            self.assertContains(response, f'value="{movie_id}"')

        study_state = self.client.session["project4_study"]
        self.assertEqual(study_state["status"], "validation")
        self.assertEqual(
            len(study_state["model_results"]["validation_predictions"]),
            2,
        )

    def test_validation_choice_is_checked_recorded_and_advances(self):
        self._complete_compact_study_conditions()
        pair = self.client.session["project4_study"]["validation_tasks"][0]

        invalid_response = self.client.post(
            reverse("project4:validation_task"),
            {"task_index": 0, "chosen_movie_id": "tt-not-displayed"},
        )
        valid_response = self.client.post(
            reverse("project4:validation_task"),
            {
                "task_index": 0,
                "chosen_movie_id": pair[1],
                "response_time_ms": 2100,
            },
        )

        self.assertEqual(invalid_response.status_code, 400)
        self.assertEqual(valid_response.status_code, 302)
        self.assertEqual(valid_response.url, reverse("project4:validation_task"))
        study_state = self.client.session["project4_study"]
        self.assertEqual(study_state["task_index"], 1)
        self.assertEqual(len(study_state["responses"]["validation"]), 1)
        saved_response = study_state["responses"]["validation"][0]
        self.assertEqual(saved_response["chosen_movie_id"], pair[1])
        self.assertEqual(saved_response["chosen_position"], "right")
        self.assertEqual(saved_response["response_time_ms"], 2100)

        stale_response = self.client.post(
            reverse("project4:validation_task"),
            {
                "task_index": 0,
                "chosen_movie_id": pair[1],
            },
        )
        self.assertEqual(stale_response.status_code, 302)
        study_state = self.client.session["project4_study"]
        self.assertEqual(study_state["task_index"], 1)
        self.assertEqual(len(study_state["responses"]["validation"]), 1)

    def test_finishing_validation_scores_models_and_shows_debrief(self):
        self._complete_compact_study_conditions()
        tasks = self.client.session["project4_study"]["validation_tasks"]

        for task_index, pair in enumerate(tasks):
            response = self.client.post(
                reverse("project4:validation_task"),
                {"task_index": task_index, "chosen_movie_id": pair[0]},
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("project4:study_complete"))
        study_state = self.client.session["project4_study"]
        self.assertEqual(study_state["status"], "complete")
        self.assertIn("completed_at", study_state)
        self.assertEqual(study_state["validation_summary"]["task_count"], 2)

        completion_page = self.client.get(reverse("project4:study_complete"))
        self.assertEqual(completion_page.status_code, 200)
        self.assertContains(completion_page, "Thank you for taking part")
        self.assertContains(completion_page, "not a result from a conducted user study")
        self.assertContains(completion_page, "Trained from pairwise choices")
        self.assertContains(completion_page, "Trained from ten-movie rankings")

        resume_response = self.client.get(reverse("project4:study_session"))
        self.assertEqual(resume_response.status_code, 302)
        self.assertEqual(resume_response.url, reverse("project4:study_complete"))

    def test_completion_page_requires_a_finished_validation_stage(self):
        self._start_pairwise_first_study()

        response = self.client.get(reverse("project4:study_complete"))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("project4:study_session"))
