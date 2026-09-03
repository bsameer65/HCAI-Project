from copy import deepcopy
from unittest import TestCase

from project4.services.movie_data import prepare_movie_data
from project4.services.study_flow import create_study_plan
from project4.services.study_results import (
    fit_study_models,
    summarize_validation,
)


def _completed_study_state():
    movies, features = prepare_movie_data()
    state = create_study_plan(movies["movie_id"].tolist(), seed=24680, cell=1)
    state["responses"] = {
        "pairwise": [
            {
                "task_index": index,
                "left_movie_id": pair[0],
                "right_movie_id": pair[1],
                "chosen_movie_id": pair[index % 2],
            }
            for index, pair in enumerate(state["pairwise_tasks"])
        ],
        "ranking": [
            {
                "task_index": index,
                "presented_movie_ids": ranking,
                "ranked_movie_ids": list(ranking),
            }
            for index, ranking in enumerate(state["ranking_tasks"])
        ],
        "validation": [],
        "questionnaires": [],
    }
    return state, movies, features


class StudyModelFittingTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.state, cls.movies, cls.features = _completed_study_state()
        cls.results = fit_study_models(cls.state, cls.movies, cls.features)

    def test_separate_models_are_fitted_with_the_expected_feature_count(self):
        self.assertEqual(self.results["feature_count"], 53)
        self.assertEqual(len(self.results["pairwise_model"]["weights"]), 53)
        self.assertEqual(len(self.results["ranking_model"]["weights"]), 53)

    def test_every_validation_pair_receives_two_valid_predictions(self):
        predictions = self.results["validation_predictions"]
        self.assertEqual(len(predictions), len(self.state["validation_tasks"]))

        for task, prediction in zip(self.state["validation_tasks"], predictions):
            self.assertEqual(
                [prediction["left_movie_id"], prediction["right_movie_id"]],
                task,
            )
            self.assertIn(prediction["pairwise_predicted_movie_id"], task)
            self.assertIn(prediction["ranking_predicted_movie_id"], task)
            self.assertGreaterEqual(prediction["pairwise_probability_left"], 0.0)
            self.assertLessEqual(prediction["pairwise_probability_left"], 1.0)
            self.assertGreaterEqual(prediction["ranking_probability_left"], 0.0)
            self.assertLessEqual(prediction["ranking_probability_left"], 1.0)

    def test_incomplete_or_inconsistent_responses_are_rejected(self):
        incomplete = deepcopy(self.state)
        incomplete["responses"]["pairwise"].pop()
        with self.assertRaisesRegex(ValueError, "Every pairwise task"):
            fit_study_models(incomplete, self.movies, self.features)

        inconsistent = deepcopy(self.state)
        inconsistent["responses"]["ranking"][0]["ranked_movie_ids"][0] = (
            inconsistent["validation_pool"][0]
        )
        with self.assertRaisesRegex(ValueError, "complete task ordering"):
            fit_study_models(inconsistent, self.movies, self.features)


class ValidationSummaryTests(TestCase):
    def test_summary_counts_each_models_matches_and_agreement(self):
        model_results = {
            "validation_predictions": [
                {
                    "task_index": 0,
                    "left_movie_id": "a",
                    "right_movie_id": "b",
                    "pairwise_predicted_movie_id": "a",
                    "ranking_predicted_movie_id": "b",
                },
                {
                    "task_index": 1,
                    "left_movie_id": "c",
                    "right_movie_id": "d",
                    "pairwise_predicted_movie_id": "d",
                    "ranking_predicted_movie_id": "d",
                },
            ]
        }
        responses = [
            {
                "task_index": 0,
                "left_movie_id": "a",
                "right_movie_id": "b",
                "chosen_movie_id": "a",
            },
            {
                "task_index": 1,
                "left_movie_id": "c",
                "right_movie_id": "d",
                "chosen_movie_id": "d",
            },
        ]

        summary = summarize_validation(model_results, responses)

        self.assertEqual(summary["pairwise_matches"], 2)
        self.assertEqual(summary["ranking_matches"], 1)
        self.assertEqual(summary["model_agreements"], 1)
        self.assertEqual(summary["pairwise_accuracy"], 1.0)
        self.assertEqual(summary["ranking_accuracy"], 0.5)
        self.assertEqual(summary["model_agreement_rate"], 0.5)

    def test_summary_rejects_incomplete_or_unassigned_choices(self):
        model_results = {
            "validation_predictions": [
                {
                    "task_index": 0,
                    "left_movie_id": "a",
                    "right_movie_id": "b",
                    "pairwise_predicted_movie_id": "a",
                    "ranking_predicted_movie_id": "b",
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "All validation tasks"):
            summarize_validation(model_results, [])
        with self.assertRaisesRegex(ValueError, "unassigned movie"):
            summarize_validation(
                model_results,
                [
                    {
                        "task_index": 0,
                        "left_movie_id": "a",
                        "right_movie_id": "b",
                        "chosen_movie_id": "c",
                    }
                ],
            )
