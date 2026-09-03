from collections import Counter
from unittest import TestCase

from project4.services.study_flow import (
    COUNTERBALANCE_CELLS,
    ELICITATION_POOL_SIZE,
    PAIRWISE_TASK_COUNT,
    RANKING_SIZE,
    RANKING_TASK_COUNT,
    REQUIRED_MOVIE_COUNT,
    VALIDATION_TASK_COUNT,
    create_study_plan,
)


class StudyPlanTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.movie_ids = [f"tt{index:07d}" for index in range(200)]

    def test_same_seed_and_cell_produce_same_plan(self):
        first_plan = create_study_plan(self.movie_ids, seed=12345, cell=1)
        second_plan = create_study_plan(self.movie_ids, seed=12345, cell=1)

        self.assertEqual(first_plan, second_plan)

    def test_different_seeds_produce_different_tasks(self):
        first_plan = create_study_plan(self.movie_ids, seed=111, cell=1)
        second_plan = create_study_plan(self.movie_ids, seed=222, cell=1)

        self.assertNotEqual(first_plan["pairwise_pool"], second_plan["pairwise_pool"])

    def test_study_uses_three_disjoint_movie_pools(self):
        plan = create_study_plan(self.movie_ids, seed=12345, cell=1)
        pairwise_movies = set(plan["pairwise_pool"])
        ranking_movies = set(plan["ranking_pool"])
        validation_movies = set(plan["validation_pool"])

        self.assertEqual(len(pairwise_movies), ELICITATION_POOL_SIZE)
        self.assertEqual(len(ranking_movies), ELICITATION_POOL_SIZE)
        self.assertEqual(len(validation_movies), VALIDATION_TASK_COUNT * 2)
        self.assertTrue(pairwise_movies.isdisjoint(ranking_movies))
        self.assertTrue(pairwise_movies.isdisjoint(validation_movies))
        self.assertTrue(ranking_movies.isdisjoint(validation_movies))
        self.assertEqual(
            len(pairwise_movies | ranking_movies | validation_movies),
            REQUIRED_MOVIE_COUNT,
        )

    def test_pairwise_tasks_are_distinct_and_balanced(self):
        plan = create_study_plan(self.movie_ids, seed=12345, cell=1)
        tasks = plan["pairwise_tasks"]
        unordered_pairs = {frozenset(task) for task in tasks}
        exposures = Counter(movie_id for task in tasks for movie_id in task)

        self.assertEqual(len(tasks), PAIRWISE_TASK_COUNT)
        self.assertEqual(len(unordered_pairs), PAIRWISE_TASK_COUNT)
        self.assertEqual(set(exposures), set(plan["pairwise_pool"]))
        self.assertLessEqual(max(exposures.values()) - min(exposures.values()), 1)

    def test_rankings_use_each_ranking_movie_once(self):
        plan = create_study_plan(self.movie_ids, seed=12345, cell=1)
        tasks = plan["ranking_tasks"]
        ranked_movies = [movie_id for task in tasks for movie_id in task]

        self.assertEqual(len(tasks), RANKING_TASK_COUNT)
        self.assertTrue(all(len(task) == RANKING_SIZE for task in tasks))
        self.assertEqual(len(ranked_movies), ELICITATION_POOL_SIZE)
        self.assertEqual(set(ranked_movies), set(plan["ranking_pool"]))

    def test_validation_tasks_use_each_validation_movie_once(self):
        plan = create_study_plan(self.movie_ids, seed=12345, cell=1)
        validation_movies = [
            movie_id for task in plan["validation_tasks"] for movie_id in task
        ]

        self.assertEqual(len(plan["validation_tasks"]), VALIDATION_TASK_COUNT)
        self.assertEqual(len(validation_movies), VALIDATION_TASK_COUNT * 2)
        self.assertEqual(set(validation_movies), set(plan["validation_pool"]))

    def test_all_four_counterbalancing_cells_are_applied(self):
        for cell, expected in COUNTERBALANCE_CELLS.items():
            with self.subTest(cell=cell):
                plan = create_study_plan(self.movie_ids, seed=12345, cell=cell)
                self.assertEqual(
                    plan["condition_order"], list(expected["condition_order"])
                )
                self.assertEqual(
                    plan["pairwise_pool_name"], expected["pairwise_pool"]
                )
                self.assertEqual(
                    plan["ranking_pool_name"], expected["ranking_pool"]
                )

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "At least"):
            create_study_plan(
                self.movie_ids[: REQUIRED_MOVIE_COUNT - 1], seed=12345
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            create_study_plan(self.movie_ids + [self.movie_ids[0]], seed=12345)
        with self.assertRaisesRegex(ValueError, "1 to 4"):
            create_study_plan(self.movie_ids, seed=12345, cell=5)
        with self.assertRaisesRegex(ValueError, "1 to 4"):
            create_study_plan(self.movie_ids, seed=12345, cell=1.0)
