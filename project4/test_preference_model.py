from itertools import permutations
from unittest import TestCase

import numpy as np

from project4.services.movie_data import prepare_movie_data
from project4.services.preference_model import (
    _objective_and_gradient,
    fit_preference_model,
    pairwise_log_probability,
    pairwise_probability,
    ranking_log_probability,
    ranking_probability,
)


class BradleyTerryTests(TestCase):
    def test_equal_utilities_give_equal_probability(self):
        weights = np.array([1.0, -0.5])
        movie = np.array([0.4, 0.2])

        self.assertAlmostEqual(
            pairwise_probability(weights, movie, movie),
            0.5,
        )

    def test_pairwise_probabilities_are_symmetric(self):
        weights = np.array([0.8, -0.3])
        first_movie = np.array([1.0, 0.0])
        second_movie = np.array([0.0, 1.0])

        probability_first = pairwise_probability(
            weights, first_movie, second_movie
        )
        probability_second = pairwise_probability(
            weights, second_movie, first_movie
        )

        self.assertAlmostEqual(probability_first + probability_second, 1.0)

    def test_pairwise_log_probability_is_stable_for_large_utilities(self):
        log_probability = pairwise_log_probability(
            np.array([1000.0]),
            np.array([-1.0]),
            np.array([1.0]),
        )

        self.assertTrue(np.isfinite(log_probability))
        self.assertLess(log_probability, -1000.0)

    def test_non_finite_pairwise_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            pairwise_probability(
                np.array([np.nan]),
                np.array([1.0]),
                np.array([0.0]),
            )


class PlackettLuceTests(TestCase):
    def test_two_item_ranking_reduces_to_bradley_terry(self):
        weights = np.array([0.7, -0.2])
        first_movie = np.array([1.0, 0.3])
        second_movie = np.array([0.2, 1.0])
        ranking = np.vstack([first_movie, second_movie])

        self.assertAlmostEqual(
            ranking_log_probability(weights, ranking),
            pairwise_log_probability(weights, first_movie, second_movie),
        )

    def test_probabilities_of_all_rankings_sum_to_one(self):
        weights = np.array([1.2])
        movies = np.array([[1.5], [0.2], [-0.7]])

        probability_sum = sum(
            ranking_probability(weights, movies[list(order)])
            for order in permutations(range(3))
        )

        self.assertAlmostEqual(probability_sum, 1.0)

    def test_more_plausible_ranking_has_higher_probability(self):
        weights = np.array([1.0])
        best_to_worst = np.array([[2.0], [1.0], [0.0]])

        expected_order = ranking_probability(weights, best_to_worst)
        reverse_order = ranking_probability(weights, best_to_worst[::-1])

        self.assertGreater(expected_order, reverse_order)

    def test_ranking_log_probability_is_numerically_stable(self):
        log_probability = ranking_log_probability(
            np.array([1000.0]),
            np.array([[1.0], [0.0], [-1.0]]),
        )

        self.assertTrue(np.isfinite(log_probability))
        self.assertAlmostEqual(log_probability, 0.0)


class PreferenceFittingTests(TestCase):
    def test_analytic_gradient_matches_finite_difference(self):
        features = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        comparisons = ((2, 1),)
        rankings = ((2, 0, 1),)
        regularization = 0.4
        weights = np.array([0.2, -0.1])

        objective, gradient = _objective_and_gradient(
            weights,
            features,
            comparisons,
            rankings,
            regularization,
        )
        numerical_gradient = np.zeros_like(weights)
        step = 1e-6

        for feature_index in range(len(weights)):
            offset = np.zeros_like(weights)
            offset[feature_index] = step
            upper_objective, _ = _objective_and_gradient(
                weights + offset,
                features,
                comparisons,
                rankings,
                regularization,
            )
            lower_objective, _ = _objective_and_gradient(
                weights - offset,
                features,
                comparisons,
                rankings,
                regularization,
            )
            numerical_gradient[feature_index] = (
                upper_objective - lower_objective
            ) / (2 * step)

        self.assertTrue(np.isfinite(objective))
        np.testing.assert_allclose(gradient, numerical_gradient, rtol=1e-5)

    def test_pairwise_model_learns_consistent_direction(self):
        features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        comparisons = [(3, 0), (3, 1), (2, 0), (2, 1)]

        model = fit_preference_model(
            features,
            pairwise_comparisons=comparisons,
            regularization=0.2,
        )

        self.assertGreater(model.weights[0], 0.0)
        self.assertGreater(
            model.probability_first_preferred(features[3], features[0]),
            0.9,
        )

    def test_ranking_model_learns_consistent_direction(self):
        features = np.array([[-2.0], [-1.0], [1.0], [2.0]])

        model = fit_preference_model(
            features,
            rankings=[(3, 2, 1, 0)],
            regularization=0.2,
        )

        self.assertGreater(model.weights[0], 0.0)
        self.assertGreater(
            model.ranking_log_probability(features[[3, 2, 1, 0]]),
            model.ranking_log_probability(features[[0, 1, 2, 3]]),
        )

    def test_stronger_regularization_produces_smaller_weights(self):
        features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        comparisons = [(3, 0), (3, 1), (2, 0), (2, 1)]

        weakly_regularized = fit_preference_model(
            features,
            pairwise_comparisons=comparisons,
            regularization=0.05,
        )
        strongly_regularized = fit_preference_model(
            features,
            pairwise_comparisons=comparisons,
            regularization=2.0,
        )

        self.assertLess(
            np.linalg.norm(strongly_regularized.weights),
            np.linalg.norm(weakly_regularized.weights),
        )

    def test_model_fits_real_movie_feature_shape(self):
        _, movie_features = prepare_movie_data()
        sample_features = movie_features.iloc[:10]

        model = fit_preference_model(
            sample_features,
            pairwise_comparisons=[(0, 1), (2, 3)],
            rankings=[tuple(range(10))],
            regularization=1.0,
        )

        self.assertEqual(model.weights.shape, (53,))
        self.assertTrue(np.isfinite(model.weights).all())
        self.assertTrue(np.isfinite(model.objective_value))

    def test_invalid_observations_are_rejected(self):
        features = np.array([[0.0], [1.0], [2.0]])

        with self.assertRaisesRegex(ValueError, "At least one"):
            fit_preference_model(features)
        with self.assertRaisesRegex(ValueError, "repeated movie"):
            fit_preference_model(features, rankings=[(2, 1, 1)])
        with self.assertRaisesRegex(ValueError, "outside the movie catalogue"):
            fit_preference_model(features, pairwise_comparisons=[(3, 0)])
        with self.assertRaisesRegex(ValueError, "positive"):
            fit_preference_model(
                features,
                pairwise_comparisons=[(2, 0)],
                regularization=-1.0,
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            fit_preference_model(
                features,
                pairwise_comparisons=[(2, 0)],
                regularization=0.0,
            )
