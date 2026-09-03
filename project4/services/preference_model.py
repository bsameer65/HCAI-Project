"""Bradley-Terry and Plackett-Luce models for movie preferences.

Each movie has utility ``u_i = w.T @ x_i``. Pairwise observations use the
Bradley-Terry probability, while complete rankings use a Plackett-Luce
sequence of choices from the remaining movies.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


def _as_feature_matrix(features):
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Movie features must be a two-dimensional matrix.")
    if matrix.shape[0] < 2:
        raise ValueError("At least two movies are required.")
    if matrix.shape[1] == 0:
        raise ValueError("Movie features must contain at least one column.")
    if not np.isfinite(matrix).all():
        raise ValueError("Movie features contain non-finite values.")
    return matrix


def _as_weights(weights, feature_count):
    vector = np.asarray(weights, dtype=float)
    if feature_count == 0 or vector.ndim != 1 or vector.shape[0] != feature_count:
        raise ValueError(
            f"Expected {feature_count} model weights, received shape {vector.shape}."
        )
    if not np.isfinite(vector).all():
        raise ValueError("Model weights contain non-finite values.")
    return vector


def _as_feature_vector(features, feature_count, label):
    vector = np.asarray(features, dtype=float)
    if vector.ndim != 1 or vector.shape[0] != feature_count:
        raise ValueError(
            f"{label} must contain exactly {feature_count} feature values."
        )
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} contains non-finite values.")
    return vector


def pairwise_probability(weights, first_movie, second_movie):
    """Return the probability that the first movie is preferred."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("Model weights must be one-dimensional.")
    weights = _as_weights(weights, weights.shape[0])

    first_movie = _as_feature_vector(
        first_movie, weights.shape[0], "First movie"
    )
    second_movie = _as_feature_vector(
        second_movie, weights.shape[0], "Second movie"
    )
    margin = weights @ (first_movie - second_movie)
    return float(expit(margin))


def pairwise_log_probability(weights, winner, loser):
    """Return the stable log probability that ``winner`` beats ``loser``."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("Model weights must be one-dimensional.")
    weights = _as_weights(weights, weights.shape[0])

    winner = _as_feature_vector(winner, weights.shape[0], "Winner")
    loser = _as_feature_vector(loser, weights.shape[0], "Loser")
    margin = weights @ (winner - loser)
    return float(-np.logaddexp(0.0, -margin))


def ranking_log_probability(weights, ranked_movies):
    """Return the Plackett-Luce log probability of a best-to-worst ranking."""
    ranked_movies = _as_feature_matrix(ranked_movies)
    weights = _as_weights(weights, ranked_movies.shape[1])
    utilities = ranked_movies @ weights

    log_probability = 0.0
    for position in range(len(utilities) - 1):
        remaining_utilities = utilities[position:]
        log_probability += utilities[position] - logsumexp(remaining_utilities)

    return float(log_probability)


def ranking_probability(weights, ranked_movies):
    """Return the probability of a complete best-to-worst ranking."""
    return float(np.exp(ranking_log_probability(weights, ranked_movies)))


def _validate_item_index(item_index, item_count, label):
    if isinstance(item_index, bool) or not isinstance(item_index, (int, np.integer)):
        raise ValueError(f"{label} must be an integer movie index.")
    if item_index < 0 or item_index >= item_count:
        raise ValueError(f"{label} index {item_index} is outside the movie catalogue.")
    return int(item_index)


def _validate_observations(item_count, pairwise_comparisons, rankings):
    checked_comparisons = []
    for comparison_number, comparison in enumerate(pairwise_comparisons, start=1):
        if len(comparison) != 2:
            raise ValueError(
                f"Pairwise comparison {comparison_number} must contain winner and loser."
            )
        winner = _validate_item_index(
            comparison[0], item_count, f"Comparison {comparison_number} winner"
        )
        loser = _validate_item_index(
            comparison[1], item_count, f"Comparison {comparison_number} loser"
        )
        if winner == loser:
            raise ValueError("A movie cannot be compared with itself.")
        checked_comparisons.append((winner, loser))

    checked_rankings = []
    for ranking_number, ranking in enumerate(rankings, start=1):
        if len(ranking) < 2:
            raise ValueError(
                f"Ranking {ranking_number} must contain at least two movies."
            )
        checked_ranking = tuple(
            _validate_item_index(
                item_index,
                item_count,
                f"Ranking {ranking_number} position {position}",
            )
            for position, item_index in enumerate(ranking, start=1)
        )
        if len(set(checked_ranking)) != len(checked_ranking):
            raise ValueError(f"Ranking {ranking_number} contains a repeated movie.")
        checked_rankings.append(checked_ranking)

    if not checked_comparisons and not checked_rankings:
        raise ValueError("At least one pairwise comparison or ranking is required.")

    return tuple(checked_comparisons), tuple(checked_rankings)


def _objective_and_gradient(
    weights,
    movie_features,
    pairwise_comparisons,
    rankings,
    regularization,
):
    """Return negative penalized log-likelihood and its gradient."""
    log_likelihood = 0.0
    log_likelihood_gradient = np.zeros_like(weights)

    for winner_index, loser_index in pairwise_comparisons:
        feature_difference = (
            movie_features[winner_index] - movie_features[loser_index]
        )
        margin = weights @ feature_difference
        log_likelihood -= np.logaddexp(0.0, -margin)
        log_likelihood_gradient += expit(-margin) * feature_difference

    for ranking in rankings:
        ranked_features = movie_features[np.asarray(ranking, dtype=int)]
        utilities = ranked_features @ weights

        for position in range(len(ranking) - 1):
            remaining_features = ranked_features[position:]
            remaining_utilities = utilities[position:]
            log_denominator = logsumexp(remaining_utilities)
            choice_probabilities = np.exp(
                remaining_utilities - log_denominator
            )

            log_likelihood += utilities[position] - log_denominator
            expected_features = choice_probabilities @ remaining_features
            log_likelihood_gradient += (
                ranked_features[position] - expected_features
            )

    penalty = 0.5 * regularization * (weights @ weights)
    objective = -log_likelihood + penalty
    gradient = -log_likelihood_gradient + regularization * weights
    return float(objective), gradient


@dataclass
class PreferenceModel:
    """Fitted linear movie-preference model."""

    weights: np.ndarray
    objective_value: float
    iterations: int

    def utility(self, movie_features):
        features = np.asarray(movie_features, dtype=float)
        if features.ndim not in (1, 2):
            raise ValueError("Movie features must be one- or two-dimensional.")
        if features.shape[-1] != len(self.weights):
            raise ValueError(
                f"Expected {len(self.weights)} features per movie."
            )
        if not np.isfinite(features).all():
            raise ValueError("Movie features contain non-finite values.")
        return features @ self.weights

    def probability_first_preferred(self, first_movie, second_movie):
        return pairwise_probability(self.weights, first_movie, second_movie)

    def ranking_log_probability(self, ranked_movies):
        return ranking_log_probability(self.weights, ranked_movies)


def fit_preference_model(
    movie_features,
    pairwise_comparisons=None,
    rankings=None,
    regularization=1.0,
    initial_weights=None,
):
    """Fit one preference vector to pairwise choices, rankings, or both.

    Pairwise comparisons are ``(winner_index, loser_index)`` tuples. Rankings
    are movie-index sequences ordered from most to least preferred.
    """
    movie_features = _as_feature_matrix(movie_features)
    pairwise_comparisons = (
        [] if pairwise_comparisons is None else pairwise_comparisons
    )
    rankings = [] if rankings is None else rankings

    pairwise_comparisons, rankings = _validate_observations(
        len(movie_features), pairwise_comparisons, rankings
    )

    if (
        isinstance(regularization, (bool, np.bool_))
        or not np.isscalar(regularization)
        or not np.isfinite(regularization)
    ):
        raise ValueError("Regularization must be a finite positive number.")
    regularization = float(regularization)
    if regularization <= 0:
        raise ValueError("Regularization must be a finite positive number.")

    feature_count = movie_features.shape[1]
    if initial_weights is None:
        starting_weights = np.zeros(feature_count, dtype=float)
    else:
        starting_weights = _as_weights(initial_weights, feature_count).copy()

    result = minimize(
        _objective_and_gradient,
        starting_weights,
        args=(
            movie_features,
            pairwise_comparisons,
            rankings,
            regularization,
        ),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500},
    )

    if (
        not result.success
        or not np.isfinite(result.x).all()
        or not np.isfinite(result.fun)
    ):
        raise RuntimeError(f"Preference model fitting failed: {result.message}")

    return PreferenceModel(
        weights=np.asarray(result.x, dtype=float),
        objective_value=float(result.fun),
        iterations=int(result.nit),
    )
