"""Fit the two study models and score their held-out predictions."""

import numpy as np

from .movie_data import prepare_movie_data
from .preference_model import fit_preference_model


MODEL_REGULARIZATION = 1.0


def _prepare_catalogue(movies, features):
    if len(movies) != len(features):
        raise ValueError("Movie metadata and feature rows must have equal length.")
    if "movie_id" not in movies.columns:
        raise ValueError("Movie metadata must contain movie IDs.")

    movie_ids = movies["movie_id"].tolist()
    if len(set(movie_ids)) != len(movie_ids):
        raise ValueError("Movie IDs must be unique when fitting study models.")

    feature_matrix = np.asarray(features, dtype=float)
    if feature_matrix.ndim != 2 or not np.isfinite(feature_matrix).all():
        raise ValueError("Study model features must be a finite matrix.")

    return feature_matrix, {
        movie_id: index for index, movie_id in enumerate(movie_ids)
    }


def _movie_index(movie_id, movie_indices):
    try:
        return movie_indices[movie_id]
    except KeyError as error:
        raise ValueError(
            f"Study movie {movie_id!r} is missing from the feature matrix."
        ) from error


def _pairwise_observations(study_state, movie_indices):
    tasks = study_state["pairwise_tasks"]
    responses = study_state["responses"]["pairwise"]
    if len(responses) != len(tasks):
        raise ValueError("Every pairwise task must be answered before model fitting.")

    comparisons = []
    for task_index, (task, response) in enumerate(zip(tasks, responses)):
        if response.get("task_index") != task_index:
            raise ValueError("Pairwise responses are not in task order.")
        if [response.get("left_movie_id"), response.get("right_movie_id")] != task:
            raise ValueError("A pairwise response does not match its assigned task.")

        chosen_movie_id = response.get("chosen_movie_id")
        if chosen_movie_id not in task:
            raise ValueError("A pairwise response chose an unassigned movie.")
        loser_movie_id = task[1] if chosen_movie_id == task[0] else task[0]
        comparisons.append(
            (
                _movie_index(chosen_movie_id, movie_indices),
                _movie_index(loser_movie_id, movie_indices),
            )
        )
    return comparisons


def _ranking_observations(study_state, movie_indices):
    tasks = study_state["ranking_tasks"]
    responses = study_state["responses"]["ranking"]
    if len(responses) != len(tasks):
        raise ValueError("Every ranking task must be answered before model fitting.")

    rankings = []
    for task_index, (task, response) in enumerate(zip(tasks, responses)):
        ranked_movie_ids = response.get("ranked_movie_ids", [])
        if response.get("task_index") != task_index:
            raise ValueError("Ranking responses are not in task order.")
        if response.get("presented_movie_ids") != task:
            raise ValueError("A ranking response does not match its assigned task.")
        if (
            len(ranked_movie_ids) != len(task)
            or len(set(ranked_movie_ids)) != len(task)
            or set(ranked_movie_ids) != set(task)
        ):
            raise ValueError("A ranking response is not a complete task ordering.")

        rankings.append(
            tuple(
                _movie_index(movie_id, movie_indices)
                for movie_id in ranked_movie_ids
            )
        )
    return rankings


def fit_study_models(
    study_state,
    movies=None,
    features=None,
    regularization=MODEL_REGULARIZATION,
):
    """Fit separate models and predict every held-out validation pair."""
    if (movies is None) != (features is None):
        raise ValueError("Provide both movies and features, or provide neither.")
    if movies is None:
        movies, features = prepare_movie_data()

    feature_matrix, movie_indices = _prepare_catalogue(movies, features)
    comparisons = _pairwise_observations(study_state, movie_indices)
    rankings = _ranking_observations(study_state, movie_indices)

    pairwise_model = fit_preference_model(
        feature_matrix,
        pairwise_comparisons=comparisons,
        regularization=regularization,
    )
    ranking_model = fit_preference_model(
        feature_matrix,
        rankings=rankings,
        regularization=regularization,
    )

    predictions = []
    for task_index, task in enumerate(study_state["validation_tasks"]):
        if len(task) != 2 or task[0] == task[1]:
            raise ValueError("Every validation task must contain two different movies.")
        left_movie_id, right_movie_id = task
        left_features = feature_matrix[_movie_index(left_movie_id, movie_indices)]
        right_features = feature_matrix[_movie_index(right_movie_id, movie_indices)]

        pairwise_probability_left = pairwise_model.probability_first_preferred(
            left_features,
            right_features,
        )
        ranking_probability_left = ranking_model.probability_first_preferred(
            left_features,
            right_features,
        )
        predictions.append(
            {
                "task_index": task_index,
                "left_movie_id": left_movie_id,
                "right_movie_id": right_movie_id,
                "pairwise_probability_left": pairwise_probability_left,
                "pairwise_predicted_movie_id": (
                    left_movie_id
                    if pairwise_probability_left >= 0.5
                    else right_movie_id
                ),
                "ranking_probability_left": ranking_probability_left,
                "ranking_predicted_movie_id": (
                    left_movie_id
                    if ranking_probability_left >= 0.5
                    else right_movie_id
                ),
            }
        )

    return {
        "regularization": float(regularization),
        "feature_count": int(feature_matrix.shape[1]),
        "pairwise_model": {
            "weights": pairwise_model.weights.tolist(),
            "objective_value": pairwise_model.objective_value,
            "iterations": pairwise_model.iterations,
        },
        "ranking_model": {
            "weights": ranking_model.weights.tolist(),
            "objective_value": ranking_model.objective_value,
            "iterations": ranking_model.iterations,
        },
        "validation_predictions": predictions,
    }


def summarize_validation(model_results, validation_responses):
    """Compare both models with a participant's blind validation choices."""
    predictions = model_results.get("validation_predictions", [])
    if len(validation_responses) != len(predictions) or not predictions:
        raise ValueError("All validation tasks must be answered before scoring.")

    pairwise_matches = 0
    ranking_matches = 0
    model_agreements = 0
    for task_index, (prediction, response) in enumerate(
        zip(predictions, validation_responses)
    ):
        if (
            prediction.get("task_index") != task_index
            or response.get("task_index") != task_index
        ):
            raise ValueError("Validation responses are not in task order.")

        assigned_movies = {
            prediction["left_movie_id"],
            prediction["right_movie_id"],
        }
        if [response.get("left_movie_id"), response.get("right_movie_id")] != [
            prediction["left_movie_id"],
            prediction["right_movie_id"],
        ]:
            raise ValueError("A validation response does not match its assigned task.")
        chosen_movie_id = response.get("chosen_movie_id")
        if chosen_movie_id not in assigned_movies:
            raise ValueError("A validation response chose an unassigned movie.")

        pairwise_prediction = prediction["pairwise_predicted_movie_id"]
        ranking_prediction = prediction["ranking_predicted_movie_id"]
        pairwise_matches += int(chosen_movie_id == pairwise_prediction)
        ranking_matches += int(chosen_movie_id == ranking_prediction)
        model_agreements += int(pairwise_prediction == ranking_prediction)

    task_count = len(predictions)
    return {
        "task_count": task_count,
        "pairwise_matches": pairwise_matches,
        "ranking_matches": ranking_matches,
        "model_agreements": model_agreements,
        "pairwise_accuracy": pairwise_matches / task_count,
        "ranking_accuracy": ranking_matches / task_count,
        "model_agreement_rate": model_agreements / task_count,
    }
