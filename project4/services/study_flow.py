"""Create reproducible, counterbalanced tasks for the preference study."""

from collections import Counter
from itertools import combinations
import random


PAIRWISE_TASK_COUNT = 27
RANKING_TASK_COUNT = 3
RANKING_SIZE = 10
VALIDATION_TASK_COUNT = 20
ELICITATION_POOL_SIZE = 30
VALIDATION_POOL_SIZE = VALIDATION_TASK_COUNT * 2
REQUIRED_MOVIE_COUNT = ELICITATION_POOL_SIZE * 2 + VALIDATION_POOL_SIZE

CONDITION_LABELS = {
    "pairwise": "Pairwise choices",
    "ranking": "Ten-movie rankings",
}

# Combining order and movie-pool assignment gives four counterbalancing cells.
COUNTERBALANCE_CELLS = {
    1: {
        "condition_order": ("pairwise", "ranking"),
        "pairwise_pool": "A",
        "ranking_pool": "B",
    },
    2: {
        "condition_order": ("ranking", "pairwise"),
        "pairwise_pool": "A",
        "ranking_pool": "B",
    },
    3: {
        "condition_order": ("pairwise", "ranking"),
        "pairwise_pool": "B",
        "ranking_pool": "A",
    },
    4: {
        "condition_order": ("ranking", "pairwise"),
        "pairwise_pool": "B",
        "ranking_pool": "A",
    },
}


def _randomize_pair_order(pairs, random_generator):
    randomized_pairs = []
    for first_movie, second_movie in pairs:
        if random_generator.random() < 0.5:
            first_movie, second_movie = second_movie, first_movie
        randomized_pairs.append([first_movie, second_movie])
    return randomized_pairs


def _build_pairwise_tasks(movie_pool, random_generator):
    """Build distinct pairs while keeping movie exposure approximately equal."""
    shuffled_movies = list(movie_pool)
    random_generator.shuffle(shuffled_movies)

    # The first 15 pairs expose every movie once.
    selected_pairs = [
        (shuffled_movies[index], shuffled_movies[index + 1])
        for index in range(0, ELICITATION_POOL_SIZE, 2)
    ]
    exposure_counts = Counter(shuffled_movies)
    used_pairs = {frozenset(pair) for pair in selected_pairs}

    candidate_pairs = [
        pair
        for pair in combinations(shuffled_movies, 2)
        if frozenset(pair) not in used_pairs
    ]
    random_generator.shuffle(candidate_pairs)

    while len(selected_pairs) < PAIRWISE_TASK_COUNT:
        # The shuffled candidate order breaks ties randomly. The count terms
        # keep repeated exposure balanced across the 30-movie pool.
        best_index = min(
            range(len(candidate_pairs)),
            key=lambda index: (
                max(
                    exposure_counts[candidate_pairs[index][0]],
                    exposure_counts[candidate_pairs[index][1]],
                ),
                exposure_counts[candidate_pairs[index][0]]
                + exposure_counts[candidate_pairs[index][1]],
            ),
        )
        pair = candidate_pairs.pop(best_index)
        selected_pairs.append(pair)
        exposure_counts.update(pair)

    random_generator.shuffle(selected_pairs)
    return _randomize_pair_order(selected_pairs, random_generator)


def _build_ranking_tasks(movie_pool, random_generator):
    shuffled_movies = list(movie_pool)
    random_generator.shuffle(shuffled_movies)
    return [
        shuffled_movies[start : start + RANKING_SIZE]
        for start in range(0, RANKING_TASK_COUNT * RANKING_SIZE, RANKING_SIZE)
    ]


def _build_validation_tasks(movie_pool, random_generator):
    shuffled_movies = list(movie_pool)
    random_generator.shuffle(shuffled_movies)
    pairs = [
        (shuffled_movies[index], shuffled_movies[index + 1])
        for index in range(0, VALIDATION_POOL_SIZE, 2)
    ]
    return _randomize_pair_order(pairs, random_generator)


def create_study_plan(movie_ids, seed, cell=None):
    """Return one JSON-serializable task plan for a participant.

    The same movie IDs, seed, and cell always produce the same plan. Sampling
    is uniform without replacement from the eligible movie catalogue.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Study seed must be an integer.")

    movie_ids = list(movie_ids)
    if any(not isinstance(movie_id, str) or not movie_id for movie_id in movie_ids):
        raise ValueError("Every movie must have a non-empty string ID.")
    if len(set(movie_ids)) != len(movie_ids):
        raise ValueError("Movie IDs must be unique before study sampling.")
    if len(movie_ids) < REQUIRED_MOVIE_COUNT:
        raise ValueError(
            f"At least {REQUIRED_MOVIE_COUNT} unique movies are required."
        )

    if cell is None:
        cell = seed % len(COUNTERBALANCE_CELLS) + 1
    if (
        isinstance(cell, bool)
        or not isinstance(cell, int)
        or cell not in COUNTERBALANCE_CELLS
    ):
        raise ValueError("Counterbalancing cell must be an integer from 1 to 4.")

    random_generator = random.Random(seed)
    selected_movies = random_generator.sample(movie_ids, REQUIRED_MOVIE_COUNT)
    pools = {
        "A": selected_movies[:ELICITATION_POOL_SIZE],
        "B": selected_movies[
            ELICITATION_POOL_SIZE : ELICITATION_POOL_SIZE * 2
        ],
    }
    validation_pool = selected_movies[ELICITATION_POOL_SIZE * 2 :]

    assignment = COUNTERBALANCE_CELLS[cell]
    pairwise_pool = pools[assignment["pairwise_pool"]]
    ranking_pool = pools[assignment["ranking_pool"]]

    return {
        "seed": seed,
        "counterbalance_cell": cell,
        "condition_order": list(assignment["condition_order"]),
        "pairwise_pool_name": assignment["pairwise_pool"],
        "ranking_pool_name": assignment["ranking_pool"],
        "pairwise_pool": list(pairwise_pool),
        "ranking_pool": list(ranking_pool),
        "validation_pool": list(validation_pool),
        "pairwise_tasks": _build_pairwise_tasks(
            pairwise_pool, random_generator
        ),
        "ranking_tasks": _build_ranking_tasks(
            ranking_pool, random_generator
        ),
        "validation_tasks": _build_validation_tasks(
            validation_pool, random_generator
        ),
    }
