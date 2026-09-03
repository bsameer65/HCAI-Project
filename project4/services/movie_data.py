"""Load the IMDb movie catalogue and create content-based movie features."""

from pathlib import Path
import re

import numpy as np
import pandas as pd


DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "movie_metadata.csv"

REQUIRED_COLUMNS = {
    "movie_title",
    "movie_imdb_link",
    "genres",
    "duration",
    "title_year",
    "language",
    "country",
    "content_rating",
    "director_name",
    "actor_1_name",
    "actor_2_name",
    "actor_3_name",
}

GENRE_CATEGORIES = (
    "Action",
    "Adventure",
    "Animation",
    "Biography",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "Film-Noir",
    "Game-Show",
    "History",
    "Horror",
    "Music",
    "Musical",
    "Mystery",
    "News",
    "Reality-TV",
    "Romance",
    "Sci-Fi",
    "Short",
    "Sport",
    "Thriller",
    "War",
    "Western",
)

ERA_CATEGORIES = (
    "Pre-1960",
    "1960s",
    "1970s",
    "1980s",
    "1990s",
    "2000s",
    "2010s",
    "2020s+",
    "Unknown",
)

LANGUAGE_CATEGORIES = (
    "English",
    "French",
    "Spanish",
    "Hindi",
    "Mandarin",
    "Other",
    "Unknown",
)

CONTENT_RATING_CATEGORIES = (
    "G",
    "PG",
    "PG-13",
    "R",
    "Not Rated",
    "Unrated",
    "Approved",
    "Other",
    "Unknown",
)


def _clean_text(series):
    """Normalize whitespace while keeping missing values as pandas NA."""
    return (
        series.astype("string")
        .str.replace("\u00a0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _filled_text(series, fallback="Unknown"):
    cleaned = _clean_text(series)
    return cleaned.mask(cleaned.eq("") | cleaned.isna(), fallback)


def _feature_name(prefix, value):
    safe_value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return f"{prefix}_{safe_value}"


def load_movie_catalog(csv_path=DATASET_PATH):
    """Load, validate, clean, and deduplicate the IMDb movie metadata."""
    raw_movies = pd.read_csv(
        csv_path,
        usecols=lambda column: column in REQUIRED_COLUMNS,
    )

    missing_columns = sorted(REQUIRED_COLUMNS.difference(raw_movies.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"IMDb dataset is missing required columns: {missing_text}")

    movie_ids = raw_movies["movie_imdb_link"].astype("string").str.extract(
        r"(tt\d+)", expand=False
    )
    if movie_ids.isna().any():
        invalid_count = int(movie_ids.isna().sum())
        raise ValueError(
            f"Could not extract an IMDb ID from {invalid_count} movie links."
        )

    titles = _clean_text(raw_movies["movie_title"])
    if titles.isna().any() or titles.eq("").any():
        raise ValueError("IMDb dataset contains movies without a usable title.")

    genres = (
        _clean_text(raw_movies["genres"])
        .str.replace(r"\s*\|\s*", "|", regex=True)
    )
    if genres.isna().any() or genres.eq("").any():
        raise ValueError("IMDb dataset contains movies without genre information.")

    catalog = pd.DataFrame(
        {
            "movie_id": movie_ids,
            "title": titles,
            "year": pd.to_numeric(raw_movies["title_year"], errors="coerce")
            .round()
            .astype("Int64"),
            "genres": genres,
            "duration": pd.to_numeric(raw_movies["duration"], errors="coerce"),
            "language": _filled_text(raw_movies["language"]),
            "country": _filled_text(raw_movies["country"]),
            "content_rating": _filled_text(raw_movies["content_rating"]),
            "director": _filled_text(raw_movies["director_name"]),
            "actor_1": _filled_text(raw_movies["actor_1_name"]),
            "actor_2": _filled_text(raw_movies["actor_2_name"]),
            "actor_3": _filled_text(raw_movies["actor_3_name"]),
            "imdb_url": _filled_text(raw_movies["movie_imdb_link"]),
        }
    )

    # Use the first source row as a clear, deterministic rule for repeated IDs.
    # The feature fields agree across the duplicate groups in this dataset.
    catalog = catalog.drop_duplicates(subset="movie_id", keep="first")
    return catalog.reset_index(drop=True)


def _release_era(year):
    if pd.isna(year):
        return "Unknown"

    year = int(year)
    if year < 1960:
        return "Pre-1960"
    if year < 1970:
        return "1960s"
    if year < 1980:
        return "1970s"
    if year < 1990:
        return "1980s"
    if year < 2000:
        return "1990s"
    if year < 2010:
        return "2000s"
    if year < 2020:
        return "2010s"
    return "2020s+"


def _language_group(language):
    if pd.isna(language) or str(language).strip().casefold() == "unknown":
        return "Unknown"

    common_languages = {
        language_name.casefold(): language_name
        for language_name in LANGUAGE_CATEGORIES[:-2]
    }
    return common_languages.get(str(language).strip().casefold(), "Other")


def _content_rating_group(content_rating):
    if pd.isna(content_rating) or str(content_rating).strip().casefold() == "unknown":
        return "Unknown"

    known_ratings = {
        rating.casefold(): rating for rating in CONTENT_RATING_CATEGORIES[:-2]
    }
    return known_ratings.get(str(content_rating).strip().casefold(), "Other")


def _one_hot(values, categories, prefix):
    columns = {
        _feature_name(prefix, category): values.eq(category).astype(float)
        for category in categories
    }
    return pd.DataFrame(columns, index=values.index)


def extract_movie_features(movies, reference_movies=None):
    """Convert cleaned movies into a stable numerical feature matrix.

    ``reference_movies`` supplies the duration scaling values. Pass the full
    catalogue when transforming a subset so that a movie keeps the same vector
    wherever it appears in the study. When omitted, ``movies`` is used.
    """
    required_columns = {"genres", "year", "duration", "language", "content_rating"}
    missing_columns = sorted(required_columns.difference(movies.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Movie catalogue is missing columns: {missing_text}")
    if movies.empty:
        raise ValueError("Cannot extract features from an empty movie catalogue.")

    genre_lists = movies["genres"].str.split("|")
    observed_genres = {
        genre.strip()
        for movie_genres in genre_lists
        for genre in movie_genres
        if genre.strip()
    }
    unsupported_genres = sorted(observed_genres.difference(GENRE_CATEGORIES))
    if unsupported_genres:
        unsupported_text = ", ".join(unsupported_genres)
        raise ValueError(f"Movie catalogue contains unsupported genres: {unsupported_text}")

    genre_features = pd.DataFrame(
        {
            _feature_name("genre", genre): genre_lists.map(
                lambda movie_genres, genre=genre: float(genre in movie_genres)
            )
            for genre in GENRE_CATEGORIES
        },
        index=movies.index,
    )

    eras = movies["year"].map(_release_era)
    era_features = _one_hot(eras, ERA_CATEGORIES, "era")

    language_groups = movies["language"].map(_language_group)
    language_features = _one_hot(
        language_groups, LANGUAGE_CATEGORIES, "language"
    )

    content_rating_groups = movies["content_rating"].map(_content_rating_group)
    rating_features = _one_hot(
        content_rating_groups, CONTENT_RATING_CATEGORIES, "content_rating"
    )

    durations = pd.to_numeric(movies["duration"], errors="coerce")
    duration_missing = durations.isna()
    duration_reference = movies if reference_movies is None else reference_movies
    if "duration" not in duration_reference.columns:
        raise ValueError("Duration reference is missing the duration column.")

    reference_durations = pd.to_numeric(
        duration_reference["duration"], errors="coerce"
    )
    if reference_durations.isna().all():
        raise ValueError("Cannot extract duration features when every value is missing.")

    duration_median = reference_durations.median()
    filled_reference = reference_durations.fillna(duration_median)
    lower_limit = filled_reference.quantile(0.01)
    upper_limit = filled_reference.quantile(0.99)
    clipped_reference = filled_reference.clip(
        lower=lower_limit, upper=upper_limit
    )
    duration_mean = clipped_reference.mean()
    duration_scale = clipped_reference.std(ddof=0)

    filled_durations = durations.fillna(duration_median)
    clipped_durations = filled_durations.clip(lower=lower_limit, upper=upper_limit)

    if duration_scale == 0:
        standardized_duration = pd.Series(0.0, index=movies.index)
    else:
        standardized_duration = (clipped_durations - duration_mean) / duration_scale

    duration_features = pd.DataFrame(
        {
            "duration_standardized": standardized_duration.astype(float),
            "duration_missing": duration_missing.astype(float),
        },
        index=movies.index,
    )

    feature_matrix = pd.concat(
        [
            genre_features,
            era_features,
            language_features,
            rating_features,
            duration_features,
        ],
        axis=1,
    )

    if not np.isfinite(feature_matrix.to_numpy(dtype=float)).all():
        raise ValueError("Movie feature matrix contains non-finite values.")

    return feature_matrix


def prepare_movie_data(csv_path=DATASET_PATH):
    """Return matching cleaned movie metadata and feature rows."""
    movies = load_movie_catalog(csv_path)
    features = extract_movie_features(movies)
    return movies, features
