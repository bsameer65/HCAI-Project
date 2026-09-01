from unittest import TestCase, mock

import numpy as np
import pandas as pd

from project4.services.movie_data import (
    CONTENT_RATING_CATEGORIES,
    DATASET_PATH,
    ERA_CATEGORIES,
    GENRE_CATEGORIES,
    LANGUAGE_CATEGORIES,
    extract_movie_features,
    load_movie_catalog,
    prepare_movie_data,
)


class MovieCatalogTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.movies = load_movie_catalog()

    def test_source_dataset_exists(self):
        self.assertTrue(DATASET_PATH.exists())

    def test_catalog_is_cleaned_and_deduplicated(self):
        self.assertEqual(len(self.movies), 4919)
        self.assertTrue(self.movies["movie_id"].is_unique)
        self.assertFalse(self.movies["title"].str.contains("\u00a0").any())
        self.assertFalse(self.movies["title"].str.startswith(" ").any())
        self.assertFalse(self.movies["title"].str.endswith(" ").any())

    def test_avatar_record_has_expected_display_values(self):
        avatar = self.movies.loc[self.movies["movie_id"] == "tt0499549"].iloc[0]

        self.assertEqual(avatar["title"], "Avatar")
        self.assertEqual(avatar["year"], 2009)
        self.assertEqual(avatar["language"], "English")
        self.assertIn("Sci-Fi", avatar["genres"].split("|"))

    def test_catalog_does_not_expose_population_or_financial_fields(self):
        excluded_columns = {
            "imdb_score",
            "num_voted_users",
            "gross",
            "budget",
            "movie_facebook_likes",
        }

        self.assertTrue(excluded_columns.isdisjoint(self.movies.columns))

    def test_missing_required_columns_are_reported(self):
        incomplete_data = pd.DataFrame({"movie_title": ["Example Movie"]})

        with mock.patch(
            "project4.services.movie_data.pd.read_csv",
            return_value=incomplete_data,
        ):
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_movie_catalog("unused.csv")


class MovieFeatureTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.movies, cls.features = prepare_movie_data()

    def test_feature_rows_match_movie_rows(self):
        self.assertEqual(len(self.movies), len(self.features))
        self.assertEqual(self.movies.index.tolist(), self.features.index.tolist())

    def test_feature_matrix_is_finite(self):
        self.assertTrue(
            np.isfinite(self.features.to_numpy(dtype=float)).all()
        )

    def test_genre_features_cover_the_dataset_genres(self):
        genre_columns = [
            column for column in self.features if column.startswith("genre_")
        ]

        self.assertEqual(len(genre_columns), len(GENRE_CATEGORIES))
        self.assertTrue((self.features[genre_columns].sum(axis=1) >= 1).all())

    def test_feature_schema_and_values_stay_stable_for_a_subset(self):
        subset = self.movies.iloc[:10]
        subset_features = extract_movie_features(
            subset,
            reference_movies=self.movies,
        )

        self.assertEqual(
            tuple(subset_features.columns),
            tuple(self.features.columns),
        )
        np.testing.assert_allclose(
            subset_features.to_numpy(dtype=float),
            self.features.iloc[:10].to_numpy(dtype=float),
        )

    def test_each_grouped_category_has_one_active_feature(self):
        expected_groups = {
            "era_": len(ERA_CATEGORIES),
            "language_": len(LANGUAGE_CATEGORIES),
            "content_rating_": len(CONTENT_RATING_CATEGORIES),
        }

        for prefix, expected_count in expected_groups.items():
            columns = [
                column for column in self.features if column.startswith(prefix)
            ]
            self.assertEqual(len(columns), expected_count)
            self.assertTrue((self.features[columns].sum(axis=1) == 1).all())

    def test_duration_missingness_is_preserved(self):
        self.assertEqual(int(self.features["duration_missing"].sum()), 15)
        self.assertAlmostEqual(
            float(self.features["duration_standardized"].mean()),
            0.0,
            places=10,
        )

    def test_popularity_and_financial_fields_are_not_features(self):
        excluded_terms = (
            "imdb_score",
            "facebook",
            "gross",
            "budget",
            "votes",
            "reviews",
        )

        for feature_name in self.features.columns:
            self.assertFalse(
                any(term in feature_name for term in excluded_terms),
                feature_name,
            )

    def test_avatar_has_expected_feature_values(self):
        avatar_index = self.movies.index[
            self.movies["movie_id"] == "tt0499549"
        ][0]

        self.assertEqual(self.features.loc[avatar_index, "genre_action"], 1.0)
        self.assertEqual(self.features.loc[avatar_index, "era_2000s"], 1.0)
        self.assertEqual(self.features.loc[avatar_index, "language_english"], 1.0)
        self.assertEqual(
            self.features.loc[avatar_index, "content_rating_pg_13"], 1.0
        )
