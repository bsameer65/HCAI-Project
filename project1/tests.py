import tempfile
from pathlib import Path

import joblib
import pandas as pd

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from .forms import RegressionDatasetSetupForm, RegressionTrainingForm
from .services.algorithms import REGRESSOR_CHOICES, create_regressor
from .services.dataset import (
    DatasetValidationError,
    get_regression_target_choices,
    inspect_dataset,
    prepare_regression_dataset,
)
from .services.evaluation import evaluate_regressor
from .services.persistence import clear_regression_artifacts


class RegressionDatasetTests(SimpleTestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_directory.cleanup()

    def write_csv(self, content, name="dataset.csv"):
        path = Path(self.temp_directory.name) / name
        path.write_text(content, encoding="utf-8")
        return str(path)

    def test_numeric_only_dataset(self):
        path = self.write_csv("size,rooms,price\n80,2,200000\n100,3,260000\n")
        prepared = prepare_regression_dataset(path, "price")
        self.assertEqual(prepared["numeric_features"], ["size", "rooms"])
        self.assertEqual(prepared["categorical_features"], [])

    def test_mixed_features_are_detected(self):
        path = self.write_csv("size,city,price\n80,Hamburg,200000\n100,Berlin,260000\n")
        prepared = prepare_regression_dataset(path, "price")
        self.assertEqual(prepared["categorical_features"], ["city"])

    def test_identifier_name_is_removed_but_unique_feature_is_retained(self):
        path = self.write_csv("Id,measurement,price\n1,10,100\n2,20,200\n")
        df, _, removed = inspect_dataset(path)
        self.assertEqual(removed, ["Id"])
        self.assertIn("measurement", df.columns)

    def test_no_numeric_target_choice_is_rejected(self):
        path = self.write_csv("city,condition\nHamburg,new\nBerlin,used\n")
        with self.assertRaises(DatasetValidationError):
            get_regression_target_choices(path)

    def test_missing_or_constant_target_is_rejected(self):
        missing = self.write_csv("size,price\n80,100\n90,\n", "missing.csv")
        constant = self.write_csv("size,price\n80,100\n90,100\n", "constant.csv")
        with self.assertRaises(DatasetValidationError):
            prepare_regression_dataset(missing, "price")
        with self.assertRaises(DatasetValidationError):
            prepare_regression_dataset(constant, "price")

    def test_invalid_encoding_and_duplicate_headers_are_rejected(self):
        invalid = Path(self.temp_directory.name) / "invalid.csv"
        invalid.write_bytes(b"\xff\xfe\x00")
        duplicate = self.write_csv("size,size,price\n1,2,3\n4,5,6\n", "duplicate.csv")
        with self.assertRaises(DatasetValidationError):
            inspect_dataset(str(invalid))
        with self.assertRaises(DatasetValidationError):
            inspect_dataset(duplicate)

    def test_target_must_exist_and_be_numeric(self):
        path = self.write_csv("size,city,price\n80,Hamburg,100\n90,Berlin,120\n")
        with self.assertRaises(DatasetValidationError):
            prepare_regression_dataset(path, "unknown")
        with self.assertRaises(DatasetValidationError):
            prepare_regression_dataset(path, "city")


class RegressionFormsAndServicesTests(SimpleTestCase):
    def test_setup_form_rejects_non_choice_target(self):
        form = RegressionDatasetSetupForm(
            {"target_column": "city"}, target_choices=["price"]
        )
        self.assertFalse(form.is_valid())

    def test_training_form_supports_splits_metrics_and_bad_parameters(self):
        for split in ("0.2", "0.3", "0.4"):
            for metric in ("mae", "mse", "rmse", "r2"):
                form = RegressionTrainingForm({
                    "model": "linear_regression", "test_size": split, "metric": metric,
                })
                self.assertTrue(form.is_valid(), form.errors)
        invalid = RegressionTrainingForm({
            "model": "knn_regressor", "test_size": "0.2", "metric": "mae",
            "knn_neighbors": 0, "knn_weights": "uniform", "knn_metric": "euclidean",
        })
        self.assertFalse(invalid.is_valid())

    def test_all_four_regressors_support_mixed_data_and_unknown_category(self):
        X = pd.DataFrame({
            "size": range(80, 100), "city": ["Hamburg", "Berlin"] * 10,
        })
        y = pd.Series([value * 2 for value in range(80, 100)])
        parameters = {
            "n_estimators": 10, "max_depth": 4, "min_samples_split": 2,
            "n_neighbors": 3, "weights": "uniform", "metric": "euclidean",
        }
        for model_key in REGRESSOR_CHOICES:
            model = create_regressor(model_key, ["size"], ["city"], parameters)
            model.fit(X, y)
            prediction = model.predict(pd.DataFrame([{"size": 101, "city": "Munich"}]))
            self.assertEqual(len(prediction), 1)

    def test_regression_metrics_include_all_supported_values(self):
        metrics = evaluate_regressor([1, 2, 3], [1, 2, 4])
        self.assertEqual(set(metrics), {"mae", "mse", "rmse", "r2"})
        self.assertAlmostEqual(metrics["rmse"], metrics["mse"] ** 0.5)

    def test_clearing_regression_artifacts_preserves_classification(self):
        with tempfile.TemporaryDirectory() as media_root:
            model_directory = Path(media_root) / "models"
            model_directory.mkdir()
            classification = model_directory / "classification_model.pkl"
            regression = model_directory / "regression_model.pkl"
            joblib.dump({"classification": True}, classification)
            joblib.dump({"regression": True}, regression)
            clear_regression_artifacts(media_root)
            self.assertTrue(classification.exists())
            self.assertFalse(regression.exists())


class RegressionWorkflowTests(TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.settings_override = override_settings(MEDIA_ROOT=self.temp_directory.name)
        self.settings_override.enable()
        self.client = Client()

    def tearDown(self):
        self.settings_override.disable()
        self.temp_directory.cleanup()

    def dataset_bytes(self):
        rows = ["Id,size,rooms,city,price"]
        for index in range(1, 41):
            city = "Hamburg" if index % 2 else "Berlin"
            rows.append(f"{index},{80 + index},{1 + index % 4},{city},{150000 + index * 7500}")
        return ("\n".join(rows) + "\n").encode()

    def upload(self, content=None):
        return self.client.post(
            reverse("project1:regression"),
            {"csv_file": SimpleUploadedFile(
                "homes.csv", content or self.dataset_bytes(), content_type="text/csv"
            )},
        )

    def configure(self, target="price"):
        self.assertRedirects(self.upload(), reverse("project1:regression_setup"))
        response = self.client.post(
            reverse("project1:regression_setup"), {"target_column": target}
        )
        self.assertRedirects(response, reverse("project1:regression_analyze"))

    def training_data(self, model="linear_regression", split="0.2", metric="mae"):
        data = {"model": model, "test_size": split, "metric": metric}
        if model == "decision_tree_regressor":
            data.update(tree_max_depth=5, tree_min_samples_split=2)
        elif model == "random_forest_regressor":
            data.update(rf_n_estimators=10, rf_max_depth=5, rf_min_samples_split=2)
        elif model == "knn_regressor":
            data.update(knn_neighbors=3, knn_weights="uniform", knn_metric="euclidean")
        return data

    def train(self, model="linear_regression", split="0.2", metric="mae"):
        response = self.client.post(
            reverse("project1:regression_train"),
            self.training_data(model, split, metric),
        )
        self.assertRedirects(response, reverse("project1:regression_result"))

    def test_setup_suggestion_and_user_target_override(self):
        self.assertEqual(self.upload().status_code, 302)
        setup = self.client.get(reverse("project1:regression_setup"))
        self.assertEqual(setup.context["suggested_target"], "price")
        changed = self.client.post(
            reverse("project1:regression_setup"), {"target_column": "size"}
        )
        self.assertRedirects(changed, reverse("project1:regression_analyze"))
        self.assertEqual(self.client.session["regression_target"], "size")

    def test_last_numeric_is_suggested_when_last_column_is_categorical(self):
        content = b"size,price,city\n80,100,Hamburg\n90,120,Berlin\n"
        self.upload(content)
        response = self.client.get(reverse("project1:regression_setup"))
        self.assertEqual(response.context["suggested_target"], "price")

    def test_analysis_supports_all_visualization_modes(self):
        self.configure()
        for mode in ("feature_target", "feature_feature", "target_distribution"):
            response = self.client.post(
                reverse("project1:regression_analyze"),
                {"plot_type": mode, "x_col": "size", "y_col": "rooms"},
            )
            self.assertEqual(response.status_code, 200)

    def test_all_models_train_across_supported_splits_and_metrics(self):
        self.configure()
        combinations = [
            ("linear_regression", "0.2", "mae"),
            ("decision_tree_regressor", "0.3", "mse"),
            ("random_forest_regressor", "0.4", "rmse"),
            ("knn_regressor", "0.2", "r2"),
        ]
        for model, split, metric in combinations:
            self.train(model, split, metric)
            result = self.client.get(reverse("project1:regression_result"))
            self.assertEqual(result.status_code, 200)

    def test_prediction_accepts_unknown_category_through_saved_pipeline(self):
        self.configure()
        self.train()
        response = self.client.post(
            reverse("project1:regression_test"),
            {"size": "125", "rooms": "3", "city": "Munich"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["has_prediction"])
        self.assertContains(response, "Typical test error")

    def test_navigation_does_not_retrain(self):
        self.configure()
        self.train()
        model_path = Path(self.temp_directory.name) / "models" / "regression_model.pkl"
        original_modified = model_path.stat().st_mtime_ns
        for route_name in (
            "regression_result", "regression_test", "regression_explain", "regression_compare",
        ):
            response = self.client.get(reverse(f"project1:{route_name}"))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(model_path.stat().st_mtime_ns, original_modified)

    def test_new_upload_clears_only_regression_artifacts(self):
        self.configure()
        self.train()
        model_directory = Path(self.temp_directory.name) / "models"
        classification = model_directory / "classification_model.pkl"
        joblib.dump({"classification": True}, classification)
        self.assertEqual(self.upload().status_code, 302)
        self.assertTrue(classification.exists())
        self.assertFalse((model_directory / "regression_model.pkl").exists())

    def test_classification_upload_training_and_comparison_remain_available(self):
        rows = ["feature_one,feature_two,target"]
        for index in range(1, 41):
            rows.append(f"{index},{index * 2},{index % 2}")
        response = self.client.post(
            reverse("project1:classification"),
            {"csv_file": SimpleUploadedFile(
                "classes.csv", ("\n".join(rows) + "\n").encode(),
                content_type="text/csv",
            )},
        )
        self.assertRedirects(response, reverse("project1:classification_analyze"))
        trained = self.client.post(reverse("project1:classification_train"), {
            "model": "decision_tree", "test_size": "0.2", "metric": "accuracy",
            "tree_max_depth": 5, "tree_min_samples_split": 2,
        })
        self.assertEqual(trained.status_code, 200)
        self.assertTrue(trained.context["training_complete"])
        compared = self.client.get(reverse("project1:classification_compare"))
        self.assertEqual(compared.status_code, 200)
        self.assertEqual(len(compared.context["results"]), 4)
