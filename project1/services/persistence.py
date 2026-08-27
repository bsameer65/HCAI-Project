import os

import joblib


CLASSIFICATION_MODEL_FILENAME = (
    "classification_model.pkl"
)

CLASSIFICATION_METADATA_FILENAME = (
    "classification_metadata.pkl"
)

REGRESSION_ARTIFACT_FILENAMES = (
    "regression_model.pkl",
    "regression_metadata.pkl",
    "regression_results.pkl",
)

REGRESSION_MODEL_FILENAME = REGRESSION_ARTIFACT_FILENAMES[0]
REGRESSION_METADATA_FILENAME = REGRESSION_ARTIFACT_FILENAMES[1]
REGRESSION_RESULTS_FILENAME = REGRESSION_ARTIFACT_FILENAMES[2]


class ModelNotFoundError(FileNotFoundError):
    """Raised when no trained model is available."""
    pass


def _get_model_directory(media_root):
    """
    Return the directory used for persisted ML models.
    """

    model_directory = os.path.join(
        media_root,
        "models",
    )

    os.makedirs(
        model_directory,
        exist_ok=True,
    )

    return model_directory


def save_classification_model(
    model,
    metadata,
    media_root,
):
    """
    Save a trained classification model and its metadata.
    """

    model_directory = _get_model_directory(
        media_root
    )

    model_path = os.path.join(
        model_directory,
        CLASSIFICATION_MODEL_FILENAME,
    )

    metadata_path = os.path.join(
        model_directory,
        CLASSIFICATION_METADATA_FILENAME,
    )

    joblib.dump(
        model,
        model_path,
    )

    joblib.dump(
        metadata,
        metadata_path,
    )


def load_classification_model(
    media_root,
):
    """
    Load the most recently trained classification model
    and associated metadata.
    """

    model_directory = _get_model_directory(
        media_root
    )

    model_path = os.path.join(
        model_directory,
        CLASSIFICATION_MODEL_FILENAME,
    )

    metadata_path = os.path.join(
        model_directory,
        CLASSIFICATION_METADATA_FILENAME,
    )

    if (
        not os.path.exists(model_path)
        or not os.path.exists(metadata_path)
    ):
        raise ModelNotFoundError(
            "No trained classification model was found."
        )

    model = joblib.load(
        model_path
    )

    metadata = joblib.load(
        metadata_path
    )

    return model, metadata


def clear_regression_artifacts(media_root):
    """Remove only artifacts belonging to the previous regression run."""
    model_directory = _get_model_directory(media_root)
    for filename in REGRESSION_ARTIFACT_FILENAMES:
        path = os.path.join(model_directory, filename)
        if os.path.exists(path):
            os.remove(path)


def save_regression_model(model, metadata, media_root):
    model_directory = _get_model_directory(media_root)
    joblib.dump(model, os.path.join(model_directory, REGRESSION_MODEL_FILENAME))
    joblib.dump(metadata, os.path.join(model_directory, REGRESSION_METADATA_FILENAME))


def load_regression_model(media_root):
    model_directory = _get_model_directory(media_root)
    model_path = os.path.join(model_directory, REGRESSION_MODEL_FILENAME)
    metadata_path = os.path.join(model_directory, REGRESSION_METADATA_FILENAME)
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        raise ModelNotFoundError("No trained regression model was found.")
    return joblib.load(model_path), joblib.load(metadata_path)


def save_regression_results(results, media_root):
    model_directory = _get_model_directory(media_root)
    joblib.dump(results, os.path.join(model_directory, REGRESSION_RESULTS_FILENAME))


def load_regression_results(media_root):
    path = os.path.join(_get_model_directory(media_root), REGRESSION_RESULTS_FILENAME)
    if not os.path.exists(path):
        raise ModelNotFoundError("No regression training results were found.")
    return joblib.load(path)
