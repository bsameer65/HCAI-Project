import os
import joblib


CLASSIFICATION_MODEL_FILENAME = (
    "classification_model.pkl"
)

CLASSIFICATION_METADATA_FILENAME = (
    "classification_metadata.pkl"
)

CLASSIFICATION_RESULTS_FILENAME = (
    "classification_results.pkl"
)


class ModelNotFoundError(FileNotFoundError):
    """Raised when no trained classification model is available."""
    pass


class ResultsNotFoundError(FileNotFoundError):
    """Raised when no saved classification results are available."""
    pass


def _get_model_directory(media_root):
    """
    Return the directory used for persisted machine-learning artifacts.
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
    Persist the most recently trained classification model
    and its metadata.
    """

    model_directory = (
        _get_model_directory(
            media_root
        )
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
    Load the most recently trained classification model.
    """

    model_directory = (
        _get_model_directory(
            media_root
        )
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


def save_classification_results(
    results,
    media_root,
):
    """
    Persist the latest classification training results.

    This makes the results page reloadable and allows users
    to navigate between Test, Explain and Compare pages
    without retraining the model.
    """

    model_directory = (
        _get_model_directory(
            media_root
        )
    )

    results_path = os.path.join(
        model_directory,
        CLASSIFICATION_RESULTS_FILENAME,
    )

    joblib.dump(
        results,
        results_path,
    )


def load_classification_results(
    media_root,
):
    """
    Load the most recently saved classification results.
    """

    model_directory = (
        _get_model_directory(
            media_root
        )
    )

    results_path = os.path.join(
        model_directory,
        CLASSIFICATION_RESULTS_FILENAME,
    )

    if not os.path.exists(
        results_path
    ):
        raise ResultsNotFoundError(
            "No classification training results were found."
        )

    return joblib.load(
        results_path
    )


def clear_classification_artifacts(
    media_root,
):
    """
    Remove the previous trained model/results.

    Useful when a new classification dataset is uploaded so
    results from the previous dataset cannot accidentally be reused.
    """

    model_directory = (
        _get_model_directory(
            media_root
        )
    )

    filenames = [
        CLASSIFICATION_MODEL_FILENAME,
        CLASSIFICATION_METADATA_FILENAME,
        CLASSIFICATION_RESULTS_FILENAME,
    ]

    for filename in filenames:

        path = os.path.join(
            model_directory,
            filename,
        )

        if os.path.exists(path):
            os.remove(path)