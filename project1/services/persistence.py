import os
import joblib


# ==============================================================
# Classification artifacts
# ==============================================================

CLASSIFICATION_MODEL_FILENAME = (
    "classification_model.pkl"
)

CLASSIFICATION_METADATA_FILENAME = (
    "classification_metadata.pkl"
)

CLASSIFICATION_RESULTS_FILENAME = (
    "classification_results.pkl"
)


# ==============================================================
# Regression artifacts
# ==============================================================

REGRESSION_ARTIFACT_FILENAMES = (
    "regression_model.pkl",
    "regression_metadata.pkl",
    "regression_results.pkl",
)

REGRESSION_MODEL_FILENAME = (
    REGRESSION_ARTIFACT_FILENAMES[0]
)

REGRESSION_METADATA_FILENAME = (
    REGRESSION_ARTIFACT_FILENAMES[1]
)

REGRESSION_RESULTS_FILENAME = (
    REGRESSION_ARTIFACT_FILENAMES[2]
)


# ==============================================================
# Exceptions
# ==============================================================

class ModelNotFoundError(FileNotFoundError):
    """
    Raised when a requested trained model
    is not available.
    """
    pass


class ResultsNotFoundError(FileNotFoundError):
    """
    Raised when saved training results
    are not available.
    """
    pass


# ==============================================================
# Shared helper
# ==============================================================

def _get_model_directory(media_root):
    """
    Return the directory used for persisted
    machine-learning artifacts.
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


# ==============================================================
# Classification persistence
# ==============================================================

def save_classification_model(
    model,
    metadata,
    media_root,
):
    """
    Save the latest trained classification model
    and associated metadata.
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
    Load the latest trained classification model
    and associated metadata.
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
    Save the latest classification training results.

    Keeping results separately makes the results page
    reusable without retraining the model.
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
    Load the latest classification training results.
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
    Remove all artifacts belonging to the previous
    classification run.

    This should be called when a new classification
    dataset is uploaded.
    """

    model_directory = (
        _get_model_directory(
            media_root
        )
    )

    classification_files = (
        CLASSIFICATION_MODEL_FILENAME,
        CLASSIFICATION_METADATA_FILENAME,
        CLASSIFICATION_RESULTS_FILENAME,
    )

    for filename in classification_files:

        path = os.path.join(
            model_directory,
            filename,
        )

        if os.path.exists(path):
            os.remove(path)


# ==============================================================
# Regression persistence
# ==============================================================

def clear_regression_artifacts(
    media_root,
):
    """
    Remove only artifacts belonging to
    the previous regression run.
    """

    model_directory = (
        _get_model_directory(
            media_root
        )
    )

    for filename in (
        REGRESSION_ARTIFACT_FILENAMES
    ):

        path = os.path.join(
            model_directory,
            filename,
        )

        if os.path.exists(path):
            os.remove(path)


def save_regression_model(
    model,
    metadata,
    media_root,
):
    """
    Save the latest trained regression model
    and associated metadata.
    """

    model_directory = (
        _get_model_directory(
            media_root
        )
    )

    model_path = os.path.join(
        model_directory,
        REGRESSION_MODEL_FILENAME,
    )

    metadata_path = os.path.join(
        model_directory,
        REGRESSION_METADATA_FILENAME,
    )

    joblib.dump(
        model,
        model_path,
    )

    joblib.dump(
        metadata,
        metadata_path,
    )


def load_regression_model(
    media_root,
):
    """
    Load the latest trained regression model
    and associated metadata.
    """

    model_directory = (
        _get_model_directory(
            media_root
        )
    )

    model_path = os.path.join(
        model_directory,
        REGRESSION_MODEL_FILENAME,
    )

    metadata_path = os.path.join(
        model_directory,
        REGRESSION_METADATA_FILENAME,
    )

    if (
        not os.path.exists(model_path)
        or not os.path.exists(metadata_path)
    ):
        raise ModelNotFoundError(
            "No trained regression model was found."
        )

    model = joblib.load(
        model_path
    )

    metadata = joblib.load(
        metadata_path
    )

    return model, metadata


def save_regression_results(
    results,
    media_root,
):
    """
    Save the latest regression training results.
    """

    model_directory = (
        _get_model_directory(
            media_root
        )
    )

    results_path = os.path.join(
        model_directory,
        REGRESSION_RESULTS_FILENAME,
    )

    joblib.dump(
        results,
        results_path,
    )


def load_regression_results(
    media_root,
):
    """
    Load the latest regression training results.
    """

    model_directory = (
        _get_model_directory(
            media_root
        )
    )

    results_path = os.path.join(
        model_directory,
        REGRESSION_RESULTS_FILENAME,
    )

    if not os.path.exists(
        results_path
    ):
        raise ResultsNotFoundError(
            "No regression training results were found."
        )

    return joblib.load(
        results_path
    )