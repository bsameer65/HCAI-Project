from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Hugging Face offline configuration
#
# IMPORTANT:
# These variables must be defined BEFORE importing the datasets package.
# ---------------------------------------------------------------------------

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from datasets import (
    load_dataset,
    load_from_disk,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT3_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

ARTIFACT_DIR = (
    PROJECT3_DIR
    / "artifacts"
)

DATASET_DIR = (
    ARTIFACT_DIR
    / "datasets"
)

LOCAL_AG_NEWS_PATH = (
    DATASET_DIR
    / "ag_news"
)


# ---------------------------------------------------------------------------
# Public dataset object
# ---------------------------------------------------------------------------

@dataclass
class AGNewsDataset:
    """
    Container used by the Project 3 services.

    Access:
        dataset.train
        dataset.test
    """

    train: pd.DataFrame
    test: pd.DataFrame


# ---------------------------------------------------------------------------
# Internal dataset loading
# ---------------------------------------------------------------------------

def _load_project_local_dataset():
    """
    Try loading the persistent project-local AG News copy.
    """

    if not LOCAL_AG_NEWS_PATH.exists():
        return None

    try:

        return load_from_disk(
            str(
                LOCAL_AG_NEWS_PATH
            )
        )

    except Exception as exc:

        print(
            "[AG News] Project-local dataset exists but "
            f"could not be loaded: {exc}"
        )

        return None


def _load_huggingface_cache():
    """
    Load AG News from the already existing Hugging Face cache.

    Because HF_HUB_OFFLINE and HF_DATASETS_OFFLINE are enabled,
    this should not make a request to huggingface.co.
    """

    try:

        dataset = load_dataset(
            "fancyzhx/ag_news",
        )

        return dataset

    except Exception as exc:

        raise RuntimeError(
            "\n"
            "AG News could not be loaded locally.\n"
            "\n"
            "The application is configured to avoid Hugging Face "
            "network requests because repeated Hub requests were "
            "causing HTTP 429 rate-limit delays.\n"
            "\n"
            "Expected one of the following:\n"
            f"1. Project-local dataset: {LOCAL_AG_NEWS_PATH}\n"
            "2. An existing Hugging Face AG News cache.\n"
            "\n"
            f"Original error: {exc}"
        ) from exc


def _persist_project_copy(
    dataset,
):
    """
    Save an independent project-local copy of AG News.

    This allows future runs to use load_from_disk() directly instead
    of depending on Hugging Face's cache structure.
    """

    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if LOCAL_AG_NEWS_PATH.exists():
        return

    try:

        dataset.save_to_disk(
            str(
                LOCAL_AG_NEWS_PATH
            )
        )

        print(
            "[AG News] Saved project-local dataset to: "
            f"{LOCAL_AG_NEWS_PATH}"
        )

    except Exception as exc:

        # Do not fail the application merely because persistence failed.
        print(
            "[AG News] Dataset loaded successfully, but the "
            "project-local copy could not be saved: "
            f"{exc}"
        )


# ---------------------------------------------------------------------------
# Cached raw dataset
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_raw_ag_news():
    """
    Load the Hugging Face DatasetDict once per Django process.

    Priority:
        1. project3/artifacts/datasets/ag_news
        2. existing Hugging Face local cache

    No network request should normally occur.
    """

    # ------------------------------------------------------------------
    # First choice: our own local project dataset
    # ------------------------------------------------------------------

    dataset = (
        _load_project_local_dataset()
    )

    if dataset is not None:

        print(
            "[AG News] Loaded from project-local cache."
        )

        return dataset


    # ------------------------------------------------------------------
    # Second choice: existing Hugging Face cache
    # ------------------------------------------------------------------

    print(
        "[AG News] Project-local copy not found. "
        "Trying existing Hugging Face cache in offline mode."
    )

    dataset = (
        _load_huggingface_cache()
    )


    # ------------------------------------------------------------------
    # Create our own project-local copy for subsequent application runs
    # ------------------------------------------------------------------

    _persist_project_copy(
        dataset
    )

    return dataset


# ---------------------------------------------------------------------------
# DataFrame conversion
# ---------------------------------------------------------------------------

def _dataset_to_dataframe(
    dataset_split,
) -> pd.DataFrame:
    """
    Convert one Hugging Face Dataset split into a normalized DataFrame.
    """

    dataframe = (
        dataset_split
        .to_pandas()
        .copy()
    )


    required_columns = {
        "text",
        "label",
    }


    missing_columns = (
        required_columns
        - set(
            dataframe.columns
        )
    )


    if missing_columns:

        raise ValueError(
            "AG News dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )


    # Stable integer index is important for Human Expert query selection.
    dataframe = dataframe.reset_index(
        drop=True
    )


    dataframe["text"] = (
        dataframe["text"]
        .fillna("")
        .astype(str)
    )


    dataframe["label"] = (
        dataframe["label"]
        .astype(int)
    )


    return dataframe


# ---------------------------------------------------------------------------
# Cached pandas representation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_ag_news_frames():
    """
    Convert AG News to pandas only once per Django process.
    """

    dataset = (
        _load_raw_ag_news()
    )


    if "train" not in dataset:

        raise ValueError(
            "AG News does not contain a training split."
        )


    if "test" not in dataset:

        raise ValueError(
            "AG News does not contain a test split."
        )


    train_dataframe = (
        _dataset_to_dataframe(
            dataset["train"]
        )
    )


    test_dataframe = (
        _dataset_to_dataframe(
            dataset["test"]
        )
    )


    return (
        train_dataframe,
        test_dataframe,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ag_news() -> AGNewsDataset:
    """
    Return the AG News training and test splits.

    The expensive dataset loading/conversion is cached internally.

    A copy of each DataFrame is returned so individual experiments can add
    temporary columns without modifying the shared cached dataset.
    """

    (
        train_dataframe,
        test_dataframe,
    ) = _load_ag_news_frames()


    return AGNewsDataset(
        train=(
            train_dataframe.copy()
        ),
        test=(
            test_dataframe.copy()
        ),
    )