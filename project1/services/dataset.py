import os

import pandas as pd


class DatasetValidationError(Exception):
    """
    Raised when an uploaded dataset cannot be used
    by the supervised-learning pipeline.
    """
    pass


IDENTIFIER_COLUMNS = {
    "id",
    "index",
    "row_id",
    "rowid",
    "sample_id",
    "sampleid",
}


# ==============================================================
# Internal helpers
# ==============================================================

def _clean_column_names(df):
    """
    Remove leading/trailing whitespace from column names.
    """

    df = df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


def _remove_identifier_columns(df):
    """
    Remove commonly named identifier columns.

    Identifier columns should not be used as ML features.
    """

    columns_to_drop = [
        column
        for column in df.columns
        if str(column).strip().lower()
        in IDENTIFIER_COLUMNS
    ]

    if columns_to_drop:
        df = df.drop(
            columns=columns_to_drop
        )

    return df, columns_to_drop


def _read_csv(file_path):
    """
    Read and perform basic validation of a CSV dataset.
    """

    if not os.path.exists(file_path):

        raise DatasetValidationError(
            "No dataset has been uploaded."
        )

    try:

        df = pd.read_csv(
            file_path
        )

    except (
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as exc:

        raise DatasetValidationError(
            "The uploaded file could not be read "
            "as a valid CSV file."
        ) from exc

    except OSError as exc:

        raise DatasetValidationError(
            "The uploaded dataset could not be opened."
        ) from exc

    if df.empty:

        raise DatasetValidationError(
            "The uploaded dataset is empty."
        )

    df = _clean_column_names(
        df
    )

    if df.columns.duplicated().any():

        duplicates = (
            df.columns[
                df.columns.duplicated()
            ]
            .unique()
            .tolist()
        )

        raise DatasetValidationError(
            "The dataset contains duplicate column names: "
            + ", ".join(duplicates)
        )

    df, removed_identifiers = (
        _remove_identifier_columns(
            df
        )
    )

    if len(df.columns) < 2:

        raise DatasetValidationError(
            "The dataset must contain at least one "
            "feature column and one target column."
        )

    return (
        df,
        removed_identifiers,
    )


# ==============================================================
# Generic dataset inspection
# ==============================================================

def inspect_dataset(file_path):
    """
    Load a dataset without deciding the target column.

    Used when the human should choose the target.

    Returns:
        df
        column_information
        removed_identifier_columns
    """

    (
        df,
        removed_identifier_columns,
    ) = _read_csv(
        file_path
    )

    column_information = []

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            column_type = "numeric"

        else:

            column_type = "categorical"

        column_information.append({
            "name":
                column,

            "type":
                column_type,

            "missing":
                int(
                    df[column]
                    .isna()
                    .sum()
                ),

            "unique":
                int(
                    df[column]
                    .nunique(
                        dropna=True
                    )
                ),
        })

    return (
        df,
        column_information,
        removed_identifier_columns,
    )


# ==============================================================
# Classification
# ==============================================================

def load_dataset(file_path):
    """
    Load a classification-style dataset following
    the assignment convention:

        feature_1, feature_2, ..., feature_n, target

    The last column is interpreted as the target.

    This function is retained so the current
    classification implementation continues working.
    """

    (
        df,
        removed_identifier_columns,
    ) = _read_csv(
        file_path
    )

    feature_columns = (
        df.columns[:-1]
        .tolist()
    )

    target_column = (
        df.columns[-1]
    )

    if df[target_column].isna().any():

        raise DatasetValidationError(
            f"The target column '{target_column}' "
            "contains missing values."
        )

    if (
        df[target_column]
        .nunique()
        < 2
    ):

        raise DatasetValidationError(
            f"The target column '{target_column}' "
            "must contain at least two different values."
        )

    return (
        df,
        feature_columns,
        target_column,
        removed_identifier_columns,
    )


# ==============================================================
# Regression
# ==============================================================

def get_regression_target_choices(
    file_path
):
    """
    Return columns that are valid regression targets.

    Regression targets must be numeric.
    """

    (
        df,
        column_information,
        removed_identifier_columns,
    ) = inspect_dataset(
        file_path
    )

    numeric_columns = [
        column["name"]
        for column
        in column_information

        if column["type"]
        == "numeric"
    ]

    if not numeric_columns:

        raise DatasetValidationError(
            "Regression requires at least one "
            "numeric column that can be used as the target."
        )

    return (
        numeric_columns,
        column_information,
        removed_identifier_columns,
    )


def prepare_regression_dataset(
    file_path,
    target_column,
):
    """
    Prepare a dataset for regression using a user-selected
    numeric target.

    Both numeric and categorical input features are retained.
    Categorical features will later be encoded by the sklearn
    preprocessing pipeline.
    """

    (
        df,
        column_information,
        removed_identifier_columns,
    ) = inspect_dataset(
        file_path
    )

    if target_column not in df.columns:

        raise DatasetValidationError(
            "The selected target column does not exist "
            "in the uploaded dataset."
        )

    if not pd.api.types.is_numeric_dtype(
        df[target_column]
    ):

        raise DatasetValidationError(
            "Regression requires a numeric target. "
            f"'{target_column}' is not numeric."
        )

    if df[target_column].isna().any():

        raise DatasetValidationError(
            f"The selected target '{target_column}' "
            "contains missing values."
        )

    if (
        df[target_column]
        .nunique()
        < 2
    ):

        raise DatasetValidationError(
            "The regression target must contain "
            "at least two different values."
        )

    feature_columns = [
        column
        for column in df.columns

        if column
        != target_column
    ]

    numeric_features = [
        column
        for column in feature_columns

        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    categorical_features = [
        column
        for column in feature_columns

        if column
        not in numeric_features
    ]

    if not feature_columns:

        raise DatasetValidationError(
            "At least one input feature is required."
        )

    return {
        "dataframe":
            df,

        "feature_columns":
            feature_columns,

        "target_column":
            target_column,

        "numeric_features":
            numeric_features,

        "categorical_features":
            categorical_features,

        "column_information":
            column_information,

        "removed_identifier_columns":
            removed_identifier_columns,
    }