import os

import pandas as pd


class DatasetValidationError(Exception):
    """Raised when an uploaded dataset cannot be used by the ML pipeline."""
    pass


# Common names used for identifier columns.
IDENTIFIER_COLUMNS = {
    "id",
    "index",
    "row_id",
    "rowid",
    "sample_id",
    "sampleid",
}


def _remove_identifier_columns(df):
    """
    Remove optional identifier columns.

    Identifier columns are not useful description features for
    supervised learning and should therefore not be passed to the model.
    """
    columns_to_drop = [
        column
        for column in df.columns
        if str(column).strip().lower() in IDENTIFIER_COLUMNS
    ]

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    return df, columns_to_drop


def load_dataset(file_path):
    """
    Load and validate a supervised-learning CSV dataset.

    Expected structure:
        feature_1, feature_2, ..., feature_n, target

    The final column is interpreted as the prediction target.
    All preceding columns are interpreted as description features.

    Returns:
        df
        feature_columns
        target_column
        removed_identifier_columns
    """

    # ---------------------------------------------------------
    # 1. Check that the file exists
    # ---------------------------------------------------------
    if not os.path.exists(file_path):
        raise DatasetValidationError(
            "No dataset has been uploaded."
        )

    # ---------------------------------------------------------
    # 2. Read CSV
    # ---------------------------------------------------------
    try:
        df = pd.read_csv(file_path)

    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetValidationError(
            "The uploaded file could not be read as a valid CSV file."
        ) from exc

    except OSError as exc:
        raise DatasetValidationError(
            "The uploaded dataset could not be opened."
        ) from exc

    # ---------------------------------------------------------
    # 3. Dataset must not be empty
    # ---------------------------------------------------------
    if df.empty:
        raise DatasetValidationError(
            "The uploaded dataset is empty."
        )

    # ---------------------------------------------------------
    # 4. Clean column names
    # ---------------------------------------------------------
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # ---------------------------------------------------------
    # 5. Check for duplicate column names
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 6. Remove optional ID columns
    # ---------------------------------------------------------
    df, removed_identifier_columns = (
        _remove_identifier_columns(df)
    )

    # ---------------------------------------------------------
    # 7. We need at least one feature + one target
    # ---------------------------------------------------------
    if len(df.columns) < 2:
        raise DatasetValidationError(
            "The dataset must contain at least one feature "
            "column and one target column."
        )

    # ---------------------------------------------------------
    # 8. Last column = target
    #    Everything before it = features
    # ---------------------------------------------------------
    feature_columns = df.columns[:-1].tolist()
    target_column = df.columns[-1]

    # ---------------------------------------------------------
    # 9. Validate target
    # ---------------------------------------------------------
    if df[target_column].isna().any():
        raise DatasetValidationError(
            f"The target column '{target_column}' contains missing values."
        )

    # ---------------------------------------------------------
    # 10. Target must contain actual information
    # ---------------------------------------------------------
    if df[target_column].nunique() < 2:
        raise DatasetValidationError(
            f"The target column '{target_column}' must contain "
            "at least two different values."
        )

    return (
        df,
        feature_columns,
        target_column,
        removed_identifier_columns,
    )