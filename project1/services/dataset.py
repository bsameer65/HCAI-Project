import csv
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


def _read_csv(file_path):
    """Read a CSV while preserving useful validation error messages."""
    if not os.path.exists(file_path):
        raise DatasetValidationError("No dataset has been uploaded.")

    try:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as source:
            header = next(csv.reader(source), None)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise DatasetValidationError(
            "The uploaded file could not be read as a valid CSV file."
        ) from exc
    except OSError as exc:
        raise DatasetValidationError(
            "The uploaded dataset could not be opened."
        ) from exc

    if not header:
        raise DatasetValidationError("The uploaded dataset is empty.")

    cleaned_header = [str(column).strip() for column in header]
    duplicates = sorted({
        column for column in cleaned_header
        if cleaned_header.count(column) > 1
    })
    if duplicates:
        raise DatasetValidationError(
            "The dataset contains duplicate column names: "
            + ", ".join(duplicates)
        )

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

    if df.empty:
        raise DatasetValidationError("The uploaded dataset is empty.")

    df.columns = cleaned_header
    return df


def inspect_dataset(file_path):
    """Inspect a dataset without deciding which column is the target."""
    df = _read_csv(file_path)
    df, removed_identifier_columns = _remove_identifier_columns(df)

    if len(df.columns) < 2:
        raise DatasetValidationError(
            "The dataset must contain at least one feature column and one target column."
        )

    column_information = [
        {
            "name": column,
            "type": (
                "numeric"
                if pd.api.types.is_numeric_dtype(df[column])
                else "categorical"
            ),
            "missing": int(df[column].isna().sum()),
            "unique": int(df[column].nunique(dropna=True)),
        }
        for column in df.columns
    ]

    return df, column_information, removed_identifier_columns


def get_regression_target_choices(file_path):
    """Return numeric columns that can be selected as a regression target."""
    _, column_information, _ = inspect_dataset(file_path)
    choices = [
        item["name"] for item in column_information
        if item["type"] == "numeric"
    ]
    if not choices:
        raise DatasetValidationError(
            "Regression requires at least one numerical column that can be used as the target."
        )
    return choices


def prepare_regression_dataset(file_path, target_column):
    """Validate the selected target and describe regression features."""
    df, column_information, removed_identifier_columns = inspect_dataset(file_path)

    if target_column not in df.columns:
        raise DatasetValidationError(
            f"The selected target column '{target_column}' does not exist."
        )
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise DatasetValidationError(
            f"The regression target '{target_column}' must be numerical."
        )
    if df[target_column].isna().any():
        raise DatasetValidationError(
            f"The target column '{target_column}' contains missing values."
        )
    if df[target_column].nunique() < 2:
        raise DatasetValidationError(
            f"The target column '{target_column}' must contain at least two different values."
        )

    feature_columns = [column for column in df.columns if column != target_column]
    if not feature_columns:
        raise DatasetValidationError("At least one input feature must remain.")

    numeric_features = [
        column for column in feature_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    categorical_features = [
        column for column in feature_columns
        if column not in numeric_features
    ]
    return {
        "dataframe": df,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "column_information": column_information,
        "removed_identifier_columns": removed_identifier_columns,
    }


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

    df = _read_csv(file_path)

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
