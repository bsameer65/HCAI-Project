

import pandas as pd
from palmerpenguins import load_penguins
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# ── Column definitions ────────────────────────────────────────────────────────

TARGET_COL = "species"

NUMERICAL_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

CATEGORICAL_FEATURES = [
    "island",
    "sex",
    "year",   # year is treated as categorical because it only has
              # three discrete values (2007, 2008, 2009) — using it as a number
              # would imply a false linear ordering that the model should not learn
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_penguin_data():
    """
    Load and clean the Palmer Penguins dataset.

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataframe (no missing values).
    X : pd.DataFrame
        Feature matrix with original (un-encoded) values.
    y : pd.Series
        Target labels (species strings).
    numerical_features : list[str]
    categorical_features : list[str]
    class_names : list[str]
        Sorted list of unique species names.
    """
    df = load_penguins()

    # Drop rows that have any missing value in our columns of interest
    cols_needed = [TARGET_COL] + ALL_FEATURES
    df = df[cols_needed].dropna().reset_index(drop=True)

    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()

    class_names = sorted(y.unique().tolist())

    return df, X, y, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, class_names


def get_preprocessor():
    """
    Build a reusable ColumnTransformer that:
      - StandardScales numerical features  (needed for Logistic Regression)
      - OneHotEncodes categorical features (needed for both DT and LR)

    wrapping preprocessing in a ColumnTransformer keeps it
    consistent between Decision Tree and Logistic Regression, so comparisons
    are fair (both see the same encoded input).
    """
    numerical_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def encode_target(y):
    """
    Encode species strings to integers (Adelie=0, Chinstrap=1, Gentoo=2).
    Returns encoded array and the fitted LabelEncoder so callers can
    map predictions back to species names.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le