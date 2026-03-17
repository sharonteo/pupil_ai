# src/features.py
# This module handles:
# - Loading the synthetic dataset
# - Building the preprocessing pipeline (OneHotEncoder + passthrough numerics)
# - Train/test split with stratification
# - Returning clean, reusable components for the rest of the system

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.config import (
    DATA_PATH,
    TARGET,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TEST_SIZE,
    RANDOM_STATE,
)

def load_dataset():
    """Load the synthetic pupillometry dataset."""
    df = pd.read_csv(DATA_PATH)
    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df[TARGET]
    return df, X, y

def build_preprocessor():
    """Create preprocessing pipeline for categorical + numeric features."""
    categorical = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical, CATEGORICAL_COLS),
            ("numeric", "passthrough", NUMERIC_COLS),
        ]
    )
    return preprocessor

def split_data(X, y):
    """Train/test split with stratification."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )