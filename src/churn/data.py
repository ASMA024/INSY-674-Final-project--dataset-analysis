"""Loading and splitting the raw Telco CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import INPUT_COLUMNS, RANDOM_STATE, TARGET, TEST_SIZE


def load_raw(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[INPUT_COLUMNS].copy()
    y = (df[TARGET] == "Yes").astype(int)
    return X, y


def train_test(df: pd.DataFrame):
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
