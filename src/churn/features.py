"""Feature engineering for the Telco churn model.

This class used to live in the notebook (``__main__.TelcoFeatures``). It must live in
an importable module so that a pickled pipeline can be loaded by the API, tests and
Docker image.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ADDON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
TENURE_BINS = [-1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-72"]


class TelcoFeatures(BaseEstimator, TransformerMixin):
    """Adds engineered columns; stateless, so ``fit`` learns nothing."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # "No phone service" / "No internet service" carry no extra info -> "No"
        X = X.replace({"No phone service": "No", "No internet service": "No"})
        # Blank TotalCharges (tenure == 0 customers) -> 0
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0.0)
        X["num_addons"] = (X[ADDON_COLS] == "Yes").sum(axis=1)
        # Positive = paid more historically than current monthly rate implies
        X["price_change"] = X["TotalCharges"] - X["MonthlyCharges"] * X["tenure"]
        X["tenure_group"] = pd.cut(
            X["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS
        ).astype(str)
        X["auto_payment"] = (
            X["PaymentMethod"].str.contains("automatic", case=False).astype(int)
        )
        return X
