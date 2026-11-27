"""Model loading and scoring."""

from __future__ import annotations

import os
from functools import lru_cache

import joblib
import pandas as pd

from churn import config as c

DEFAULT_MODEL_PATH = "models/churn_model.joblib"


def load_bundle(path: str | None = None) -> dict:
    """Load from MLflow registry if MODEL_URI is set, otherwise a local joblib."""
    uri = os.getenv("MODEL_URI")
    if uri:  # e.g. models:/churn-model@champion
        import mlflow

        pipe = mlflow.sklearn.load_model(uri)
        return {
            "model": pipe,
            "model_name": c.MODEL_NAME,
            "threshold": c.THRESHOLD,
            "assumptions": c.ASSUMPTIONS,
            "input_columns": c.INPUT_COLUMNS,
            "version": uri,
        }
    path = path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    bundle = joblib.load(path)
    bundle.setdefault("version", os.path.basename(path))
    return bundle


@lru_cache(maxsize=1)
def get_bundle() -> dict:
    return load_bundle()


def score(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    df = df[bundle["input_columns"]]
    proba = bundle["model"].predict_proba(df)[:, 1]
    # Same rule as score_customers() in the notebook: the threshold was already chosen
    # to minimise total cost, so it is the whole decision. Probabilities come from a
    # class-weighted model and are not calibrated, so don't read them as literal odds.
    return pd.DataFrame(
        {
            "churn_probability": proba,
            "send_offer": proba >= bundle["threshold"],
        }
    )
