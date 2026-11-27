"""Model loading + scoring + retention-offer business logic."""

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
        return {"model": pipe, "model_name": c.MODEL_NAME, "threshold": c.THRESHOLD,
                "assumptions": c.ASSUMPTIONS, "input_columns": c.INPUT_COLUMNS,
                "version": uri}
    path = path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    bundle = joblib.load(path)
    bundle.setdefault("version", os.path.basename(path))
    return bundle


@lru_cache(maxsize=1)
def get_bundle() -> dict:
    return load_bundle()


def offer_value(proba: float, monthly_charges: float, assumptions: dict) -> float:
    """Expected net value of sending a retention offer to one customer.

    = P(churn) * P(offer works) * revenue saved over clv_months - offer cost
    """
    saved = monthly_charges * assumptions["clv_months"]
    return proba * assumptions["success_rate"] * saved - assumptions["offer_cost"]


def score(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    df = df[bundle["input_columns"]]
    proba = bundle["model"].predict_proba(df)[:, 1]
    out = pd.DataFrame({"churn_probability": proba})
    out["will_churn"] = out["churn_probability"] >= bundle["threshold"]
    out["expected_offer_value"] = [
        offer_value(p, m, bundle["assumptions"])
        for p, m in zip(proba, df["MonthlyCharges"], strict=True)
    ]
    out["send_offer"] = out["will_churn"] & (out["expected_offer_value"] > 0)
    return out
