"""Shared fixtures: synthetic Telco-like data so tests never need the real CSV."""

import numpy as np
import pandas as pd
import pytest

from churn.config import INPUT_COLUMNS

YN = ["Yes", "No"]
YNI = ["Yes", "No", "No internet service"]


def make_customers(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    monthly = rng.uniform(18, 120, n).round(2)
    df = pd.DataFrame(
        {
            "gender": rng.choice(["Female", "Male"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(YN, n),
            "Dependents": rng.choice(YN, n),
            "tenure": tenure,
            "PhoneService": rng.choice(YN, n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            **{
                c: rng.choice(YNI, n)
                for c in [
                    "OnlineSecurity",
                    "OnlineBackup",
                    "DeviceProtection",
                    "TechSupport",
                    "StreamingTV",
                    "StreamingMovies",
                ]
            },
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(YN, n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": (monthly * tenure).round(2).astype(str),
        }
    )
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "  # mimic the real blanks
    return df[INPUT_COLUMNS]


@pytest.fixture(scope="session")
def customers() -> pd.DataFrame:
    return make_customers()


@pytest.fixture(scope="session")
def labels(customers) -> pd.Series:
    # make churn depend on contract/tenure so the model learns something
    logit = 1.5 * (customers["Contract"] == "Month-to-month") - 0.04 * customers["tenure"]
    rng = np.random.default_rng(1)
    return pd.Series((rng.uniform(size=len(customers)) < 1 / (1 + np.exp(-logit))).astype(int))


@pytest.fixture(scope="session")
def model_path(tmp_path_factory, customers, labels):
    import joblib

    from churn.pipeline import build_pipeline
    from churn.train import make_bundle

    pipe = build_pipeline().fit(customers, labels)
    path = tmp_path_factory.mktemp("models") / "churn_model.joblib"
    joblib.dump(make_bundle(pipe), path)
    return path
