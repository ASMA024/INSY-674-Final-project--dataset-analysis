"""FastAPI prediction service.

Run locally:  uvicorn churn.api:app --reload
Docs:         http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from churn import __version__
from churn.predict import get_bundle, score

YesNo = Literal["Yes", "No"]
YesNoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoInternet
    OnlineBackup: YesNoInternet
    DeviceProtection: YesNoInternet
    TechSupport: YesNoInternet
    StreamingTV: YesNoInternet
    StreamingMovies: YesNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.5,
                "TotalCharges": 190.0,
            }
        }
    }


class Prediction(BaseModel):
    churn_probability: float
    send_offer: bool


class BatchResponse(BaseModel):
    model_version: str
    threshold: float
    predictions: list[Prediction]


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_bundle()  # fail fast at startup if the model can't load
    yield


app = FastAPI(title="Telco Churn API", version=__version__, lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    b = get_bundle()
    return {"status": "ok", "model": b["model_name"], "model_version": b["version"]}


def _score(customers: list[Customer]) -> BatchResponse:
    b = get_bundle()
    df = pd.DataFrame([c.model_dump() for c in customers])
    res = score(df, b)
    return BatchResponse(
        model_version=b["version"],
        threshold=b["threshold"],
        predictions=[Prediction(**r) for r in res.to_dict(orient="records")],
    )


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    return _score([customer]).predictions[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(customers: list[Customer]) -> BatchResponse:
    return _score(customers)
