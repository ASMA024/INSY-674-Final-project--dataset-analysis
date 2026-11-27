"""Calls a *running* API over HTTP (e.g. the Docker container).

    API_URL=http://localhost:8000 pytest -m integration tests/integration/test_api_live.py

`make integration-test` builds the image, starts it, runs this, and stops it.
"""

import json
import os
from pathlib import Path

import httpx
import pytest

API_URL = os.getenv("API_URL")
SAMPLE = json.loads((Path(__file__).resolve().parents[1] / "sample_request.json").read_text())

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not API_URL, reason="set API_URL to a running API"),
]


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=API_URL, timeout=10) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_high_risk_customer_gets_offer(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["churn_probability"] > 0.5
    assert body["send_offer"] is True


def test_low_risk_customer_no_offer(client):
    loyal = {
        **SAMPLE,
        "tenure": 60,
        "Contract": "Two year",
        "InternetService": "DSL",
        "PaymentMethod": "Credit card (automatic)",
        "TotalCharges": 5700.0,
    }
    r = client.post("/predict", json=loyal)
    assert r.status_code == 200
    assert r.json()["send_offer"] is False


def test_batch(client):
    r = client.post("/predict/batch", json=[SAMPLE, SAMPLE, SAMPLE])
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 3


def test_invalid_input_rejected(client):
    r = client.post("/predict", json={**SAMPLE, "Contract": "Three year"})
    assert r.status_code == 422
