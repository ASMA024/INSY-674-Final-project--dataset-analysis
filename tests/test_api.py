import pytest
from fastapi.testclient import TestClient

from churn.api import Customer


@pytest.fixture
def client(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.delenv("MODEL_URI", raising=False)
    from churn import predict

    predict.get_bundle.cache_clear()
    from churn.api import app

    with TestClient(app) as c:
        yield c
    predict.get_bundle.cache_clear()


@pytest.fixture
def payload():
    return Customer.model_config["json_schema_extra"]["example"].copy()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert isinstance(body["send_offer"], bool)


def test_predict_batch(client, payload):
    r = client.post("/predict/batch", json=[payload, payload])
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_missing_total_charges_ok(client, payload):
    payload["tenure"], payload["TotalCharges"] = 0, None
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize(
    "field,bad",
    [
        ("Contract", "Three year"),
        ("tenure", -1),
        ("MonthlyCharges", "abc"),
    ],
)
def test_invalid_input_rejected(client, payload, field, bad):
    payload[field] = bad
    assert client.post("/predict", json=payload).status_code == 422
