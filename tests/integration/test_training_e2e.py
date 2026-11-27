"""End-to-end: real CSV -> train -> MLflow registry -> load @champion -> score.

Exercises data loading, the full pipeline, MLflow logging/registration and model
loading together, so it is slower than the unit tests and runs with `make integration-test`.
"""

from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[2] / "data" / "raw" / "telco.csv"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATA.exists(), reason="data/raw/telco.csv not found"),
]


def test_train_register_and_serve_from_registry(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    from churn import predict, train

    metrics = train.main(["--data", str(DATA), "--out", "model.joblib", "--register"])
    assert metrics["roc_auc"] > 0.80
    assert metrics["recall"] > 0.75

    monkeypatch.setenv("MODEL_URI", "models:/churn-model@champion")
    bundle = predict.load_bundle()
    customers = pd.read_csv(DATA).head(50)
    out = predict.score(customers, bundle)
    assert len(out) == 50
    assert out["churn_probability"].between(0, 1).all()
