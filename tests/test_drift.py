import pytest

pytest.importorskip("evidently")

from churn.drift import run_drift, simulate  # noqa: E402


def test_price_hike_detected(model_path, customers):
    import joblib

    bundle = joblib.load(model_path)
    ref, cur = customers.iloc[:150], customers.iloc[150:]
    summary = run_drift(ref, simulate(cur, "price_hike"), bundle)
    assert "MonthlyCharges" in summary["drifted_columns"]


def test_no_drift_baseline_mostly_clean(model_path, customers):
    import joblib

    bundle = joblib.load(model_path)
    ref, cur = customers.iloc[:150], customers.iloc[150:]
    summary = run_drift(ref, simulate(cur, "none"), bundle)
    assert summary["share_drifted"] < 0.3
