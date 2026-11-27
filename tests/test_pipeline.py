import joblib
import numpy as np

from churn.pipeline import build_pipeline
from churn.predict import offer_value, score


def test_pipeline_fits_and_outputs_probabilities(customers, labels):
    pipe = build_pipeline().fit(customers, labels)
    proba = pipe.predict_proba(customers)[:, 1]
    assert proba.shape == (len(customers),)
    assert np.all((proba >= 0) & (proba <= 1))
    # 5 num + 12 binary + 1 ordinal + 11 one-hot + 2 passthrough = 31 features
    assert len(pipe[-1].coef_[0]) == 31


def test_bundle_roundtrip(model_path, customers):
    """The saved model must load in a fresh process context (no notebook __main__)."""
    bundle = joblib.load(model_path)
    assert bundle["model"].named_steps["features"].__class__.__module__ == "churn.features"
    out = score(customers.head(10), bundle)
    assert set(out.columns) == {
        "churn_probability", "will_churn", "expected_offer_value", "send_offer"}


def test_offer_value_formula():
    a = {"offer_cost": 50, "success_rate": 0.3, "clv_months": 12}
    # 0.8 * 0.3 * (100 * 12) - 50 = 238
    assert offer_value(0.8, 100.0, a) == 238.0
    assert offer_value(0.0, 100.0, a) == -50.0
