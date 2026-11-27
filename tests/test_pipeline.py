import joblib
import numpy as np

from churn.pipeline import build_pipeline
from churn.predict import score


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
    assert list(out.columns) == ["churn_probability", "send_offer"]
    assert (out["send_offer"] == (out["churn_probability"] >= bundle["threshold"])).all()
