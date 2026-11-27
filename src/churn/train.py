"""Train the churn pipeline, log it to MLflow and (optionally) register it.

Usage:
    python -m churn.train --data data/raw/telco.csv
    python -m churn.train --data data/raw/telco.csv --register
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from churn import config as c
from churn.data import load_raw, train_test
from churn.pipeline import build_pipeline

PKG_DIR = Path(__file__).resolve().parent


def evaluate(pipe, X, y, threshold: float) -> dict[str, float]:
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "recall": recall_score(y, pred),
        "precision": precision_score(y, pred),
        "f1": f1_score(y, pred),
    }


def make_bundle(pipe) -> dict:
    """Same structure as the original notebook joblib."""
    return {
        "model": pipe,
        "model_name": c.MODEL_NAME,
        "threshold": c.THRESHOLD,
        "assumptions": c.ASSUMPTIONS,
        "input_columns": c.INPUT_COLUMNS,
    }


def main(argv=None) -> dict[str, float]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/telco.csv")
    ap.add_argument("--out", default="models/churn_model.joblib")
    ap.add_argument("--experiment", default="telco-churn")
    ap.add_argument("--register", action="store_true", help="register + set @champion")
    args = ap.parse_args(argv)

    import mlflow
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(args.experiment)

    df = load_raw(args.data)
    X_train, X_test, y_train, y_test = train_test(df)

    with mlflow.start_run(run_name=c.MODEL_NAME) as run:
        pipe = build_pipeline().fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test, c.THRESHOLD)

        mlflow.log_params({**c.MODEL_PARAMS, "threshold": c.THRESHOLD,
                           "test_size": c.TEST_SIZE, "random_state": c.RANDOM_STATE})
        mlflow.log_params({f"biz_{k}": v for k, v in c.ASSUMPTIONS.items()})
        mlflow.log_metrics(metrics)

        sample = X_train.head(5)
        mlflow.sklearn.log_model(
            pipe,
            name="model",
            # ship our package with the model so TelcoFeatures can be unpickled anywhere
            code_paths=[str(PKG_DIR)],
            # MLflow 3 serializes sklearn with skops, which rejects custom classes by default
            skops_trusted_types=["churn.features.TelcoFeatures"],
            signature=infer_signature(sample, pipe.predict_proba(sample)[:, 1]),
            input_example=sample,
            registered_model_name=c.REGISTERED_MODEL_NAME if args.register else None,
        )

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(make_bundle(pipe), out)
        mlflow.log_artifact(str(out))

        if args.register:
            client = mlflow.MlflowClient()
            latest = client.search_model_versions(
                f"name='{c.REGISTERED_MODEL_NAME}' and run_id='{run.info.run_id}'"
            )[0]
            client.set_registered_model_alias(
                c.REGISTERED_MODEL_NAME, "champion", latest.version
            )
            print(f"Registered {c.REGISTERED_MODEL_NAME} v{latest.version} as @champion")

    print({k: round(v, 4) for k, v in metrics.items()})
    return metrics


if __name__ == "__main__":
    main()
