"""Simulated drift monitoring with Evidently.

The Telco data has no timestamps, so we cannot split "last month" vs "this month".
Instead: reference = training set, current = test set (no drift baseline), and
current + synthetic shifts (the drift scenarios). We then check Evidently flags them.

Usage:
    python -m churn.drift --data data/raw/telco.csv --scenario price_hike
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from churn.data import load_raw, train_test
from churn.predict import load_bundle


def _price_hike(df, rng):
    """Company raises prices 20% for everyone."""
    df["MonthlyCharges"] = df["MonthlyCharges"] * 1.20
    return df


def _fiber_rollout(df, rng):
    """Half of DSL customers switch to fiber; more electronic check payers."""
    dsl = df.index[df["InternetService"] == "DSL"]
    df.loc[rng.choice(dsl, len(dsl) // 2, replace=False), "InternetService"] = "Fiber optic"
    auto = df.index[df["PaymentMethod"].str.contains("automatic")]
    df.loc[rng.choice(auto, len(auto) // 3, replace=False), "PaymentMethod"] = "Electronic check"
    return df


def _new_customers(df, rng):
    """Marketing campaign: influx of short-tenure month-to-month customers."""
    df["tenure"] = (df["tenure"] * rng.uniform(0.2, 0.6, len(df))).round().astype(int)
    df["TotalCharges"] = df["MonthlyCharges"] * df["tenure"]
    m2m = df.index[df["Contract"] != "Month-to-month"]
    df.loc[rng.choice(m2m, len(m2m) // 2, replace=False), "Contract"] = "Month-to-month"
    return df


SCENARIOS = {
    "none": lambda df, rng: df,
    "price_hike": _price_hike,
    "fiber_rollout": _fiber_rollout,
    "new_customers": _new_customers,
}


def simulate(current: pd.DataFrame, scenario: str, seed: int = 0) -> pd.DataFrame:
    df = current.copy().reset_index(drop=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return SCENARIOS[scenario](df, np.random.default_rng(seed))


def run_drift(
    reference: pd.DataFrame, current: pd.DataFrame, bundle: dict, out_html: str | Path | None = None
) -> dict:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    ref, cur = reference.copy(), current.copy()
    for d in (ref, cur):
        d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0.0)
        d["prediction"] = bundle["model"].predict_proba(d[bundle["input_columns"]])[:, 1]

    snapshot = Report([DataDriftPreset()]).run(current_data=cur, reference_data=ref)
    if out_html:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(out_html))

    summary = {"drifted_columns": [], "share_drifted": None}
    for m in snapshot.dict()["metrics"]:
        name, val = m["metric_name"], m["value"]
        if name.startswith("DriftedColumnsCount"):
            summary["share_drifted"] = round(val["share"], 3)
            summary["n_drifted"] = int(val["count"])
        elif name.startswith("ValueDrift") and isinstance(val, int | float):
            col = m["config"]["column"]
            thr = m["config"].get("threshold")
            method = m["config"].get("method", "")
            # p-value tests drift when value < threshold; distance metrics when value > threshold
            is_pvalue = "p_value" in method or "test" in method.lower()
            drifted = val < thr if is_pvalue else val > thr
            if drifted:
                summary["drifted_columns"].append(col)
    summary["prediction_drift"] = "prediction" in summary["drifted_columns"]
    return summary


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/telco.csv")
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="price_hike")
    ap.add_argument("--out", default=None, help="HTML report path")
    args = ap.parse_args(argv)

    bundle = load_bundle()
    X_train, X_test, _, _ = train_test(load_raw(args.data))
    current = simulate(X_test, args.scenario)
    out = args.out or f"reports/drift_{args.scenario}.html"
    summary = run_drift(X_train, current, bundle, out)
    print(json.dumps({"scenario": args.scenario, **summary, "report": out}, indent=2))
    return summary


if __name__ == "__main__":
    main()
