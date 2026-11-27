"""Builds the (unfitted) sklearn pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from churn import config as c
from churn.features import TelcoFeatures


def build_pipeline(**model_params) -> Pipeline:
    params = {**c.MODEL_PARAMS, **model_params}
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), c.NUM_COLS),
            ("bin", OneHotEncoder(drop="if_binary", sparse_output=False), c.BIN_COLS),
            ("ord", OrdinalEncoder(categories=[c.CONTRACT_ORDER]), c.ORD_COLS),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), c.OHE_COLS),
            ("pass", "passthrough", c.PASS_COLS),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return Pipeline(
        [
            ("features", TelcoFeatures()),
            ("prep", prep),
            ("model", LogisticRegression(**params)),
        ]
    )
