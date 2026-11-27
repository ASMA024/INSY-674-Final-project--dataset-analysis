import pandas as pd
import pytest

from churn.features import TelcoFeatures

from .conftest import make_customers


@pytest.fixture
def row():
    return make_customers(1).iloc[[0]].copy()


def transform(df):
    return TelcoFeatures().fit_transform(df)


def test_blank_total_charges_becomes_zero(row):
    row["tenure"], row["TotalCharges"] = 0, " "
    assert transform(row)["TotalCharges"].iloc[0] == 0.0


def test_num_addons_counts_yes_only(row):
    row[["OnlineSecurity", "OnlineBackup", "DeviceProtection"]] = "Yes"
    row[["TechSupport", "StreamingTV"]] = "No"
    row["StreamingMovies"] = "No internet service"
    assert transform(row)["num_addons"].iloc[0] == 3


def test_price_change(row):
    row["tenure"], row["MonthlyCharges"], row["TotalCharges"] = 10, 50.0, "520"
    assert transform(row)["price_change"].iloc[0] == pytest.approx(20.0)


@pytest.mark.parametrize(
    "tenure,group",
    [
        (0, "0-12"),
        (12, "0-12"),
        (13, "13-24"),
        (24, "13-24"),
        (25, "25-48"),
        (48, "25-48"),
        (49, "49-72"),
        (72, "49-72"),
        (90, "49-72"),
    ],
)
def test_tenure_group_boundaries(row, tenure, group):
    row["tenure"] = tenure
    assert transform(row)["tenure_group"].iloc[0] == group


@pytest.mark.parametrize(
    "method,expected",
    [
        ("Bank transfer (automatic)", 1),
        ("Credit card (automatic)", 1),
        ("Electronic check", 0),
        ("Mailed check", 0),
    ],
)
def test_auto_payment(row, method, expected):
    row["PaymentMethod"] = method
    assert transform(row)["auto_payment"].iloc[0] == expected


def test_no_service_values_collapsed(row):
    row["MultipleLines"], row["OnlineBackup"] = "No phone service", "No internet service"
    out = transform(row)
    assert out["MultipleLines"].iloc[0] == "No"
    assert out["OnlineBackup"].iloc[0] == "No"


def test_does_not_mutate_input(row):
    before = row.copy()
    transform(row)
    pd.testing.assert_frame_equal(row, before)
