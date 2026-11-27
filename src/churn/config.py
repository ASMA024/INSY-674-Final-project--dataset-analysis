"""Central place for column lists, hyperparameters and business assumptions."""

INPUT_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
TARGET = "Churn"

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "num_addons", "price_change"]
BIN_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "PaperlessBilling",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
ORD_COLS = ["Contract"]
CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]
OHE_COLS = ["InternetService", "PaymentMethod", "tenure_group"]
PASS_COLS = ["SeniorCitizen", "auto_payment"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_PARAMS = {"C": 1.0129197956845732, "class_weight": "balanced", "max_iter": 3000}
THRESHOLD = 0.45
ASSUMPTIONS = {"offer_cost": 50, "success_rate": 0.3, "clv_months": 12}
MODEL_NAME = "Logistic Regression"
REGISTERED_MODEL_NAME = "churn-model"
