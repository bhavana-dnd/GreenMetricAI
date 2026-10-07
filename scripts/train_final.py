import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


DATA_PATH = "data/processed/mbpp_task_level_final.csv"

FEATURES = [
    "cc",
    "mi",
    "hv",
    "loc",
    "depth_nesting",
    "num_functions",
    "avg_function_length",
    "comment_to_code_ratio",
    "technical_debt_index",
    "api_call_density",
    "loop_depth",
    "branching_factor",
    "coupling_between_classes",
    "num_parameters",
]
RUNTIME_FEATURES = FEATURES + [
    "runtime_sec_per_execution",
]


def evaluate(name, y_true, predictions):
    mae = mean_absolute_error(y_true, predictions)
    rmse = mean_squared_error(y_true, predictions) ** 0.5
    r2 = r2_score(y_true, predictions)

    print(name)
    print(f"  MAE  (log10): {mae:.6f}")
    print(f"  RMSE (log10): {rmse:.6f}")
    print(f"  R2:           {r2:.6f}")
    print(f"  Median actual energy:    {np.median(10 ** y_true):.6e} kWh")
    print(f"  Median predicted energy: {np.median(10 ** predictions):.6e} kWh")
    print()


df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = np.log10(df["energy_kwh_per_execution"])

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

print("=" * 70)
print("GREENMETRICAI FINAL ML EXPERIMENT")
print("=" * 70)
print(f"Dataset shape: {X.shape}")
print(f"Features: {len(FEATURES)}")
print(f"Target: log10(energy_kwh_per_execution)")
print(
    f"Target range: {y.min():.4f} to {y.max():.4f}"
)
print()

# Baseline
baseline_prediction = np.full(len(y), y.mean())
evaluate("BASELINE", y, baseline_prediction)

# Random Forest
rf = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

rf_predictions = cross_val_predict(
    rf,
    X,
    y,
    cv=cv,
    n_jobs=1,
)

evaluate("RANDOM FOREST", y, rf_predictions)

# XGBoost
xgb = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

xgb_predictions = cross_val_predict(
    xgb,
    X,
    y,
    cv=cv,
    n_jobs=1,
)

evaluate("XGBOOST", y, xgb_predictions)

print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
# Runtime-inclusive experiment
X_runtime = df[RUNTIME_FEATURES]

rf_runtime = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

rf_runtime_predictions = cross_val_predict(
    rf_runtime,
    X_runtime,
    y,
    cv=cv,
    n_jobs=1,
)

evaluate(
    "RANDOM FOREST + RUNTIME",
    y,
    rf_runtime_predictions,
)