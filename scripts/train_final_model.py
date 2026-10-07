import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor


DATA_PATH = "data/processed/mbpp_task_level_final.csv"
MODEL_PATH = "models/greenmetric_rf_final.joblib"

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


df = pd.read_csv(DATA_PATH)

X = df[FEATURES].astype(float)

y = np.log10(
    df["energy_kwh_per_execution"].astype(float)
)

model = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

model.fit(X, y)

Path(MODEL_PATH).parent.mkdir(
    parents=True,
    exist_ok=True,
)

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "target": "log10_energy_kwh_per_execution",
    },
    MODEL_PATH,
)

print("=" * 70)
print("FINAL GREENMETRICAI MODEL")
print("=" * 70)
print(f"Training rows: {len(df)}")
print(f"Features: {len(FEATURES)}")
print("Target: log10(energy_kwh_per_execution)")
print(f"Model: Random Forest")
print()
print(f"Saved: {MODEL_PATH}")
print("=" * 70)