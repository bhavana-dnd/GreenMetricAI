from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


DATA_PATH = Path(
    "data/processed/mbpp_task_level_normalized.csv"
)

RESULT_PATH = Path(
    "reports/normalized_model_results.csv"
)

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

TARGET = "energy_kwh_per_execution"


def rmse(y_true, y_pred):
    return np.sqrt(
        mean_squared_error(y_true, y_pred)
    )


def evaluate(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    model.fit(X_train, y_train)

    pred_log = model.predict(X_test)

    # Convert back to kWh.
    pred = 10 ** pred_log
    actual = 10 ** y_test.to_numpy()

    return {
        "model": name,
        "MAE_kWh": mean_absolute_error(
            actual,
            pred,
        ),
        "RMSE_kWh": rmse(
            actual,
            pred,
        ),
        "R2_original": r2_score(
            actual,
            pred,
        ),
        "R2_log": r2_score(
            y_test,
            pred_log,
        ),
    }, model


def main():

    print("=" * 70)
    print("GREENMETRICAI - NORMALIZED ENERGY MODEL")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    print()
    print("DATA")
    print("-" * 70)
    print("Rows:", len(df))
    print("Tasks:", df["task_id"].nunique())

    missing = [
        f
        for f in FEATURES
        if df[f].isna().any()
    ]

    if missing:
        raise ValueError(
            f"Missing feature values: {missing}"
        )

    if df[TARGET].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    # Log target because energy spans many orders of magnitude.
    df["log_energy"] = np.log10(
        df[TARGET].clip(lower=1e-15)
    )

    X = df[FEATURES]
    y = df["log_energy"]

    # Task-level split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print()
    print("SPLIT")
    print("-" * 70)
    print("Training:", len(X_train))
    print("Testing :", len(X_test))

    # ---------------------------------------------------------------
    # RANDOM FOREST
    # ---------------------------------------------------------------

    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    )

    rf_result, rf = evaluate(
        "RandomForest",
        rf,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # ---------------------------------------------------------------
    # XGBOOST
    # ---------------------------------------------------------------

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

    xgb_result, xgb = evaluate(
        "XGBoost",
        xgb,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    results = pd.DataFrame(
        [rf_result, xgb_result]
    )

    print()
    print("=" * 70)
    print("MODEL RESULTS")
    print("=" * 70)

    print(
        results.to_string(index=False)
    )

    # ---------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------------

    rf_importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": rf.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    xgb_importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": xgb.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    print()
    print("=" * 70)
    print("RANDOM FOREST IMPORTANCE")
    print("=" * 70)
    print(
        rf_importance.to_string(index=False)
    )

    print()
    print("=" * 70)
    print("XGBOOST IMPORTANCE")
    print("=" * 70)
    print(
        xgb_importance.to_string(index=False)
    )

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULT_PATH,
        index=False,
    )

    rf_importance.to_csv(
        "reports/normalized_rf_importance.csv",
        index=False,
    )

    xgb_importance.to_csv(
        "reports/normalized_xgb_importance.csv",
        index=False,
    )

    print()
    print("Saved:")
    print(RESULT_PATH)
    print("reports/normalized_rf_importance.csv")
    print("reports/normalized_xgb_importance.csv")


if __name__ == "__main__":
    main()