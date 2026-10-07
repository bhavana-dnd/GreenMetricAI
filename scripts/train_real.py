from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor


DATA_PATH = Path(
    "data/processed/mbpp_100_real_complete.csv"
)

TASK_LEVEL_PATH = Path(
    "data/processed/mbpp_task_level.csv"
)

RESULT_PATH = Path(
    "reports/model_results.csv"
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


TARGET = "energy_kwh"


def rmse(y_true, y_pred):
    return np.sqrt(
        mean_squared_error(y_true, y_pred)
    )


def evaluate_model(name, model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results = {
        "model": name,
        "MAE": mean_absolute_error(
            y_test,
            predictions
        ),
        "RMSE": rmse(
            y_test,
            predictions
        ),
        "R2": r2_score(
            y_test,
            predictions
        ),
    }

    return results, predictions


def main():

    print("=" * 70)
    print("GREENMETRICAI - REAL DATA MODEL TRAINING")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print()
    print("RAW DATA")
    print("-" * 70)
    print("Rows:", len(df))
    print("Tasks:", df["task_id"].nunique())

    # ---------------------------------------------------------------
    # TASK-LEVEL AGGREGATION
    # ---------------------------------------------------------------

    print()
    print("AGGREGATING 5 RUNS PER TASK")
    print("-" * 70)

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}"
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET}"
        )

    task_level = (
        df.groupby("task_id")
        .agg(
            {
                **{
                    feature: "first"
                    for feature in FEATURES
                },
                "energy_kwh": "median",
                "runtime_sec": "median",
                "carbon_emissions_g": "median",
            }
        )
        .reset_index()
    )

    print(
        "Task-level rows:",
        len(task_level)
    )

    print(
        "Expected:",
        df["task_id"].nunique()
    )

    # ---------------------------------------------------------------
    # CHECK FEATURES
    # ---------------------------------------------------------------

    print()
    print("FEATURE QUALITY")
    print("-" * 70)

    feature_nulls = task_level[
        FEATURES
    ].isna().sum()

    print(
        feature_nulls[
            feature_nulls > 0
        ]
    )

    if feature_nulls.sum() > 0:
        raise ValueError(
            "Feature columns contain missing values."
        )

    target_nulls = task_level[
        TARGET
    ].isna().sum()

    print(
        "Target nulls:",
        target_nulls
    )

    if target_nulls > 0:
        raise ValueError(
            "Target contains missing values."
        )

    # Save task-level dataset.
    TASK_LEVEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    task_level.to_csv(
        TASK_LEVEL_PATH,
        index=False,
    )

    print()
    print(
        "Saved:",
        TASK_LEVEL_PATH
    )

    # ---------------------------------------------------------------
    # DATA DISTRIBUTION
    # ---------------------------------------------------------------

    print()
    print("TARGET DISTRIBUTION")
    print("-" * 70)

    print(
        task_level[TARGET].describe()
    )

    # Energy values can span orders of magnitude.
    # Train on log10(energy) while reporting metrics
    # back on the original kWh scale.
    task_level["log_energy_kwh"] = np.log10(
        task_level[TARGET].clip(lower=1e-15)
    )

    X = task_level[FEATURES]
    y = task_level["log_energy_kwh"]

    # ---------------------------------------------------------------
    # TASK-LEVEL TRAIN / TEST SPLIT
    # ---------------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print()
    print("DATA SPLIT")
    print("-" * 70)
    print("Training tasks:", len(X_train))
    print("Testing tasks:", len(X_test))

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

    rf_results, rf_pred_log = evaluate_model(
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

    xgb_results, xgb_pred_log = evaluate_model(
        "XGBoost",
        xgb,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # ---------------------------------------------------------------
    # CONVERT PREDICTIONS BACK TO kWh
    # ---------------------------------------------------------------

    y_test_kwh = 10 ** y_test.to_numpy()

    rf_pred_kwh = 10 ** rf_pred_log
    xgb_pred_kwh = 10 ** xgb_pred_log

    rf_results_original = {
        "model": "RandomForest_original_scale",
        "MAE": mean_absolute_error(
            y_test_kwh,
            rf_pred_kwh,
        ),
        "RMSE": rmse(
            y_test_kwh,
            rf_pred_kwh,
        ),
        "R2": r2_score(
            y_test_kwh,
            rf_pred_kwh,
        ),
    }

    xgb_results_original = {
        "model": "XGBoost_original_scale",
        "MAE": mean_absolute_error(
            y_test_kwh,
            xgb_pred_kwh,
        ),
        "RMSE": rmse(
            y_test_kwh,
            xgb_pred_kwh,
        ),
        "R2": r2_score(
            y_test_kwh,
            xgb_pred_kwh,
        ),
    }

    results = pd.DataFrame(
        [
            rf_results,
            xgb_results,
            rf_results_original,
            xgb_results_original,
        ]
    )

    # ---------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------------

    rf_importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": rf.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    xgb_importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": xgb.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print()
    print("=" * 70)
    print("MODEL RESULTS")
    print("=" * 70)

    print(results.to_string(index=False))

    print()
    print("=" * 70)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        rf_importance.to_string(index=False)
    )

    print()
    print("=" * 70)
    print("XGBOOST FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        xgb_importance.to_string(index=False)
    )

    # ---------------------------------------------------------------
    # SAVE RESULTS
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
        "reports/random_forest_importance.csv",
        index=False,
    )

    xgb_importance.to_csv(
        "reports/xgboost_importance.csv",
        index=False,
    )

    print()
    print("Saved:")
    print(
        "  ",
        RESULT_PATH
    )
    print(
        "  reports/random_forest_importance.csv"
    )
    print(
        "  reports/xgboost_importance.csv"
    )


if __name__ == "__main__":
    main()