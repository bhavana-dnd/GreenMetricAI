import pandas as pd
import shap

from sklearn.ensemble import RandomForestRegressor


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


df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = __import__("numpy").log10(df["energy_kwh_per_execution"])


model = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

importance = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": abs(shap_values).mean(axis=0),
})

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False,
)

print("=" * 70)
print("GREENMETRICAI SHAP FEATURE IMPORTANCE")
print("=" * 70)
print()
print(importance.to_string(index=False))

importance.to_csv(
    "reports/shap_feature_importance.csv",
    index=False,
)

print()
print("Saved:")
print("reports/shap_feature_importance.csv")