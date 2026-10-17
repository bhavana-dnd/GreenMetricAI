
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from greenmetricai.features import extract_code_metrics

st.set_page_config(
    page_title="GreenMetricAI",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 GreenMetricAI")
st.caption(
    "Measured MBPP execution data, energy prediction, "
    "SHAP explainability, and optimization validation"
)

MODEL_CANDIDATE = "models/greenmetric_rf_final.joblib"
MODEL_TRAINED = "models/greenmetric_model.joblib"
DATA = "data/processed/mbpp_100_final_5sec.csv"
OPTIMIZATION_DATA = "data/processed/optimization_comparison.csv"
DEMO_OPTIMIZATION_DATA = (
    "data/processed/optimization_comparison_ui_demo.csv"
)


@st.cache_resource
def load_candidate_model(path):
    """Load the saved Random Forest artifact."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("Expected a model artifact dictionary.")
    for key in ("model", "features", "target"):
        if key not in bundle:
            raise ValueError(f"Model artifact is missing '{key}'.")
    return bundle


@st.cache_resource
def load_trained_model(path):
    """Load the separate GreenMetricModel artifact."""
    from greenmetricai.model import GreenMetricModel
    return GreenMetricModel.load(path)


def format_energy(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.4g} kWh"


# Load the available prediction model.
model = None
model_kind = None
model_error = None

if os.path.exists(MODEL_TRAINED):
    try:
        model = load_trained_model(MODEL_TRAINED)
        model_kind = "greenmetric"
    except Exception as exc:
        model_error = f"{MODEL_TRAINED}: {exc}"

if model is None and os.path.exists(MODEL_CANDIDATE):
    try:
        model = load_candidate_model(MODEL_CANDIDATE)
        model_kind = "random_forest"
        model_error = None
    except Exception as exc:
        model_error = f"{MODEL_CANDIDATE}: {exc}"

if model is None:
    st.warning(
        "The prediction model could not be loaded. "
        "The Energy Validation tab can still be used."
    )
    if model_error:
        st.caption(f"Model loading detail: {model_error}")

analyze_tab, shap_tab, energy_tab = st.tabs(
    ["Analyze", "SHAP", "Energy Validation"]
)

# ---------------------------------------------------------
# ANALYZE
# ---------------------------------------------------------
with analyze_tab:
    st.subheader("Python Code Sustainability Analysis")

    if model is None:
        st.info(
            "To enable prediction, provide a compatible model artifact. "
            "The Energy Validation tab remains available independently."
        )
    else:
        code = st.text_area(
            "Python code",
            height=300,
            value=(
                "def square_sum(n):\n"
                "    return sum(i*i for i in range(n))"
            ),
        )

        if st.button("Analyze", type="primary"):
            try:
                metrics = extract_code_metrics(code)
                feature_values = metrics.to_dict()

                if model_kind == "random_forest":
                    feature_names = list(model["features"])
                    missing_features = [
                        name for name in feature_names
                        if name not in feature_values
                    ]

                    if missing_features:
                        st.error(
                            "The feature extractor did not produce: "
                            + ", ".join(missing_features)
                        )
                    else:
                        X = pd.DataFrame(
                            [[feature_values[name] for name in feature_names]],
                            columns=feature_names,
                        )
                        predicted_value = float(
                            model["model"].predict(X)[0]
                        )
                        target = str(model["target"])

                        if target == "log10_energy_kwh_per_execution":
                            if (
                                not np.isfinite(predicted_value)
                                or not -300 <= predicted_value <= 100
                            ):
                                st.error(
                                    "The model returned an invalid "
                                    "log-energy prediction."
                                )
                            else:
                                predicted_energy = 10.0 ** predicted_value
                                st.metric(
                                    "Predicted energy per execution",
                                    format_energy(predicted_energy),
                                )
                                st.caption(
                                    "Prediction from the saved Random Forest "
                                    "model. This is an estimate, not a "
                                    "direct measurement."
                                )
                        else:
                            st.error(
                                f"Unsupported model target: {target}. "
                                "Expected log10_energy_kwh_per_execution."
                            )

                elif model_kind == "greenmetric":
                    predicted_energy = float(model.predict(metrics))
                    energy_values = getattr(model, "energy_values", None)

                    if energy_values is not None:
                        from greenmetricai.scoring import SustainabilityScorer

                        scores = SustainabilityScorer(
                            energy_values
                        ).score(predicted_energy)

                        a, b, c, d = st.columns(4)
                        a.metric(
                            "Predicted energy",
                            format_energy(predicted_energy),
                        )
                        b.metric(
                            "Impact index",
                            f"{scores['impact_index']:.1f}/100",
                        )
                        c.metric(
                            "Sustainability",
                            f"{scores['sustainability_score']:.1f}/100",
                        )
                        d.metric("Class", scores["classification"])
                    else:
                        st.metric(
                            "Predicted energy",
                            format_energy(predicted_energy),
                        )

                st.subheader("Static Features")
                st.dataframe(
                    pd.DataFrame([feature_values]).T.rename(
                        columns={0: "value"}
                    ),
                    use_container_width=True,
                )

                if model_kind == "greenmetric":
                    st.subheader("Model Validation Metrics")
                    st.json(getattr(model, "metrics", {}))
                else:
                    st.caption(
                        f"Model target: {model['target']} · "
                        f"Features used: {len(model['features'])}"
                    )

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")


# ---------------------------------------------------------
# SHAP
# ---------------------------------------------------------
with shap_tab:
    st.subheader("Model Explainability")

    if model is None:
        st.info("SHAP is unavailable until a compatible model is loaded.")

    elif model_kind == "random_forest":
        try:
            import shap

            feature_names = list(model["features"])
            rf_model = model["model"]

            if hasattr(rf_model, "feature_importances_"):
                importance = pd.DataFrame({
                    "feature": feature_names,
                    "importance": rf_model.feature_importances_,
                }).sort_values("importance", ascending=False)

                fig = px.bar(
                    importance.head(12).sort_values("importance"),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Random Forest Feature Importance",
                    labels={
                        "importance": "Feature importance",
                        "feature": "Feature",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "This chart shows built-in Random Forest feature "
                    "importance, not SHAP values. SHAP values require "
                    "compatible input data."
                )

                if os.path.exists(DATA):
                    measured_df = pd.read_csv(DATA)
                    missing = [
                        name for name in feature_names
                        if name not in measured_df.columns
                    ]

                    if missing:
                        st.warning(
                            "The measured dataset does not contain all "
                            "model features required for SHAP: "
                            + ", ".join(missing)
                        )
                    else:
                        X = measured_df[feature_names].apply(
                            pd.to_numeric, errors="coerce"
                        ).dropna()

                        if X.empty:
                            st.warning(
                                "No complete numeric rows are available "
                                "for SHAP."
                            )
                        else:
                            sample = X.head(200)
                            explainer = shap.TreeExplainer(rf_model)
                            shap_values = explainer.shap_values(sample)
                            values = np.asarray(shap_values)

                            if values.ndim == 3:
                                values = values[0]

                            mean_abs = np.abs(values).mean(axis=0)
                            shap_importance = pd.DataFrame({
                                "feature": feature_names,
                                "mean_abs_shap": mean_abs,
                            }).sort_values(
                                "mean_abs_shap", ascending=False
                            )

                            st.subheader("SHAP Importance")
                            shap_fig = px.bar(
                                shap_importance.head(12).sort_values(
                                    "mean_abs_shap"
                                ),
                                x="mean_abs_shap",
                                y="feature",
                                orientation="h",
                                title="Mean Absolute SHAP Values",
                                labels={
                                    "mean_abs_shap": (
                                        "Mean absolute SHAP value"
                                    ),
                                    "feature": "Feature",
                                },
                            )
                            st.plotly_chart(
                                shap_fig, use_container_width=True
                            )
                            st.dataframe(
                                shap_importance,
                                use_container_width=True,
                                hide_index=True,
                            )
                else:
                    st.info(
                        f"Measured dataset not found: {DATA}. "
                        "Built-in feature importance is available; SHAP "
                        "values need compatible measured feature rows."
                    )
            else:
                st.warning(
                    "The loaded model does not expose Random Forest "
                    "feature_importances_."
                )

        except ImportError:
            st.warning(
                "The `shap` package is not installed. Install it in the "
                "project environment to calculate SHAP values."
            )
        except Exception as exc:
            st.error(f"SHAP analysis failed: {exc}")

    else:
        if not os.path.exists(DATA):
            st.warning(
                f"Measured dataset not found: {DATA}. "
                "SHAP analysis requires compatible measured data."
            )
        else:
            try:
                from greenmetricai.explain import global_shap

                measured_df = pd.read_csv(DATA)
                importance, *_ = global_shap(model, measured_df)

                fig = px.bar(
                    importance.head(12).sort_values("mean_abs_shap"),
                    x="mean_abs_shap",
                    y="feature",
                    orientation="h",
                    title="Feature Importance from Measured Data",
                    labels={
                        "mean_abs_shap": "Mean Absolute SHAP Value",
                        "feature": "Feature",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(importance, use_container_width=True)

            except Exception as exc:
                st.error(f"SHAP analysis failed: {exc}")


# ---------------------------------------------------------
# ENERGY VALIDATION
# ---------------------------------------------------------
with energy_tab:
    st.subheader("Original vs Optimized Code")
    st.write(
        "Compare repeated energy and runtime measurements for "
        "original and optimized MBPP task implementations."
    )

    # Build the list of available benchmark datasets.
    data_sources = {}

    if os.path.exists(OPTIMIZATION_DATA):
        data_sources["Historical benchmark results"] = OPTIMIZATION_DATA

    if os.path.exists(DEMO_OPTIMIZATION_DATA):
        data_sources["Latest demo benchmark"] = DEMO_OPTIMIZATION_DATA

    if not data_sources:
        st.info(
            "No optimization results are available yet. Run "
            "`python -m scripts.compare_optimization --task-id 2` "
            "to generate the local comparison CSV."
        )
    else:
        selected_source = st.radio(
            "Benchmark dataset",
            options=list(data_sources.keys()),
            horizontal=True,
        )
        selected_path = data_sources[selected_source]
        st.caption(f"Loaded measurements from: {selected_path}")

        try:
            results = pd.read_csv(selected_path)

            required = {
                "task_id",
                "version",
                "run",
                "runtime_sec_per_execution",
                "energy_kwh_per_execution",
                "carbon_emissions_g_per_execution",
            }
            missing = required - set(results.columns)

            if missing:
                st.error(
                    "The results file is missing required columns: "
                    + ", ".join(sorted(missing))
                )
            else:
                results["version"] = (
                    results["version"].astype(str).str.lower()
                )
                results["task_id"] = pd.to_numeric(
                    results["task_id"], errors="coerce"
                )
                results["run"] = pd.to_numeric(
                    results["run"], errors="coerce"
                )

                for col in [
                    "runtime_sec_per_execution",
                    "energy_kwh_per_execution",
                    "carbon_emissions_g_per_execution",
                ]:
                    results[col] = pd.to_numeric(
                        results[col], errors="coerce"
                    )

                for col in [
                    "candidate_different",
                    "original_tests_passed",
                    "candidate_tests_passed",
                ]:
                    if col in results.columns:
                        results[col] = (
                            results[col].astype(str).str.lower().eq("true")
                        )

                valid_tasks = sorted(
                    results["task_id"].dropna().unique().tolist()
                )

                if not valid_tasks:
                    st.warning(
                        "No valid task IDs were found in the results file."
                    )
                else:
                    selected_task = st.selectbox(
                        "MBPP task ID",
                        valid_tasks,
                        format_func=lambda x: str(int(x)),
                    )

                    # Show the saved original and accepted candidate code.
                    artifact_path = (
                        f"data/processed/optimization_code_task_"
                        f"{int(selected_task)}.json"
                    )

                    st.subheader("Original vs Optimized Code")

                    if os.path.exists(artifact_path):
                        with open(
                            artifact_path, "r", encoding="utf-8"
                        ) as artifact_file:
                            code_artifact = json.load(artifact_file)

                        if code_artifact.get("candidate_different"):
                            st.success(
                                "The accepted candidate differs from "
                                "the original implementation."
                            )
                        else:
                            st.warning(
                                "The accepted candidate is unchanged. "
                                "This is not a successful code optimization."
                            )

                        st.write(
                            "Original MBPP validation: "
                            + (
                                "Passed"
                                if code_artifact.get("original_tests_passed")
                                else "Failed"
                            )
                        )
                        st.write(
                            "Optimized candidate validation: "
                            + (
                                "Passed"
                                if code_artifact.get("candidate_tests_passed")
                                else "Failed"
                            )
                        )

                        original_col, optimized_col = st.columns(2)

                        with original_col:
                            st.markdown("**Original Python code**")
                            st.code(
                                code_artifact["original_code"],
                                language="python",
                            )

                        with optimized_col:
                            st.markdown("**Optimized Python code**")
                            st.code(
                                code_artifact["optimized_code"],
                                language="python",
                            )

                        st.caption(
                            "Code is loaded from the saved benchmark artifact. "
                            "It is not regenerated when the dashboard opens."
                        )
                    else:
                        st.info(
                            "No saved code artifact exists for this task yet. "
                            "Run the benchmark to save the original and "
                            "accepted optimized code."
                        )

                    filtered = results[
                        results["task_id"] == selected_task
                    ].copy()

                    versions_present = set(filtered["version"])

                    if not {"original", "optimized"}.issubset(
                        versions_present
                    ):
                        st.warning(
                            "Both original and optimized measurements "
                            "are required for comparison."
                        )
                    else:
                        summary = (
                            filtered.groupby("version")
                            .agg(
                                runs=("run", "nunique"),
                                mean_energy=(
                                    "energy_kwh_per_execution", "mean"
                                ),
                                median_energy=(
                                    "energy_kwh_per_execution", "median"
                                ),
                                mean_runtime=(
                                    "runtime_sec_per_execution", "mean"
                                ),
                                median_runtime=(
                                    "runtime_sec_per_execution", "median"
                                ),
                                mean_carbon=(
                                    "carbon_emissions_g_per_execution", "mean"
                                ),
                            )
                            .reindex(["original", "optimized"])
                        )

                        original_energy = summary.loc[
                            "original", "mean_energy"
                        ]
                        optimized_energy = summary.loc[
                            "optimized", "mean_energy"
                        ]
                        original_runtime = summary.loc[
                            "original", "mean_runtime"
                        ]
                        optimized_runtime = summary.loc[
                            "optimized", "mean_runtime"
                        ]

                        energy_change = None
                        if (
                            pd.notna(original_energy)
                            and original_energy > 0
                            and pd.notna(optimized_energy)
                        ):
                            energy_change = (
                                (original_energy - optimized_energy)
                                / original_energy
                            ) * 100

                        runtime_change = None
                        if (
                            pd.notna(original_runtime)
                            and original_runtime > 0
                            and pd.notna(optimized_runtime)
                        ):
                            runtime_change = (
                                (original_runtime - optimized_runtime)
                                / original_runtime
                            ) * 100

                        a, b, c, d = st.columns(4)
                        a.metric(
                            "Original mean energy",
                            format_energy(original_energy),
                        )
                        b.metric(
                            "Optimized mean energy",
                            format_energy(optimized_energy),
                        )

                        if energy_change is None:
                            c.metric("Energy reduction", "N/A")
                        else:
                            c.metric(
                                "Energy reduction",
                                f"{energy_change:.2f}%",
                                help=(
                                    "Positive means lower optimized energy; "
                                    "negative means higher optimized energy."
                                ),
                            )

                        if runtime_change is None:
                            d.metric("Runtime reduction", "N/A")
                        else:
                            d.metric(
                                "Runtime reduction",
                                f"{runtime_change:.2f}%",
                                help=(
                                    "Positive means faster optimized "
                                    "execution; negative means slower "
                                    "execution."
                                ),
                            )

                        st.caption(
                            f"Task {int(selected_task)} · "
                            f"{len(filtered)} measurement rows"
                        )

                        st.subheader("Mean Energy per Execution")
                        energy_chart = summary.reset_index()[
                            ["version", "mean_energy"]
                        ]
                        fig_energy = px.bar(
                            energy_chart,
                            x="version",
                            y="mean_energy",
                            color="version",
                            title="Original vs Optimized Energy",
                            labels={
                                "version": "Code version",
                                "mean_energy": (
                                    "Mean energy (kWh/execution)"
                                ),
                            },
                        )
                        st.plotly_chart(
                            fig_energy, use_container_width=True
                        )

                        st.subheader("Runtime per Measurement Run")
                        runtime_data = filtered.dropna(
                            subset=["runtime_sec_per_execution"]
                        ).sort_values(["run", "version"])

                        fig_runtime = px.line(
                            runtime_data,
                            x="run",
                            y="runtime_sec_per_execution",
                            color="version",
                            markers=True,
                            title="Runtime Across Repeated Runs",
                            labels={
                                "run": "Run number",
                                "runtime_sec_per_execution": (
                                    "Runtime (seconds/execution)"
                                ),
                                "version": "Code version",
                            },
                        )
                        st.plotly_chart(
                            fig_runtime, use_container_width=True
                        )

                        st.subheader("Mean Carbon Emissions")
                        carbon_chart = summary.reset_index()[
                            ["version", "mean_carbon"]
                        ]
                        fig_carbon = px.bar(
                            carbon_chart,
                            x="version",
                            y="mean_carbon",
                            color="version",
                            title="Estimated Carbon Emissions per Execution",
                            labels={
                                "version": "Code version",
                                "mean_carbon": (
                                    "Mean carbon (g/execution)"
                                ),
                            },
                        )
                        st.plotly_chart(
                            fig_carbon, use_container_width=True
                        )

                        st.subheader("Correctness and Candidate Status")
                        status_columns = [
                            col for col in [
                                "candidate_different",
                                "original_tests_passed",
                                "candidate_tests_passed",
                            ]
                            if col in filtered.columns
                        ]

                        if status_columns:
                            status = filtered[
                                ["task_id", "version", "run"] + status_columns
                            ].drop_duplicates()
                            st.dataframe(
                                status,
                                use_container_width=True,
                                hide_index=True,
                            )

                        st.subheader("Measurement Summary")
                        st.dataframe(
                            summary.reset_index(),
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.subheader("Individual Measurements")
                        display_columns = [
                            col for col in [
                                "task_id",
                                "version",
                                "run",
                                "measurement_repeats",
                                "measurement_class",
                                "runtime_sec_per_execution",
                                "energy_kwh_per_execution",
                                "carbon_emissions_g_per_execution",
                                "wall_clock_sec",
                                "platform",
                                "python_version",
                                "cpu_model",
                                "ram_gb",
                                "candidate_different",
                                "original_tests_passed",
                                "candidate_tests_passed",
                            ]
                            if col in filtered.columns
                        ]

                        st.dataframe(
                            filtered[display_columns].sort_values(
                                ["run", "version"]
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.info(
                            "Interpretation: these results describe the "
                            "selected MBPP task and recorded machine. "
                            "They do not establish that optimization will "
                            "reduce energy for all programs. Compare "
                            "repeated runs and investigate measurement "
                            "variability before drawing broader conclusions."
                        )

        except Exception as exc:
            st.error(f"Could not load optimization results: {exc}")
