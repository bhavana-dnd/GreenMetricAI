
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from greenmetricai.features import extract_code_metrics

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # The optimizer also accepts a key entered in the page if python-dotenv
    # has not been installed yet.
    pass

st.set_page_config(
    page_title="GreenMetricAI",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 GreenMetricAI")
st.caption(
    "Measured MBPP execution data, energy prediction, "
    "SHAP explainability, and behavior-checked code alternatives"
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


def render_local_shap(title, explanation):
    """Render local SHAP contributions without implying measured effects."""
    st.markdown(f"**{title}**")
    if not explanation or not explanation.get("available"):
        reason = explanation.get("reason") if explanation else "No explanation was computed."
        st.info(f"Local SHAP explanation unavailable: {reason}")
        return

    contributions = pd.DataFrame(explanation["features"])
    top = contributions.head(10).sort_values("shap_value")
    fig = px.bar(
        top,
        x="shap_value",
        y="feature",
        color="effect",
        orientation="h",
        color_discrete_map={
            "Raises predicted energy": "#d95f59",
            "Lowers predicted energy": "#20a779",
        },
        labels={
            "shap_value": "SHAP contribution (log10 kWh/execution)",
            "feature": "Code feature",
            "effect": "Effect on prediction",
        },
        title="Largest local feature contributions",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Positive SHAP values raise the model's predicted energy; negative "
        "values lower it. These explain the model prediction, not measured "
        "energy or causal effects."
    )
    st.dataframe(
        contributions,
        use_container_width=True,
        hide_index=True,
        column_config={
            "shap_value": st.column_config.NumberColumn(
                "SHAP contribution (log10 kWh)", format="%.5f"
            )
        },
    )


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

analyze_tab, shap_tab, energy_tab, response_tab = st.tabs(
    ["Analyze", "SHAP", "Energy Validation", "Optimize LLM code"]
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
            "`python -m scripts.compare_optimization --task-id 2 "
            "--reuse-saved-candidate` to measure the saved task-2 candidate "
            "and create the local comparison CSV. This uses fresh local "
            "measurements; it does not recreate another computer's history."
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

                        measured_validation = code_artifact.get(
                            "measured_validation"
                        )
                        if measured_validation:
                            st.subheader("Noise-aware paired validation")
                            evidence_status = measured_validation.get(
                                "status", "unknown"
                            )
                            status_messages = {
                                "confirmed_reduction": (
                                    "Measured reduction cleared the minimum "
                                    "effect threshold with the adjusted interval."
                                ),
                                "below_minimum_effect": (
                                    "The interval places the measured effect "
                                    "below the minimum reduction threshold."
                                ),
                                "inconclusive": (
                                    "The maximum number of pairs was reached "
                                    "without a decisive interval."
                                ),
                                "insufficient_measurements": (
                                    "The maximum number of pairs was reached, "
                                    "but too few produced valid energy values."
                                ),
                            }
                            message = status_messages.get(
                                evidence_status,
                                "More paired measurements are needed.",
                            )
                            if evidence_status == "confirmed_reduction":
                                st.success(message)
                            elif evidence_status in {
                                "below_minimum_effect",
                                "insufficient_measurements",
                            }:
                                st.warning(message)
                            else:
                                st.info(message)

                            lower = measured_validation.get("ci_lower_pct")
                            upper = measured_validation.get("ci_upper_pct")
                            mean_reduction = measured_validation.get(
                                "geometric_mean_reduction_pct"
                            )
                            ci_text = (
                                f"{lower:.2f}% to {upper:.2f}%"
                                if lower is not None and upper is not None
                                else "Not enough valid pairs"
                            )
                            a, b, c = st.columns(3)
                            a.metric(
                                "Paired geometric reduction",
                                (
                                    f"{mean_reduction:.2f}%"
                                    if mean_reduction is not None
                                    else "N/A"
                                ),
                            )
                            b.metric("Adjusted confidence interval", ci_text)
                            c.metric(
                                "Valid pairs",
                                str(measured_validation.get("paired_runs", 0)),
                            )
                            st.caption(
                                "Original and optimized runs are paired by run "
                                "number. The interval uses log energy ratios and "
                                "a Bonferroni adjustment across sequential checks. "
                                "The target is a measured reduction greater than "
                                f"{measured_validation.get('minimum_reduction_pct', 3):.1f}%."
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


# ---------------------------------------------------------
# LLM RESPONSE OPTIMIZATION
# ---------------------------------------------------------
with response_tab:
    st.subheader("Find behavior-preserving code alternatives")
    st.write(
        "Paste Python code from an LLM, provide checks for its behavior, and "
        "GreenMetricAI will generate alternatives, reject candidates that fail "
        "those checks, and rank the survivors by predicted energy."
    )
    st.caption(
        "A working square-sum example and its checks are prefilled. Add a "
        "Gemini API key and confirm the code is trusted to enable the run button."
    )

    st.warning(
        "Your code and checks will execute on this computer in a timed child "
        "process. A timeout is not a security sandbox. Run only code you trust."
    )

    st.session_state.setdefault(
        "response_optimizer_code",
        "def square_sum(n):\n    return sum(i * i for i in range(n))",
    )
    st.session_state.setdefault(
        "response_optimizer_checks",
        "assert square_sum(3) == 5\nassert square_sum(10) == 285",
    )
    response_code = st.text_area(
        "Python code to improve",
        height=240,
        key="response_optimizer_code",
        placeholder="Paste the Python code returned by an LLM...",
    )
    if any(
        marker in response_code.lower()
        for marker in ("read_csv", "pandas", "dataframe", "groupby")
    ):
        st.warning(
            "This looks like a data-processing script. GreenMetricAI's energy "
            "model was trained on measured MBPP code snippets, not large pandas "
            "pipelines, so treat its score as out-of-domain and verify changes "
            "with CodeCarbon on the target workload."
        )
    response_checks = st.text_area(
        "Behavior checks (required)",
        height=140,
        key="response_optimizer_checks",
        help=(
            "Write Python assertions that pass for the original code and every "
            "acceptable alternative. The checks are run in a separate process."
        ),
    )
    working_directory = st.text_input(
        "Project directory for relative data paths",
        value=os.getcwd(),
        key="response_optimizer_working_directory",
        help=(
            "Choose the folder that contains paths referenced by your code, "
            "such as an ml/ directory. The candidate and checks run with this "
            "folder as their working directory."
        ),
    ).strip()
    working_directory_valid = Path(working_directory).expanduser().is_dir()
    if not working_directory_valid:
        st.error("The project directory does not exist. Enter an existing folder.")
    candidate_count = st.slider(
        "Alternatives to generate",
        min_value=1,
        max_value=3,
        value=3,
        key="response_optimizer_count",
    )
    trusted_to_run = st.checkbox(
        "I trust this code and these checks to run on my computer.",
        key="response_optimizer_trusted",
    )

    configured_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if configured_api_key.lower() in {
        "your_key_here",
        "your_api_key_here",
        "replace_me",
    }:
        configured_api_key = ""
    if configured_api_key:
        api_key = configured_api_key
        st.caption("Gemini API key loaded from the environment or .env file.")
    else:
        api_key = st.text_input(
            "Gemini API key",
            type="password",
            key="response_optimizer_api_key",
            help=(
                "Used only for this Streamlit session. It is not saved to the "
                "project files. You can also put GEMINI_API_KEY in a local .env file."
            ),
        ).strip()
    api_key_ready = bool(api_key)
    if model is None:
        st.info("Load a compatible GreenMetricAI model to rank alternatives.")

    requirements_missing = []
    if not response_code.strip():
        requirements_missing.append("paste Python code")
    if not response_checks.strip():
        requirements_missing.append("provide behavior checks")
    if not api_key_ready:
        requirements_missing.append("provide a Gemini API key")
    if not trusted_to_run:
        requirements_missing.append("confirm the code is trusted")
    if model is None:
        requirements_missing.append("load the GreenMetricAI model")
    if not working_directory_valid:
        requirements_missing.append("choose an existing project directory")
    if requirements_missing:
        st.caption("Before you can run it: " + "; ".join(requirements_missing) + ".")

    run_optimizer = st.button(
        "Generate and evaluate alternatives",
        type="primary",
        disabled=(
            not response_code.strip()
            or not response_checks.strip()
            or not trusted_to_run
            or not api_key_ready
            or model is None
            or not working_directory_valid
        ),
        key="response_optimizer_run",
    )

    if run_optimizer:
        try:
            from greenmetricai.optimizer import GeminiProvider
            from greenmetricai.response_optimizer import (
                explain_prediction,
                generate_energy_alternatives,
            )

            with st.spinner(
                f"Generating and checking {candidate_count} alternatives..."
            ):
                result = generate_energy_alternatives(
                    response_code,
                    response_checks,
                    model=model,
                    model_kind=model_kind,
                    provider=GeminiProvider(api_key=api_key),
                    candidate_count=candidate_count,
                    working_directory=working_directory,
                )
                result["original_explanation"] = explain_prediction(
                    response_code,
                    model,
                    model_kind,
                )
                for candidate in result["candidates"]:
                    if candidate.get("passed_checks") and candidate.get(
                        "predicted_energy_kwh"
                    ) is not None:
                        candidate["explanation"] = explain_prediction(
                            candidate["code"],
                            model,
                            model_kind,
                        )
            st.session_state["response_optimizer_result"] = {
                "source_code": response_code,
                "checks": response_checks,
                "candidate_count": candidate_count,
                "working_directory": working_directory,
                "result": result,
            }
        except Exception as exc:
            st.session_state.pop("response_optimizer_result", None)
            st.error(f"Could not evaluate alternatives: {exc}")

    saved = st.session_state.get("response_optimizer_result")
    if saved and (
        saved["source_code"] == response_code
        and saved["checks"] == response_checks
        and saved.get("candidate_count") == candidate_count
        and saved.get("working_directory", os.getcwd()) == working_directory
    ):
        result = saved["result"]
        from greenmetricai.response_optimizer import explain_prediction

        if "original_explanation" not in result:
            result["original_explanation"] = explain_prediction(
                result["original_code"],
                model,
                model_kind,
            )
        st.subheader("Original code baseline")
        st.metric(
            "Predicted energy per execution",
            format_energy(result["original_energy_kwh"]),
        )
        st.caption(
            "This is a model prediction, not a CodeCarbon measurement. "
            "Passing the supplied checks does not prove equivalence for all inputs."
        )
        render_local_shap(
            "Why did the model give the original code this score?",
            result.get("original_explanation"),
        )

        ranked_number = result.get("recommended_number")
        if ranked_number is None:
            st.warning(
                "No alternative both passed the supplied checks and received "
                "a valid energy estimate. Keep the original code."
            )

        for candidate in result["candidates"]:
            number = candidate["number"]
            with st.expander(
                f"Alternative {number}"
                + (" · lowest predicted energy" if number == ranked_number else ""),
                expanded=(number == ranked_number),
            ):
                if not candidate.get("passed_checks"):
                    st.error(candidate.get("reason", "Candidate was rejected."))
                elif candidate.get("predicted_energy_kwh") is None:
                    st.warning(candidate.get("reason", "Energy estimate unavailable."))
                else:
                    if "explanation" not in candidate:
                        candidate["explanation"] = explain_prediction(
                            candidate["code"],
                            model,
                            model_kind,
                        )
                    reduction = candidate["predicted_reduction_pct"]
                    st.success("Supplied checks passed.")
                    a, b = st.columns(2)
                    a.metric(
                        "Predicted energy per execution",
                        format_energy(candidate["predicted_energy_kwh"]),
                    )
                    b.metric(
                        "Predicted change vs original",
                        f"{reduction:+.2f}%",
                        help="Positive means lower predicted energy.",
                    )
                    if reduction <= 0:
                        st.info(
                            "This candidate passed the checks, but its predicted "
                            "energy is not lower than the original."
                        )
                    st.code(candidate["code"], language="python")
                    render_local_shap(
                        f"Why did the model score alternative {number} this way?",
                        candidate.get("explanation"),
                    )
                    st.download_button(
                        "Download this alternative",
                        data=candidate["code"],
                        file_name=f"greenmetricai_alternative_{number}.py",
                        mime="text/x-python",
                        key=f"download_response_candidate_{number}",
                    )

        st.caption(
            "The alternatives are ranked using GreenMetricAI predictions only. "
            "A measured energy reduction and an LLM-generation break-even estimate "
            "are not available in this workflow yet."
        )
