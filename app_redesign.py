"""GreenMetricAI dashboard (redesign).

Keeps the existing architecture: same model artifacts, same data files,
same feature extractor. Only the presentation layer is rebuilt.
"""
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from greenmetricai.features import extract_code_metrics

# ---------------------------------------------------------------------
# Paths (unchanged from the previous app.py)
# ---------------------------------------------------------------------
MODEL_CANDIDATE = "models/greenmetric_rf_final.joblib"
MODEL_TRAINED = "models/greenmetric_model.joblib"
DATA = "data/processed/mbpp_100_final_5sec.csv"
OPTIMIZATION_DATA = "data/processed/optimization_comparison.csv"
DEMO_OPTIMIZATION_DATA = "data/processed/optimization_comparison_ui_demo.csv"
ARTIFACT_PATTERN = "data/processed/optimization_code_task_{task}.json"

GREEN = "#2ecc71"
TEAL = "#1abc9c"
AMBER = "#f5b041"
RED = "#e74c3c"
COLORS = {"original": "#7f8c8d", "optimized": GREEN}

st.set_page_config(
    page_title="GreenMetricAI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1280px;}
.hero {
  background: linear-gradient(120deg, #0b3d2e 0%, #11694b 55%, #1abc9c 100%);
  border-radius: 18px; padding: 28px 34px; color: #fff; margin-bottom: 18px;
}
.hero h1 {margin: 0; font-size: 2.1rem; letter-spacing: .3px;}
.hero p {margin: 6px 0 0 0; opacity: .92; font-size: 1.0rem;}
.kpi {
  background: rgba(46,204,113,.08); border: 1px solid rgba(46,204,113,.35);
  border-radius: 14px; padding: 14px 16px; height: 100%;
}
.kpi .label {font-size: .78rem; text-transform: uppercase; letter-spacing: .8px; opacity: .75;}
.kpi .value {font-size: 1.7rem; font-weight: 700; margin-top: 2px;}
.kpi .sub {font-size: .78rem; opacity: .7; margin-top: 2px;}
.pill {display:inline-block; padding: 3px 12px; border-radius: 999px;
  font-size: .78rem; font-weight: 600; margin-right: 6px;}
.pill.ok {background: rgba(46,204,113,.18); color: #2ecc71;}
.pill.warn {background: rgba(245,176,65,.18); color: #f5b041;}
.pill.bad {background: rgba(231,76,60,.18); color: #e74c3c;}
.pill.info {background: rgba(52,152,219,.18); color: #5dade2;}
.note {border-left: 4px solid #f5b041; background: rgba(245,176,65,.08);
  padding: 10px 14px; border-radius: 6px; font-size: .88rem;}
</style>
""",
    unsafe_allow_html=True,
)


def kpi(label, value, sub=""):
    st.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def pill(text, kind="info"):
    return f'<span class="pill {kind}">{text}</span>'


def format_energy(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.3e} kWh"


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------
@st.cache_resource
def load_candidate_model(path):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("Expected a model artifact dictionary.")
    for key in ("model", "features", "target"):
        if key not in bundle:
            raise ValueError(f"Model artifact is missing '{key}'.")
    return bundle


@st.cache_resource
def load_trained_model(path):
    from greenmetricai.model import GreenMetricModel

    return GreenMetricModel.load(path)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data
def load_results(path):
    df = pd.read_csv(path)
    required = {
        "task_id", "version", "runtime_sec_per_execution",
        "energy_kwh_per_execution", "carbon_emissions_g_per_execution",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    df["version"] = df["version"].astype(str).str.lower()
    df["task_id"] = pd.to_numeric(df["task_id"], errors="coerce")
    for col in (
        "runtime_sec_per_execution",
        "energy_kwh_per_execution",
        "carbon_emissions_g_per_execution",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("candidate_different", "original_tests_passed",
                "candidate_tests_passed"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().eq("true")
    return df.dropna(subset=["task_id"])


def load_artifact(task_id):
    path = ARTIFACT_PATTERN.format(task=int(task_id))
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def summarize(results):
    """Per-task original-vs-optimized summary with variability."""
    rows = []
    for task_id, grp in results.groupby("task_id"):
        orig = grp[grp["version"] == "original"]["energy_kwh_per_execution"].dropna()
        opt = grp[grp["version"] == "optimized"]["energy_kwh_per_execution"].dropna()
        if orig.empty or opt.empty:
            continue
        o_med, p_med = orig.median(), opt.median()
        reduction = (o_med - p_med) / o_med * 100 if o_med else np.nan
        cv = orig.std(ddof=1) / orig.mean() * 100 if len(orig) > 1 and orig.mean() else np.nan
        # Heuristic: change smaller than the original's own run-to-run
        # spread cannot be separated from measurement noise.
        within_noise = bool(pd.notna(cv) and abs(reduction) < cv)
        rt_o = grp[grp["version"] == "original"]["runtime_sec_per_execution"].median()
        rt_p = grp[grp["version"] == "optimized"]["runtime_sec_per_execution"].median()
        row = {
            "task_id": int(task_id),
            "runs (orig/opt)": f"{len(orig)}/{len(opt)}",
            "original_energy_kwh": o_med,
            "optimized_energy_kwh": p_med,
            "energy_reduction_%": reduction,
            "original_run_CV_%": cv,
            "within_noise": within_noise,
            "original_runtime_s": rt_o,
            "optimized_runtime_s": rt_p,
        }
        if "candidate_different" in grp.columns:
            row["candidate_different"] = bool(grp["candidate_different"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("task_id") if rows else pd.DataFrame()


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
model, model_kind, model_error = None, None, None
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

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
sources = {}
if os.path.exists(OPTIMIZATION_DATA):
    sources["Historical benchmark results"] = OPTIMIZATION_DATA
if os.path.exists(DEMO_OPTIMIZATION_DATA):
    sources["Latest demo benchmark"] = DEMO_OPTIMIZATION_DATA

with st.sidebar:
    st.markdown("## 🌱 GreenMetricAI")
    st.caption("Sustainable Python code optimization")
    st.divider()
    if sources:
        source_name = st.radio("Benchmark dataset", list(sources.keys()))
        results = None
        try:
            results = load_results(sources[source_name])
        except Exception as exc:
            st.error(f"Could not load results: {exc}")
    else:
        source_name, results = None, None
        st.info("No optimization results found yet.")
    st.divider()
    st.markdown("**System status**")
    st.markdown(
        pill("Model loaded", "ok") if model is not None
        else pill("Model missing", "bad"),
        unsafe_allow_html=True,
    )
    st.markdown(
        pill("Benchmark data", "ok") if results is not None
        else pill("No benchmark data", "warn"),
        unsafe_allow_html=True,
    )
    if model_error:
        st.caption(model_error)

summary = summarize(results) if results is not None else pd.DataFrame()

# ---------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>🌱 GreenMetricAI</h1>
  <p>Predict the energy cost of Python code from its source, explain it with SHAP,
  optimize it with an LLM, then verify correctness and measure the real energy change.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab_over, tab_analyze, tab_opt, tab_bench, tab_explain, tab_tools, tab_method = st.tabs(
    ["Overview", "Analyze code", "Optimization", "Benchmarks",
     "Explainability", "Tool comparison", "Method & limits"]
)

# ---------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    n_tasks = len(summary)
    changed = int(summary["candidate_different"].sum()) if "candidate_different" in summary else 0
    if n_tasks:
        valid = summary[~summary["within_noise"]] if "within_noise" in summary else summary
        med_red = summary["energy_reduction_%"].median()
        clear_wins = int((valid["energy_reduction_%"] > 0).sum())
    else:
        med_red, clear_wins = np.nan, 0
    with c1:
        kpi("Tasks benchmarked", f"{n_tasks}", "original vs optimized")
    with c2:
        kpi("Median energy change",
            "—" if pd.isna(med_red) else f"{med_red:+.1f}%",
            "positive = less energy (measured)")
    with c3:
        kpi("Clear improvements", f"{clear_wins}",
            "reduction larger than run-to-run noise")
    with c4:
        kpi("Candidates changed", f"{changed}/{n_tasks}" if n_tasks else "—",
            "unchanged candidates are not optimizations")

    st.write("")
    if n_tasks:
        left, right = st.columns([3, 2])
        with left:
            st.markdown("#### Energy per execution: original vs optimized")
            long = summary.melt(
                id_vars="task_id",
                value_vars=["original_energy_kwh", "optimized_energy_kwh"],
                var_name="version", value_name="energy_kwh",
            )
            long["version"] = long["version"].str.replace("_energy_kwh", "")
            fig = px.bar(
                long, x=long["task_id"].astype(str), y="energy_kwh",
                color="version", barmode="group",
                color_discrete_map=COLORS,
                labels={"x": "MBPP task", "energy_kwh": "Median kWh / execution"},
            )
            fig.update_layout(height=380, margin=dict(t=10, b=10),
                              legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("#### Reduction by task")
            fig2 = go.Figure(go.Bar(
                x=summary["energy_reduction_%"],
                y=summary["task_id"].astype(str),
                orientation="h",
                marker_color=[GREEN if v > 0 else RED
                              for v in summary["energy_reduction_%"]],
            ))
            fig2.update_layout(height=380, margin=dict(t=10, b=10),
                               xaxis_title="Energy reduction (%)",
                               yaxis_title="MBPP task")
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown(
            '<div class="note"><b>Read with care:</b> these are CodeCarbon '
            "<i>estimates</i>, and the values are tiny. A change smaller than the "
            "run-to-run variation is flagged as within noise on the Benchmarks tab.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "Run `python -m scripts.compare_optimization --task-id 2` to "
            "generate results, then refresh."
        )

    st.markdown("#### Pipeline")
    steps = ["Source code", "Static features (Radon + custom)", "RF energy prediction",
             "SHAP", "Gemini optimization", "MBPP tests", "CodeCarbon re-measurement"]
    st.markdown(" → ".join(f"**{s}**" for s in steps))

# ---------------------------------------------------------------------
# ANALYZE
# ---------------------------------------------------------------------
with tab_analyze:
    st.markdown("### Analyze Python code")
    if model is None:
        st.info("Prediction is unavailable until a compatible model artifact is present.")
    else:
        code = st.text_area(
            "Python code", height=260,
            value="def square_sum(n):\n    return sum(i*i for i in range(n))",
        )
        if st.button("Analyze", type="primary"):
            try:
                metrics = extract_code_metrics(code)
                values = metrics.to_dict()
                predicted = None
                if model_kind == "random_forest":
                    names = list(model["features"])
                    missing = [n for n in names if n not in values]
                    if missing:
                        st.error("Feature extractor did not produce: " + ", ".join(missing))
                    elif str(model["target"]) != "log10_energy_kwh_per_execution":
                        st.error(f"Unsupported model target: {model['target']}")
                    else:
                        X = pd.DataFrame([[values[n] for n in names]], columns=names)
                        log_pred = float(model["model"].predict(X)[0])
                        if np.isfinite(log_pred) and -300 <= log_pred <= 100:
                            predicted = 10.0 ** log_pred
                        else:
                            st.error("Model returned an invalid log-energy prediction.")
                else:
                    predicted = float(model.predict(metrics))

                if predicted is not None:
                    a, b = st.columns([1, 2])
                    with a:
                        kpi("Predicted energy", format_energy(predicted),
                            "per execution - an estimate, not a measurement")
                    with b:
                        st.markdown(
                            pill("Static prediction", "info")
                            + pill("Limited accuracy (CV R² ≈ 0.08)", "warn"),
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            "Use for relative comparison between code variants, "
                            "not as an absolute energy figure."
                        )
                st.markdown("#### Static features")
                feat = pd.DataFrame(values.items(), columns=["feature", "value"])
                st.dataframe(feat, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

# ---------------------------------------------------------------------
# OPTIMIZATION (per task)
# ---------------------------------------------------------------------
with tab_opt:
    st.markdown("### Original vs optimized code")
    if summary.empty:
        st.info("No optimization results available.")
    else:
        task = st.selectbox("MBPP task", summary["task_id"].tolist(),
                            format_func=lambda x: f"Task {x}")
        art = load_artifact(task)
        row = summary[summary["task_id"] == task].iloc[0]
        tags = ""
        if art:
            tags += pill("Original tests passed", "ok") if art.get("original_tests_passed") \
                else pill("Original tests failed", "bad")
            tags += pill("Candidate tests passed", "ok") if art.get("candidate_tests_passed") \
                else pill("Candidate tests failed", "bad")
            tags += pill("Code changed", "ok") if art.get("candidate_different") \
                else pill("Unchanged - not an optimization", "warn")
        tags += pill("Within noise", "warn") if row["within_noise"] else pill("Beyond noise", "ok")
        st.markdown(tags, unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi("Measured energy change", f"{row['energy_reduction_%']:+.1f}%", "median of runs")
        with k2:
            kpi("Original", format_energy(row["original_energy_kwh"]), "measured median")
        with k3:
            kpi("Optimized", format_energy(row["optimized_energy_kwh"]), "measured median")
        with k4:
            pe = art.get("predicted_energy_kwh") if art else None
            kpi("Model-predicted (optimized)", format_energy(pe), "prediction, not measurement")

        if art:
            if art.get("candidate_validation_reason"):
                st.warning(f"Validation note: {art['candidate_validation_reason']}")
            lc, rc = st.columns(2)
            with lc:
                st.markdown("**Original**")
                st.code(art.get("original_code", ""), language="python")
            with rc:
                st.markdown("**Optimized**")
                st.code(art.get("optimized_code", ""), language="python")

            # Radon / static metric comparison, computed live from the saved code
            try:
                m_o = extract_code_metrics(art["original_code"]).to_dict()
                m_p = extract_code_metrics(art["optimized_code"]).to_dict()
                keys = [k for k in ("cc", "mi", "hv", "loc", "loop_depth",
                                    "branching_factor", "technical_debt_index")
                        if k in m_o and k in m_p]
                comp = pd.DataFrame({
                    "metric": keys,
                    "original": [m_o[k] for k in keys],
                    "optimized": [m_p[k] for k in keys],
                })
                comp["change"] = comp["optimized"] - comp["original"]
                st.markdown("#### Static metrics (Radon-based)")
                st.dataframe(comp, use_container_width=True, hide_index=True)
                st.caption("cc = cyclomatic complexity, mi = maintainability index "
                           "(higher is better), hv = Halstead volume, loc = lines of code.")
            except Exception as exc:
                st.caption(f"Static metric comparison unavailable: {exc}")
        else:
            st.info(f"No saved code artifact for task {task}.")

        run_df = results[results["task_id"] == task]
        st.markdown("#### Individual runs")
        fig = px.box(
            run_df, x="version", y="energy_kwh_per_execution", color="version",
            points="all", color_discrete_map=COLORS,
            labels={"energy_kwh_per_execution": "kWh / execution"},
        )
        fig.update_layout(height=340, showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# BENCHMARKS (across tasks)
# ---------------------------------------------------------------------
with tab_bench:
    st.markdown("### Results across tasks")
    if summary.empty:
        st.info("No benchmark data available.")
    else:
        show_noise = st.toggle("Include results within noise", value=True)
        view = summary if show_noise else summary[~summary["within_noise"]]
        if view.empty:
            st.warning("Every result is within measurement noise.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                kpi("Mean reduction", f"{view['energy_reduction_%'].mean():+.1f}%",
                    f"over {len(view)} tasks")
            with c2:
                kpi("Median reduction", f"{view['energy_reduction_%'].median():+.1f}%", "")
            with c3:
                kpi("Spread (std)", f"{view['energy_reduction_%'].std(ddof=1):.1f} pts"
                    if len(view) > 1 else "—", "variability between tasks")
            fig = px.scatter(
                view, x="original_runtime_s", y="energy_reduction_%",
                color=view["within_noise"].map({True: "within noise", False: "beyond noise"}),
                hover_data=["task_id"],
                color_discrete_map={"within noise": AMBER, "beyond noise": GREEN},
                labels={"original_runtime_s": "Original runtime (s)",
                        "energy_reduction_%": "Energy reduction (%)"},
            )
            fig.add_hline(y=0, line_dash="dot")
            fig.update_layout(height=360, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                view.style.format({
                    "original_energy_kwh": "{:.3e}", "optimized_energy_kwh": "{:.3e}",
                    "energy_reduction_%": "{:+.1f}", "original_run_CV_%": "{:.1f}",
                    "original_runtime_s": "{:.2e}", "optimized_runtime_s": "{:.2e}",
                }),
                use_container_width=True, hide_index=True,
            )
        st.caption("'Within noise' = the change is smaller than the original code's own "
                   "run-to-run variation (CV). This is a screening heuristic, not a "
                   "significance test; add more runs for confidence intervals.")

# ---------------------------------------------------------------------
# EXPLAINABILITY
# ---------------------------------------------------------------------
with tab_explain:
    st.markdown("### Model explainability")
    if model is None:
        st.info("Explainability is unavailable until a model is loaded.")
    elif model_kind != "random_forest":
        try:
            from greenmetricai.explain import global_shap

            df = load_csv(DATA)
            importance, *_ = global_shap(model, df)
            fig = px.bar(importance.head(12).sort_values("mean_abs_shap"),
                         x="mean_abs_shap", y="feature", orientation="h")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"SHAP analysis failed: {exc}")
    else:
        names = list(model["features"])
        rf = model["model"]
        left, right = st.columns(2)
        with left:
            if hasattr(rf, "feature_importances_"):
                imp = pd.DataFrame({"feature": names, "importance": rf.feature_importances_})
                fig = px.bar(imp.sort_values("importance").tail(12), x="importance",
                             y="feature", orientation="h",
                             title="Random Forest feature importance",
                             color_discrete_sequence=[TEAL])
                st.plotly_chart(fig, use_container_width=True)
        with right:
            try:
                import shap

                if not os.path.exists(DATA):
                    st.info(f"Measured dataset not found: {DATA}")
                else:
                    df = load_csv(DATA)
                    missing = [n for n in names if n not in df.columns]
                    if missing:
                        st.warning("Dataset lacks model features: " + ", ".join(missing))
                    else:
                        X = df[names].apply(pd.to_numeric, errors="coerce").dropna().head(200)
                        sv = np.asarray(shap.TreeExplainer(rf).shap_values(X))
                        if sv.ndim == 3:
                            sv = sv[0]
                        sh = pd.DataFrame({"feature": names,
                                           "mean_abs_shap": np.abs(sv).mean(axis=0)})
                        fig = px.bar(sh.sort_values("mean_abs_shap").tail(12),
                                     x="mean_abs_shap", y="feature", orientation="h",
                                     title="Mean |SHAP| (first 200 rows)",
                                     color_discrete_sequence=[GREEN])
                        st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.warning("The shap package is not installed.")
            except Exception as exc:
                st.error(f"SHAP analysis failed: {exc}")
        st.markdown(
            '<div class="note">SHAP shows how much each feature influenced the '
            "model's predictions. It does <b>not</b> prove that a feature causes "
            "higher or lower energy use.</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------
# TOOL COMPARISON
# ---------------------------------------------------------------------
with tab_tools:
    st.markdown("### What each tool contributes")
    tools = pd.DataFrame([
        {"Tool": "Radon", "What it measures": "Static complexity: cyclomatic complexity, "
         "Halstead, maintainability index, LOC",
         "Needs execution?": "No", "Energy output?": "No (proxy only)",
         "Role here": "Features for the energy model; before/after code-quality comparison"},
        {"Tool": "CodeCarbon", "What it measures": "Estimated energy (kWh) and CO₂e of a run, "
         "from hardware power estimates",
         "Needs execution?": "Yes", "Energy output?": "Yes (estimate)",
         "Role here": "Ground-truth label for training; original vs optimized re-measurement"},
        {"Tool": "GreenMetricAI model", "What it measures": "Predicted log10 energy from 14 "
         "static features (Random Forest)",
         "Needs execution?": "No", "Energy output?": "Yes (prediction)",
         "Role here": "Pre-execution screening; guides the LLM optimizer"},
        {"Tool": "MBPP tests", "What it measures": "Functional correctness",
         "Needs execution?": "Yes", "Energy output?": "No",
         "Role here": "Gate: rejects candidates that fail"},
    ])
    st.dataframe(tools, use_container_width=True, hide_index=True)
    st.caption("This table describes each tool's role. It contains no measured values.")

    if not summary.empty:
        st.markdown("#### Does static complexity track measured energy?")
        rows = []
        for t in summary["task_id"]:
            art = load_artifact(t)
            if not art:
                continue
            try:
                mo = extract_code_metrics(art["original_code"]).to_dict()
                mp = extract_code_metrics(art["optimized_code"]).to_dict()
            except Exception:
                continue
            r = summary[summary["task_id"] == t].iloc[0]
            rows.append({
                "task_id": t,
                "Δcc (Radon)": mp.get("cc", np.nan) - mo.get("cc", np.nan),
                "Δmi (Radon)": mp.get("mi", np.nan) - mo.get("mi", np.nan),
                "measured energy change %": -r["energy_reduction_%"],
            })
        if rows:
            cmp_df = pd.DataFrame(rows)
            fig = px.scatter(cmp_df, x="Δcc (Radon)", y="measured energy change %",
                             hover_data=["task_id"],
                             labels={"measured energy change %":
                                     "Measured energy change (%, negative = saved)"})
            fig.add_hline(y=0, line_dash="dot")
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("With only a few tasks this is illustrative, not a statistical "
                       "claim. Expand the task count before drawing conclusions.")
        else:
            st.info("Saved code artifacts are needed for this comparison.")

# ---------------------------------------------------------------------
# METHOD & LIMITS
# ---------------------------------------------------------------------
with tab_method:
    st.markdown("### Method and limitations")
    st.markdown(
        """
**Data.** Real MBPP sanitized tasks, with 100 tasks and 5 runs each in the training set. No synthetic data.

**Measurement.** CodeCarbon *estimates* energy. It is not laboratory-grade metering,
and repeated tests run in one process, so very small values should not be read as precise.

**Model.** Random Forest on 14 static features; 5-fold CV R² ≈ 0.085 (baseline 0.0).
Static features alone have limited predictive power, so predictions are best used for
relative comparison. The deployed model is trained on all 100 tasks, so its in-sample
predictions are not cross-validation results.

**Optimization.** Candidates must parse, pass the official MBPP tests, and meet the
predicted-improvement threshold. Unchanged candidates are flagged.

**Predicted vs measured.** The predicted saving comes from the model; the measured saving
comes from CodeCarbon re-measurement. They are reported separately and should not be merged.
"""
    )