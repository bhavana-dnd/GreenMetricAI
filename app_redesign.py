"""GreenMetricAI dashboard - simple flow.

1. Paste Python code
2. See plain-language results (energy estimate, complexity, why)
3. Let the AI optimize it, with safety checks
4. Compare Radon, GreenMetricAI and CodeCarbon on YOUR code
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from greenmetricai.features import extract_code_metrics

MODEL_CANDIDATE = "models/greenmetric_rf_final.joblib"
MODEL_TRAINED = "models/greenmetric_model.joblib"
DATA = "data/processed/mbpp_100_final_5sec.csv"
OPTIMIZATION_DATA = "data/processed/optimization_comparison.csv"
DEMO_OPTIMIZATION_DATA = "data/processed/optimization_comparison_ui_demo.csv"

GREEN, AMBER, RED, GREY = "#2ecc71", "#f5b041", "#e74c3c", "#7f8c8d"
MIN_IMPROVEMENT = 0.03   # same 3% threshold the optimizer uses
MAX_ROUNDS = 3

st.set_page_config(page_title="GreenMetricAI", page_icon="🌱", layout="centered")

st.markdown(
    """
<style>
.block-container {max-width: 980px; padding-top: 1.4rem;}
.hero {background: linear-gradient(120deg,#0b3d2e,#11694b 60%,#1abc9c);
  border-radius: 16px; padding: 22px 28px; color: #fff; margin-bottom: 14px;}
.hero h1 {margin: 0; font-size: 1.9rem;}
.hero p {margin: 6px 0 0; opacity: .92;}
.card {border: 1px solid rgba(128,128,128,.28); border-radius: 14px;
  padding: 14px 16px; height: 100%;}
.card .t {font-size: .75rem; text-transform: uppercase; letter-spacing: .8px; opacity: .7;}
.card .v {font-size: 1.45rem; font-weight: 700; margin: 2px 0;}
.card .d {font-size: .85rem; opacity: .85;}
.badge {display:inline-block; padding: 2px 10px; border-radius: 999px;
  font-size: .78rem; font-weight: 600;}
.good {background: rgba(46,204,113,.18); color: #2ecc71;}
.mid  {background: rgba(245,176,65,.18); color: #f5b041;}
.bad  {background: rgba(231,76,60,.18); color: #e74c3c;}
.step {font-size: 1.15rem; font-weight: 700; margin: 22px 0 6px;}
</style>
""",
    unsafe_allow_html=True,
)


def card(title, value, desc, badge=None, kind="good"):
    b = f'<span class="badge {kind}">{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="card"><div class="t">{title}</div><div class="v">{value}</div>'
        f'{b}<div class="d">{desc}</div></div>',
        unsafe_allow_html=True,
    )


def fmt_kwh(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:.2e} kWh"


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
@st.cache_resource
def load_candidate_model(path):
    bundle = joblib.load(path)
    for key in ("model", "features", "target"):
        if key not in bundle:
            raise ValueError(f"Model artifact is missing '{key}'.")
    return bundle


@st.cache_resource
def load_trained_model(path):
    from greenmetricai.model import GreenMetricModel
    return GreenMetricModel.load(path)


model, model_kind = None, None
if os.path.exists(MODEL_TRAINED):
    try:
        model, model_kind = load_trained_model(MODEL_TRAINED), "greenmetric"
    except Exception:
        model = None
if model is None and os.path.exists(MODEL_CANDIDATE):
    try:
        model, model_kind = load_candidate_model(MODEL_CANDIDATE), "random_forest"
    except Exception:
        model = None


def predict_kwh(code):
    """Return (kWh per execution, feature dict, feature row or None)."""
    metrics = extract_code_metrics(code)
    values = metrics.to_dict()
    if model_kind == "random_forest":
        names = list(model["features"])
        X = pd.DataFrame([[float(values[n]) for n in names]], columns=names)
        log_pred = float(model["model"].predict(X)[0])
        if not np.isfinite(log_pred) or not -300 <= log_pred <= 100:
            raise ValueError("The model returned an invalid prediction.")
        return 10.0 ** log_pred, values, X
    return float(model.predict(metrics)), values, None


@st.cache_data
def reference_log_energy(data_path, _bundle):
    """Model predictions for the benchmark programs, used for 'Low/Typical/High'."""
    if not os.path.exists(data_path):
        return None
    df = pd.read_csv(data_path)
    names = list(_bundle["features"])
    if any(n not in df.columns for n in names):
        return None
    X = df[names].apply(pd.to_numeric, errors="coerce").dropna()
    return None if X.empty else _bundle["model"].predict(X)


PLAIN = {
    "cc": "number of decision paths (if/loops)",
    "mi": "how maintainable the code is",
    "hv": "amount of operations and variables",
    "loc": "length of the code",
    "depth_nesting": "how deeply code is nested",
    "num_functions": "number of functions",
    "avg_function_length": "average function length",
    "comment_to_code_ratio": "amount of comments",
    "technical_debt_index": "estimated technical debt",
    "api_call_density": "how many function/library calls it makes",
    "loop_depth": "how deeply loops are nested",
    "branching_factor": "amount of branching",
    "coupling_between_classes": "coupling between classes",
    "num_parameters": "number of function parameters",
}


@st.cache_resource
def get_explainer(_rf):
    import shap
    return shap.TreeExplainer(_rf)


def top_reasons(X, k=3):
    """Plain-language top factors for this one prediction (None if unavailable)."""
    try:
        sv = np.asarray(get_explainer(model["model"]).shap_values(X))
        if sv.ndim == 3:
            sv = sv[0]
        vals = sv[0]
        total = np.abs(vals).sum() or 1.0
        order = np.argsort(-np.abs(vals))[:k]
        out = []
        for i in order:
            share = abs(vals[i]) / total
            strength = "strongly" if share >= .30 else "moderately" if share >= .15 else "slightly"
            name = list(model["features"])[i]
            out.append((PLAIN.get(name, name), vals[i] > 0, strength))
        return out
    except Exception:
        return None


def complexity_label(cc):
    return ("Simple", "good") if cc <= 5 else ("Moderate", "mid") if cc <= 10 else ("Complex", "bad")


def maintain_label(mi):
    return (("Easy to maintain", "good") if mi >= 65
            else ("Okay", "mid") if mi >= 40 else ("Hard to maintain", "bad"))


# ---------------------------------------------------------------------
# Running code safely-ish (your own machine; timeouts applied)
# ---------------------------------------------------------------------
def run_checks(code, checks, timeout=30):
    """Run code + the user's check lines in a separate process."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "run.py"
        p.write_text(code + "\n\n" + checks + "\n", encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, str(p)], capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"Timed out after {timeout}s."
        if r.returncode != 0:
            return False, (r.stderr.strip() or r.stdout.strip() or "Failed.")[-600:]
    return True, None


MEASURE_SCRIPT = r'''
import sys, json, time, io, contextlib, os
os.environ["CODECARBON_LOG_LEVEL"] = "error"
d, target = sys.argv[1], float(sys.argv[2])
ns = {"__name__": "__bench__"}
exec(compile(open(d + "/defs.py").read(), "<code>", "exec"), ns)
calls = compile(open(d + "/calls.py").read(), "<calls>", "exec")
sink = io.StringIO()
with contextlib.redirect_stdout(sink):
    exec(calls, ns)
    t = time.perf_counter(); exec(calls, ns)
    t1 = max(time.perf_counter() - t, 1e-7)
n = min(max(int(target / t1), 1), 20_000_000)
try:
    from codecarbon import EmissionsTracker
    tr = EmissionsTracker(save_to_file=False, log_level="error", measure_power_secs=1)
except Exception as exc:
    print("RESULT:" + json.dumps({"error": "CodeCarbon unavailable: %s" % exc})); sys.exit(0)
tr.start()
t0 = time.perf_counter()
with contextlib.redirect_stdout(sink):
    for _ in range(n):
        exec(calls, ns)
dur = time.perf_counter() - t0
tr.stop()
data = tr.final_emissions_data
print("RESULT:" + json.dumps({"energy_kwh": data.energy_consumed / n,
      "runtime_s": dur / n, "repeats": n}))
'''


def measure_once(code, calls, target=2.0):
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "defs.py").write_text(code, encoding="utf-8")
        Path(tmp, "calls.py").write_text(calls, encoding="utf-8")
        Path(tmp, "runner.py").write_text(MEASURE_SCRIPT, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, str(Path(tmp, "runner.py")), tmp, str(target)],
                               capture_output=True, text=True, timeout=240)
        except subprocess.TimeoutExpired:
            return {"error": "Measurement timed out."}
        for line in r.stdout.splitlines():
            if line.startswith("RESULT:"):
                return json.loads(line[7:])
        return {"error": (r.stderr.strip() or "No result produced.")[-400:]}


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>🌱 GreenMetricAI</h1>'
    "<p>Paste Python code. Get an energy estimate, a plain-English explanation, "
    "and an AI-optimized version that is checked before you see it.</p></div>",
    unsafe_allow_html=True,
)

if model is None:
    st.error("The prediction model could not be loaded. Check the models/ folder.")
    st.stop()

# ---------------------------------------------------------------------
# STEP 1 - input
# ---------------------------------------------------------------------
st.markdown('<div class="step">1 · Paste your Python code</div>', unsafe_allow_html=True)
code = st.text_area(
    "Python code", height=220, label_visibility="collapsed",
    value="def square_sum(n):\n    total = 0\n    for i in range(n):\n        total = total + i * i\n    return total",
)
checks = st.text_area(
    "Optional: lines that test your code (recommended)",
    height=90,
    placeholder="assert square_sum(3) == 5\nassert square_sum(10) == 285",
    help="Used in two ways: (1) to confirm the optimized code still gives the same "
         "answers, and (2) to run the code repeatedly when measuring real energy. "
         "Without it, we cannot verify correctness or measure real energy.",
)

if st.button("Analyze my code", type="primary"):
    try:
        kwh, values, X = predict_kwh(code)
        st.session_state["analysis"] = {
            "code": code, "kwh": kwh, "values": values, "X": X,
            "reasons": top_reasons(X) if X is not None else None,
        }
        for key in ("opt", "meas"):
            st.session_state.pop(key, None)
    except Exception as exc:
        st.session_state.pop("analysis", None)
        st.error(f"Could not analyze this code: {exc}")

analysis = st.session_state.get("analysis")
if analysis and analysis["code"] != code:
    st.info("You changed the code. Click **Analyze my code** to refresh the results.")
    analysis = None

# ---------------------------------------------------------------------
# STEP 2 - results in plain language
# ---------------------------------------------------------------------
if analysis:
    values, kwh = analysis["values"], analysis["kwh"]
    st.markdown('<div class="step">2 · What we found</div>', unsafe_allow_html=True)

    rating, kind, desc = "—", "mid", "Estimated energy for one run."
    if model_kind == "random_forest":
        ref = reference_log_energy(DATA, model)
        if ref is not None and len(ref):
            pct = float((ref < np.log10(kwh)).mean() * 100)
            rating, kind = (("Low", "good") if pct < 33 else ("Typical", "mid") if pct < 67
                            else ("High", "bad"))
            desc = f"Uses more than about {pct:.0f}% of the {len(ref)} benchmark programs."
    cc_text, cc_kind = complexity_label(values.get("cc", 0))
    mi_text, mi_kind = maintain_label(values.get("mi", 0))

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Estimated energy", fmt_kwh(kwh), desc, rating + " energy", kind)
    with c2:
        card("Complexity", f"{values.get('cc', 0):.0f}",
             "Number of decision paths. Lower is simpler.", cc_text, cc_kind)
    with c3:
        card("Maintainability", f"{values.get('mi', 0):.0f} / 100",
             "How easy the code is to read and change.", mi_text, mi_kind)
    st.caption(
        "The energy figure is a **prediction from the code's structure**, not a measurement. "
        "It is a rough guide that works best for comparing two versions of the same code."
    )

    if analysis["reasons"]:
        st.markdown("**Why this estimate?**")
        for label, up, strength in analysis["reasons"]:
            arrow = "⬆️ raises" if up else "⬇️ lowers"
            st.markdown(f"- The **{label}** {strength} {arrow} the estimate.")
        st.caption("These are the factors the model leaned on most. They show influence, "
                   "not proven cause.")

    # -----------------------------------------------------------------
    # STEP 3 - optimize
    # -----------------------------------------------------------------
    st.markdown('<div class="step">3 · Optimize with AI</div>', unsafe_allow_html=True)
    key_ok = bool(os.getenv("GEMINI_API_KEY"))
    if not key_ok:
        st.warning("Set the GEMINI_API_KEY environment variable to enable AI optimization.")

    if st.button("✨ Optimize my code", disabled=not key_ok):
        try:
            from greenmetricai.optimizer import (GeminiProvider, extract_code,
                                                 validate_python_syntax)
            provider = GeminiProvider()
        except Exception as exc:
            st.error(f"Could not start the AI optimizer: {exc}")
            provider = None

        if provider is not None:
            orig_ok, orig_err = (True, None)
            if checks.strip():
                orig_ok, orig_err = run_checks(code, checks)
            attempts, accepted = [], None
            with st.spinner("Asking the AI for an optimized version..."):
                for rnd in range(1, MAX_ROUNDS + 1):
                    prompt = (
                        "Optimize this Python code to use less execution energy while "
                        "preserving EXACT behavior. Keep function names, arguments and "
                        "return values. Do not add dependencies. Return ONLY complete "
                        "Python source code, no explanation.\n\n```python\n" + code + "\n```\n"
                        f"\nCurrent predicted energy: {kwh:.6g} kWh per execution."
                    )
                    try:
                        cand = extract_code(provider.generate(prompt))
                    except Exception as exc:
                        attempts.append((rnd, None, f"AI request failed: {exc}"))
                        continue
                    ok, err = validate_python_syntax(cand)
                    if not ok:
                        attempts.append((rnd, cand, f"Not valid Python ({err})"))
                        continue
                    if cand.strip() == code.strip():
                        attempts.append((rnd, cand, "The AI returned your code unchanged."))
                        continue
                    if checks.strip():
                        tests_ok, tests_err = run_checks(cand, checks)
                        if not tests_ok:
                            attempts.append((rnd, cand, f"Failed your checks: {tests_err}"))
                            continue
                    try:
                        c_kwh, c_vals, _ = predict_kwh(cand)
                    except Exception as exc:
                        attempts.append((rnd, cand, f"Could not analyze candidate: {exc}"))
                        continue
                    gain = (kwh - c_kwh) / kwh if kwh else 0.0
                    if gain < MIN_IMPROVEMENT:
                        attempts.append((rnd, cand,
                                         f"Predicted energy not lower enough ({gain*100:+.1f}%)."))
                        continue
                    accepted = {"code": cand, "kwh": c_kwh, "values": c_vals, "gain": gain}
                    attempts.append((rnd, cand, "Accepted"))
                    break
            st.session_state["opt"] = {"source": code, "accepted": accepted,
                                       "attempts": attempts, "orig_ok": orig_ok,
                                       "orig_err": orig_err, "had_checks": bool(checks.strip())}
            st.session_state.pop("meas", None)

    opt = st.session_state.get("opt")
    if opt and opt["source"] == code:
        if opt["had_checks"] and not opt["orig_ok"]:
            st.warning("Your own checks fail on the ORIGINAL code, so correctness of any "
                       f"optimized version can't be trusted. Details: {opt['orig_err']}")
        acc = opt["accepted"]
        if acc:
            st.success(f"Optimized version found. Predicted energy is "
                       f"{acc['gain']*100:.1f}% lower.")
            left, right = st.columns(2)
            with left:
                st.markdown("**Your code**")
                st.code(code, language="python")
            with right:
                st.markdown("**Optimized**")
                st.code(acc["code"], language="python")
            st.markdown("**Safety checks**")
            st.markdown("- ✅ Valid Python\n- ✅ Code is different from yours")
            if opt["had_checks"]:
                st.markdown("- ✅ Still passes your checks")
            else:
                st.markdown("- ⚠️ Not verified: you did not provide checks, so we can't "
                            "confirm it behaves the same. Review it yourself.")
            st.markdown("- ✅ Predicted energy is lower (a prediction, not yet measured)")
        else:
            st.warning("No safe improvement found. Every attempt was rejected (see below). "
                       "That is a valid result: the checks are working.")
        with st.expander("What the AI tried"):
            for rnd, cand, status in opt["attempts"]:
                st.markdown(f"**Attempt {rnd}:** {status}")
                if cand:
                    st.code(cand, language="python")

        # -------------------------------------------------------------
        # STEP 4 - tool comparison on the user's own code
        # -------------------------------------------------------------
        st.markdown('<div class="step">4 · Tool comparison on your code</div>',
                    unsafe_allow_html=True)
        t1, t2, t3 = st.columns(3)
        with t1:
            card("Radon", "Is it simple?", "Reads the code and scores complexity and "
                 "maintainability. Does not run it and says nothing about energy.")
        with t2:
            card("GreenMetricAI model", "How much energy, before running?",
                 "Predicts energy from the code's structure. Instant, but approximate.")
        with t3:
            card("CodeCarbon", "How much energy did it really use?",
                 "Runs the code and estimates actual energy. Most realistic, but slower.")
        st.write("")

        meas = st.session_state.get("meas")
        can_measure = bool(acc) and bool(checks.strip())
        if st.button("⚡ Measure real energy with CodeCarbon", disabled=not can_measure,
                     help="Needs an optimized version and your test lines."):
            runs = {"original": [], "optimized": []}
            err = None
            with st.spinner("Running both versions with CodeCarbon (about 15-30 seconds)..."):
                for _ in range(3):   # alternate so drift hits both equally
                    for name, src in (("original", code), ("optimized", acc["code"])):
                        r = measure_once(src, checks)
                        if "error" in r:
                            err = r["error"]
                            break
                        runs[name].append(r)
                    if err:
                        break
            st.session_state["meas"] = {"runs": runs, "error": err, "key": (code, acc["code"])}
            meas = st.session_state["meas"]

        if not can_measure and acc:
            st.caption("Add test lines in step 1 to unlock the CodeCarbon measurement.")
        if meas and meas.get("key") != (code, acc["code"] if acc else None):
            meas = None

        rows = []
        if acc:
            o, p = values, acc["values"]
            rows += [
                ("Radon", "Complexity (lower = simpler)", f"{o.get('cc', 0):.0f}",
                 f"{p.get('cc', 0):.0f}"),
                ("Radon", "Maintainability (higher = better)", f"{o.get('mi', 0):.0f}",
                 f"{p.get('mi', 0):.0f}"),
                ("GreenMetricAI", "Predicted energy", fmt_kwh(kwh), fmt_kwh(acc["kwh"])),
            ]
            if meas and not meas.get("error"):
                def med(name, k):
                    return float(np.median([r[k] for r in meas["runs"][name]]))
                e_o, e_p = med("original", "energy_kwh"), med("optimized", "energy_kwh")
                rows.append(("CodeCarbon", "Measured energy (median of 3)", fmt_kwh(e_o), fmt_kwh(e_p)))
            else:
                rows.append(("CodeCarbon", "Measured energy", "not run yet", "not run yet"))
            st.dataframe(pd.DataFrame(rows, columns=["Tool", "What it reports", "Your code", "Optimized"]),
                         use_container_width=True, hide_index=True)

        if meas and meas.get("error"):
            st.error(f"Measurement failed: {meas['error']}")
        elif meas and acc:
            def energies(name):
                return np.array([r["energy_kwh"] for r in meas["runs"][name]])
            eo, ep = energies("original"), energies("optimized")
            change = (np.median(eo) - np.median(ep)) / np.median(eo) * 100 if np.median(eo) else 0.0
            spread = max(np.std(eo) / np.mean(eo), np.std(ep) / np.mean(ep)) * 100 if len(eo) > 1 else 0.0
            if abs(change) <= spread:
                st.warning(f"Measured change: {change:+.1f}%. That is smaller than the normal "
                           f"run-to-run variation (about {spread:.0f}%), so the real saving can't "
                           "be confirmed for this code.")
            elif change > 0:
                st.success(f"Measured energy is about {change:.1f}% lower, beyond normal variation.")
            else:
                st.error(f"Measured energy went UP by about {abs(change):.1f}%. The prediction "
                         "was wrong for this code.")
            st.caption("Quick estimate from 3 short runs with a small fixed test overhead. "
                       "Your benchmark dataset used a longer, calibrated protocol.")

# ---------------------------------------------------------------------
# Evidence from the benchmark (collapsed, one chart)
# ---------------------------------------------------------------------
source = OPTIMIZATION_DATA if os.path.exists(OPTIMIZATION_DATA) else (
    DEMO_OPTIMIZATION_DATA if os.path.exists(DEMO_OPTIMIZATION_DATA) else None)

with st.expander("📊 How well does this work? Results on benchmark tasks"):
    if source is None:
        st.info("No benchmark results yet. Run `python -m scripts.compare_optimization --task-id 2`.")
    else:
        df = pd.read_csv(source)
        df["version"] = df["version"].astype(str).str.lower()
        rows = []
        for tid, g in df.groupby("task_id"):
            o = g[g.version == "original"]["energy_kwh_per_execution"].dropna()
            p = g[g.version == "optimized"]["energy_kwh_per_execution"].dropna()
            if o.empty or p.empty:
                continue
            red = (o.median() - p.median()) / o.median() * 100
            cv = o.std(ddof=1) / o.mean() * 100 if len(o) > 1 else np.nan
            changed = True
            if "candidate_different" in g.columns:
                changed = str(g["candidate_different"].iloc[0]).lower() == "true"
            if not changed:
                verdict = "AI returned same code"
            elif pd.notna(cv) and abs(red) < cv:
                verdict = "Too small to tell"
            else:
                verdict = "Less energy" if red > 0 else "More energy"
            rows.append({"Task": int(tid), "Energy change (%)": round(red, 1), "Result": verdict})
        out = pd.DataFrame(rows).sort_values("Task")
        if out.empty:
            st.info("Benchmark file has no complete original/optimized pairs.")
        else:
            palette = {"Less energy": GREEN, "More energy": RED,
                       "Too small to tell": AMBER, "AI returned same code": GREY}
            fig = go.Figure(go.Bar(
                x=out["Task"].astype(str), y=-out["Energy change (%)"],
                marker_color=[palette[v] for v in out["Result"]]))
            fig.update_layout(height=300, margin=dict(t=10, b=10),
                              xaxis_title="Benchmark task",
                              yaxis_title="Energy change (%) - below zero = saved")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(out, use_container_width=True, hide_index=True)
            useful = int((out["Result"] == "Less energy").sum())
            st.caption(f"{useful} of {len(out)} tasks showed a clear energy saving. "
                       "Green = saved, red = got worse, amber = within normal variation, "
                       "grey = the AI changed nothing.")

with st.expander("ℹ️ How to read these results"):
    st.markdown(
        "- **Predicted** energy comes from our model (instant, approximate).\n"
        "- **Measured** energy comes from CodeCarbon actually running the code (an estimate "
        "from hardware power models, not a lab meter).\n"
        "- The model learned from 100 small benchmark programs, so predictions for very "
        "different code are less reliable.\n"
        "- Always check the optimized code yourself. Passing your test lines is good evidence, "
        "not proof."
    )