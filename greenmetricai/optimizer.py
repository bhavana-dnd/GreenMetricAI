from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import pandas as pd

from .features import extract_code_metrics
from .mbpp import build_benchmark_code


MODEL_PATH = "models/greenmetric_rf_final.joblib"
GEMINI_MODEL = "gemini-3.6-flash"


class GeminiProvider:
    """
    Gemini API provider used to generate candidate Python optimizations.
    """

    def __init__(self, model=None, max_retries=3, retry_delay=5):
        from google import genai

        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(api_key=key)
        self.model = model or os.getenv("GEMINI_MODEL", GEMINI_MODEL)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(self, prompt):
        """Send a prompt to Gemini, retrying temporary 429/503 errors."""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )

                if not response.text:
                    raise RuntimeError("Gemini returned an empty response.")

                return response.text

            except Exception as exc:
                last_error = exc
                error_text = str(exc)

                temporary_error = (
                    "503" in error_text
                    or "429" in error_text
                    or "UNAVAILABLE" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                )

                if not temporary_error or attempt >= self.max_retries:
                    raise

                wait_time = self.retry_delay * attempt
                print(
                    f"Gemini temporary error "
                    f"(attempt {attempt}/{self.max_retries}). "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

        raise RuntimeError(f"Gemini request failed: {last_error}")


class GreenMetricEnergyModel:
    """
    Loads the GreenMetricAI Random Forest model.

    The model predicts log10 energy in kWh per execution;
    predict() converts that prediction back into kWh.
    """

    def __init__(self, path=MODEL_PATH):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"GreenMetricAI model not found: {path}")

        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.features = artifact["features"]
        self.target = artifact.get(
            "target",
            "log10_energy_kwh_per_execution",
        )

    def predict_log10(self, metrics):
        """Return predicted log10 energy."""
        values = metrics.to_dict() if hasattr(metrics, "to_dict") else dict(metrics)

        row = pd.DataFrame(
            [[float(values[name]) for name in self.features]],
            columns=self.features,
        )

        prediction = self.model.predict(row)[0]
        return float(prediction)

    def predict(self, metrics):
        """Return predicted energy in kWh per execution."""
        log_energy = self.predict_log10(metrics)
        return float(10 ** log_energy)


def extract_code(text):
    """Extract Python source from a Gemini response."""
    if not text:
        raise ValueError("Empty Gemini response.")

    match = re.search(
        r"```(?:python|py)?\s*(.*?)```",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    return text.strip()


def validate_python_syntax(code):
    """Check whether generated code is valid Python."""
    try:
        compile(code, "<greenmetricai_candidate>", "exec")
        return True, None
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} (line {exc.lineno})"
    except Exception as exc:
        return False, str(exc)


def validate_mbpp_candidate(task, candidate, timeout_seconds=30):
    """
    Validate a candidate against the task's MBPP tests.

    Returns (True, None) when validation succeeds,
    otherwise (False, reason).
    """
    syntax_ok, syntax_error = validate_python_syntax(candidate)
    if not syntax_ok:
        return False, syntax_error

    test_task = dict(task)
    test_task["code"] = candidate
    benchmark = build_benchmark_code(test_task)

    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "candidate.py"
        script_path.write_text(benchmark, encoding="utf-8")

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return (
                False,
                f"MBPP validation timed out after {timeout_seconds} seconds.",
            )
        except Exception as exc:
            return False, str(exc)

        if result.returncode != 0:
            if result.stderr.strip():
                return False, result.stderr.strip()[-2000:]
            if result.stdout.strip():
                return False, result.stdout.strip()[-2000:]
            return False, f"Process exited with code {result.returncode}."

    return True, None


def optimize_mbpp_task(
    task,
    model,
    provider,
    rounds=3,
    min_improvement=0.03,
    validation_timeout=30,
):
    """
    Optimize one MBPP task.

    A candidate is accepted only when:
    1. Its syntax is valid.
    2. It passes MBPP tests.
    3. Its predicted energy is lower.
    4. Relative predicted improvement meets min_improvement.

    Predicted improvement is not a measured energy saving.
    Actual energy measurement must be performed separately.
    """
    original_code = task["code"]
    current_code = original_code
    current_metrics = extract_code_metrics(current_code)
    current_energy = model.predict(current_metrics)
    history = []

    for round_number in range(1, rounds + 1):
        prompt = (
            "You are optimizing Python code for the GreenMetricAI research project.\n\n"
            "OBJECTIVE:\n"
            "Reduce predicted execution energy while preserving EXACT behavior.\n\n"
            "STRICT REQUIREMENTS:\n"
            "1. Preserve all required function names.\n"
            "2. Preserve all function arguments.\n"
            "3. Preserve return values.\n"
            "4. Preserve externally observable behavior.\n"
            "5. Do not remove required functionality.\n"
            "6. Do not add external dependencies.\n"
            "7. Do not change input/output format.\n"
            "8. Do not modify the test cases.\n"
            "9. Return ONLY complete Python source code.\n"
            "10. Do not explain the changes.\n\n"
            "CURRENT PYTHON CODE:\n"
            "```python\n"
            + current_code
            + "\n```\n\n"
            "CURRENT STATIC CODE METRICS:\n"
            + str(current_metrics.to_dict())
            + "\n\n"
            "CURRENT PREDICTED ENERGY:\n"
            + f"{current_energy:.12g}"
            + " kWh per execution.\n\n"
            "Generate one behavior-preserving optimization."
        )

        print(f"\nOptimization round {round_number}/{rounds}")

        # Gemini API failures stop the loop. The provider itself retries
        # temporary API errors according to its configured retry policy.
        try:
            response = provider.generate(prompt)
            candidate = extract_code(response)
        except Exception as exc:
            history.append(
                {
                    "round": round_number,
                    "accepted": False,
                    "reason": f"llm_error: {exc}",
                }
            )
            break

        # Syntax validation: reject and continue to the next round.
        syntax_ok, syntax_error = validate_python_syntax(candidate)
        if not syntax_ok:
            history.append(
                {
                    "round": round_number,
                    "accepted": False,
                    "tests_passed": False,
                    "reason": f"syntax_error: {syntax_error}",
                }
            )
            continue

        # Extract candidate features and predict energy.
        try:
            candidate_metrics = extract_code_metrics(candidate)
            candidate_energy = model.predict(candidate_metrics)
        except Exception as exc:
            history.append(
                {
                    "round": round_number,
                    "accepted": False,
                    "reason": f"metric_error: {exc}",
                }
            )
            continue
        try:
            candidate_metrics = extract_code_metrics(candidate)
            candidate_energy = model.predict(candidate_metrics)
        except Exception as exc:
            history.append(
                {
                    "round": round_number,
                    "accepted": False,
                    "reason": f"metric_error: {exc}",
                }
            )
            continue

        improvement = (
            (current_energy - candidate_energy)
            / max(abs(current_energy), 1e-30)
        )

        improvement = (
            (current_energy - candidate_energy)
            / max(abs(current_energy), 1e-30)
        )

        print("Validating candidate against official MBPP tests...")
        tests_passed, validation_reason = validate_mbpp_candidate(
            task,
            candidate,
            timeout_seconds=validation_timeout,
        )

        accepted = tests_passed and improvement >= min_improvement

        if not tests_passed:
            rejection_reason = (
                f"mbpp_failed: {validation_reason or 'Tests did not pass.'}"
            )
        elif improvement < min_improvement:
            rejection_reason = (
                f"below_energy_threshold: predicted improvement "
                f"{improvement:.4%} is below required "
                f"{min_improvement:.4%}."
            )
        else:
            rejection_reason = None

        history.append(
            {
                "round": round_number,
                "accepted": accepted,
                "tests_passed": tests_passed,
                "before_energy_kwh": current_energy,
                "candidate_energy_kwh": candidate_energy,
                "relative_improvement": improvement,
                "validation_reason": validation_reason,
                "reason": rejection_reason,
            }
        )

        print(f"Tests passed: {tests_passed}")
        print(
            "Predicted candidate energy: "
            f"{candidate_energy:.12g} kWh/execution"
        )
        print(f"Predicted improvement: {improvement * 100:.4f}%")
        print(f"Accepted: {accepted}")

        if accepted:
            current_code = candidate
            current_metrics = candidate_metrics
            current_energy = candidate_energy

        if accepted:
            current_code = candidate
            current_metrics = candidate_metrics
            current_energy = candidate_energy

        # If rejected, continue to the next bounded round.

    return {
        "original_code": original_code,
        "optimized_code": current_code,
        "predicted_energy_kwh": current_energy,
        "metrics": current_metrics.to_dict(),
        "history": history,
    }