from greenmetricai.response_optimizer import (
    explain_prediction,
    generate_energy_alternatives,
    validate_code_with_checks,
)


ORIGINAL = """\
def square_sum(n):
    return sum(i * i for i in range(n))
"""

CHECKS = """\
assert square_sum(3) == 5
assert square_sum(10) == 285
"""

ALTERNATIVE = """\
def square_sum(n):
    return n * (n - 1) * (2 * n - 1) // 6
"""


def test_example_behavior_checks_pass():
    passed, reason = validate_code_with_checks(ORIGINAL, CHECKS)

    assert passed is True
    assert reason is None


def test_failed_behavior_check_is_rejected():
    passed, reason = validate_code_with_checks(
        ORIGINAL,
        "assert square_sum(3) == 6",
    )

    assert passed is False
    assert reason


def test_behavior_checks_can_read_project_relative_data(tmp_path, monkeypatch):
    import pandas as pd

    data_path = tmp_path / "ml" / "data" / "raw" / "sample.csv.gz"
    data_path.parent.mkdir(parents=True)
    pd.DataFrame({"value": [2, 4]}).to_csv(
        data_path,
        index=False,
        compression="gzip",
    )
    monkeypatch.chdir(tmp_path)

    code = """\
from pathlib import Path
import pandas as pd

def total():
    path = Path(__file__).parent / "ml/data/raw/sample.csv.gz"
    return int(pd.read_csv(path)["value"].sum())
"""
    passed, reason = validate_code_with_checks(code, "assert total() == 6")

    assert passed is True
    assert reason is None


def test_missing_project_data_reports_expected_working_directory(tmp_path):
    code = """\
from pathlib import Path
Path(__file__).parent.joinpath("ml/data/raw/missing.csv").read_text()
"""
    passed, reason = validate_code_with_checks(
        code,
        "assert True",
        working_directory=tmp_path,
    )

    assert passed is False
    assert "file used by this code was not found" in reason
    assert str(tmp_path) in reason


def test_passing_candidate_is_scored_and_recommended():
    class FakeModel:
        def __init__(self):
            self.energies = iter([1e-6, 5e-7])

        def predict(self, _metrics):
            return next(self.energies)

    class FakeProvider:
        def generate(self, _prompt):
            return ALTERNATIVE

    result = generate_energy_alternatives(
        ORIGINAL,
        CHECKS,
        model=FakeModel(),
        model_kind="greenmetric",
        provider=FakeProvider(),
        candidate_count=1,
    )

    candidate = result["candidates"][0]
    assert candidate["passed_checks"] is True
    assert candidate["predicted_reduction_pct"] == 50.0
    assert result["recommended_number"] == 1


def test_random_forest_local_shap_values_explain_prediction():
    import numpy as np
    import pandas as pd
    import pytest
    from sklearn.ensemble import RandomForestRegressor

    pytest.importorskip("shap")
    from greenmetricai.features import FEATURES, extract_code_metrics

    training = pd.DataFrame(
        [
            [float(i + j) for j in range(len(FEATURES))]
            for i in range(8)
        ],
        columns=FEATURES,
    )
    model = RandomForestRegressor(n_estimators=8, random_state=3).fit(
        training,
        np.linspace(-12, -8, len(training)),
    )
    bundle = {
        "model": model,
        "features": FEATURES,
        "target": "log10_energy_kwh_per_execution",
    }

    explanation = explain_prediction(ORIGINAL, bundle, "random_forest")

    assert explanation["available"] is True
    row = pd.DataFrame(
        [extract_code_metrics(ORIGINAL).to_dict()],
        columns=FEATURES,
    )
    assert np.isclose(
        explanation["predicted_log10_energy"],
        model.predict(row)[0],
    )
    assert explanation["features"]
