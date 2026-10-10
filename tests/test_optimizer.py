from greenmetricai.optimizer import (
    validate_python_syntax,
    validate_mbpp_candidate,
)


def test_valid_python_syntax_is_accepted():
    valid, reason = validate_python_syntax(
        "def answer():\n    return 42\n"
    )

    assert valid is True
    assert reason is None


def test_invalid_python_syntax_is_rejected():
    valid, reason = validate_python_syntax(
        "def answer(:\n    return 42\n"
    )

    assert valid is False
    assert "SyntaxError" in reason


def test_candidate_failing_mbpp_test_is_rejected():
    task = {
        "task_id": 2,
        "prompt": "Find shared elements from two lists.",
        "code": (
            "def similar_elements(test_tup1, test_tup2):\n"
            "    return ()\n"
        ),
        "test_list": [
            "assert set(similar_elements([1, 2], [2, 3])) == {2}"
        ],
        "test_setup_code": "",
    }

    candidate = (
        "def similar_elements(test_tup1, test_tup2):\n"
        "    return ()\n"
    )

    passed, reason = validate_mbpp_candidate(task, candidate)

    assert passed is False
    assert reason is not None

def test_gemini_failure_preserves_original_code():
    from greenmetricai.optimizer import optimize_mbpp_task

    original_code = (
        "def answer():\n"
        "    return 42\n"
    )

    class FakeModel:
        def predict(self, metrics):
            return 1e-10

    class FailingProvider:
        def generate(self, prompt):
            raise RuntimeError("Simulated Gemini failure")

    task = {
        "task_id": 999,
        "code": original_code,
    }

    result = optimize_mbpp_task(
        task=task,
        model=FakeModel(),
        provider=FailingProvider(),
        rounds=1,
    )

    assert result["optimized_code"] == original_code
    assert len(result["history"]) == 1
    assert result["history"][0]["accepted"] is False
    assert "llm_error" in result["history"][0]["reason"]


def test_candidate_below_energy_threshold_is_rejected(monkeypatch):
    import greenmetricai.optimizer as optimizer

    original_code = "def answer():\n    return 42\n"
    candidate_code = "def answer():\n    return 42  # optimized\n"

    class FakeModel:
        def __init__(self):
            self.predictions = iter([0.100, 0.098])

        def predict(self, metrics):
            return next(self.predictions)

    class FakeProvider:
        def generate(self, prompt):
            return candidate_code

    monkeypatch.setattr(
        optimizer,
        "validate_mbpp_candidate",
        lambda *args, **kwargs: (True, None),
    )

    result = optimizer.optimize_mbpp_task(
        task={"task_id": 999, "code": original_code},
        model=FakeModel(),
        provider=FakeProvider(),
        rounds=1,
        min_improvement=0.03,
    )

    assert result["optimized_code"] == original_code
    assert result["history"][0]["accepted"] is False
    assert result["history"][0]["relative_improvement"] < 0.03


def test_candidate_at_energy_threshold_is_accepted(monkeypatch):
    import greenmetricai.optimizer as optimizer

    original_code = "def answer():\n    return 42\n"
    candidate_code = "def answer():\n    return 42  # optimized\n"

    class FakeModel:
        def __init__(self):
            self.predictions = iter([0.100, 0.097])

        def predict(self, metrics):
            return next(self.predictions)

    class FakeProvider:
        def generate(self, prompt):
            return candidate_code

    monkeypatch.setattr(
        optimizer,
        "validate_mbpp_candidate",
        lambda *args, **kwargs: (True, None),
    )

    result = optimizer.optimize_mbpp_task(
        task={"task_id": 999, "code": original_code},
        model=FakeModel(),
        provider=FakeProvider(),
        rounds=1,
        min_improvement=0.03,
    )

    assert result["optimized_code"].strip() == candidate_code.strip()
    assert result["history"][0]["accepted"] is True


def test_candidate_failing_mbpp_is_rejected_even_with_lower_energy(monkeypatch):
    import greenmetricai.optimizer as optimizer

    original_code = "def answer():\n    return 42\n"
    candidate_code = "def answer():\n    return 100\n"

    class FakeModel:
        def __init__(self):
            self.predictions = iter([0.100, 0.050])

        def predict(self, metrics):
            return next(self.predictions)

    class FakeProvider:
        def generate(self, prompt):
            return candidate_code

    monkeypatch.setattr(
        optimizer,
        "validate_mbpp_candidate",
        lambda *args, **kwargs: (False, "MBPP tests failed"),
    )

    result = optimizer.optimize_mbpp_task(
        task={"task_id": 999, "code": original_code},
        model=FakeModel(),
        provider=FakeProvider(),
        rounds=1,
        min_improvement=0.03,
    )

    assert result["optimized_code"].strip() == original_code.strip()
    assert result["history"][0]["tests_passed"] is False
    assert result["history"][0]["accepted"] is False
