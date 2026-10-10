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
