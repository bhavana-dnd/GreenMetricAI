from __future__ import annotations

import json
import textwrap
from pathlib import Path


MBPP_RAW_URL = (
    "https://raw.githubusercontent.com/google-research/"
    "google-research/master/mbpp/sanitized-mbpp.json"
)


# Standard-library modules that may be referenced by
# official MBPP test cases.
#
# This is only an execution-environment compatibility layer.
# We are NOT modifying the benchmark programs or generating data.
TEST_SUPPORT_IMPORTS = """
import math
import cmath
import statistics
import itertools
import functools
import collections
import re
import string
import random
"""


def load_mbpp(path="data/raw/sanitized-mbpp.json"):
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def _build_common_code(task):
    code = task["code"]

    tests = task.get("test_list", [])

    setup = task.get(
        "test_setup_code",
        ""
    )

    test_block = "\n".join(
        textwrap.indent(test, "    ")
        for test in tests
    )

    if not test_block:
        test_block = "    pass"

    return (
        f"{TEST_SUPPORT_IMPORTS}\n"
        f"{setup}\n"
        f"{code}\n\n"
        "def __greenmetricai_run_tests__():\n"
        f"{test_block}\n"
    )


def build_benchmark_code(task):

    common = _build_common_code(task)

    return (
        f"{common}\n"
        "if __name__ == '__main__':\n"
        "    __greenmetricai_run_tests__()\n"
    )


def build_repeated_code(task, repeats=20):

    common = _build_common_code(task)

    return (
        f"{common}\n"
        "if __name__ == '__main__':\n"
        f"    for __r in range({int(repeats)}):\n"
        "        __greenmetricai_run_tests__()\n"
    )