"""Create a PR report from unit-test results and recorded benchmark evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def _read_test_results(path: Path) -> dict[str, int | str]:
    if not path.exists():
        return {"status": "not run", "tests": 0, "failures": 0, "errors": 0, "skipped": 0}

    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    totals = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for key in totals:
            totals[key] += int(suite.attrib.get(key, 0))

    totals["status"] = (
        "passed" if totals["tests"] > 0 and not totals["failures"] and not totals["errors"]
        else "failed"
    )
    return totals


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_report(test_results: dict, artifact_path: Path) -> str:
    tests_passed = test_results["status"] == "passed"
    lines = [
        "## GreenMetricAI pull request validation",
        "",
        f"**Project checks:** {test_results['status']} - "
        f"{test_results['tests']} tests, {test_results['failures']} failures, "
        f"{test_results['errors']} errors, {test_results['skipped']} skipped.",
        "",
    ]

    if artifact_path.exists():
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            evidence = artifact.get("measured_validation") or {}
            task_id = artifact.get("task_id", "unknown")
            status = evidence.get("status", "not available")
            reduction = _number(evidence.get("geometric_mean_reduction_pct"))
            lower = _number(evidence.get("ci_lower_pct"))
            upper = _number(evidence.get("ci_upper_pct"))
            pairs = evidence.get("paired_runs", 0)
            attempted = evidence.get("attempted_pairs", 0)
            confidence = _number(evidence.get("family_confidence_pct"))
            minimum = _number(evidence.get("minimum_reduction_pct"))

            lines.extend([
                f"### Recorded benchmark evidence (MBPP task {task_id})",
                "",
                "This is the checked-in experiment, not a measurement of this PR. "
                "Any candidate conclusion below applies only to this recorded MBPP task, "
                "not to code changed by the PR. "
                "GitHub-hosted runner energy is not measured because its hardware "
                "and background load are not controlled.",
                "",
                f"- Evidence status: **{status}**",
                f"- Valid paired runs: {pairs} of {attempted} attempted",
            ])
            if reduction is not None:
                lines.append(f"- Geometric mean energy reduction: {reduction:.3f}%")
            if lower is not None and upper is not None:
                conf_label = f"{confidence:g}%" if confidence is not None else "reported"
                lines.append(
                    f"- Family-wise {conf_label} interval: [{lower:.3f}%, {upper:.3f}%]"
                )
            if minimum is not None:
                lines.append(f"- Minimum reduction threshold: {minimum:.3f}%")

            candidate_checks_passed = (
                artifact.get("original_tests_passed") is True
                and artifact.get("candidate_tests_passed") is True
            )
            recommendation = (
                tests_passed
                and candidate_checks_passed
                and status == "confirmed_reduction"
                and lower is not None
                and minimum is not None
                and lower > minimum
            )
            lines.extend([
                "",
                (
                    "**Measured candidate recommendation:** eligible for human review. "
                    "Its recorded checks passed and the interval clears the configured threshold."
                    if recommendation else
                    "**Measured candidate recommendation:** none. A recommendation requires "
                    "passing project checks, passing candidate checks, and a confidence "
                    "interval that clears the minimum-reduction threshold."
                ),
            ])
        except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
            lines.extend([
                "### Recorded benchmark evidence",
                "",
                "The saved benchmark artifact could not be read; no energy recommendation is made.",
            ])
    else:
        lines.extend([
            "### Recorded benchmark evidence",
            "",
            "No checked-in paired benchmark artifact is available; no energy recommendation is made.",
        ])

    lines.extend([
        "",
        "Energy results are experiment-specific and do not establish savings for other "
        "hardware or workloads. CI does not rerun CodeCarbon measurements. Review any "
        "code alternative before applying it.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--junit", type=Path, default=Path("reports/ci-junit.xml"))
    parser.add_argument(
        "--benchmark", type=Path,
        default=Path("data/processed/optimization_code_task_2.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/ci-pr-report.md"))
    args = parser.parse_args()

    report = build_report(_read_test_results(args.junit), args.benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write(report)
    print(report)


if __name__ == "__main__":
    main()
