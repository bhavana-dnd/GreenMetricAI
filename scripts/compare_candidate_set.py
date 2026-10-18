"""Compare several behavior-checked MBPP candidates with family-wise evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from greenmetricai.mbpp import load_mbpp
from greenmetricai.optimizer import validate_mbpp_candidate
from greenmetricai.paired_validation import (
    balanced_candidate_order,
    summarize_paired_candidates,
)
from scripts.compare_optimization import (
    FAMILY_CONFIDENCE,
    MAXIMUM_PAIRED_RUNS,
    MINIMUM_MEASURED_REDUCTION,
    MINIMUM_PAIRED_RUNS,
    measure_version,
)


_CANDIDATE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,39}\Z")


def _load_candidates(path: Path, original_task: dict):
    task_id = original_task["task_id"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("task_id") != task_id:
        raise ValueError(f"Candidate set task_id must equal {task_id}.")
    if manifest.get("original_code", "").strip() != original_task["code"].strip():
        raise ValueError(
            "Candidate set baseline does not match the official MBPP task."
        )

    items = manifest.get("candidates")
    if not isinstance(items, list) or not 1 <= len(items) <= 10:
        raise ValueError("Provide between one and ten candidates.")

    candidates = {}
    checks = {}
    details = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each candidate must be an object.")
        candidate_id = item.get("candidate_id")
        code = item.get("code")
        if not isinstance(candidate_id, str) or not _CANDIDATE_ID.fullmatch(candidate_id):
            raise ValueError(
                "candidate_id must be 1-40 letters, numbers, dots, underscores, or hyphens."
            )
        if candidate_id == "original" or candidate_id in candidates:
            raise ValueError(f"Candidate ID is reserved or duplicated: {candidate_id}")
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"Candidate {candidate_id} has no Python code.")

        task = dict(original_task)
        task["code"] = code
        passed, reason = validate_mbpp_candidate(task, code)
        checks[candidate_id] = passed
        details[candidate_id] = {"passed": passed, "reason": reason}
        if passed:
            candidates[candidate_id] = task
        print(
            f"{candidate_id}: "
            + ("behavior checks passed" if passed else f"rejected: {reason}")
        )
    return candidates, checks, details


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Measure and compare behavior-passing alternatives for one MBPP task. "
            "Run only trusted candidate code."
        )
    )
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--candidate-set", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/candidate_set_comparison.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/processed/candidate_set_validation.json"),
    )
    args = parser.parse_args()
    if args.output.exists() or args.summary_output.exists():
        raise FileExistsError(
            "Refusing to overwrite a prior experiment. Choose new --output and "
            "--summary-output paths."
        )

    tasks = load_mbpp()
    matches = [item for item in tasks if item.get("task_id") == args.task_id]
    if not matches:
        raise ValueError(f"MBPP task ID {args.task_id} was not found.")
    original_task = dict(matches[0])
    original_ok, original_reason = validate_mbpp_candidate(
        original_task,
        original_task["code"],
    )
    if not original_ok:
        raise RuntimeError(f"Original task failed validation: {original_reason}")

    candidates, checks, check_details = _load_candidates(
        args.candidate_set,
        original_task,
    )
    if not candidates:
        summary = summarize_paired_candidates(
            [],
            checks,
            minimum_pairs=MINIMUM_PAIRED_RUNS,
            maximum_pairs=MAXIMUM_PAIRED_RUNS,
            minimum_reduction=MINIMUM_MEASURED_REDUCTION,
            family_confidence=FAMILY_CONFIDENCE,
        )
        summary["behavior_checks"] = check_details
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("No candidates passed the MBPP tests; no energy measurements were run.")
        print(f"Summary saved to {args.summary_output}")
        return

    rows = []
    summary = None
    for run_number in range(1, MAXIMUM_PAIRED_RUNS + 1):
        block_rows = {}
        order = balanced_candidate_order(run_number, candidates)
        for version in order:
            task = original_task if version == "original" else candidates[version]
            print(
                f"Measuring task={args.task_id}, version={version}, "
                f"pair={run_number}/{MAXIMUM_PAIRED_RUNS}"
            )
            row = measure_version(task, version, args.task_id, run_number)
            row["candidate_id"] = version
            row["behavior_checks_passed"] = (
                True if version == "original" else checks[version]
            )
            block_rows[version] = row

        # The single baseline reading is intentionally shared by each paired
        # comparison in the block; label those records to preserve pair keys.
        for candidate_id in candidates:
            baseline = dict(block_rows["original"])
            baseline["candidate_id"] = candidate_id
            rows.append(baseline)
            rows.append(dict(block_rows[candidate_id]))

        summary = summarize_paired_candidates(
            rows,
            checks,
            minimum_pairs=MINIMUM_PAIRED_RUNS,
            maximum_pairs=MAXIMUM_PAIRED_RUNS,
            minimum_reduction=MINIMUM_MEASURED_REDUCTION,
            family_confidence=FAMILY_CONFIDENCE,
        )
        print(
            f"Set status={summary['status']} | "
            f"recommended={summary['recommended_candidate']} | "
            f"pairs={summary['paired_runs']}"
        )
        for candidate_id, result in summary["candidates"].items():
            print(
                f"  {candidate_id}: reduction="
                f"{result['geometric_mean_reduction_pct']}%, "
                f"family-wise interval=[{result['ci_lower_pct']}%, "
                f"{result['ci_upper_pct']}%], status={result['status']}"
            )
        if summary["status"] in {
            "recommended",
            "no_candidate_meets_threshold",
            "inconclusive",
            "insufficient_measurements",
        }:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    summary["behavior_checks"] = check_details
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Measurements saved to {args.output}")
    print(f"Evidence summary saved to {args.summary_output}")


if __name__ == "__main__":
    main()
