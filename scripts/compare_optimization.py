from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from greenmetricai.dataset import determine_duration_repeats
from greenmetricai.executor import execute_python_code, hardware_metadata
from greenmetricai.mbpp import build_repeated_code, load_mbpp
from greenmetricai.optimizer import (GeminiProvider, GreenMetricEnergyModel,
                                     optimize_mbpp_task,
                                     validate_mbpp_candidate)

TARGET_SECONDS = 5.0
MEASUREMENTS_PER_VERSION = 5
COUNTRY = "IND"


def measure_version(task, version, task_id, run_number):
    """Calibrate and measure one version of an MBPP task."""

    plan = determine_duration_repeats(
        task,
        target_seconds=TARGET_SECONDS,
        calibration_timeout=5,
        max_repeats=10_000_000,
    )

    repeats = plan["measurement_repeats"]

    code = build_repeated_code(task, repeats=repeats)

    result = execute_python_code(
        code,
        timeout_seconds=180,
        country_iso=COUNTRY,
    )

    energy = result.get("energy_kwh")
    carbon = result.get("carbon_emissions_g")
    benchmark_runtime = result.get("benchmark_runtime_sec")

    return {
        "task_id": task_id,
        "version": version,
        "run": run_number,
        "measurement_repeats": repeats,
        "measurement_class": plan["measurement_class"],
        "estimated_runtime_sec": plan["estimated_runtime_sec"],
        "runtime_sec_per_execution": (
            benchmark_runtime / repeats
            if benchmark_runtime is not None
            else result["runtime_sec"] / repeats
        ),
        "energy_kwh_per_execution": (
            energy / repeats if energy is not None else None
        ),
        "carbon_emissions_g_per_execution": (
            carbon / repeats if carbon is not None else None
        ),
        "benchmark_runtime_sec": benchmark_runtime,
        "wall_clock_sec": result.get("wall_clock_sec"),
        **hardware_metadata(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare original and optimized MBPP measurements."
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=2,
        help="Official MBPP task ID to test first (default: 2).",
    )
    parser.add_argument(
        "--output",
        default="data/processed/optimization_comparison.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    tasks = load_mbpp()
    matches = [t for t in tasks if t.get("task_id") == args.task_id]

    if not matches:
        raise ValueError(f"MBPP task ID {args.task_id} was not found.")

    original_task = dict(matches[0])

    # Validate the baseline before requesting an optimization.
    original_ok, original_reason = validate_mbpp_candidate(
        original_task,
        original_task["code"],
    )
    if not original_ok:
        raise RuntimeError(
            f"Original task failed validation: {original_reason}"
        )

    print(f"Optimizing MBPP task {args.task_id}...")
    model = GreenMetricEnergyModel()
    provider = GeminiProvider()

    optimization = optimize_mbpp_task(
        original_task,
        model=model,
        provider=provider,
    )

    candidate_code = optimization["optimized_code"]
    candidate_task = dict(original_task)
    candidate_task["code"] = candidate_code

    candidate_ok, candidate_reason = validate_mbpp_candidate(
        candidate_task,
        candidate_code,
    )
    if not candidate_ok:
        raise RuntimeError(
            f"Candidate failed validation: {candidate_reason}"
        )

    if candidate_code.strip() == original_task["code"].strip():
        print(
            "No different optimized candidate was accepted. "
            "Measurements will still run, but this is not an "
            "optimization-success result."
        )

    rows = []

    # Alternate order between runs to reduce simple order bias.
    for run_number in range(1, MEASUREMENTS_PER_VERSION + 1):
        versions = [
            ("original", original_task),
            ("optimized", candidate_task),
        ]
        if run_number % 2 == 0:
            versions.reverse()

        for version, task in versions:
            print(
                f"Measuring task={args.task_id}, "
                f"version={version}, run={run_number}/"
                f"{MEASUREMENTS_PER_VERSION}"
            )
            row = measure_version(
                task,
                version,
                args.task_id,
                run_number,
            )
            row["candidate_different"] = (
                candidate_code.strip()
                != original_task["code"].strip()
            )
            row["original_tests_passed"] = original_ok
            row["candidate_tests_passed"] = candidate_ok
            row["predicted_energy_kwh"] = optimization.get(
                "predicted_energy_kwh"
            )
            rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_data = pd.DataFrame(rows)

    # Preserve previous experiments instead of silently overwriting them.
    if output_path.exists():
        new_data.to_csv(
            output_path,
            mode="a",
            header=False,
            index=False,
        )
    else:
        new_data.to_csv(output_path, index=False)

    print(f"\nMeasurements saved to: {output_path}")
    print(f"Rows added: {len(new_data)}")

    summary = new_data.groupby("version").agg(
        mean_runtime_sec=("runtime_sec_per_execution", "mean"),
        mean_energy_kwh=("energy_kwh_per_execution", "mean"),
        mean_carbon_g=("carbon_emissions_g_per_execution", "mean"),
    )
    print("\nSummary of this run:")
    print(summary)

    original_energy = summary.loc["original", "mean_energy_kwh"]
    optimized_energy = summary.loc["optimized", "mean_energy_kwh"]

    if pd.notna(original_energy) and original_energy > 0:
        change = (
            (original_energy - optimized_energy) / original_energy
        ) * 100
        print(f"\nMeasured energy reduction: {change:.3f}%")
        print("Positive = reduction; negative = energy increased.")
    else:
        print("\nEnergy reduction could not be calculated from these measurements.")


if __name__ == "__main__":
    main()