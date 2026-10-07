from greenmetricai.mbpp import load_mbpp, build_repeated_code
from greenmetricai.dataset import determine_duration_repeats
from greenmetricai.executor import execute_python_code


def main():
    tasks = load_mbpp()

    task = next(
        task
        for task in tasks
        if task["task_id"] == 67
    )

    print("=" * 70)
    print("TASK 67 - DURATION PROTOCOL VALIDATION")
    print("=" * 70)

    plan = determine_duration_repeats(
        task,
        target_seconds=10.0,
        calibration_timeout=5,
    )

    repeats = plan["measurement_repeats"]

    print()
    print(f"Selected repeats: {repeats:,}")
    print(
        f"Estimated benchmark runtime: "
        f"{plan['estimated_runtime_sec']:.3f}s"
    )

    code = build_repeated_code(
        task,
        repeats=repeats,
    )

    print()
    print("Starting 5 measurements...")
    print()

    for run in range(1, 6):

        result = execute_python_code(
            code,
            timeout_seconds=60,
            country_iso="IND",
        )

        energy = result["energy_kwh"]

        energy_per_execution = (
            energy / repeats
        )

        benchmark_runtime = (
            result["benchmark_runtime_sec"]
        )

        runtime_per_execution = (
            benchmark_runtime / repeats
        )

        print(
            f"Run {run}: "
            f"benchmark_runtime="
            f"{benchmark_runtime:.4f}s | "
            f"runtime/execution="
            f"{runtime_per_execution:.9f}s | "
            f"energy="
            f"{energy:.6e} kWh | "
            f"energy/execution="
            f"{energy_per_execution:.6e} kWh | "
            f"wall_clock="
            f"{result['wall_clock_sec']:.3f}s"
        )


if __name__ == "__main__":
    main()