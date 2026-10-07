from statistics import mean, stdev

from greenmetricai.mbpp import (
    load_mbpp,
    build_repeated_code,
)
from greenmetricai.dataset import (
    determine_duration_repeats,
)
from greenmetricai.executor import (
    execute_python_code,
)


TASK_IDS = [
    2,
    20,
    67,
    80,
    98,
    123,
    124,
    137,
    139,
    163,
]

MEASUREMENTS_PER_TASK = 5
TARGET_SECONDS = 5.0


def calculate_cv(values):
    """
    Calculate sample coefficient of variation (%).
    """

    if len(values) < 2:
        return None

    avg = mean(values)

    if avg == 0:
        return None

    return (
        stdev(values) / avg
    ) * 100


def main():

    tasks = load_mbpp()

    task_map = {
        task["task_id"]: task
        for task in tasks
    }

    missing = [
        task_id
        for task_id in TASK_IDS
        if task_id not in task_map
    ]

    if missing:

        raise ValueError(
            f"These task IDs are not present "
            f"in the downloaded MBPP dataset: "
            f"{missing}"
        )

    print("=" * 70)
    print(
        "DURATION-TARGETED MEASUREMENT "
        "PROTOCOL VALIDATION"
    )
    print("=" * 70)

    print()
    print(
        f"Tasks: {TASK_IDS}"
    )

    print(
        f"Measurements per task: "
        f"{MEASUREMENTS_PER_TASK}"
    )

    print(
        f"Target benchmark duration: "
        f"{TARGET_SECONDS} second"
    )

    all_results = {}

    for task_id in TASK_IDS:

        task = task_map[task_id]

        print()
        print("=" * 70)
        print(f"TASK {task_id}")
        print("=" * 70)

        plan = determine_duration_repeats(
            task,
            target_seconds=TARGET_SECONDS,
            calibration_timeout=5,
        )

        repeats = plan[
            "measurement_repeats"
        ]

        estimated = plan.get(
            "estimated_runtime_sec"
        )

        print()
        print(
            f"Selected repeats: "
            f"{repeats:,}"
        )

        if estimated is None:

            print(
                "Estimated benchmark runtime: "
                "unavailable"
            )

        else:

            print(
                f"Estimated benchmark runtime: "
                f"{estimated:.3f}s"
            )

        print(
            f"Measurement class: "
            f"{plan.get('measurement_class')}"
        )

        code = build_repeated_code(
            task,
            repeats=repeats,
        )

        print()
        print(
            f"Starting "
            f"{MEASUREMENTS_PER_TASK} "
            f"measurements..."
        )

        energies = []
        runtimes = []

        for run in range(
            1,
            MEASUREMENTS_PER_TASK + 1,
        ):

            result = execute_python_code(
                code,
                timeout_seconds=60,
                country_iso="IND",
            )

            energy = result[
                "energy_kwh"
            ]

            benchmark_runtime = result[
                "benchmark_runtime_sec"
            ]

            energy_per_execution = (
                energy / repeats
            )

            runtime_per_execution = (
                benchmark_runtime /
                repeats
            )

            energies.append(
                energy_per_execution
            )

            runtimes.append(
                runtime_per_execution
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

        energy_cv = calculate_cv(
            energies
        )

        runtime_cv = calculate_cv(
            runtimes
        )

        all_results[task_id] = {
            "repeats": repeats,
            "energy_cv_percent": energy_cv,
            "runtime_cv_percent": runtime_cv,
            "mean_energy_per_execution": mean(
                energies
            ),
            "mean_runtime_per_execution": mean(
                runtimes
            ),
        }

        print()
        print(
            f"Task {task_id} summary:"
        )

        print(
            f"  Mean energy/execution: "
            f"{mean(energies):.6e} kWh"
        )

        print(
            f"  Energy CV: "
            f"{energy_cv:.2f}%"
        )

        print(
            f"  Mean runtime/execution: "
            f"{mean(runtimes):.9f}s"
        )

        print(
            f"  Runtime CV: "
            f"{runtime_cv:.2f}%"
        )

    print()
    print("=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)

    for task_id, result in all_results.items():

        print(
            f"Task {task_id}: "
            f"repeats={result['repeats']:,} | "
            f"energy CV="
            f"{result['energy_cv_percent']:.2f}% | "
            f"runtime CV="
            f"{result['runtime_cv_percent']:.2f}%"
        )


if __name__ == "__main__":
    main()