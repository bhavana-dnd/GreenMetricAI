from __future__ import annotations

import math

import pandas as pd

from greenmetricai.features import extract_code_metrics
from greenmetricai.mbpp import load_mbpp, build_repeated_code
from greenmetricai.executor import (
    execute_python_code,
    calibrate_python_code,
    hardware_metadata,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FAST_REPEATS = 100_000
SLOW_REPEATS = 1

CALIBRATION_STAGES = [
    1,
    10,
    100,
    1_000,
    10_000,
]

CALIBRATION_STAGE_TIMEOUT_SEC = 5

MEASUREMENT_TIMEOUT_SEC = 180


# ---------------------------------------------------------------------
# Original safe-repeat calibration
# ---------------------------------------------------------------------

def determine_safe_repeats(
    task,
    calibration_timeout=CALIBRATION_STAGE_TIMEOUT_SEC,
):
    """
    Calibrate a task and determine a safe repeat count.

    This function is retained for the original collection workflow.

    The largest calibration stage that completes within the
    calibration timeout is selected.
    """

    last_safe_repeats = 1

    for repeats in CALIBRATION_STAGES:

        code = build_repeated_code(
            task,
            repeats=repeats,
        )

        try:

            result = calibrate_python_code(
                code,
                timeout_seconds=calibration_timeout,
            )

            runtime = result["runtime_sec"]

            print(
                f"    calibration: "
                f"repeats={repeats}, "
                f"runtime={runtime:.3f}s"
            )

            # If the calibration reaches the timeout,
            # stop and use the previous safe stage.
            if runtime >= calibration_timeout:

                print(
                    f"    calibration limit reached at "
                    f"{repeats} repeats"
                )

                return last_safe_repeats, "slow"

            last_safe_repeats = repeats

        except Exception as e:

            print(
                f"    calibration stopped at "
                f"repeats={repeats}: "
                f"{type(e).__name__}: {e}"
            )

            return last_safe_repeats, "slow"

    # Every calibration stage was fast enough.
    return FAST_REPEATS, "fast"


# ---------------------------------------------------------------------
# Prepare original measurement plan
# ---------------------------------------------------------------------

def prepare_measurement_plan(
    tasks,
    calibration_timeout=CALIBRATION_STAGE_TIMEOUT_SEC,
):
    """
    Calibrate every task exactly once.

    Returns a fixed measurement plan that can then be reused
    for multiple measurement runs.
    """

    plan = {}

    total_tasks = len(tasks)

    print()
    print("=" * 70)
    print("CALIBRATION PHASE")
    print("=" * 70)

    for idx, task in enumerate(tasks, 1):

        task_id = task.get("task_id")

        print(
            f"\n[{idx}/{total_tasks}] "
            f"Calibrating task={task_id}"
        )

        duration_plan = determine_duration_repeats(
            task,
            target_seconds=5.0,
            calibration_timeout=calibration_timeout,
            max_repeats=10_000_000,
        )

        repeats = duration_plan["measurement_repeats"]
        task_class = duration_plan["measurement_class"]

        plan[task_id] = {
            "measurement_repeats": repeats,
            "measurement_class": task_class,
            "target_seconds": 5.0,
            "estimated_runtime_sec":
                duration_plan["estimated_runtime_sec"],
        }

        print(
            f"    FINAL PLAN: "
            f"class={task_class}, "
            f"repeats={repeats}"
        )

    print()
    print("=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)

    return plan


# ---------------------------------------------------------------------
# Measure one MBPP task
# ---------------------------------------------------------------------

def measure_mbpp_task(
    task,
    plan,
    country="IND",
    measurement_timeout=MEASUREMENT_TIMEOUT_SEC,
    task_index=None,
):
    """
    Execute one MBPP task using a previously calibrated
    measurement plan.
    """

    current_task_id = task.get("task_id")

    code = task["code"]

    metrics = extract_code_metrics(
        code
    ).to_dict()

    repeats = plan[current_task_id]["measurement_repeats"]

    task_class = plan[current_task_id]["measurement_class"]

    measured_code = build_repeated_code(
        task,
        repeats=repeats,
    )

    result = execute_python_code(
        measured_code,
        timeout_seconds=measurement_timeout,
        country_iso=country,
    )

    energy_kwh = result.get(
        "energy_kwh"
    )

    carbon_g = result.get(
        "carbon_emissions_g"
    )

    runtime_sec = result["runtime_sec"]

    # This is the actual benchmark execution time measured
    # inside the CodeCarbon subprocess.
    benchmark_runtime_sec = result.get(
        "benchmark_runtime_sec"
    )

    energy_per_execution = (
        energy_kwh / repeats
        if energy_kwh is not None
        else None
    )

    carbon_per_execution = (
        carbon_g / repeats
        if carbon_g is not None
        else None
    )

    # Prefer actual benchmark execution time rather than
    # runtime_sec, because runtime_sec includes CodeCarbon
    # startup/shutdown overhead.
    if benchmark_runtime_sec is not None:

        runtime_per_execution = (
            benchmark_runtime_sec / repeats
        )

    else:

        runtime_per_execution = (
            runtime_sec / repeats
        )

    row = {
        **metrics,
        **result,

        "energy_kwh_per_execution":
            energy_per_execution,

        "carbon_emissions_g_per_execution":
            carbon_per_execution,

        "runtime_sec_per_execution":
            runtime_per_execution,

        "measurement_repeats":
            repeats,

        "measurement_class":
            task_class,

        "task_id":
            current_task_id,

        "benchmark":
            "MBPP-sanitized",

        "task_index":
            task_index,

        **hardware_metadata(),
    }

    return row


# ---------------------------------------------------------------------
# Duration-targeted calibration
# ---------------------------------------------------------------------

def determine_duration_repeats(
    task,
    target_seconds=10.0,
    calibration_timeout=5,
    max_repeats=10_000_000,
):
    """
    Determine a repeat count that targets approximately
    target_seconds of actual benchmark execution time.

    Calibration is performed without CodeCarbon.

    Very fast tasks require progressively larger calibration
    workloads. We therefore do not extrapolate from a single
    execution.

    The calibration stops once a workload reaches approximately
    one second of benchmark execution. This gives a much more
    reliable estimate of time per execution than a single tiny
    measurement.
    """

    stages = [
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        200,
        500,
        1_000,
        2_000,
        5_000,
        10_000,
        20_000,
        50_000,
        100_000,
        200_000,
        500_000,
        1_000_000,
    ]

    last_safe_repeats = None
    last_safe_runtime = None

    for repeats in stages:

        if repeats > max_repeats:
            break

        code = build_repeated_code(
            task,
            repeats=repeats,
        )

        try:

            result = calibrate_python_code(
                code,
                timeout_seconds=calibration_timeout,
            )

        except (TimeoutError, RuntimeError):

            break

        runtime = result.get(
            "runtime_sec"
        )

        if runtime is None or runtime <= 0:
            break

        last_safe_repeats = repeats
        last_safe_runtime = runtime

        print(
            f"    calibration: "
            f"{repeats:,} repeats -> "
            f"{runtime:.6f}s"
        )

        # Once the benchmark itself reaches around one second,
        # timing noise becomes much smaller relative to the workload.
        if runtime >= 1.0:
            break

    if last_safe_repeats is None:

        return {
            "measurement_repeats": 1,
            "estimated_runtime_sec": None,
            "measurement_class": "calibration_failed",
        }

    seconds_per_repeat = (
        last_safe_runtime /
        last_safe_repeats
    )

    estimated_repeats = math.ceil(
        target_seconds /
        seconds_per_repeat
    )

    estimated_repeats = max(
        1,
        min(
            estimated_repeats,
            max_repeats,
        ),
    )

    estimated_runtime = (
        estimated_repeats *
        seconds_per_repeat
    )

    return {
        "measurement_repeats":
            estimated_repeats,

        "estimated_runtime_sec":
            estimated_runtime,

        "measurement_class":
            "duration_targeted",
    }


# ---------------------------------------------------------------------
# Collect MBPP measurements
# ---------------------------------------------------------------------

def collect_mbpp(
    tasks="all",
    country="IND",
    measurement_timeout=MEASUREMENT_TIMEOUT_SEC,
    task_id=None,
    runs=1,
    calibration_timeout=CALIBRATION_STAGE_TIMEOUT_SEC,
    **kwargs,
):
    """
    Collect repeated real measurements.

    Calibration happens once per task before measurement runs.

    The same repeat count is reused for every measurement run.
    """

    data = load_mbpp()

    # ---------------------------------------------------------------
    # Select specific task IDs
    # ---------------------------------------------------------------

    if task_id is not None:

        if isinstance(
            task_id,
            (list, tuple, set),
        ):

            requested_ids = set(task_id)

            data = [
                task
                for task in data
                if task.get("task_id") in requested_ids
            ]

            found_ids = {
                task.get("task_id")
                for task in data
            }

            missing_ids = (
                requested_ids -
                found_ids
            )

            if missing_ids:

                raise ValueError(
                    f"MBPP task IDs not found: "
                    f"{sorted(missing_ids)}"
                )

            data.sort(
                key=lambda x: x.get("task_id")
            )

        else:

            data = [
                task
                for task in data
                if task.get("task_id") == task_id
            ]

            if not data:

                raise ValueError(
                    f"MBPP task_id {task_id} "
                    f"was not found."
                )

    # ---------------------------------------------------------------
    # Select first N tasks
    # ---------------------------------------------------------------

    elif tasks != "all":

        n = int(tasks)

        data = data[:n]

    if not data:

        raise ValueError(
            "No MBPP tasks selected."
        )

    # ---------------------------------------------------------------
    # CALIBRATE ONCE
    # ---------------------------------------------------------------

    measurement_plan = prepare_measurement_plan(
        data,
        calibration_timeout=calibration_timeout,
    )

    rows = []

    total_tasks = len(data)

    # ---------------------------------------------------------------
    # MEASURE
    # ---------------------------------------------------------------

    for run_number in range(
        1,
        runs + 1,
    ):

        print()
        print("=" * 70)
        print(
            f"MEASUREMENT RUN "
            f"{run_number}/{runs}"
        )
        print("=" * 70)

        for idx, task in enumerate(
            data,
            1,
        ):

            current_task_id = task.get(
                "task_id"
            )

            print(
                f"\n[{idx}/{total_tasks}] "
                f"task={current_task_id}"
            )

            plan = measurement_plan[
                current_task_id
            ]

            try:

                row = measure_mbpp_task(
                    task=task,
                    plan=measurement_plan,
                    country=country,
                    measurement_timeout=measurement_timeout,
                    task_index=idx,
                )

                row["measurement_run"] = (
                    run_number
                )

                rows.append(row)

                print(
                    f"    class="
                    f"{row['measurement_class']} "
                    f"repeats="
                    f"{row['measurement_repeats']} "
                    f"runtime="
                    f"{row['runtime_sec']:.3f}s "
                    f"energy="
                    f"{row['energy_kwh']} kWh "
                    f"energy/exec="
                    f"{row['energy_kwh_per_execution']}"
                )

            except Exception as e:

                print(
                    f"[SKIP] "
                    f"task={current_task_id}: "
                    f"{type(e).__name__}: {e}"
                )

    return pd.DataFrame(rows)



