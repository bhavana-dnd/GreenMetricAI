import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil


def _run_untracked_benchmark(code, result_path):
    """
    Run a benchmark without CodeCarbon.

    Used only for calibration so we can estimate how many
    repetitions are appropriate for the real measurement.
    """

    start = time.perf_counter()

    exec(
        code,
        {
            "__name__": "__main__"
        }
    )

    runtime = time.perf_counter() - start

    result = {
        "runtime_sec": runtime,
    }

    Path(result_path).write_text(
        json.dumps(result),
        encoding="utf-8"
    )


def calibrate_python_code(
    code,
    timeout_seconds=30
):
    """
    Run benchmark code without CodeCarbon.

    This is used only for calibration to determine an appropriate
    repeat count for the real energy measurement.

    Returns:
        {
            "runtime_sec": float
        }

    A subprocess timeout is converted into TimeoutError so callers
    can safely handle calibration failures.
    """

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        code_path = temp_dir / "benchmark.py"
        result_path = temp_dir / "result.json"
        runner_path = temp_dir / "runner.py"

        code_path.write_text(
            code,
            encoding="utf-8",
        )

        runner_code = f"""
import json
import time
from pathlib import Path

code_path = Path(r"{code_path}")
result_path = Path(r"{result_path}")

code = code_path.read_text(
    encoding="utf-8"
)

start = time.perf_counter()

exec(
    code,
    {{
        "__name__": "__main__"
    }}
)

runtime = time.perf_counter() - start

result_path.write_text(
    json.dumps({{
        "runtime_sec": runtime
    }}),
    encoding="utf-8"
)
"""

        runner_path.write_text(
            runner_code,
            encoding="utf-8",
        )

        proc = subprocess.Popen(
            [
                sys.executable,
                str(runner_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:

            out, err = proc.communicate(
                timeout=timeout_seconds
            )

        except subprocess.TimeoutExpired:

            proc.kill()

            try:
                proc.communicate(timeout=2)
            except Exception:
                pass

            raise TimeoutError(
                f"Calibration exceeded "
                f"{timeout_seconds} seconds."
            )

        if proc.returncode != 0:

            raise RuntimeError(
                "Calibration process failed.\n"
                f"stdout:\n{out}\n"
                f"stderr:\n{err}"
            )

        if not result_path.exists():

            raise RuntimeError(
                "Calibration completed but did not "
                "produce a result file."
            )

        result = json.loads(
            result_path.read_text(
                encoding="utf-8"
            )
        )

        return result


def _run_tracked_benchmark(
    code,
    country_iso,
    result_path
):
    """
    Runs INSIDE the benchmark subprocess.

    CodeCarbon measures this process directly.

    runtime_sec:
        Total tracked-process time including CodeCarbon
        start/stop overhead.

    benchmark_runtime_sec:
        Time spent executing the benchmark code itself.
    """

    from codecarbon import OfflineEmissionsTracker

    tracker = OfflineEmissionsTracker(
        country_iso_code=country_iso,
        project_name="greenmetricai_mbpp",
        tracking_mode="process",
        save_to_file=False,
        measure_power_secs=1,
    )

    process = psutil.Process(
        os.getpid()
    )

    start = time.perf_counter()

    peak_memory_mb = (
        process.memory_info().rss
        / 1024**2
    )

    tracker.start()

    benchmark_runtime = None

    try:

        current_memory_mb = (
            process.memory_info().rss
            / 1024**2
        )

        peak_memory_mb = max(
            peak_memory_mb,
            current_memory_mb
        )

        benchmark_start = time.perf_counter()

        exec(
            code,
            {
                "__name__": "__main__"
            }
        )

        benchmark_runtime = (
            time.perf_counter()
            - benchmark_start
        )

        current_memory_mb = (
            process.memory_info().rss
            / 1024**2
        )

        peak_memory_mb = max(
            peak_memory_mb,
            current_memory_mb
        )

    finally:

        emissions_kg = tracker.stop()

    runtime = (
        time.perf_counter()
        - start
    )

    data = tracker.final_emissions_data

    result = {
        "runtime_sec":
            runtime,

        "benchmark_runtime_sec":
            benchmark_runtime,

        "memory_peak_mb":
            peak_memory_mb,

        "energy_kwh":
            getattr(
                data,
                "energy_consumed",
                None
            ),

        "carbon_emissions_g":
            emissions_kg * 1000,
    }

    Path(result_path).write_text(
        json.dumps(result),
        encoding="utf-8"
    )


def execute_python_code(
    code,
    timeout_seconds=120,
    country_iso="IND"
):
    """
    Launch a separate Python process.

    CodeCarbon is started inside that process.
    """

    with tempfile.TemporaryDirectory(
        prefix="greenmetricai_"
    ) as td:

        td_path = Path(td)

        benchmark_path = (
            td_path / "benchmark.py"
        )

        result_path = (
            td_path / "result.json"
        )

        runner_path = (
            td_path / "runner.py"
        )

        benchmark_path.write_text(
            code,
            encoding="utf-8"
        )

        runner_code = f'''
import sys
from pathlib import Path

sys.path.insert(
    0,
    {str(Path.cwd())!r}
)

from greenmetricai.executor import (
    _run_tracked_benchmark
)

code = Path(
    {str(benchmark_path)!r}
).read_text(
    encoding="utf-8"
)

_run_tracked_benchmark(
    code=code,
    country_iso={country_iso!r},
    result_path={str(result_path)!r}
)
'''

        runner_path.write_text(
            runner_code,
            encoding="utf-8"
        )

        start = time.perf_counter()

        proc = subprocess.Popen(
            [
                sys.executable,
                str(runner_path)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:

            out, err = proc.communicate(
                timeout=timeout_seconds
            )

        except subprocess.TimeoutExpired:

            proc.kill()

            out, err = proc.communicate()

            raise TimeoutError(
                f"Benchmark exceeded "
                f"{timeout_seconds} seconds"
            )

        if proc.returncode != 0:

            raise RuntimeError(
                err[-5000:]
            )

        if not result_path.exists():

            raise RuntimeError(
                "Benchmark completed but did not "
                "produce a measurement result."
            )

        result = json.loads(
            result_path.read_text(
                encoding="utf-8"
            )
        )

        result["wall_clock_sec"] = (
            time.perf_counter()
            - start
        )

        return result


def hardware_metadata():

    return {
        "platform":
            platform.platform(),

        "python_version":
            platform.python_version(),

        "cpu_count":
            os.cpu_count(),

        "cpu_model":
            platform.processor(),

        "ram_gb":
            round(
                psutil.virtual_memory()
                .total
                / 1024**3,
                2
            )
    }