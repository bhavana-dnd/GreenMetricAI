from __future__ import annotations

import argparse
from pathlib import Path

from greenmetricai.dataset import collect_mbpp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect real MBPP CodeCarbon measurements."
    )

    parser.add_argument(
        "--task-id",
        type=int,
        action="append",
        help="Specific MBPP task ID. Can be supplied multiple times.",
    )

    parser.add_argument(
        "--tasks",
        type=int,
        default=None,
        help="Number of tasks from the beginning of MBPP.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Independent measurements per task.",
    )

    parser.add_argument(
        "--country",
        default="IND",
        help="Country ISO code for CodeCarbon.",
    )

    parser.add_argument(
        "--calibration-timeout",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--measurement-timeout",
        type=float,
        default=180.0,
    )

    parser.add_argument(
        "--output",
        default="data/processed/mbpp_missing_6.csv",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("GREENMETRICAI REAL MBPP COLLECTION")
    print("=" * 70)

    if args.task_id:
        print("Task IDs:", sorted(args.task_id))
    else:
        print(
            "Tasks:",
            args.tasks if args.tasks else "all"
        )

    print("Runs per task:", args.runs)
    print()

    df = collect_mbpp(
        tasks=args.tasks if args.tasks else "all",
        task_id=args.task_id,
        country=args.country,
        runs=args.runs,
        calibration_timeout=args.calibration_timeout,
        measurement_timeout=args.measurement_timeout,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)

    print(f"Rows collected: {len(df)}")
    print(f"Tasks collected: {df['task_id'].nunique()}")

    print()
    print("ROWS PER TASK:")
    print(df.groupby("task_id").size())

    print()
    print("REPEAT COUNTS PER TASK:")
    print(
        df.groupby("task_id")["measurement_repeats"].unique()
    )

    print()
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()