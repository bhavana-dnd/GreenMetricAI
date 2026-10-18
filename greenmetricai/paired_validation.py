"""Sequential, noise-aware validation for paired energy measurements."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, stdev

from scipy.stats import t


def summarize_paired_energy(
    rows,
    *,
    minimum_pairs=5,
    maximum_pairs=15,
    minimum_reduction=0.03,
    family_confidence=0.95,
):
    """Estimate energy reduction from paired runs and decide whether to stop.

    Each pair is keyed by ``run`` and compares the optimized/original energy
    ratio. A Student-t interval is calculated on log ratios; Bonferroni
    adjustment across the possible interim looks limits false discoveries from
    repeated checking. The returned reduction and bounds are percentages.
    """
    by_run = defaultdict(dict)
    attempted_runs = set()
    for row in rows:
        version = str(row.get("version", "")).lower()
        if version not in {"original", "optimized"}:
            continue
        try:
            run = int(row["run"])
        except (KeyError, TypeError, ValueError):
            continue
        attempted_runs.add(run)
        try:
            energy = float(row["energy_kwh_per_execution"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(energy) and energy > 0:
            by_run[run][version] = energy

    log_ratios = [
        math.log(pair["optimized"]) - math.log(pair["original"])
        for pair in by_run.values()
        if {"original", "optimized"}.issubset(pair)
    ]
    count = len(log_ratios)
    result = {
        "paired_runs": count,
        "attempted_pairs": len(attempted_runs),
        "minimum_reduction_pct": minimum_reduction * 100,
        "family_confidence_pct": family_confidence * 100,
        "geometric_mean_reduction_pct": None,
        "ci_lower_pct": None,
        "ci_upper_pct": None,
        "status": "collecting",
    }

    if count == 0:
        if len(attempted_runs) >= maximum_pairs:
            result["status"] = "insufficient_measurements"
        return result

    average_log_ratio = mean(log_ratios)
    result["geometric_mean_reduction_pct"] = (
        1 - math.exp(average_log_ratio)
    ) * 100

    if count == 1:
        if len(attempted_runs) >= maximum_pairs:
            result["status"] = "insufficient_measurements"
        return result

    # Spend the family-wise error rate evenly across all planned interim looks.
    per_look_alpha = (1 - family_confidence) / maximum_pairs
    critical = t.ppf(1 - per_look_alpha / 2, df=count - 1)
    margin = critical * stdev(log_ratios) / math.sqrt(count)
    low_log = average_log_ratio - margin
    high_log = average_log_ratio + margin

    # Reduction is 1 - exp(log ratio), so the endpoints reverse.
    result["ci_lower_pct"] = (1 - math.exp(high_log)) * 100
    result["ci_upper_pct"] = (1 - math.exp(low_log)) * 100

    if count < minimum_pairs:
        result["status"] = (
            "insufficient_measurements"
            if len(attempted_runs) >= maximum_pairs
            else "collecting"
        )
    elif result["ci_lower_pct"] > result["minimum_reduction_pct"]:
        result["status"] = "confirmed_reduction"
    elif result["ci_upper_pct"] < result["minimum_reduction_pct"]:
        result["status"] = "below_minimum_effect"
    elif count >= maximum_pairs:
        result["status"] = "inconclusive"
    else:
        result["status"] = "continue"

    return result
