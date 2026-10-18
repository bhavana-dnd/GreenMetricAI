"""Sequential, noise-aware validation for paired energy measurements."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, stdev

from scipy.stats import t


def balanced_candidate_order(run_number, candidate_ids):
    """Return a deterministic, position-balanced order for one measurement block.

    Each block contains the shared baseline and every behavior-passing
    candidate once. Rotating the order and reversing every second block
    balances simple warm-up and thermal-order effects across repeated runs.
    """
    candidates = sorted({str(value) for value in candidate_ids})
    if not candidates:
        raise ValueError("At least one candidate is required.")
    if "original" in candidates:
        raise ValueError("'original' is reserved for the shared baseline.")
    if run_number < 1:
        raise ValueError("run_number must be positive.")

    versions = ["original", *candidates]
    cycle, offset = divmod(run_number - 1, len(versions))
    if cycle % 2:
        versions.reverse()
    ordered = versions[offset:] + versions[:offset]
    return ordered


def summarize_paired_candidates(
    rows,
    candidate_checks,
    *,
    minimum_pairs=5,
    maximum_pairs=15,
    minimum_reduction=0.03,
    family_confidence=0.95,
):
    """Select a measured candidate while controlling repeated and multi-candidate looks.

    ``rows`` contains one shared ``original`` reading per block and one reading
    per candidate, with ``run``, ``version``, and positive
    ``energy_kwh_per_execution`` fields. Candidate names are the keys in
    ``candidate_checks``; only candidates whose behavior checks passed enter
    the statistical comparison. Bonferroni-adjusted Student-t intervals on
    paired log energy ratios cover all passing candidates and all planned
    interim looks. A candidate is recommended only if its lower bound clears
    the practical reduction threshold and separates from every other passing
    candidate's upper bound.
    """
    if not 1 <= minimum_pairs <= maximum_pairs:
        raise ValueError("Require 1 <= minimum_pairs <= maximum_pairs.")
    if maximum_pairs < 2:
        raise ValueError("maximum_pairs must be at least 2.")
    if not 0 < minimum_reduction < 1:
        raise ValueError("minimum_reduction must be between 0 and 1.")
    if not 0 < family_confidence < 1:
        raise ValueError("family_confidence must be between 0 and 1.")

    checked_ids = [str(candidate_id) for candidate_id in candidate_checks]
    passing_ids = [
        candidate_id for candidate_id in checked_ids
        if candidate_checks[candidate_id] is True
    ]
    grouped = defaultdict(lambda: defaultdict(dict))
    attempted_runs = set()
    for row in rows:
        try:
            run = int(row["run"])
            version = str(row["version"])
        except (KeyError, TypeError, ValueError):
            continue
        if version != "original" and version not in passing_ids:
            continue
        attempted_runs.add(run)
        try:
            energy = float(row["energy_kwh_per_execution"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(energy) and energy > 0:
            grouped[version][run] = energy

    candidate_results = {}
    valid_count = 0
    attempted_count = len(attempted_runs)
    # Family-wise correction spans every passing candidate and every planned
    # look, including candidates selected after seeing the same measurements.
    comparisons = max(1, len(passing_ids))
    alpha_per_interval = (
        (1 - family_confidence) / (maximum_pairs * comparisons)
    )
    for candidate_id in checked_ids:
        result = {
            "candidate_id": candidate_id,
            "behavior_checks_passed": candidate_checks[candidate_id] is True,
            "paired_runs": 0,
            "attempted_pairs": attempted_count,
            "geometric_mean_reduction_pct": None,
            "ci_lower_pct": None,
            "ci_upper_pct": None,
            "minimum_reduction_pct": minimum_reduction * 100,
            "family_confidence_pct": family_confidence * 100,
            "status": "rejected_behavior_checks",
        }
        if candidate_id not in passing_ids:
            candidate_results[candidate_id] = result
            continue

        ratios = []
        candidate_runs = grouped[candidate_id]
        baseline_runs = grouped["original"]
        paired_runs = sorted(set(candidate_runs) & set(baseline_runs))
        for run in paired_runs:
            ratios.append(
                math.log(candidate_runs[run]) - math.log(baseline_runs[run])
            )
        count = len(ratios)
        valid_count = max(valid_count, count)
        result["paired_runs"] = count

        if count:
            average_log_ratio = mean(ratios)
            result["geometric_mean_reduction_pct"] = (
                1 - math.exp(average_log_ratio)
            ) * 100
        if count < 2:
            result["status"] = (
                "insufficient_measurements"
                if attempted_count >= maximum_pairs
                else "collecting"
            )
            candidate_results[candidate_id] = result
            continue

        critical = t.ppf(
            1 - alpha_per_interval / 2,
            df=count - 1,
        )
        margin = critical * stdev(ratios) / math.sqrt(count)
        # Reduction is 1-exp(log(candidate / baseline)); reverse endpoints.
        result["ci_lower_pct"] = (
            1 - math.exp(average_log_ratio + margin)
        ) * 100
        result["ci_upper_pct"] = (
            1 - math.exp(average_log_ratio - margin)
        ) * 100
        if count < minimum_pairs:
            result["status"] = (
                "insufficient_measurements"
                if attempted_count >= maximum_pairs
                else "collecting"
            )
        elif result["ci_lower_pct"] > minimum_reduction * 100:
            result["status"] = "confirmed_reduction"
        elif result["ci_upper_pct"] < minimum_reduction * 100:
            result["status"] = "below_minimum_effect"
        elif count >= maximum_pairs:
            result["status"] = "inconclusive"
        else:
            result["status"] = "continue"
        candidate_results[candidate_id] = result

    passing_results = [candidate_results[value] for value in passing_ids]
    confirmed = [
        result for result in passing_results
        if result["status"] == "confirmed_reduction"
    ]
    recommended = None
    if confirmed:
        for result in sorted(
            confirmed,
            key=lambda item: item["ci_lower_pct"],
            reverse=True,
        ):
            competitors = [
                other for other in passing_results
                if other["candidate_id"] != result["candidate_id"]
            ]
            if all(
                other["ci_upper_pct"] is not None
                and result["ci_lower_pct"] > other["ci_upper_pct"]
                for other in competitors
            ):
                recommended = result["candidate_id"]
                break

    if not passing_ids:
        overall_status = "no_behavior_passing_candidates"
    elif recommended is not None:
        overall_status = "recommended"
    elif passing_results and all(
        result["ci_upper_pct"] is not None
        and result["ci_upper_pct"] < minimum_reduction * 100
        for result in passing_results
    ):
        overall_status = "no_candidate_meets_threshold"
    elif attempted_count >= maximum_pairs:
        overall_status = (
            "insufficient_measurements"
            if valid_count < minimum_pairs
            else "inconclusive"
        )
    else:
        overall_status = "continue"

    return {
        "status": overall_status,
        "recommended_candidate": recommended,
        "paired_runs": valid_count,
        "attempted_pairs": attempted_count,
        "minimum_reduction_pct": minimum_reduction * 100,
        "family_confidence_pct": family_confidence * 100,
        "candidate_count": len(checked_ids),
        "passing_candidate_count": len(passing_ids),
        "candidates": candidate_results,
    }


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
