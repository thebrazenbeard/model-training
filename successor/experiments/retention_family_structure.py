from __future__ import annotations

from collections import Counter, defaultdict
import math


def required_family_counts(
    allocation: dict[str, int],
    *,
    max_cases_per_family: int = 10,
) -> dict[str, int]:
    if max_cases_per_family < 1:
        raise ValueError("max_cases_per_family must be positive")
    return {
        category: math.ceil(count / max_cases_per_family)
        for category, count in allocation.items()
    }


def assess_retention_family_structure(
    rows: list[dict],
    *,
    allocation: dict[str, int],
    max_cases_per_family: int = 10,
) -> dict:
    required = required_family_counts(
        allocation,
        max_cases_per_family=max_cases_per_family,
    )
    counts = Counter()
    families: dict[str, set[str]] = defaultdict(set)
    family_sizes: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        category = row.get("category")
        if category not in allocation:
            continue
        counts[category] += 1
        family_id = str(row.get("family_id", ""))
        families[category].add(family_id)
        family_sizes[category][family_id] += 1

    reasons = []
    categories = {}
    for category, expected_rows in allocation.items():
        observed_rows = counts[category]
        observed_families = len(families[category])
        max_family = max(family_sizes[category].values(), default=0)
        if observed_rows != expected_rows:
            reasons.append(
                f"row_count:{category}:{observed_rows}!={expected_rows}"
            )
        if observed_families < required[category]:
            reasons.append(
                f"family_count:{category}:{observed_families}<"
                f"{required[category]}"
            )
        categories[category] = {
            "rows": observed_rows,
            "families": observed_families,
            "required_families": required[category],
            "max_family_size": max_family,
            "mean_cases_per_family": (
                observed_rows / observed_families
                if observed_families
                else None
            ),
        }
    return {
        "schema": "V10_RETENTION_FAMILY_STRUCTURE_V1",
        "status": "PASS" if not reasons else "HOLD",
        "max_cases_per_family_target": max_cases_per_family,
        "categories": categories,
        "reasons": sorted(reasons),
    }
