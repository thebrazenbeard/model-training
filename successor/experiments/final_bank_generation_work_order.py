from __future__ import annotations

from successor.experiments.final_bank_custodian_binding import (
    validate_custodian_binding,
)


DIMENSIONS = tuple(f"H{i:02d}" for i in range(1, 21))


def _family_rows(
    *,
    lane: str,
    dimension: str,
    provenance_class: str,
    count: int,
    cases_per_family: int,
    source_actor_ids: list[str],
    start_index: int,
) -> list[dict]:
    if not source_actor_ids:
        raise ValueError(f"source actor pool empty:{provenance_class}")
    rows: list[dict] = []
    for offset in range(count):
        index = start_index + offset
        actor_id = source_actor_ids[offset % len(source_actor_ids)]
        rows.append({
            "family_id": (
                f"v10-final-{lane}-{dimension.lower()}-"
                f"{provenance_class.lower()}-{index:03d}"
            ),
            "lane": lane,
            "dimension": dimension,
            "family_index": index,
            "provenance_class": provenance_class,
            "source_actor_id": actor_id,
            "cases_per_family": cases_per_family,
        })
    return rows


def prepare_generation_work_order(binding: dict) -> dict:
    gate = validate_custodian_binding(binding)
    if gate["status"] != "CUSTODIAN_BINDING_PASS":
        raise RuntimeError(
            "custodian binding HOLD: " + " | ".join(gate["reasons"])
        )

    synthetic = binding["synthetic_custodians"]
    authors = [entry["actor_id"] for entry in binding["human_authors"]]
    families: list[dict] = []

    for dimension in DIMENSIONS:
        families.extend(_family_rows(
            lane="behavioral",
            dimension=dimension,
            provenance_class="synthetic_A",
            count=20,
            cases_per_family=10,
            source_actor_ids=[synthetic["A"]["actor_id"]],
            start_index=1,
        ))
        families.extend(_family_rows(
            lane="behavioral",
            dimension=dimension,
            provenance_class="synthetic_B",
            count=20,
            cases_per_family=10,
            source_actor_ids=[synthetic["B"]["actor_id"]],
            start_index=21,
        ))
        families.extend(_family_rows(
            lane="behavioral",
            dimension=dimension,
            provenance_class="human_seeded",
            count=10,
            cases_per_family=10,
            source_actor_ids=authors,
            start_index=41,
        ))

        families.extend(_family_rows(
            lane="adversarial",
            dimension=dimension,
            provenance_class="synthetic_A",
            count=8,
            cases_per_family=5,
            source_actor_ids=[synthetic["A"]["actor_id"]],
            start_index=1,
        ))
        families.extend(_family_rows(
            lane="adversarial",
            dimension=dimension,
            provenance_class="synthetic_B",
            count=8,
            cases_per_family=5,
            source_actor_ids=[synthetic["B"]["actor_id"]],
            start_index=9,
        ))
        families.extend(_family_rows(
            lane="adversarial",
            dimension=dimension,
            provenance_class="human_seeded",
            count=4,
            cases_per_family=5,
            source_actor_ids=authors,
            start_index=17,
        ))

    lane_family_counts = {
        lane: sum(1 for row in families if row["lane"] == lane)
        for lane in ("behavioral", "adversarial")
    }
    planned_case_count = sum(row["cases_per_family"] for row in families)

    return {
        "schema": "V10_FINAL_BANK_GENERATION_WORK_ORDER_V1",
        "status": "READY_FOR_CUSTODY_GENERATION",
        "family_count": len(families),
        "lane_family_counts": lane_family_counts,
        "planned_case_count": planned_case_count,
        "dimensions": list(DIMENSIONS),
        "families": families,
        "custody_surface_id": binding["custody"]["surface_id"],
        "human_reviewer_ids": [
            entry["actor_id"] for entry in binding["human_reviewers"]
        ],
        "boundaries": {
            "contains_final_plaintext": False,
            "generation_must_execute_inside_bound_custody_surface": True,
            "training_lane_access_to_generated_plaintext": False,
        },
    }
