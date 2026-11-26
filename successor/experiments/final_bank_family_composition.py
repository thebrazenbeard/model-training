from __future__ import annotations

from collections import Counter, defaultdict


PROVENANCE_CLASSES = {
    "synthetic_A",
    "synthetic_B",
    "human_seeded",
}


def _nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _group_key(lane: str, dimension: str) -> str:
    return f"{lane}:{dimension}"


def _actor_binding(custodian_binding: dict) -> tuple[str, str, set[str]]:
    synthetic = custodian_binding.get("synthetic_custodians")
    if not isinstance(synthetic, dict):
        raise ValueError("synthetic custodian binding missing")
    a = synthetic.get("A")
    b = synthetic.get("B")
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise ValueError("synthetic custodian A/B binding missing")
    actor_a = a.get("actor_id")
    actor_b = b.get("actor_id")
    if not _nonempty(actor_a) or not _nonempty(actor_b):
        raise ValueError("synthetic custodian actor IDs missing")
    if actor_a == actor_b:
        raise ValueError("synthetic custodian actor IDs not distinct")

    humans = custodian_binding.get("human_authors")
    if not isinstance(humans, list) or not humans:
        raise ValueError("human author binding missing")
    human_ids = {
        item.get("actor_id")
        for item in humans
        if isinstance(item, dict) and _nonempty(item.get("actor_id"))
    }
    if not human_ids:
        raise ValueError("human author actor IDs missing")
    return actor_a, actor_b, human_ids


def _spec_lane(generation_spec: dict, lane: str) -> dict:
    value = generation_spec.get(lane)
    if not isinstance(value, dict):
        raise ValueError(f"generation spec missing lane: {lane}")
    dimensions = value.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError(f"generation spec {lane} dimensions missing")
    for field in (
        "rows_per_dimension",
        "families_per_dimension",
        "cases_per_family",
    ):
        if not isinstance(value.get(field), int) or value[field] < 1:
            raise ValueError(
                f"generation spec {lane} {field} invalid"
            )
    provenance = value.get("family_provenance_per_dimension")
    if not isinstance(provenance, dict):
        raise ValueError(
            f"generation spec {lane} provenance mix missing"
        )
    if set(provenance) != PROVENANCE_CLASSES:
        raise ValueError(
            f"generation spec {lane} provenance key set invalid"
        )
    if any(
        not isinstance(count, int) or count < 0
        for count in provenance.values()
    ):
        raise ValueError(
            f"generation spec {lane} provenance counts invalid"
        )
    if sum(provenance.values()) != value["families_per_dimension"]:
        raise ValueError(
            f"generation spec {lane} provenance/family total mismatch"
        )
    if (
        value["families_per_dimension"] * value["cases_per_family"]
        != value["rows_per_dimension"]
    ):
        raise ValueError(
            f"generation spec {lane} row/family total mismatch"
        )
    return value


def validate_final_bank_family_composition(
    rows: list[dict],
    *,
    family_manifest: list[dict],
    generation_spec: dict,
    custodian_binding: dict,
) -> dict:
    reasons: list[str] = []
    actor_a, actor_b, human_ids = _actor_binding(custodian_binding)
    specs = {
        lane: _spec_lane(generation_spec, lane)
        for lane in ("behavioral", "adversarial")
    }

    row_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    row_family_ids_by_lane: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            reasons.append(f"invalid_row:{index}")
            continue
        lane = row.get("lane")
        if lane not in specs:
            continue
        dimension = row.get("dimension")
        family_id = row.get("family_id")
        actor_id = row.get("generation_actor_id")
        if dimension not in specs[lane]["dimensions"]:
            reasons.append(
                f"invalid_dimension:{lane}:{dimension}"
            )
            continue
        if not _nonempty(family_id):
            reasons.append(f"invalid_family_id:{lane}:{dimension}:{index}")
            continue
        if not _nonempty(actor_id):
            reasons.append(
                f"invalid_generation_actor:{lane}:{dimension}:{family_id}"
            )
        key = (lane, dimension, family_id)
        row_groups[key].append(row)
        row_family_ids_by_lane[lane].add(family_id)

    for family_id in sorted(
        row_family_ids_by_lane["behavioral"]
        & row_family_ids_by_lane["adversarial"]
    ):
        reasons.append(f"cross_lane_family_id:{family_id}")

    manifest_index: dict[tuple[str, str, str], dict] = {}
    for index, record in enumerate(family_manifest):
        if not isinstance(record, dict):
            reasons.append(f"invalid_manifest_entry:{index}")
            continue
        lane = record.get("lane")
        if lane not in specs:
            continue
        dimension = record.get("dimension")
        family_id = record.get("family_id")
        if dimension not in specs[lane]["dimensions"]:
            reasons.append(
                f"manifest_invalid_dimension:{lane}:{dimension}"
            )
            continue
        if not _nonempty(family_id):
            reasons.append(
                f"manifest_invalid_family_id:{lane}:{dimension}:{index}"
            )
            continue
        key = (lane, dimension, family_id)
        if key in manifest_index:
            reasons.append(
                f"duplicate_manifest_family:{lane}:{dimension}:{family_id}"
            )
            continue
        provenance = record.get("provenance_class")
        if provenance not in PROVENANCE_CLASSES:
            reasons.append(
                f"invalid_provenance:{lane}:{dimension}:{family_id}"
            )
        for field in (
            "source_actor_id",
            "failure_mechanism",
            "independence_claim",
        ):
            if not _nonempty(record.get(field)):
                reasons.append(
                    f"manifest_missing_{field}:"
                    f"{lane}:{dimension}:{family_id}"
                )
        manifest_index[key] = record

    row_keys = set(row_groups)
    manifest_keys = set(manifest_index)
    for lane, dimension, family_id in sorted(row_keys - manifest_keys):
        reasons.append(
            f"rows_family_without_manifest:{lane}:{dimension}:{family_id}"
        )
    for lane, dimension, family_id in sorted(manifest_keys - row_keys):
        reasons.append(
            f"manifest_family_without_rows:{lane}:{dimension}:{family_id}"
        )

    groups = {}
    for lane in ("behavioral", "adversarial"):
        spec = specs[lane]
        for dimension in spec["dimensions"]:
            group_key = _group_key(lane, dimension)
            keys = [
                key
                for key in row_keys | manifest_keys
                if key[0] == lane and key[1] == dimension
            ]
            row_count = sum(
                len(row_groups.get(key, []))
                for key in keys
            )
            family_ids = {key[2] for key in keys}
            if row_count != spec["rows_per_dimension"]:
                reasons.append(
                    f"row_count:{group_key}:{row_count}!="
                    f"{spec['rows_per_dimension']}"
                )
            if len(family_ids) != spec["families_per_dimension"]:
                reasons.append(
                    f"family_count:{group_key}:{len(family_ids)}!="
                    f"{spec['families_per_dimension']}"
                )

            provenance_counts = Counter()
            for family_id in sorted(family_ids):
                key = (lane, dimension, family_id)
                family_rows = row_groups.get(key, [])
                record = manifest_index.get(key)
                if len(family_rows) != spec["cases_per_family"]:
                    reasons.append(
                        f"family_case_count:{group_key}:{family_id}:"
                        f"{len(family_rows)}!={spec['cases_per_family']}"
                    )
                if record is None:
                    continue

                provenance = record.get("provenance_class")
                if provenance in PROVENANCE_CLASSES:
                    provenance_counts[provenance] += 1

                source_actor = record.get("source_actor_id")
                expected_actor = None
                if provenance == "synthetic_A":
                    expected_actor = actor_a
                elif provenance == "synthetic_B":
                    expected_actor = actor_b
                elif provenance == "human_seeded":
                    if source_actor not in human_ids:
                        reasons.append(
                            f"source_actor_mismatch:"
                            f"{group_key}:{family_id}"
                        )
                if (
                    expected_actor is not None
                    and source_actor != expected_actor
                ):
                    reasons.append(
                        f"source_actor_mismatch:"
                        f"{group_key}:{family_id}"
                    )

                family_actor_ids = {
                    row.get("generation_actor_id")
                    for row in family_rows
                }
                if _nonempty(source_actor) and source_actor not in family_actor_ids:
                    reasons.append(
                        f"source_actor_has_no_case:"
                        f"{group_key}:{family_id}"
                    )
                if (
                    provenance == "human_seeded"
                    and not (family_actor_ids & human_ids)
                ):
                    reasons.append(
                        f"human_seeded_without_human_case:"
                        f"{group_key}:{family_id}"
                    )

            expected_mix = spec[
                "family_provenance_per_dimension"
            ]
            for provenance in sorted(PROVENANCE_CLASSES):
                actual = provenance_counts[provenance]
                expected = expected_mix[provenance]
                if actual != expected:
                    reasons.append(
                        f"provenance_count:{group_key}:{provenance}:"
                        f"{actual}!={expected}"
                    )

            groups[group_key] = {
                "row_count": row_count,
                "family_count": len(family_ids),
                "cases_per_family_expected": spec["cases_per_family"],
                "provenance_family_counts": dict(
                    sorted(provenance_counts.items())
                ),
            }

    reasons = sorted(set(reasons))
    return {
        "schema": "V10_FINAL_BANK_FAMILY_COMPOSITION_CHECK_V1",
        "status": "COMPOSITION_PASS" if not reasons else "HOLD",
        "reasons": reasons,
        "groups": groups,
        "plaintext_in_receipt": False,
        "claim_ceiling": (
            "FAMILY_QUOTA_AND_PROVENANCE_COMPOSITION_ONLY / "
            "DOES NOT PROVE SEMANTIC FAMILY INDEPENDENCE / "
            "DOES NOT ADMIT OR SCORE FINAL BANK"
        ),
    }
