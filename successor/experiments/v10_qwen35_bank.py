from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re
import unicodedata

DIMENSIONS = tuple(f"H{i:02d}" for i in range(1, 21))
RETENTION_ALLOCATION = {
    "knowledge_factuality": 300,
    "reasoning_math": 300,
    "coding": 250,
    "instruction_following": 250,
    "extraction_structured": 200,
    "truthfulness_factual_calibration": 200,
}
LANE_COUNTS = {"behavioral": 10_000, "adversarial": 2_000, "retention": 1_500}
REQUIRED_FIELDS = {
    "case_id",
    "lane",
    "family_id",
    "prompt",
    "source_id",
    "source_revision",
    "source_terms",
    "source_hash",
    "generation_method",
    "generation_actor_id",
    "grader_contract",
    "review_receipt",
}
REVIEWER_CLASSES = {"HUMAN_REVIEW", "EXTERNAL_MODEL_REVIEW", "EXTERNAL_REVIEW"}
REVIEW_SUBJECT_FIELDS = (
    "case_id",
    "lane",
    "dimension",
    "category",
    "family_id",
    "prompt",
    "source_id",
    "source_revision",
    "source_terms",
    "source_hash",
    "generation_method",
    "generation_actor_id",
    "grader_contract",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def normalize_prompt(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("prompt must be nonempty text")
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def prompt_fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_prompt(text).encode("utf-8")).hexdigest()


def review_subject_payload(row: dict) -> dict:
    return {
        field: row.get(field)
        for field in REVIEW_SUBJECT_FIELDS
        if field in row
    }


def review_subject_digest(row: dict) -> str:
    payload = json.dumps(
        review_subject_payload(row),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha(value) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _review_ok(value, generation_actor_id: str, grader_kind: str) -> bool:
    if not isinstance(value, dict):
        return False
    reviewer_class = value.get("reviewer_class")
    allowed_classes = {"HUMAN_REVIEW"} if grader_kind == "semantic_review" else REVIEWER_CLASSES
    return (
        value.get("verdict") == "ADMIT"
        and _nonempty(value.get("reviewer_id"))
        and reviewer_class in allowed_classes
        and _nonempty(value.get("provider"))
        and _nonempty(value.get("runtime"))
        and value.get("independent_from_generation") is True
        and value.get("reviewer_id") != generation_actor_id
        and _sha(value.get("artifact_digest"))
        and _sha(value.get("subject_digest"))
    )


def review_receipt_matches_case(row: dict) -> bool:
    grader_kind = row.get("grader_contract", {}).get("kind")
    receipt = row.get("review_receipt")
    return (
        _review_ok(
            receipt,
            row.get("generation_actor_id"),
            grader_kind,
        )
        and receipt["subject_digest"] == review_subject_digest(row)
    )


def _pending_review_ok(value, generation_actor_id: str) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("verdict") == "PENDING_INDEPENDENT_REVIEW"
        and value.get("reviewer_id") == "UNBOUND"
        and value.get("reviewer_class") == "UNBOUND"
        and value.get("provider") == "UNBOUND"
        and value.get("runtime") == "UNBOUND"
        and value.get("independent_from_generation") is False
        and _sha(value.get("artifact_digest"))
        and _sha(value.get("subject_digest"))
    )


def _source_audit_ok(value, generation_actor_id: str) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("verdict") == "SOURCE_CONTRACT_VALID"
        and _nonempty(value.get("auditor_id"))
        and value.get("auditor_class") == "SOURCE_BOUND_DETERMINISTIC_AUDIT"
        and _nonempty(value.get("provider"))
        and _nonempty(value.get("runtime"))
        and value.get("independent_review") is False
        and value.get("auditor_id") != generation_actor_id
        and _sha(value.get("artifact_digest"))
        and _sha(value.get("subject_digest"))
    )


def _grader_ok(value) -> bool:
    if not isinstance(value, dict):
        return False
    common = _nonempty(value.get("grader_id")) and _nonempty(value.get("grader_version"))
    if not common:
        return False
    if value.get("kind") == "deterministic":
        return _sha(value.get("answer_key_digest"))
    if value.get("kind") == "semantic_review":
        return _sha(value.get("rubric_digest"))
    return False


def preflight_bank(rows: list[dict], *, exclusion_hashes: set[str]) -> dict:
    reasons: list[str] = []
    ids: set[str] = set()
    prompts: set[str] = set()
    lanes = Counter()
    dimensions = defaultdict(Counter)
    families = defaultdict(set)
    retention = Counter()

    if not isinstance(exclusion_hashes, set) or any(not _sha(x) for x in exclusion_hashes):
        raise ValueError("exclusion_hashes must be a set of SHA-256 strings")

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            reasons.append(f"invalid_row:{index}")
            continue
        case_id = row.get("case_id")
        if not _nonempty(case_id):
            reasons.append(f"invalid_case_id:{index}")
            continue

        missing = REQUIRED_FIELDS - set(row)
        if missing:
            reasons.append(f"missing_fields:{case_id}:{','.join(sorted(missing))}")
            continue
        if case_id in ids:
            reasons.append(f"duplicate_case_id:{case_id}")
        ids.add(case_id)

        try:
            fingerprint = prompt_fingerprint(row["prompt"])
        except ValueError:
            reasons.append(f"invalid_prompt:{case_id}")
            continue
        if fingerprint in prompts:
            reasons.append(f"duplicate_prompt:{case_id}")
        prompts.add(fingerprint)
        if fingerprint in exclusion_hashes:
            reasons.append(f"excluded_prompt:{case_id}")

        for field in (
            "family_id",
            "source_id",
            "source_revision",
            "source_terms",
            "generation_method",
            "generation_actor_id",
        ):
            if not _nonempty(row.get(field)):
                reasons.append(f"invalid_{field}:{case_id}")
        if not _sha(row.get("source_hash")):
            reasons.append(f"invalid_source_hash:{case_id}")
        if not _grader_ok(row.get("grader_contract")):
            reasons.append(f"invalid_grader_contract:{case_id}")
        grader_kind = row.get("grader_contract", {}).get("kind")
        if not _review_ok(
            row.get("review_receipt"),
            row.get("generation_actor_id"),
            grader_kind,
        ):
            reasons.append(f"invalid_review_receipt:{case_id}")
        elif row["review_receipt"]["subject_digest"] != review_subject_digest(row):
            reasons.append(f"review_subject_digest_mismatch:{case_id}")

        lane = row.get("lane")
        if lane not in LANE_COUNTS:
            reasons.append(f"invalid_lane:{case_id}")
            continue
        lanes[lane] += 1
        if lane in {"behavioral", "adversarial"}:
            dim = row.get("dimension")
            if dim not in DIMENSIONS:
                reasons.append(f"invalid_dimension:{case_id}")
            else:
                dimensions[lane][dim] += 1
                if lane == "behavioral":
                    families[dim].add(row["family_id"])
        else:
            category = row.get("category")
            if category not in RETENTION_ALLOCATION:
                reasons.append(f"invalid_retention_category:{case_id}")
            else:
                retention[category] += 1

    for lane, expected in LANE_COUNTS.items():
        if lanes[lane] != expected:
            reasons.append(f"lane_count:{lane}:{lanes[lane]}!={expected}")
    for dim in DIMENSIONS:
        if dimensions["behavioral"][dim] != 500:
            reasons.append(
                f"behavioral_dimension_count:{dim}:{dimensions['behavioral'][dim]}!=500"
            )
        if len(families[dim]) < 50:
            reasons.append(f"behavioral_family_count:{dim}:{len(families[dim])}<50")
        if dimensions["adversarial"][dim] != 100:
            reasons.append(
                f"adversarial_dimension_count:{dim}:{dimensions['adversarial'][dim]}!=100"
            )
    for category, expected in RETENTION_ALLOCATION.items():
        if retention[category] != expected:
            reasons.append(f"retention_count:{category}:{retention[category]}!={expected}")

    reasons = sorted(set(reasons))
    return {
        "schema": "V10_QWEN35_FINAL_BANK_PREFLIGHT_V1",
        "status": "READY_TO_FREEZE" if not reasons else "HOLD",
        "case_count": len(rows),
        "lane_counts": dict(sorted(lanes.items())),
        "behavioral_family_counts": {d: len(families[d]) for d in DIMENSIONS},
        "reasons": reasons,
        "plaintext_prompts_in_receipt": False,
        "independent_review_metadata_required": True,
    }


def preflight_retention_rows(
    rows: list[dict],
    *,
    exclusion_hashes: set[str],
    expected_allocation: dict[str, int] | None = None,
) -> dict:
    reasons: list[str] = []
    allocation = dict(expected_allocation or RETENTION_ALLOCATION)
    ids: set[str] = set()
    prompts: set[str] = set()
    retention = Counter()

    if not isinstance(exclusion_hashes, set) or any(not _sha(x) for x in exclusion_hashes):
        raise ValueError("exclusion_hashes must be a set of SHA-256 strings")

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            reasons.append(f"invalid_row:{index}")
            continue
        case_id = row.get("case_id")
        if not _nonempty(case_id):
            reasons.append(f"invalid_case_id:{index}")
            continue
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            reasons.append(f"missing_fields:{case_id}:{','.join(sorted(missing))}")
            continue
        if case_id in ids:
            reasons.append(f"duplicate_case_id:{case_id}")
        ids.add(case_id)

        if row.get("lane") != "retention":
            reasons.append(f"invalid_lane:{case_id}")
            continue
        category = row.get("category")
        if category not in allocation:
            reasons.append(f"invalid_retention_category:{case_id}")
        else:
            retention[category] += 1

        try:
            fingerprint = prompt_fingerprint(row["prompt"])
        except ValueError:
            reasons.append(f"invalid_prompt:{case_id}")
            continue
        if fingerprint in prompts:
            reasons.append(f"duplicate_prompt:{case_id}")
        prompts.add(fingerprint)
        if fingerprint in exclusion_hashes:
            reasons.append(f"excluded_prompt:{case_id}")

        for field in (
            "family_id",
            "source_id",
            "source_revision",
            "source_terms",
            "generation_method",
            "generation_actor_id",
        ):
            if not _nonempty(row.get(field)):
                reasons.append(f"invalid_{field}:{case_id}")
        if not _sha(row.get("source_hash")):
            reasons.append(f"invalid_source_hash:{case_id}")
        if not _grader_ok(row.get("grader_contract")):
            reasons.append(f"invalid_grader_contract:{case_id}")
        if not _pending_review_ok(
            row.get("review_receipt"),
            row.get("generation_actor_id"),
        ):
            reasons.append(f"invalid_pending_review_receipt:{case_id}")
        elif row["review_receipt"]["subject_digest"] != review_subject_digest(row):
            reasons.append(f"pending_review_subject_digest_mismatch:{case_id}")
        if not _source_audit_ok(
            row.get("source_audit_receipt"),
            row.get("generation_actor_id"),
        ):
            reasons.append(f"invalid_source_audit_receipt:{case_id}")
        elif row["source_audit_receipt"]["subject_digest"] != row.get("source_hash"):
            reasons.append(f"source_audit_subject_digest_mismatch:{case_id}")

    for category, expected in allocation.items():
        if retention[category] != expected:
            reasons.append(f"retention_count:{category}:{retention[category]}!={expected}")

    reasons = sorted(set(reasons))
    return {
        "schema": "V10_QWEN35_RETENTION_PREFLIGHT_V1",
        "status": "STRUCTURE_READY" if not reasons else "HOLD",
        "case_count": len(rows),
        "retention_counts": {
            category: retention[category] for category in allocation
        },
        "reasons": reasons,
        "plaintext_prompts_in_receipt": False,
        "semantic_independent_review_required": False,
        "deterministic_source_audit_required": True,
        "review_policy": "SEE_V10_QWEN35_RETENTION_ADMISSION_V3",
        "admission_review_status": "OUT_OF_SCOPE_FOR_STRUCTURE_PREFLIGHT",
        "source_audit_is_not_independent_review": True,
    }
