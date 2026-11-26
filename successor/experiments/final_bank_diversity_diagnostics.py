from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
import unicodedata


_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_QUOTED = re.compile(r"""(["']).*?\1""")
_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_HEX = re.compile(r"\b(?:0x)?[0-9a-f]{12,}\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<!\w)[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?!\w)")
_TOKEN = re.compile(r"[\w<>]+", re.UNICODE)


def _collapse(text: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", text).casefold().split()
    )


def lexical_template_key(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("prompt must be nonempty text")
    value = _collapse(text)
    value = _URL.sub("<url>", value)
    value = _EMAIL.sub("<email>", value)
    value = _UUID.sub("<id>", value)
    value = _HEX.sub("<id>", value)
    value = _QUOTED.sub("<quoted>", value)
    value = _NUMBER.sub("<num>", value)
    return " ".join(value.split())


def _token_set(text: str) -> set[str]:
    return set(_TOKEN.findall(lexical_template_key(text)))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _share(counter: Counter) -> float:
    total = sum(counter.values())
    return max(counter.values(), default=0) / total if total else 0.0


def _effective_count(counter: Counter) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    concentration = sum((count / total) ** 2 for count in counter.values())
    return 1.0 / concentration if concentration else 0.0


def _scope(row: dict) -> tuple[str, str, str]:
    lane = row.get("lane")
    if lane in {"behavioral", "adversarial"}:
        dimension = row.get("dimension")
        if not isinstance(dimension, str) or not dimension:
            raise ValueError("behavioral/adversarial row missing dimension")
        return lane, dimension, f"{lane}:{dimension}"
    if lane == "retention":
        category = row.get("category")
        if not isinstance(category, str) or not category:
            raise ValueError("retention row missing category")
        return lane, category, f"retention:{category}"
    raise ValueError(f"unsupported lane: {lane}")


def _family_scope(record: dict) -> tuple[str, str, str]:
    lane = record.get("lane")
    if lane in {"behavioral", "adversarial"}:
        dimension = record.get("dimension")
        if not isinstance(dimension, str) or not dimension:
            raise ValueError("family manifest missing dimension")
        return lane, dimension, f"{lane}:{dimension}"
    if lane == "retention":
        category = record.get("category")
        if not isinstance(category, str) or not category:
            raise ValueError("retention family manifest missing category")
        return lane, category, f"retention:{category}"
    raise ValueError(f"unsupported family lane: {lane}")


def _validate_family_manifest(
    rows: list[dict],
    family_manifest: list[dict],
) -> dict[tuple[str, str, str], dict]:
    index: dict[tuple[str, str, str], dict] = {}
    for record in family_manifest:
        if not isinstance(record, dict):
            raise ValueError("family manifest entry must be object")
        lane, scope, _ = _family_scope(record)
        family_id = record.get("family_id")
        if not isinstance(family_id, str) or not family_id:
            raise ValueError("family manifest entry missing family_id")
        key = (lane, scope, family_id)
        if key in index:
            raise ValueError(f"duplicate family manifest entry: {key}")
        for field in (
            "provenance_class",
            "source_actor_id",
            "failure_mechanism",
            "independence_claim",
        ):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"family manifest {family_id} missing {field}"
                )
        index[key] = record

    for row in rows:
        lane, scope, _ = _scope(row)
        family_id = row.get("family_id")
        key = (lane, scope, family_id)
        record = index.get(key)
        if record is None:
            # Distinguish a missing family from a same-id wrong-scope entry.
            same_id = [
                item
                for item in family_manifest
                if item.get("family_id") == family_id
            ]
            if same_id:
                raise ValueError(
                    f"family manifest scope mismatch: {family_id}"
                )
            raise ValueError(
                f"family manifest missing referenced family: {family_id}"
            )
    return index


def _validate_semantic_assignments(
    rows: list[dict],
    assignments: dict[str, dict[str, str]],
) -> None:
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("semantic cluster assignments missing")
    case_ids = {row.get("case_id") for row in rows}
    if None in case_ids or len(case_ids) != len(rows):
        raise ValueError("case IDs must be unique and nonempty")
    for threshold, mapping in assignments.items():
        if not isinstance(threshold, str) or not threshold:
            raise ValueError("semantic threshold key invalid")
        if not isinstance(mapping, dict) or set(mapping) != case_ids:
            raise ValueError(
                f"semantic cluster case set mismatch at {threshold}"
            )
        if any(
            not isinstance(cluster, str) or not cluster
            for cluster in mapping.values()
        ):
            raise ValueError(
                f"semantic cluster ID invalid at {threshold}"
            )


def _near_duplicate_stats(
    prompts: list[str],
    thresholds: tuple[float, ...],
) -> dict[str, dict]:
    token_sets = [_token_set(prompt) for prompt in prompts]
    total_pairs = len(token_sets) * (len(token_sets) - 1) // 2
    counts = {threshold: 0 for threshold in thresholds}
    for i, left in enumerate(token_sets):
        for right in token_sets[i + 1 :]:
            similarity = _jaccard(left, right)
            for threshold in thresholds:
                if similarity >= threshold:
                    counts[threshold] += 1
    return {
        f"{threshold:.2f}": {
            "pair_count": counts[threshold],
            "total_pairs": total_pairs,
            "density": (
                counts[threshold] / total_pairs
                if total_pairs
                else 0.0
            ),
            "metric": "SET_JACCARD_OVER_MASKED_LEXICAL_TOKENS",
        }
        for threshold in thresholds
    }


def _distribution(values) -> dict:
    counter = Counter(value for value in values if value is not None)
    return dict(sorted(counter.items()))


def compute_diversity_diagnostics(
    rows: list[dict],
    *,
    family_manifest: list[dict],
    semantic_cluster_assignments: dict[str, dict[str, str]],
    near_duplicate_thresholds: tuple[float, ...] = (0.80, 0.90),
    rejected_records: list[dict] | None = None,
) -> dict:
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a nonempty list")
    if any(
        not isinstance(value, (int, float))
        or not 0.0 <= float(value) <= 1.0
        for value in near_duplicate_thresholds
    ):
        raise ValueError("near-duplicate thresholds must be within [0,1]")
    thresholds = tuple(
        sorted({float(value) for value in near_duplicate_thresholds})
    )
    family_index = _validate_family_manifest(rows, family_manifest)
    _validate_semantic_assignments(
        rows,
        semantic_cluster_assignments,
    )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        case_id = row.get("case_id")
        prompt = row.get("prompt")
        family_id = row.get("family_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("row missing case_id")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"row missing prompt: {case_id}")
        if not isinstance(family_id, str) or not family_id:
            raise ValueError(f"row missing family_id: {case_id}")
        _, _, group_key = _scope(row)
        grouped[group_key].append(row)

    groups = {}
    for group_key in sorted(grouped):
        group_rows = grouped[group_key]
        family_counts = Counter(row["family_id"] for row in group_rows)
        actor_counts = Counter(
            row.get("generation_actor_id") for row in group_rows
        )
        method_counts = Counter(
            row.get("generation_method") for row in group_rows
        )
        source_counts = Counter(row.get("source_id") for row in group_rows)
        template_counts = Counter(
            lexical_template_key(row["prompt"]) for row in group_rows
        )

        provenance_counts = Counter()
        for family_id in family_counts:
            lane, scope, _ = _scope(
                next(
                    row
                    for row in group_rows
                    if row["family_id"] == family_id
                )
            )
            record = family_index[(lane, scope, family_id)]
            provenance_counts[record["provenance_class"]] += 1

        semantic = {}
        group_case_ids = {row["case_id"] for row in group_rows}
        for threshold in sorted(semantic_cluster_assignments):
            cluster_counts = Counter(
                cluster_id
                for case_id, cluster_id
                in semantic_cluster_assignments[threshold].items()
                if case_id in group_case_ids
            )
            semantic[threshold] = {
                "cluster_count": len(cluster_counts),
                "max_cluster_share": _share(cluster_counts),
                "effective_cluster_count": _effective_count(
                    cluster_counts
                ),
            }

        groups[group_key] = {
            "row_count": len(group_rows),
            "family_count": len(family_counts),
            "max_family_share": _share(family_counts),
            "effective_family_count": _effective_count(
                family_counts
            ),
            "generation_actor_count": len(actor_counts),
            "max_generation_actor_share": _share(actor_counts),
            "generation_method_count": len(method_counts),
            "max_generation_method_share": _share(method_counts),
            "source_count": len(source_counts),
            "max_source_share": _share(source_counts),
            "lexical_template_count": len(template_counts),
            "max_lexical_template_share": _share(template_counts),
            "effective_lexical_template_count": _effective_count(
                template_counts
            ),
            "near_duplicate": _near_duplicate_stats(
                [row["prompt"] for row in group_rows],
                thresholds,
            ),
            "semantic_clusters": semantic,
            "provenance_family_counts": dict(
                sorted(provenance_counts.items())
            ),
            "difficulty_distribution": _distribution(
                row.get("difficulty") for row in group_rows
            ),
            "severity_distribution": _distribution(
                row.get("severity") for row in group_rows
            ),
        }

    rejected = Counter()
    for record in rejected_records or []:
        if not isinstance(record, dict):
            raise ValueError("rejected record must be object")
        reason = record.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("rejected record missing reason")
        rejected[reason] += 1

    return {
        "schema": "V10_FINAL_BANK_DIVERSITY_DIAGNOSTICS_V1",
        "status": "DIAGNOSTICS_COMPLETE",
        "row_count": len(rows),
        "group_count": len(groups),
        "groups": groups,
        "rejected_quarantined_counts_by_reason": dict(
            sorted(rejected.items())
        ),
        "method": {
            "family_effective_count": "INVERSE_SIMPSON",
            "lexical_template_normalization": (
                "UNICODE_NFKC_CASEFOLD_COLLAPSE_WHITESPACE_"
                "MASK_URL_EMAIL_UUID_HEX_QUOTED_NUMBER_V1"
            ),
            "near_duplicate_metric": (
                "SET_JACCARD_OVER_MASKED_LEXICAL_TOKENS"
            ),
            "near_duplicate_reporting_thresholds": [
                f"{value:.2f}" for value in thresholds
            ],
            "semantic_clusters": (
                "CUSTODY_SUPPLIED_CASE_CLUSTER_ASSIGNMENTS; "
                "RECEIPT_REPORTS_ONLY AGGREGATES"
            ),
        },
        "hard_diversity_thresholds": "NONE_PREDECLARED",
        "plaintext_in_receipt": False,
        "claim_ceiling": (
            "AGGREGATE_DIVERSITY_DIAGNOSTICS_ONLY / "
            "NO POST_HOC ADMISSION THRESHOLD / "
            "DOES NOT ESTABLISH FAMILY INDEPENDENCE OR MODEL BENEFIT"
        ),
    }
