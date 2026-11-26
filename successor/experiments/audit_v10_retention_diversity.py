from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata


_NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:\.\d+)?(?!\w)")
_QUOTED = re.compile(r"""(["']).*?\1""")
_CODE = re.compile(r"\b[A-Z]{2,5}\b")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
NEAR_DUPLICATE_THRESHOLDS = (0.80, 0.90, 0.95)


def _normalize(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("prompt must be nonempty text")
    return " ".join(unicodedata.normalize("NFKC", text).split())


def lexical_template(prompt: str) -> str:
    value = _normalize(prompt)
    value = _QUOTED.sub("<STR>", value)
    value = _NUMBER.sub("<NUM>", value)
    value = _CODE.sub("<CODE>", value)
    return value.casefold()


def _template_tokens(prompt: str) -> frozenset[str]:
    return frozenset(_WORD.findall(lexical_template(prompt)))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _effective_family_count(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    concentration = sum((count / total) ** 2 for count in counts.values())
    return round(1.0 / concentration, 6)


def _near_duplicate_report(rows: list[dict]) -> dict:
    tokens = [_template_tokens(row["prompt"]) for row in rows]
    total_pairs = math.comb(len(rows), 2) if len(rows) >= 2 else 0
    pair_counts = {threshold: 0 for threshold in NEAR_DUPLICATE_THRESHOLDS}
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            score = _jaccard(tokens[i], tokens[j])
            for threshold in NEAR_DUPLICATE_THRESHOLDS:
                if score >= threshold:
                    pair_counts[threshold] += 1
    return {
        "total_pairs": total_pairs,
        "thresholds": {
            f"{threshold:.2f}": {
                "pairs": pair_counts[threshold],
                "density": (
                    round(pair_counts[threshold] / total_pairs, 8)
                    if total_pairs
                    else 0.0
                ),
            }
            for threshold in NEAR_DUPLICATE_THRESHOLDS
        },
        "method": "token-set Jaccard over value-collapsed lexical templates",
    }


def _difficulty(rows: list[dict]) -> dict:
    values = [row.get("difficulty") for row in rows]
    present = [value for value in values if isinstance(value, str) and value.strip()]
    if not present:
        return {"status": "UNAVAILABLE"}
    counts = dict(sorted(Counter(present).items()))
    if len(present) != len(rows):
        return {
            "status": "PARTIAL",
            "labeled_rows": len(present),
            "rows": len(rows),
            "counts": counts,
        }
    return {"status": "AVAILABLE", "counts": counts}


def _category_report(rows: list[dict]) -> dict:
    family_counts = Counter(row["family_id"] for row in rows)
    template_counts = Counter(lexical_template(row["prompt"]) for row in rows)
    source_terms = Counter(row.get("source_terms", "UNKNOWN") for row in rows)
    generation_methods = Counter(
        row.get("generation_method", "UNKNOWN") for row in rows
    )
    source_revisions = Counter(
        row.get("source_revision", "UNKNOWN") for row in rows
    )
    total = len(rows)
    return {
        "rows": total,
        "unique_families": len(family_counts),
        "family_counts": dict(sorted(family_counts.items())),
        "max_family_share": round(max(family_counts.values()) / total, 8),
        "effective_family_count": _effective_family_count(family_counts),
        "unique_lexical_templates": len(template_counts),
        "max_lexical_template_share": round(
            max(template_counts.values()) / total, 8
        ),
        "unique_source_ids": len({row.get("source_id") for row in rows}),
        "source_id_unique_ratio": round(
            len({row.get("source_id") for row in rows}) / total, 8
        ),
        "source_terms": dict(sorted(source_terms.items())),
        "source_revisions": dict(sorted(source_revisions.items())),
        "generation_methods": dict(sorted(generation_methods.items())),
        "near_duplicate_pairs": _near_duplicate_report(rows),
        "difficulty": _difficulty(rows),
    }


def analyze_retention_diversity(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("retention rows must be nonempty")
    grouped: dict[str, list[dict]] = defaultdict(list)
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} is not an object")
        case_id = row.get("case_id")
        category = row.get("category")
        family_id = row.get("family_id")
        prompt = row.get("prompt")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"row {index} has invalid case_id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen_ids.add(case_id)
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"{case_id}: invalid category")
        if not isinstance(family_id, str) or not family_id.strip():
            raise ValueError(f"{case_id}: invalid family_id")
        _normalize(prompt)
        grouped[category].append(row)

    return {
        "schema": "V10_RETENTION_DIVERSITY_DIAGNOSTIC_V1",
        "status": "DIAGNOSTIC_ONLY_NO_PASS_FAIL_GATE",
        "rows": len(rows),
        "categories": {
            category: _category_report(grouped[category])
            for category in sorted(grouped)
        },
        "claim_ceiling": (
            "DESCRIPTIVE_DIVERSITY_DIAGNOSTICS_ONLY / "
            "NO_PROMOTION_THRESHOLD / NOT_QUALITY_PROOF"
        ),
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.input.read_bytes()
    rows = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    receipt = analyze_retention_diversity(rows)
    receipt["input_sha256"] = hashlib.sha256(raw).hexdigest()
    receipt["input_rows"] = len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
