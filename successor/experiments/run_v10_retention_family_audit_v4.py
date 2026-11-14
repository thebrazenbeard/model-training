from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path

from successor.experiments.run_v10_retention_family_audit import (
    RUNTIME_FIELDS,
    _read_jsonl,
    _reviewer_matches_protocol,
    canonical_json,
    group_packet_by_family,
    sha256_bytes,
)
from successor.experiments.run_v10_retention_independent_audit import (
    _ollama_request,
    reviewer_identity,
)


REQUIRED_JUDGMENT_KEYS = {
    "prompt_well_posed",
    "grader_matches_prompt",
    "source_evidence_sufficient",
    "issue_code",
    "reason",
    "confidence",
}


def family_review_prompt_v4(cases: list[dict]) -> str:
    if not cases:
        raise ValueError("family review requires at least one case")
    family_ids = {case.get("family_id") for case in cases}
    if len(family_ids) != 1 or None in family_ids:
        raise ValueError("family review cases must share one family_id")

    subjects = [
        {
            "case_id": case["case_id"],
            "category": case["category"],
            "family_id": case["family_id"],
            "prompt": case["prompt"],
            "grader_contract": case["grader_contract"],
            "source_evidence": case.get("source_evidence"),
        }
        for case in cases
    ]
    return (
        "You are an independent benchmark-quality reviewer. "
        "Audit each case below independently; do not grade any model response. "
        "For each case, judge three questions only: whether the prompt is "
        "well-posed and unambiguous, whether the deterministic grader checks "
        "what the prompt actually asks, and whether supplied source evidence "
        "is sufficient where source evidence is relevant. "
        "For executable tests, mark grader_matches_prompt=false when tests "
        "permit materially wrong implementations or miss required behavior. "
        "Do not return a verdict field. The caller derives the verdict "
        "mechanically: ADMIT iff all three boolean judgments are true; "
        "otherwise REJECT. "
        "Return JSON only as an object with exactly one key named reviews. "
        "reviews must be an array with exactly one object per supplied case. "
        "Each object must contain exactly: case_id, "
        "prompt_well_posed (bool), grader_matches_prompt (bool), "
        "source_evidence_sufficient (bool), issue_code (NONE or short code), "
        "reason (short string), confidence (high|medium|low). "
        "Do not omit, add, merge, reorder, or rename case IDs.\nSUBJECTS="
        + canonical_json(subjects)
    )


def _derive_verdict(judgment: dict) -> str:
    return (
        "ADMIT"
        if all(
            judgment[field]
            for field in (
                "prompt_well_posed",
                "grader_matches_prompt",
                "source_evidence_sufficient",
            )
        )
        else "REJECT"
    )


def parse_family_judgments_v4(
    text: str,
    *,
    expected_case_ids: list[str],
) -> list[dict]:
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"reviews"}:
        raise ValueError("family review must use exact object schema")
    rows = value["reviews"]
    if not isinstance(rows, list):
        raise ValueError("reviews must be an array")

    expected = list(expected_case_ids)
    if len(expected) != len(set(expected)):
        raise ValueError("expected case IDs must be unique")

    parsed_by_id: dict[str, dict] = {}
    allowed_keys = {"case_id"} | REQUIRED_JUDGMENT_KEYS
    for row in rows:
        if not isinstance(row, dict) or set(row) != allowed_keys:
            raise ValueError("family review row must use closed schema")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("family review row has invalid case_id")
        if case_id in parsed_by_id:
            raise ValueError(
                f"duplicate family review case_id: {case_id}"
            )

        judgment = {
            key: row[key]
            for key in REQUIRED_JUDGMENT_KEYS
        }
        for field in (
            "prompt_well_posed",
            "grader_matches_prompt",
            "source_evidence_sufficient",
        ):
            if not isinstance(judgment[field], bool):
                raise ValueError(f"{field} must be boolean")
        if judgment["confidence"] not in {"high", "medium", "low"}:
            raise ValueError("invalid confidence")
        if (
            not isinstance(judgment["issue_code"], str)
            or not judgment["issue_code"]
        ):
            raise ValueError("invalid issue_code")
        if (
            not isinstance(judgment["reason"], str)
            or not judgment["reason"].strip()
        ):
            raise ValueError("invalid reason")

        parsed_by_id[case_id] = {
            "case_id": case_id,
            **judgment,
            "verdict": _derive_verdict(judgment),
        }

    if set(parsed_by_id) != set(expected):
        raise ValueError(
            "family review case set mismatch: "
            f"got={sorted(parsed_by_id)} expected={sorted(expected)}"
        )
    return [parsed_by_id[case_id] for case_id in expected]


def call_family_reviewer_v4(
    cases: list[dict],
    *,
    max_attempts: int = 3,
) -> tuple[list[dict], dict]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    prompt = family_review_prompt_v4(cases)
    expected_ids = [case["case_id"] for case in cases]
    failures: list[str] = []

    for attempt in range(1, max_attempts + 1):
        try:
            outer = _ollama_request(prompt)
            response_text = outer.get("response")
            if (
                not isinstance(response_text, str)
                or not response_text.strip()
            ):
                raise ValueError("empty reviewer response")
            judgments = parse_family_judgments_v4(
                response_text,
                expected_case_ids=expected_ids,
            )
            runtime = {
                key: outer.get(key)
                for key in RUNTIME_FIELDS
            }
            runtime["attempts"] = attempt
            if failures:
                runtime["prior_attempt_failures"] = failures
            return judgments, runtime
        except (ValueError, json.JSONDecodeError) as exc:
            failures.append(
                f"attempt_{attempt}:{type(exc).__name__}:{exc}"
            )
            if attempt == max_attempts:
                raise RuntimeError(
                    "family reviewer response invalid after "
                    f"{max_attempts} attempts: {' | '.join(failures)}"
                ) from exc
    raise AssertionError("unreachable")


def summarize_family_audit_v4(
    rows: list[dict],
    *,
    expected_families: set[str],
    expected_per_family: int,
) -> dict:
    verdicts = Counter(row.get("verdict") for row in rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        family_id = row.get("family_id")
        if isinstance(family_id, str):
            grouped[family_id].append(row)

    reasons: list[str] = []
    if verdicts["REJECT"]:
        reasons.append(f"reject_count:{verdicts['REJECT']}")

    unknown = sum(
        count
        for verdict, count in verdicts.items()
        if verdict not in {"ADMIT", "REJECT"}
    )
    if unknown:
        reasons.append(f"unknown_verdict_count:{unknown}")

    for family_id, family_rows in grouped.items():
        if len(family_rows) != expected_per_family:
            reasons.append(
                f"family_count:{family_id}:{len(family_rows)}!="
                f"{expected_per_family}"
            )

    missing = expected_families - set(grouped)
    extra = set(grouped) - expected_families
    if missing:
        reasons.append(f"missing_family_count:{len(missing)}")
    if extra:
        reasons.append(f"extra_family_count:{len(extra)}")

    reasons = sorted(set(reasons))
    return {
        "status": "PASS" if not reasons else "HOLD",
        "reviewed": len(rows),
        "families_reviewed": len(grouped),
        "families_expected": len(expected_families),
        "admit": verdicts["ADMIT"],
        "reject": verdicts["REJECT"],
        "reasons": reasons,
        "by_family": {
            family_id: dict(
                sorted(
                    Counter(
                        row["verdict"]
                        for row in family_rows
                    ).items()
                )
            )
            for family_id, family_rows in sorted(grouped.items())
        },
    }


def _validate_prior_v4_rows(
    prior_rows: list[dict],
    *,
    grouped: dict[str, list[dict]],
    protocol_sha: str,
    expected_per_family: int,
) -> dict[str, list[dict]]:
    prior_by_family: dict[str, list[dict]] = defaultdict(list)
    for row in prior_rows:
        family_id = row.get("family_id")
        if family_id not in grouped:
            raise RuntimeError(
                f"prior V4 review has unknown family:{family_id}"
            )
        if row.get("protocol_sha256") != protocol_sha:
            raise RuntimeError("prior V4 protocol hash mismatch")
        prior_by_family[family_id].append(row)

    for family_id, family_rows in prior_by_family.items():
        if len(family_rows) != expected_per_family:
            raise RuntimeError(
                f"partial prior V4 family:{family_id}"
            )
        expected_ids = {
            case["case_id"]
            for case in grouped[family_id]
        }
        actual_ids = {
            row.get("case_id")
            for row in family_rows
        }
        if actual_ids != expected_ids:
            raise RuntimeError(
                f"mismatched prior V4 family:{family_id}"
            )
        prompt = family_review_prompt_v4(grouped[family_id])
        prompt_sha = sha256_bytes(prompt.encode("utf-8"))
        if any(
            row.get("family_review_prompt_sha256") != prompt_sha
            for row in family_rows
        ):
            raise RuntimeError(
                f"prior V4 family prompt hash mismatch:{family_id}"
            )
    return dict(prior_by_family)


def run_family_audit_v4(
    *,
    packet_path: Path,
    packet_manifest_path: Path,
    protocol_path: Path,
    output_path: Path,
) -> dict:
    packet_raw = packet_path.read_bytes()
    packet_sha = sha256_bytes(packet_raw)
    packet_manifest = json.loads(
        packet_manifest_path.read_text(encoding="utf-8")
    )
    if packet_manifest.get("packet_sha256") != packet_sha:
        raise RuntimeError("audit packet hash mismatch")

    protocol_raw = protocol_path.read_bytes()
    protocol_sha = sha256_bytes(protocol_raw)
    protocol = json.loads(protocol_raw.decode("utf-8"))
    if protocol.get("packet_sha256") != packet_sha:
        raise RuntimeError("protocol packet hash mismatch")
    if (
        protocol.get("candidate_sha256")
        != packet_manifest.get("candidate_sha256")
    ):
        raise RuntimeError("protocol candidate hash mismatch")

    expected_per_family = protocol.get("cases_per_family")
    if not isinstance(expected_per_family, int):
        raise RuntimeError("protocol cases_per_family missing")

    packet = _read_jsonl(packet_path)
    grouped = group_packet_by_family(
        packet,
        expected_per_family=expected_per_family,
    )
    if len(packet) != protocol.get("sample_rows"):
        raise RuntimeError("protocol sample_rows mismatch")
    if len(grouped) != protocol.get("family_count"):
        raise RuntimeError("protocol family_count mismatch")

    execution = protocol.get("execution")
    if not isinstance(execution, dict):
        raise RuntimeError("protocol execution missing")
    stop_on_first_reject = execution.get("stop_on_first_reject")
    if stop_on_first_reject is not True:
        raise RuntimeError(
            "V4 requires stop_on_first_reject=true"
        )

    identity = reviewer_identity()
    _reviewer_matches_protocol(identity, protocol)

    prior_rows = _read_jsonl(output_path) if output_path.exists() else []
    prior_by_family = _validate_prior_v4_rows(
        prior_rows,
        grouped=grouped,
        protocol_sha=protocol_sha,
        expected_per_family=expected_per_family,
    )

    all_rows: list[dict] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for family_id, cases in grouped.items():
        prompt = family_review_prompt_v4(cases)
        prompt_sha = sha256_bytes(prompt.encode("utf-8"))
        existing = prior_by_family.get(family_id, [])
        if existing:
            existing_sorted = sorted(
                existing,
                key=lambda row: row["case_id"],
            )
            all_rows.extend(existing_sorted)
            if any(
                row.get("verdict") == "REJECT"
                for row in existing_sorted
            ):
                break
            continue

        judgments, runtime = call_family_reviewer_v4(cases)
        category_by_id = {
            case["case_id"]: case["category"]
            for case in cases
        }
        family_rows = []
        for item in judgments:
            case_id = item["case_id"]
            review = {
                key: item[key]
                for key in REQUIRED_JUDGMENT_KEYS
            }
            family_rows.append({
                "case_id": case_id,
                "category": category_by_id[case_id],
                "family_id": family_id,
                "review": review,
                "verdict": item["verdict"],
                "family_review_prompt_sha256": prompt_sha,
                "runtime": runtime,
                "protocol_sha256": protocol_sha,
            })

        family_rows.sort(key=lambda row: row["case_id"])
        payload = (
            "".join(
                canonical_json(row) + "\n"
                for row in family_rows
            )
        ).encode("utf-8")
        with output_path.open("ab") as handle:
            handle.write(payload)
            handle.flush()

        all_rows.extend(family_rows)
        if any(
            row["verdict"] == "REJECT"
            for row in family_rows
        ):
            break

    raw = output_path.read_bytes()
    summary = summarize_family_audit_v4(
        all_rows,
        expected_families=set(grouped),
        expected_per_family=expected_per_family,
    )
    return {
        "schema": "V10_RETENTION_FAMILY_MODEL_AUDIT_V4",
        "candidate_sha256": packet_manifest["candidate_sha256"],
        "packet_sha256": packet_sha,
        "protocol_sha256": protocol_sha,
        "sample_rows": len(packet),
        "family_count": len(grouped),
        "cases_per_family": expected_per_family,
        "review_output_sha256": sha256_bytes(raw),
        "reviewer": identity,
        "summary": summary,
        "response_contract": {
            "reviewer_emits_verdict": False,
            "derived_verdict": (
                "ADMIT iff all three boolean judgments are true; "
                "otherwise REJECT"
            ),
        },
        "claim_ceiling": (
            "INDEPENDENT_EXTERNAL_MODEL_FAMILY_AUDIT_V4 / "
            "V3_PREDECESSOR_REMAINS_HOLD / "
            "DETERMINISTIC_SOURCE_OR_TEST_TRUTH_REMAINS_AUTHORITATIVE / "
            "NOT_BEHAVIORAL_OR_ADVERSARIAL_FINAL_BANK_REVIEW"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    try:
        receipt = run_family_audit_v4(
            packet_path=args.packet,
            packet_manifest_path=args.packet_manifest,
            protocol_path=args.protocol,
            output_path=args.output,
        )
        code = 0 if receipt["summary"]["status"] == "PASS" else 2
    except Exception as exc:
        receipt = {
            "schema": "V10_RETENTION_FAMILY_MODEL_AUDIT_V4",
            "status": "HOLD",
            "error": str(exc),
        }
        code = 2

    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_bytes(
        (
            json.dumps(receipt, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
    )
    print(json.dumps(receipt, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
