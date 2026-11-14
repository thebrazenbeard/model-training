from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import successor.experiments.run_v10_retention_family_audit_v4 as audit_v4


def _case(case_id: str, family_id: str = "fam") -> dict:
    return {
        "case_id": case_id,
        "category": "coding",
        "family_id": family_id,
        "prompt": f"Prompt {case_id}",
        "grader_contract": {
            "kind": "deterministic",
            "grader_id": "g",
            "grader_version": "1",
            "answer_key_digest": "a" * 64,
        },
        "source_evidence": {"family": family_id, "variant": 1},
    }


def _judgment(
    case_id: str,
    *,
    prompt_ok: bool = True,
    grader_ok: bool = True,
    source_ok: bool = True,
) -> dict:
    ok = prompt_ok and grader_ok and source_ok
    return {
        "case_id": case_id,
        "prompt_well_posed": prompt_ok,
        "grader_matches_prompt": grader_ok,
        "source_evidence_sufficient": source_ok,
        "issue_code": "NONE" if ok else "DEFECT",
        "reason": "No material defect." if ok else "Material defect.",
        "confidence": "high",
    }


def test_v4_prompt_forbids_model_verdict_field() -> None:
    prompt = audit_v4.family_review_prompt_v4(
        [_case(f"c{i}") for i in range(5)]
    )
    assert "Do not return a verdict field" in prompt
    assert "prompt_well_posed" in prompt


def test_v4_derives_admit_when_all_three_checks_true() -> None:
    ids = [f"c{i}" for i in range(5)]
    text = json.dumps(
        {"reviews": [_judgment(case_id) for case_id in ids]}
    )
    parsed = audit_v4.parse_family_judgments_v4(
        text,
        expected_case_ids=ids,
    )
    assert all(row["verdict"] == "ADMIT" for row in parsed)


def test_v4_derives_reject_from_false_check_without_transport_retry() -> None:
    ids = [f"c{i}" for i in range(5)]
    rows = [_judgment(case_id) for case_id in ids]
    rows[2] = _judgment(ids[2], grader_ok=False)
    text = json.dumps({"reviews": rows})
    parsed = audit_v4.parse_family_judgments_v4(
        text,
        expected_case_ids=ids,
    )
    assert parsed[2]["verdict"] == "REJECT"
    assert parsed[2]["grader_matches_prompt"] is False


def test_v4_rejects_model_supplied_verdict_as_closed_schema_violation() -> None:
    ids = [f"c{i}" for i in range(5)]
    rows = [_judgment(case_id) for case_id in ids]
    rows[0]["verdict"] = "ADMIT"
    with pytest.raises(ValueError, match="closed schema"):
        audit_v4.parse_family_judgments_v4(
            json.dumps({"reviews": rows}),
            expected_case_ids=ids,
        )


def test_v4_call_does_not_retry_valid_negative_judgment(monkeypatch) -> None:
    cases = [_case(f"c{i}") for i in range(5)]
    rows = [_judgment(case["case_id"]) for case in cases]
    rows[0] = _judgment(cases[0]["case_id"], source_ok=False)
    calls = 0

    def fake_request(prompt: str) -> dict:
        nonlocal calls
        calls += 1
        return {
            "response": json.dumps({"reviews": rows}),
            "done_reason": "stop",
        }

    monkeypatch.setattr(audit_v4, "_ollama_request", fake_request)
    parsed, runtime = audit_v4.call_family_reviewer_v4(cases)
    assert calls == 1
    assert runtime["attempts"] == 1
    assert parsed[0]["verdict"] == "REJECT"


def test_v4_call_retries_only_malformed_transport(monkeypatch) -> None:
    cases = [_case(f"c{i}") for i in range(5)]
    valid = json.dumps(
        {
            "reviews": [
                _judgment(case["case_id"])
                for case in cases
            ]
        }
    )
    responses = iter([
        {"response": "", "done_reason": "stop"},
        {"response": valid, "done_reason": "stop"},
    ])
    monkeypatch.setattr(
        audit_v4,
        "_ollama_request",
        lambda prompt: next(responses),
    )
    parsed, runtime = audit_v4.call_family_reviewer_v4(cases)
    assert len(parsed) == 5
    assert runtime["attempts"] == 2


def test_v4_summary_requires_all_families_and_zero_rejects() -> None:
    rows = []
    for family in ("a", "b"):
        for i in range(5):
            judgment = _judgment(f"{family}{i}")
            rows.append({
                "case_id": f"{family}{i}",
                "family_id": family,
                "category": "coding",
                "review": judgment,
                "verdict": "ADMIT",
            })
    passed = audit_v4.summarize_family_audit_v4(
        rows,
        expected_families={"a", "b"},
        expected_per_family=5,
    )
    assert passed["status"] == "PASS"

    rows[-1]["verdict"] = "REJECT"
    rows[-1]["review"]["grader_matches_prompt"] = False
    held = audit_v4.summarize_family_audit_v4(
        rows,
        expected_families={"a", "b"},
        expected_per_family=5,
    )
    assert held["status"] == "HOLD"
    assert "reject_count:1" in held["reasons"]


def test_v4_summary_holds_when_early_stop_leaves_families_missing() -> None:
    rows = []
    for i in range(5):
        judgment = _judgment(f"a{i}")
        if i == 0:
            judgment["source_evidence_sufficient"] = False
        rows.append({
            "case_id": f"a{i}",
            "family_id": "a",
            "category": "coding",
            "review": judgment,
            "verdict": (
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
            ),
        })
    result = audit_v4.summarize_family_audit_v4(
        rows,
        expected_families={"a", "b"},
        expected_per_family=5,
    )
    assert result["status"] == "HOLD"
    assert "missing_family_count:1" in result["reasons"]


def test_v4_refuses_prior_output_from_different_protocol(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packet = tmp_path / "packet.jsonl"
    packet_manifest = tmp_path / "packet.manifest.json"
    protocol = tmp_path / "protocol.json"
    output = tmp_path / "reviews.jsonl"

    cases = [_case(f"c{i}", "fam") for i in range(5)]
    packet_raw = (
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in cases
        )
    ).encode("utf-8")
    packet.write_bytes(packet_raw)
    packet_sha = hashlib.sha256(packet_raw).hexdigest()
    packet_manifest.write_text(
        json.dumps(
            {
                "candidate_sha256": "c" * 64,
                "packet_sha256": packet_sha,
            }
        ),
        encoding="utf-8",
    )
    protocol_obj = {
        "candidate_sha256": "c" * 64,
        "packet_sha256": packet_sha,
        "sample_rows": 5,
        "family_count": 1,
        "cases_per_family": 5,
        "reviewer": {
            "provider": "OLLAMA_LOCAL",
            "model": "ministral-3:14b",
            "model_blob_sha256": "b" * 64,
            "runtime": "ollama version is 0.34.2",
            "temperature": 0,
            "seed": 20261001,
        },
        "execution": {"stop_on_first_reject": True},
    }
    protocol.write_text(
        json.dumps(protocol_obj, indent=2) + "\n",
        encoding="utf-8",
    )
    output.write_text(
        json.dumps(
            {
                "case_id": "c0",
                "family_id": "fam",
                "category": "coding",
                "verdict": "ADMIT",
                "review": _judgment("c0"),
                "family_review_prompt_sha256": "d" * 64,
                "runtime": {},
                "protocol_sha256": "e" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        audit_v4,
        "reviewer_identity",
        lambda: protocol_obj["reviewer"],
    )
    with pytest.raises(RuntimeError, match="prior V4 protocol hash mismatch"):
        audit_v4.run_family_audit_v4(
            packet_path=packet,
            packet_manifest_path=packet_manifest,
            protocol_path=protocol,
            output_path=output,
        )
