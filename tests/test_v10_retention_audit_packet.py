from __future__ import annotations

import hashlib

from successor.experiments.build_v10_retention_audit_packet import (
    select_audit_sample,
    select_family_audit_sample,
    source_evidence_for_row,
    write_packet_files,
)


def _row(case_id: str, category: str) -> dict:
    return {
        "case_id": case_id,
        "category": category,
        "source_id": f"source:{case_id}",
        "prompt": f"Prompt {case_id}",
        "grader_contract": {"kind": "deterministic"},
    }


def test_select_audit_sample_is_deterministic_and_order_invariant() -> None:
    rows = [
        _row(f"{cat}-{i}", cat)
        for cat in ("a", "b", "c")
        for i in range(20)
    ]
    digest = hashlib.sha256(b"candidate").hexdigest()
    a = select_audit_sample(rows, per_category=3, candidate_sha256=digest)
    b = select_audit_sample(list(reversed(rows)), per_category=3, candidate_sha256=digest)
    assert [r["case_id"] for r in a] == [r["case_id"] for r in b]
    assert len(a) == 9


def test_source_evidence_for_country_and_pair() -> None:
    countries = {
        "US": {"code": "US", "name": "United States", "regions": 50, "settlements": 1},
        "CA": {"code": "CA", "name": "Canada", "regions": 13, "settlements": 2},
    }
    single = {
        "source_id": "mindstellar/location-data:release:country:US",
    }
    pair = {
        "source_id": "mindstellar/location-data:release:pair:US:CA",
    }
    assert source_evidence_for_row(single, countries) == countries["US"]
    assert source_evidence_for_row(pair, countries) == {
        "left": countries["US"],
        "right": countries["CA"],
    }


def test_source_evidence_reconstructs_internal_contracts() -> None:
    instruction = source_evidence_for_row(
        {
            "source_id": (
                "vera_model_training:V10_RETENTION_OBJECTIVE_CONTRACTS_20261001_V1:"
                "instruction:token_count:3"
            )
        },
        {},
    )
    coding = source_evidence_for_row(
        {
            "source_id": (
                "vera_model_training:V10_RETENTION_OBJECTIVE_CONTRACTS_20261001_V1:"
                "coding:digit_sum:4"
            )
        },
        {},
    )
    assert instruction["family"] == "token_count"
    assert instruction["variant"] == 3
    assert coding["family"] == "digit_sum"
    assert coding["variant"] == 4


def test_select_all_cases_preserves_every_category_row() -> None:
    rows = [_row(f"{cat}-{i}", cat) for cat in ("a", "b") for i in range(4)]
    digest = hashlib.sha256(b"candidate").hexdigest()
    selected = select_audit_sample(rows, per_category=None, candidate_sha256=digest)
    assert {row["case_id"] for row in selected} == {row["case_id"] for row in rows}


def test_packet_writer_hashes_exact_persisted_bytes(tmp_path) -> None:
    packet = [
        {"case_id": "a", "category": "x", "prompt": "Alpha"},
        {"case_id": "b", "category": "x", "prompt": "Beta"},
    ]
    manifest = {"schema": "test", "packet_sha256": "UNSET"}
    output = tmp_path / "packet.jsonl"
    manifest_path = tmp_path / "packet.manifest.json"
    written = write_packet_files(packet, manifest, output, manifest_path)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == written["packet_sha256"]
    assert b"\r\n" not in output.read_bytes()


def test_family_audit_sample_covers_every_family_deterministically() -> None:
    rows = []
    for family in ("f1", "f2", "f3"):
        for i in range(8):
            row = _row(f"{family}-{i}", "category")
            row["family_id"] = family
            rows.append(row)
    digest = hashlib.sha256(b"candidate").hexdigest()
    a = select_family_audit_sample(
        rows,
        per_family=2,
        candidate_sha256=digest,
    )
    b = select_family_audit_sample(
        list(reversed(rows)),
        per_family=2,
        candidate_sha256=digest,
    )
    assert [row["case_id"] for row in a] == [row["case_id"] for row in b]
    assert len(a) == 6
    assert {row["family_id"] for row in a} == {"f1", "f2", "f3"}
