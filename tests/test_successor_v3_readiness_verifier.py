import hashlib
import json
from pathlib import Path

import pytest

import successor.v3_readiness_verifier as verifier


HEAD = "a" * 40
UPSTREAM = "b" * 40


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")


def make_evidence(tmp_path: Path):
    repo = tmp_path / "repo"
    ready = tmp_path / "ready"
    blind = tmp_path / "blind"
    receipts = tmp_path / "receipts"
    for path in (repo, ready, blind, receipts):
        path.mkdir()

    train = [
        {"prompt": "train one", "response": "a", "source_class": "stable_identity_evidence"},
        {"prompt": "train two", "response": "b", "source_class": "ordinary_general_competence"},
    ]
    validation = [{"prompt": "validation one", "response": "c", "family": "identity"}]
    general = [{"prompt": "train two", "response": "b", "source_class": "ordinary_general_competence"}]
    train_path = ready / "train.jsonl"
    validation_path = ready / "validation.jsonl"
    general_path = ready / "general_competence_pool.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    write_jsonl(general_path, general)

    disposition = {
        "schema": "VERA_SUCCESSOR_V3_SEMANTIC_CLOSURE_DISPOSITION_V1",
        "scope": "TASK9_READINESS_CORPUS_ONLY",
        "source_commit": HEAD,
        "targets": {
            name: {"status": "DISPOSITIONED", "disposition": "EXCLUDED_PENDING_SEMANTIC_CLOSURE"}
            for name in verifier.SEMANTIC_CLOSURE_TARGETS
        },
    }
    disposition_path = ready / "semantic_closure_disposition.json"
    write_json(disposition_path, disposition)
    disposition_sha = sha256_file(disposition_path)

    corpus = {
        "schema": "VERA_SUCCESSOR_V3_READINESS_CORPUS_FREEZE_V1",
        "readiness_source_commit": HEAD,
        "upstream_vera_lab_commit": UPSTREAM,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "general_competence_pool_rows": len(general),
        "train_digest": sha256_json(train),
        "validation_digest": sha256_json(validation),
        "general_competence_pool_digest": sha256_json(general),
        "train_file_sha256": sha256_file(train_path),
        "validation_file_sha256": sha256_file(validation_path),
        "general_competence_pool_file_sha256": sha256_file(general_path),
        "train_validation_overlap": 0,
        "semantic_closure_disposition_sha256": disposition_sha,
        "semantic_closure": {
            name: {"status": "DISPOSITIONED", "receipt_digest": disposition_sha}
            for name in verifier.SEMANTIC_CLOSURE_TARGETS
        },
        "blind_plaintext_included": False,
        "weight_change_performed": False,
    }
    write_json(ready / "corpus_manifest.json", corpus)

    families = {name: 7 for name in verifier.CRITICAL_FAMILIES}
    blind_manifest = {
        "custodian_lane_key": "blind-lane",
        "item_count": 77,
        "family_counts": families,
        "leakage_check_status": "PASS",
        "plaintext_visible_to_training_lane": False,
        "set_digest": "1" * 64,
        "train_digest_checked": corpus["train_file_sha256"],
        "validation_digest_checked": corpus["validation_file_sha256"],
    }
    blind_manifest_path = blind / "manifest.json"
    write_json(blind_manifest_path, blind_manifest)
    registration = {
        "registered": True,
        "lane_key": "blind-lane",
        "manifest_digest": sha256_file(blind_manifest_path),
        "set_digest": blind_manifest["set_digest"],
        "item_count": 77,
    }
    write_json(blind / "registration_receipt.json", registration)

    source_review = {
        "source_commit": UPSTREAM,
        "review_kind": "SOURCE",
        "verdict": "PASS",
    }
    behavior_review = {
        "source_commit": UPSTREAM,
        "review_kind": "BEHAVIOR",
        "verdict": "PASS_WITH_NONBLOCKING_FINDINGS",
    }
    radical = {
        "registered": True,
        "role": "RADICAL_HOSTILE",
        "lane_key": "radical-lane",
        "readiness_source_commit": HEAD,
    }
    pragmatic = {
        "registered": True,
        "role": "PRAGMATIC_HOSTILE",
        "lane_key": "pragmatic-lane",
        "readiness_source_commit": HEAD,
    }
    write_json(receipts / "source.json", source_review)
    write_json(receipts / "behavior.json", behavior_review)
    write_json(receipts / "radical.json", radical)
    write_json(receipts / "pragmatic.json", pragmatic)
    return repo, ready, blind, receipts


def call(tmp_path, monkeypatch, **overrides):
    repo, ready, blind, receipts = make_evidence(tmp_path)
    monkeypatch.setattr(verifier, "_git", lambda *args: HEAD)
    monkeypatch.setattr(verifier.subprocess, "check_call", lambda *args, **kwargs: 0)
    args = {
        "repo_root": repo,
        "readiness_dir": ready,
        "blind_dir": blind,
        "training_lane_key": "training-lane",
        "vera_lab_source_review_receipt": receipts / "source.json",
        "vera_lab_behavior_review_receipt": receipts / "behavior.json",
        "radical_registration_receipt": receipts / "radical.json",
        "pragmatic_registration_receipt": receipts / "pragmatic.json",
    }
    args.update(overrides)
    return verifier.verify_local_v3_readiness(**args), ready, blind


def test_verified_when_all_measured_evidence_and_receipts_match(tmp_path, monkeypatch):
    decision, _, _ = call(tmp_path, monkeypatch)
    assert decision.verified is True
    assert decision.status == "VERIFIED"
    assert decision.reasons == ()
    assert len(decision.evidence_digest) == 64


def test_missing_independent_receipts_stays_hold(tmp_path, monkeypatch):
    decision, _, _ = call(
        tmp_path,
        monkeypatch,
        vera_lab_source_review_receipt=None,
        vera_lab_behavior_review_receipt=None,
        radical_registration_receipt=None,
        pragmatic_registration_receipt=None,
    )
    assert decision.verified is False
    assert decision.status == "HOLD"
    assert "missing_vera_lab_review_receipt:SOURCE" in decision.reasons
    assert "missing_vera_lab_review_receipt:BEHAVIOR" in decision.reasons
    assert "missing_reviewer_registration:RADICAL_HOSTILE" in decision.reasons
    assert "missing_reviewer_registration:PRAGMATIC_HOSTILE" in decision.reasons


def test_corpus_file_tamper_is_measured_not_trusted(tmp_path, monkeypatch):
    decision, ready, _ = call(tmp_path, monkeypatch)
    assert decision.verified
    (ready / "train.jsonl").write_text(
        (ready / "train.jsonl").read_text(encoding="utf-8") + canonical({"prompt": "tamper", "response": "x"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verifier, "_git", lambda *args: HEAD)
    monkeypatch.setattr(verifier.subprocess, "check_call", lambda *args, **kwargs: 0)
    receipts = tmp_path / "receipts"
    changed = verifier.verify_local_v3_readiness(
        repo_root=tmp_path / "repo",
        readiness_dir=ready,
        blind_dir=tmp_path / "blind",
        training_lane_key="training-lane",
        vera_lab_source_review_receipt=receipts / "source.json",
        vera_lab_behavior_review_receipt=receipts / "behavior.json",
        radical_registration_receipt=receipts / "radical.json",
        pragmatic_registration_receipt=receipts / "pragmatic.json",
    )
    assert changed.verified is False
    assert "file_digest_mismatch:train_file_sha256" in changed.reasons
    assert "canonical_digest_mismatch:train_digest" in changed.reasons
    assert "train_row_count_mismatch" in changed.reasons


def test_computed_overlap_blocks_even_when_manifest_claims_zero(tmp_path, monkeypatch):
    decision, ready, _ = call(tmp_path, monkeypatch)
    assert decision.verified
    validation = [{"prompt": "train one", "response": "different", "family": "identity"}]
    validation_path = ready / "validation.jsonl"
    write_jsonl(validation_path, validation)
    manifest_path = ready / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["validation_rows"] = 1
    manifest["validation_digest"] = sha256_json(validation)
    manifest["validation_file_sha256"] = sha256_file(validation_path)
    write_json(manifest_path, manifest)
    blind_manifest_path = tmp_path / "blind" / "manifest.json"
    blind_manifest = json.loads(blind_manifest_path.read_text(encoding="utf-8"))
    blind_manifest["validation_digest_checked"] = manifest["validation_file_sha256"]
    write_json(blind_manifest_path, blind_manifest)
    reg_path = tmp_path / "blind" / "registration_receipt.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    reg["manifest_digest"] = sha256_file(blind_manifest_path)
    write_json(reg_path, reg)
    monkeypatch.setattr(verifier, "_git", lambda *args: HEAD)
    monkeypatch.setattr(verifier.subprocess, "check_call", lambda *args, **kwargs: 0)
    receipts = tmp_path / "receipts"
    changed = verifier.verify_local_v3_readiness(
        repo_root=tmp_path / "repo",
        readiness_dir=ready,
        blind_dir=tmp_path / "blind",
        training_lane_key="training-lane",
        vera_lab_source_review_receipt=receipts / "source.json",
        vera_lab_behavior_review_receipt=receipts / "behavior.json",
        radical_registration_receipt=receipts / "radical.json",
        pragmatic_registration_receipt=receipts / "pragmatic.json",
    )
    assert "computed_train_validation_prompt_overlap" in changed.reasons


def test_stale_source_and_reviewer_registration_block(tmp_path, monkeypatch):
    repo, ready, blind, receipts = make_evidence(tmp_path)
    monkeypatch.setattr(verifier, "_git", lambda *args: "c" * 40)
    monkeypatch.setattr(verifier.subprocess, "check_call", lambda *args, **kwargs: 0)
    decision = verifier.verify_local_v3_readiness(
        repo_root=repo,
        readiness_dir=ready,
        blind_dir=blind,
        training_lane_key="training-lane",
        vera_lab_source_review_receipt=receipts / "source.json",
        vera_lab_behavior_review_receipt=receipts / "behavior.json",
        radical_registration_receipt=receipts / "radical.json",
        pragmatic_registration_receipt=receipts / "pragmatic.json",
    )
    assert "readiness_source_commit_not_current_head" in decision.reasons
    assert "semantic_closure_source_commit_not_current" in decision.reasons
    assert "reviewer_registration_stale:RADICAL_HOSTILE" in decision.reasons
    assert "reviewer_registration_stale:PRAGMATIC_HOSTILE" in decision.reasons


def test_blind_manifest_is_hash_bound_to_registration(tmp_path, monkeypatch):
    decision, _, blind = call(tmp_path, monkeypatch)
    assert decision.verified
    manifest_path = blind / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["item_count"] = 1
    write_json(manifest_path, manifest)
    monkeypatch.setattr(verifier, "_git", lambda *args: HEAD)
    monkeypatch.setattr(verifier.subprocess, "check_call", lambda *args, **kwargs: 0)
    receipts = tmp_path / "receipts"
    changed = verifier.verify_local_v3_readiness(
        repo_root=tmp_path / "repo",
        readiness_dir=tmp_path / "ready",
        blind_dir=blind,
        training_lane_key="training-lane",
        vera_lab_source_review_receipt=receipts / "source.json",
        vera_lab_behavior_review_receipt=receipts / "behavior.json",
        radical_registration_receipt=receipts / "radical.json",
        pragmatic_registration_receipt=receipts / "pragmatic.json",
    )
    assert "blind_manifest_digest_mismatch" in changed.reasons
    assert "blind_item_count_mismatch" in changed.reasons
    assert "blind_item_count_below_70" in changed.reasons


def test_source_review_receipt_cannot_substitute_for_behavior_review(tmp_path, monkeypatch):
    repo, ready, blind, receipts = make_evidence(tmp_path)
    monkeypatch.setattr(verifier, "_git", lambda *args: HEAD)
    monkeypatch.setattr(verifier.subprocess, "check_call", lambda *args, **kwargs: 0)

    decision = verifier.verify_local_v3_readiness(
        repo_root=repo,
        readiness_dir=ready,
        blind_dir=blind,
        training_lane_key="training-lane",
        vera_lab_source_review_receipt=receipts / "source.json",
        vera_lab_behavior_review_receipt=receipts / "source.json",
        radical_registration_receipt=receipts / "radical.json",
        pragmatic_registration_receipt=receipts / "pragmatic.json",
    )

    assert decision.verified is False
    assert decision.status == "HOLD"
    assert "vera_lab_review_kind_mismatch:BEHAVIOR" in decision.reasons
