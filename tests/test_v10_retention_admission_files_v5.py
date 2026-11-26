from __future__ import annotations

import hashlib
import json
from pathlib import Path

import successor.experiments.verify_v10_retention_admission_v5 as admission_files


def _write_json(path: Path, value: dict) -> None:
    path.write_bytes(
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    raw = (
        "\n".join(
            json.dumps(r, sort_keys=True, separators=(",", ":"))
            for r in rows
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    paths = {name: tmp_path / name for name in (
        "candidate.jsonl","candidate.manifest.json","exclusion.txt",
        "predecessor.jsonl","packet.jsonl","packet.manifest.json",
        "method.json","builder.py","protocol.json","runner.py",
        "semantic.json","policy.json","binding.json","audit.jsonl",
        "receipt.json"
    )}
    row = {
        "case_id":"fresh","lane":"retention","category":"knowledge_factuality",
        "family_id":"f1","prompt":"Question?","source_id":"source",
        "source_revision":"1","source_terms":"CC0","source_hash":"a"*64,
        "generation_method":"method","generation_actor_id":"generator",
        "grader_contract":{"kind":"deterministic","grader_id":"g","grader_version":"1","answer_key_digest":"b"*64},
        "review_receipt":{"verdict":"PENDING_INDEPENDENT_REVIEW","subject_digest":"c"*64},
        "source_audit_receipt":{"verdict":"SOURCE_CONTRACT_VALID"},
    }
    pred = dict(row); pred["case_id"]="old"; pred["prompt"]="Old?"
    candidate_sha=_write_jsonl(paths["candidate.jsonl"],[row,pred])
    _write_json(paths["candidate.manifest.json"],{"case_count":2,"data_sha256":candidate_sha,"category_counts":{"knowledge_factuality":2}})
    pred_sha=_write_jsonl(paths["predecessor.jsonl"],[pred])
    packet_sha=_write_jsonl(paths["packet.jsonl"],[row])
    _write_json(paths["packet.manifest.json"],{"candidate_sha256":candidate_sha,"predecessor_packet_sha256":pred_sha,"packet_sha256":packet_sha,"sample_rows":1,"family_count":1,"per_family":1})
    paths["exclusion.txt"].write_bytes(b"")
    exclusion_sha=_sha(paths["exclusion.txt"])
    for name,content in (("method.json",{"schema":"method"}),("protocol.json",{"candidate_sha256":candidate_sha,"predecessor_packet_sha256":pred_sha,"packet_sha256":packet_sha,"sample_rows":1,"family_count":1,"cases_per_family":1}),("semantic.json",{"status":"PASS","target_sha256":candidate_sha,"threshold":0.9,"model_archive_sha256":"d"*64,"registry":{"registry_sha256":exclusion_sha},"reasons":[]})):
        _write_json(paths[name],content)
    paths["builder.py"].write_text("# builder\n",encoding="utf-8",newline="\n")
    paths["runner.py"].write_text("# runner\n",encoding="utf-8",newline="\n")
    method_sha=_sha(paths["method.json"]); builder_sha=_sha(paths["builder.py"]); protocol_sha=_sha(paths["protocol.json"]); runner_sha=_sha(paths["runner.py"]); semantic_sha=_sha(paths["semantic.json"]); manifest_sha=_sha(paths["packet.manifest.json"])
    policy={"schema":"V10_QWEN35_RETENTION_ADMISSION_V5","status":"PREREGISTERED_BEFORE_V5_AUDIT_RESULT","subject":{"expected_rows":2,"expected_candidate_sha256":candidate_sha},"methodology":{"method_sha256":method_sha,"builder_sha256":builder_sha,"protocol_sha256":protocol_sha,"runner_sha256":runner_sha},"fresh_packet":{"packet_sha256":packet_sha,"packet_manifest_sha256":manifest_sha,"predecessor_packet_sha256":pred_sha,"predecessor_cases_reusable":False}}
    _write_json(paths["policy.json"],policy); policy_sha=_sha(paths["policy.json"])
    review={"prompt_well_posed":True,"grader_matches_prompt":True,"evidence_policy_supports_expected_answer":True,"issue_code":"NONE","reason":"Good.","confidence":"high"}
    audit_sha=_write_jsonl(paths["audit.jsonl"],[{"case_id":"fresh","family_id":"f1","category":"knowledge_factuality","verdict":"ADMIT","review":review,"family_review_prompt_sha256":"e"*64,"protocol_sha256":protocol_sha}])
    reviewer={"provider":"OLLAMA_LOCAL","model":"ministral-3:14b","model_blob_sha256":"bfb40fc6bb9c3b2ed529b480e04f824c005ea8f86733d4ebbf0c204de484891e","runtime":"ollama version is 0.34.2","temperature":0,"seed":20261001}
    _write_json(paths["receipt.json"],{"candidate_sha256":candidate_sha,"predecessor_packet_sha256":pred_sha,"packet_sha256":packet_sha,"protocol_sha256":protocol_sha,"review_output_sha256":audit_sha,"sample_rows":1,"family_count":1,"cases_per_family":1,"reviewer":reviewer,"summary":{"status":"PASS","reviewed":1,"families_reviewed":1,"admit":1,"reject":0,"reasons":[]}})
    binding={"schema":"V10_QWEN35_RETENTION_ADMISSION_EVIDENCE_BINDING_V3","status":"FROZEN_BEFORE_V5_AUDIT_RESULT","admission_policy":{"file_sha256":policy_sha},"candidate":{"file_sha256":candidate_sha,"expected_rows":2},"predecessor_packet":{"file_sha256":pred_sha,"case_count":1},"fresh_packet":{"file_sha256":packet_sha,"manifest_file_sha256":manifest_sha,"rows":1,"families":1,"cases_per_family":1},"methodology":{"method_file_sha256":method_sha,"builder_file_sha256":builder_sha,"protocol_file_sha256":protocol_sha,"runner_file_sha256":runner_sha},"semantic_screen":{"file_sha256":semantic_sha,"required_status":"PASS","target_sha256":candidate_sha,"threshold":0.9,"model_archive_sha256":"d"*64,"exclusion_registry_sha256":exclusion_sha},"exclusion_registry":{"file_sha256":exclusion_sha},"required_reviewer":reviewer,"final_v5_receipt_requirements":{"candidate_sha256":candidate_sha,"predecessor_packet_sha256":pred_sha,"packet_sha256":packet_sha,"protocol_sha256":protocol_sha,"sample_rows":1,"family_count":1,"cases_per_family":1,"summary_status_required":"PASS","reviewed_rows_required":1,"reviewed_families_required":1,"reject_count_required":0}}
    _write_json(paths["binding.json"],binding)
    return paths


def _verify(p):
    return admission_files.verify_retention_admission_files_v5(
        candidate_path=p["candidate.jsonl"],candidate_manifest_path=p["candidate.manifest.json"],exclusion_path=p["exclusion.txt"],
        predecessor_packet_path=p["predecessor.jsonl"],packet_path=p["packet.jsonl"],packet_manifest_path=p["packet.manifest.json"],
        method_path=p["method.json"],builder_path=p["builder.py"],protocol_path=p["protocol.json"],runner_path=p["runner.py"],
        audit_output_path=p["audit.jsonl"],audit_receipt_path=p["receipt.json"],semantic_path=p["semantic.json"],
        admission_policy_path=p["policy.json"],evidence_binding_path=p["binding.json"],
    )


def test_v5_file_binding_passes_before_core(monkeypatch,tmp_path):
    p=_fixture(tmp_path)
    monkeypatch.setattr(admission_files,"RETENTION_ALLOCATION",{"knowledge_factuality":2})
    monkeypatch.setattr(admission_files,"validate_retention_admission_v5",lambda **kwargs:{"status":"RETENTION_ADMITTED_V5","reasons":[]})
    result=_verify(p)
    assert result["status"]=="RETENTION_ADMITTED_V5"
    assert result["binding_reasons"]==[]


def test_v5_runner_substitution_holds_before_core(monkeypatch,tmp_path):
    p=_fixture(tmp_path); called=False
    def core(**kwargs):
        nonlocal called; called=True; return {"status":"RETENTION_ADMITTED_V5","reasons":[]}
    monkeypatch.setattr(admission_files,"validate_retention_admission_v5",core)
    p["runner.py"].write_text("# changed\n",encoding="utf-8")
    result=_verify(p)
    assert result["status"]=="HOLD"
    assert "runner_file_sha256_mismatch" in result["binding_reasons"]
    assert called is False


def test_v5_predecessor_packet_substitution_holds(monkeypatch,tmp_path):
    p=_fixture(tmp_path)
    p["predecessor.jsonl"].write_text("{}\n",encoding="utf-8")
    result=_verify(p)
    assert result["status"]=="HOLD"
    assert "predecessor_packet_file_sha256_mismatch" in result["binding_reasons"]
