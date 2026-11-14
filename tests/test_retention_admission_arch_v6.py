from __future__ import annotations

from successor.experiments.verify_retention_admission_arch_v6 import verify_admission_v6

EXPECTED={
    "candidate_sha256":"a"*64,
    "candidate_manifest_sha256":"b"*64,
    "candidate_verify_sha256":"2"*64,
    "exclusion_sha256":"c"*64,
    "semantic_screen_sha256":"d"*64,
    "packet_sha256":"e"*64,
    "protocol_sha256":"f"*64,
    "reviewer_identity_sha256":"1"*64,
}

def evidence():
    ids=[f"case-{i:03d}" for i in range(130)]
    return {
        "candidate_verify":{
            "status":"CANDIDATE_V2_MUTATION_ADEQUATE",
            "coding_rows":250,
            "reference_failures":[],
            "mutant_survivors":[],
            "noncoding_changed":0,
        },
        "mechanical":{
            "status":"MECHANICAL_VALIDATED",
            "reviewed":1500,
            "row_status_counts":{"MECHANICAL_VALID":1500},
            "qualification":{"status":"VALIDATORS_QUALIFIED"},
        },
        "semantic_screen":{
            "status":"PASS",
            "failure_count":0,
            "target_sha256":EXPECTED["candidate_sha256"],
        },
        "reviewer_qualification":{
            "status":"REVIEWER_QUALIFIED",
            "structured_contradictions":[],
        },
        "packet_manifest":{
            "sample_rows":130,
            "family_count":26,
            "per_family":5,
            "disjoint_from_all_predecessors":True,
            "packet_sha256":EXPECTED["packet_sha256"],
            "sample_case_ids":ids,
        },
        "semantic":{
            "status":"SEMANTIC_AUDIT_COMPLETE",
            "reviewed":130,
            "rows":[{"case_id":x} for x in ids],
        },
        "reconciliation":{
            "status":"RETENTION_ADMITTED_ARCH_V6",
            "blocking_defect_counts":{},
            "nonblocking_finding_counts":{"REVIEWER_DEFECT":2},
        },
        "bindings":dict(EXPECTED),
    }

def test_v6_admission_allows_nonblocking_reviewer_defects():
    result=verify_admission_v6(evidence(), expected_bindings=EXPECTED)
    assert result["status"] == "RETENTION_ADMITTED_ARCH_V6"
    assert result["reasons"] == []
    assert result["nonblocking_findings"] == {"REVIEWER_DEFECT":2}

def test_v6_admission_holds_on_blocking_reconciliation_defect():
    e=evidence()
    e["reconciliation"]={
        "status":"RETENTION_HOLD_ARCH_V6",
        "blocking_defect_counts":{"UNRESOLVED":1},
        "nonblocking_finding_counts":{},
    }
    result=verify_admission_v6(e, expected_bindings=EXPECTED)
    assert result["status"] == "RETENTION_HOLD_ARCH_V6"
    assert "UNRESOLVED" in result["defect_classes"]

def test_v6_admission_requires_candidate_mutation_adequacy():
    e=evidence()
    e["candidate_verify"]["mutant_survivors"]=["case-bad"]
    result=verify_admission_v6(e, expected_bindings=EXPECTED)
    assert result["status"] == "RETENTION_HOLD_ARCH_V6"
    assert "candidate_mutation_adequacy_failed" in result["reasons"]

def test_v6_admission_requires_semantic_screen_pass():
    e=evidence()
    e["semantic_screen"]["status"]="HOLD"
    result=verify_admission_v6(e, expected_bindings=EXPECTED)
    assert result["status"] == "RETENTION_HOLD_ARCH_V6"
    assert "semantic_screen_not_pass" in result["reasons"]

def test_v6_admission_rejects_noncoding_drift():
    e=evidence()
    e["candidate_verify"]["noncoding_changed"]=1
    result=verify_admission_v6(e, expected_bindings=EXPECTED)
    assert result["status"] == "RETENTION_HOLD_ARCH_V6"
    assert "candidate_mutation_adequacy_failed" in result["reasons"]
