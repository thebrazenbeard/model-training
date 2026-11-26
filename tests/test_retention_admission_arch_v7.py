from __future__ import annotations
from successor.experiments.verify_retention_admission_arch_v7 import verify_admission_v7

EXPECTED={"candidate_sha256":"a"*64,"candidate_manifest_sha256":"b"*64,"candidate_verify_sha256":"2"*64,"exclusion_sha256":"c"*64,"semantic_screen_sha256":"d"*64,"packet_sha256":"e"*64,"protocol_sha256":"f"*64,"reviewer_identity_sha256":"1"*64}

def evidence():
    ids=[f"case-{i:03d}" for i in range(130)]
    return {
      "candidate_verify":{"status":"CANDIDATE_V3_MUTATION_AND_FRESHNESS_ADEQUATE","case_count":1500,"coding_rows":250,"replacement_count":12,"reference_failures":[],"mutant_survivors":[],"fresh_deficits":{},"added_all_fresh":True,"removed_all_consumed":True,"retained_nonchunk_changed":[]},
      "mechanical":{"status":"MECHANICAL_VALIDATED","reviewed":1500,"row_status_counts":{"MECHANICAL_VALID":1500},"qualification":{"status":"VALIDATORS_QUALIFIED"}},
      "semantic_screen":{"status":"PASS","failure_count":0,"target_sha256":EXPECTED["candidate_sha256"]},
      "reviewer_qualification":{"status":"REVIEWER_QUALIFIED","structured_contradictions":[]},
      "packet_manifest":{"sample_rows":130,"family_count":26,"per_family":5,"excluded_case_count":520,"disjoint_from_all_predecessors":True,"packet_sha256":EXPECTED["packet_sha256"],"sample_case_ids":ids},
      "semantic":{"status":"SEMANTIC_AUDIT_COMPLETE","reviewed":130,"rows":[{"case_id":x} for x in ids]},
      "reconciliation":{"status":"RETENTION_ADMITTED_ARCH_V7","blocking_defect_counts":{},"nonblocking_finding_counts":{"REVIEWER_DEFECT":1}},
      "bindings":dict(EXPECTED),
    }

def test_v7_exact_evidence_admits():
    result=verify_admission_v7(evidence(),expected_bindings=EXPECTED)
    assert result["status"]=="RETENTION_ADMITTED_ARCH_V7"
    assert result["reasons"]==[]
    assert result["nonblocking_findings"]=={"REVIEWER_DEFECT":1}

def test_v7_requires_freshness_adequacy():
    e=evidence(); e["candidate_verify"]["fresh_deficits"]={"generated-code:chunk_list":1}
    result=verify_admission_v7(e,expected_bindings=EXPECTED)
    assert result["status"]=="RETENTION_HOLD_ARCH_V7"
    assert "candidate_v3_freshness_or_mutation_adequacy_failed" in result["reasons"]

def test_v7_requires_520_predecessor_exclusion():
    e=evidence(); e["packet_manifest"]["excluded_case_count"]=519
    result=verify_admission_v7(e,expected_bindings=EXPECTED)
    assert result["status"]=="RETENTION_HOLD_ARCH_V7"
    assert "packet_excluded_case_count:519" in result["reasons"]
