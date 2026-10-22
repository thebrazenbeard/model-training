"""Bounded actual-Mono source regression; PASS includes deliberate unsafe observations."""
import pytest
from evaluate_task_cas_issuer_gap_v1 import CASES, _sha, evaluate

@pytest.fixture(scope="module")
def evidence():
    return evaluate()

def test_frozen_source_cases(evidence):
    assert evidence["schema"] == "TRAINING_NATIVE_TASK_CAS_NOT_RECEIVER_ISSUER_V1"
    assert len(CASES) == evidence["cases_count"] == 8
    assert evidence["cases_sha256"] == _sha(CASES)

def test_untrusted_issuer_and_loaded_artifact_can_be_recorded_as_task_history(evidence):
    assert evidence["forged_receiver_task_checkpoint_accepted"]
    assert evidence["forged_payload_contains_untrusted_claim"]
    assert evidence["real_receiver_authenticated"] is False
    assert evidence["real_loaded_artifact_verified"] is False

def test_native_task_CAS_separates_A_stale_from_B_global_progress(evidence):
    assert evidence["stale_A_task_CAS_rejected"]
    assert evidence["unchanged_C_after_other_tasks_accepted"]
    assert evidence["old_head_replay_rejected"]

def test_task_scoped_nonce_identity_is_not_cross_task_or_cross_root(evidence):
    assert evidence["same_nonce_distinct_tasks_both_accepted"]
    assert evidence["same_nonce_other_root_accepted"]
    assert evidence["global_nonce_consume_once_proven"] is False

def test_real_cold_crash_cuts_preserve_ambiguous_claim(evidence):
    before=evidence["precommit_crash"]
    after=evidence["postcommit_crash"]
    assert before["child_exit"] == after["child_exit"] == 23
    assert before["task_event_count"] == 1
    assert after["task_event_count"] == 2
    assert before["task_still_open"] and after["task_still_open"]
    assert evidence["postcommit_claim_has_no_verified_result"]

def test_no_provider_memory_effect_or_completion(evidence):
    assert evidence["all_tasks_unclosed"]
    assert not evidence["provider_result_admitted"]
    assert not evidence["memory_promoted"]
    assert not evidence["effect_performed"]
    assert not evidence["selected_provider_route_verified"]
    assert evidence["classification"] == "SOURCE_TASK_CAS_VALID_AUTHENTICATED_PROVIDER_CLAIM_HOLD"
