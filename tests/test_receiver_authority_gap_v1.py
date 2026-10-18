"""Frozen adverse tests: PASS means source mechanism RED observed, not qualified."""
import pytest
from evaluate_receiver_authority_gap_v1 import (
    SCHEMA, FROZEN_CASES, _sha, evaluate,
)

@pytest.fixture(scope="module")
def evidence():
    return evaluate()

def by_name(evidence, name):
    return next(x for x in evidence["case_results"] if x["name"] == name)

def test_frozen_real_mono_corpus(evidence):
    assert evidence["schema"] == SCHEMA
    assert evidence["case_total"] == 8
    assert evidence["case_sha256"] == _sha(FROZEN_CASES)
    assert len(evidence["case_results"]) == len({x["name"] for x in evidence["case_results"]})
    assert evidence["all_adverse_mechanisms_observed"]

def test_foreign_task_checkpoint_can_poison_claim(evidence):
    assert by_name(evidence, "counterfeit_checkpoint")["observed"]
    assert by_name(evidence, "counterfeit_checkpoint")["classification"] == "REJECT_CHECKPOINT_AUTHORITY"

def test_actual_A_checkpoint_between_read_and_verdict_accepts_stale(evidence):
    assert by_name(evidence, "stale_A_during_preflight")["observed"]
    assert evidence["race"]["actual_A_events"] == 2
    assert evidence["race"]["canonical_task_closed"] is False

def test_unrelated_B_does_not_change_task_A_event_head(evidence):
    assert by_name(evidence, "unrelated_B_current_A")["observed"]

def test_cold_replay_and_cross_task_same_scope_differ_from_root_split(evidence):
    assert by_name(evidence, "cold_and_cross_task_same_nonce")["observed"]
    assert by_name(evidence, "different_SQLite_roots")["observed"]

def test_forged_host_generation_session_and_artifact_fixture_not_trust(evidence):
    assert by_name(evidence, "forged_generation_and_loaded_model")["observed"]
    assert evidence["coherent_fixture_forgery"]["coherent_forged_fixture_passes_preflight"]
    assert not evidence["coherent_fixture_forgery"]["real_issuer_authenticated"]
    assert not evidence["coherent_fixture_forgery"]["real_loaded_model_observed"]

def test_crash_before_after_marker_write_is_not_trusted_nonce_consumption(evidence):
    assert by_name(evidence, "crash_cut_pre_post")["observed"]
    assert by_name(evidence, "crash_cut_pre_post")["classification"] == "POSTCOMMIT_UNKNOWN_HOLD"

def test_action_lock_does_not_fence_independent_task_writer(evidence):
    assert by_name(evidence, "action_lock_nonfence")["observed"]

def test_positive_preflight_does_not_authorize_provider_or_memory(evidence):
    assert evidence["old_synthetic_preflight_positives_retained"] == 2
    assert evidence["old_synthetic_preflight_adverse_accepted"] >= 1
    assert evidence["production_disposition"] == "HOLD_NO_TRUSTED_RECEIVER_ATOMIC_AUTHORITY"
    for name in ("real_issuer_verified", "real_process_loaded_model_verified",
                 "selected_receiver_and_collision_domain_verified",
                 "durable_host_cross_root_nonce_authority_verified",
                 "provider_result_admitted", "memory_admitted", "protected_effect"):
        assert evidence[name] is False
