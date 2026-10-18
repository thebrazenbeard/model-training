"""Adverse source-only tests of #112 public marker nonce poisoning."""
import pytest
from evaluate_checkpoint_poison_v1 import CASES, digest, evaluate


@pytest.fixture(scope="module")
def result():
    return evaluate()


def test_frozen_cases_and_clean_baseline(result):
    assert len(CASES) == 8
    assert result["frozen_case_sha256"] == digest(CASES)
    assert result["clean_first_accepted"] is True
    assert result["different_nonce_accepted"] is True
    assert result["different_scope_accepted"] is True


def test_ordinary_task_checkpoint_poisons_first_nonce(result):
    assert result["counterfeit_marker_blocks_legitimate_first"]
    assert result["forged_marker_is_valid_task_history"]
    assert result["forged_marker_is_foreign_task"]
    assert result["poisoned_victim_stays_open"]


def test_separate_valid_task_database_has_no_shared_nonce_ledger(result):
    assert result["foreign_root_same_nonce_accepted"]


def test_real_cold_process_crash_before_and_after_marker_commit(result):
    assert result["precommit_crash_exit"] == 17
    assert result["postcommit_crash_exit"] == 17
    assert result["after_precommit_crash_accepted"]
    assert result["after_postcommit_crash_rejected"]
    assert result["postcommit_rejection_reason"] == "DUPLICATE_SCOPE_NONCE"


def test_unattested_generation_replays_same_nonce(result):
    assert result["first_generation_accepted"]
    assert result["spoofed_generation_reuse_accepted"]
    assert not result["receiver_generation_attested"]


def test_no_admission_or_protected_effect_promotion(result):
    assert result["all_events_open_or_checkpoint"]
    assert result["real_issuer_authenticated"] is False
    assert result["process_loaded_artifact_observed"] is False
    assert result["canonical_task_completed"] is False
    assert result["protected_effect_performed"] is False
    assert result["memory_promoted"] is False
    assert result["production_protocol_decision"] == "REJECT_CHECKPOINT_AS_NONCE_AUTHORITY"
