"""Verify exact Mono task-scoped CAS, not real provider or global nonce trust."""
from evaluate_mono_task_nonce_cas_v1 import evaluate, CASE_DESCRIPTIONS


def test_frozen_source_case_digest_and_positive_controls():
    r = evaluate()
    assert r["fixture_sha256"] == "eea18014153e9de82b82138ab8d665fd54e877223999ee1005e9be2d9cda80fe"
    assert len(CASE_DESCRIPTIONS) == 6
    assert r["positive_first"] is True
    assert r["positive_next"] is True
    assert r["task_states_unclosed"] is True


def test_two_cold_processes_reject_same_task_nonce():
    r = evaluate()
    assert r["cold_replay_rejected"]
    assert r["stale_task_head_rejected"]


def test_two_concurrent_processes_only_one_wins_task_cas():
    r = evaluate()
    assert r["concurrent_accept_count"] == 1


def test_global_nonce_reuse_remains_red_across_distinct_tasks():
    r = evaluate()
    assert r["cross_task_same_nonce_accept_count"] == 2
    assert r["global_nonce_uniqueness"] == "RED_TWO_DISTINCT_TASKS_ACCEPT_SAME_NONCE"


def test_no_provider_memory_completion_or_guest_promoted():
    r = evaluate()
    assert r["event_types_only_open_and_checkpoint"]
    assert r["task_states_unclosed"]
    for k in ("real_provider_authenticated", "real_model_loaded", "memory_written",
              "provider_effect_performed", "real_guest_boot"):
        assert r[k] is False
    assert r["decision"] == "TASK_LOCAL_CAS_SOURCE_UTILITY_ONLY_GLOBAL_NONCE_HOLD"
