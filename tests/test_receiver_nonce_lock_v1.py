"""Real Mono-source receiver-scope ablation, not a production nonce authority."""
import pytest
from evaluate_receiver_nonce_lock_v1 import evaluate

@pytest.fixture(scope="module")
def observation():
    return evaluate()

def test_durable_same_scope_nonce_replay_and_hostile_tasks(observation):
    o=observation
    assert o["first"]["accepted"] is True
    assert o["cold_replay"]["accepted"] is False
    assert o["cold_replay"]["reason"] == "DUPLICATE_SCOPE_NONCE"
    assert o["different_task_replay"]["accepted"] is False
    assert o["different_task_replay"]["reason"] == "DUPLICATE_SCOPE_NONCE"

def test_two_overlapping_real_python_processes_only_one_wins(observation):
    assert observation["race_accepts"] == 1
    assert sorted(r["accepted"] for r in observation["race"]) == [False,True]

def test_current_mono_source_digest_and_stale_task_head(observation):
    assert observation["forged_source"]["accepted"] is False
    assert observation["forged_source"]["reason"] == "CANONICAL_SOURCE_MISMATCH"
    assert observation["stale_head"]["accepted"] is False
    assert observation["stale_head"]["reason"] == "MONO_TASK_CAS_REJECT"

def test_legitimate_nonreplay_and_session_boundary(observation):
    assert observation["different_nonce"]["accepted"] is True
    assert observation["different_session"]["accepted"] is True

def test_no_provider_effect_or_runtime_promotion(observation):
    o=observation
    assert o["all_tasks_open"] and o["events_only_task_open_and_checkpoint"]
    assert not o["real_issuer_verified"]
    assert not o["real_loaded_model"]
    assert not o["real_provider_transport"]
    assert not o["memory_admitted"]
    assert not o["effect_authorized"]
    assert o["admission_status"] == "HOLD_RESEARCH_TASK_CHECKPOINT_IS_NOT_NONCE_API"
    assert o["scope"] == "receiver_id|host_generation|provider_session|nonce"


def test_unauthenticated_generation_switch_is_still_a_red(observation):
    assert observation["forged_generation"]["accepted"] is True
    assert not observation["real_issuer_verified"]
