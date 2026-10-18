"""Frozen negative source tests: Mono action_lock != task journal transaction."""
import pytest
from evaluate_action_lock_fence_v1 import CASES, source_case_digest, evaluate


@pytest.fixture(scope="module")
def result():
    return evaluate()


def test_frozen_scope_control_and_positive_task_checkpoint(result):
    assert len(CASES) == 5
    assert result["frozen_case_sha256"] == source_case_digest()
    assert result["cold_unlocked_checkpoint_accepted"] is True


def test_separate_python_process_can_commit_while_action_lock_held(result):
    assert result["cold_writer_accepted_while_action_lock_held"]
    assert result["writer_checkpoint_visible_before_unlock"]
    assert result["action_lock_still_held_at_readback"]
    assert result["finding"] == "ACTION_LOCK_NOT_IMPLICIT_LOCK_FOR_ORDINARY_TASK_CHECKPOINT"


def test_task_cas_still_fences_stale_task_specific_head(result):
    assert result["stale_task_cas_rejected"]
    assert result["stale_rejection_type"] == "TaskExecutionError"
    assert result["original_task_head_changed_by_writer"]


def test_unrelated_task_moves_global_head_but_not_A_event_head(result):
    assert result["task_head_unchanged_by_unrelated_B"]
    assert result["global_head_changed_by_unrelated_B"]
    assert result["global_chain_verified"]


def test_source_probe_does_not_admit_model_memory_or_effect(result):
    assert result["all_tasks_open"]
    assert result["events_only_open_or_checkpoint"]
    for name in ("real_provider_authenticated", "loaded_model_observed",
                 "canonical_result_accepted", "memory_admitted",
                 "protected_effect_performed"):
        assert result[name] is False
