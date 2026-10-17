"""Falsifiable Mono-only no-provider source and cold-process acceptance, not GUI."""
from __future__ import annotations
import pytest
from evaluate_offline_mono_minimal_v1 import evaluate


@pytest.fixture(scope="module")
def result():
    return evaluate()


def test_frozen_adverse_cases_and_receiver_owned_hold(result):
    assert result["fixture_sha256"] == (
       "5eb4750ddd38ecbdc5600ad1787ace67de98c67b678a71a0ca3457ea08e51c40")
    assert result["hard_negative_cases"] == result["negative_rejections"] == 6
    assert {r["name"] for r in result["negative_cases"]} == {
        "fabricated_done", "forged_provider", "reused_nonce", "wrong_route",
        "wrong_scope", "self_certified_origin"}


def test_cold_idempotence_current_source_and_literal(result):
    x, y = result["initial"], result["recovered"]
    assert x["task_id"] == y["task_id"]
    assert x["head"] == y["head"]
    assert x["source"] == y["source"]
    assert y["history"] == y["task_count"] == 1
    assert y["literal"] == "literal  content"
    assert result["cold_recovery"]
    assert result["literal_substitutions_rejected"] == 2


def test_no_optional_boot_imports_model_or_generated_completion(result):
    assert result["initial"]["no_optional_imports"]
    assert result["recovered"]["provider_transport_count"] == 0
    assert result["initial"]["event_types"] == ["TASK_OPENED"]
    assert all(r["event_only_open"] and r["held_open"] and r["no_provider"]
               and r["no_optional_imports"] for r in result["negative_cases"])
    assert not result["optional_module_imported"]
    assert not result["loaded_model"]
    assert not result["authorized_provider"]
    assert not result["effect_executed"]


def test_roots_are_not_shared_identity_or_global_writer(result):
    assert result["separate_state_roots"]
    assert result["different_root"]["authenticated"] is False
    assert result["different_root"]["provider_transport_count"] == 0
    assert result["different_root"]["event_types"] == ["TASK_OPENED"]


def test_resource_and_qualification_not_promoted(result):
    source = result["selected_prototype_source_stage"]
    assert (source["mono_files"], source["portal_files"],
            source["pre_active_files"], source["volition_files"],
            source["total"]) == (316, 89, 18, 8, 431)
    assert source["stage_is_not_strict_boot_requirement"]
    assert result["total_process_wall_ms"] > 0
    assert result["real_guest_boot"] is False
    assert result["memory_proposal_admitted"] is False
    assert result["decision"] == (
        "ACCEPT_MINIMAL_MONO_TASK_ONLY_DEFER_REAL_VERAOS_BOOT")
