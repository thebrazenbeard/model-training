"""Source-contract compatibility only; never an installed cognition/admission verifier."""
import json
import os
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/contracts/TRAINING_COGNITIVE_CAPABILITY_INTERFACE_V1.json"


@pytest.fixture(scope="module")
def spec():
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_single_mono_owner_and_no_shadow_stores(spec):
    assert spec["canonical_owner"] == "vera_core.QualifiedVeraRuntime"
    for key in ("allow_new_task_store", "allow_new_memory_store",
                "allow_new_supervisor", "allow_self_certified_provider"):
        assert spec[key] is False
    assert spec["status"] == "SOURCE_RESEARCH_NOT_INSTALLED"


def test_ordered_capability_exchange_and_no_effect_grant(spec):
    ids = [s["id"] for s in spec["stages"]]
    assert len(ids) == len(set(ids)) == 9
    assert ids == ["canonical_intake", "source_context", "initiative_advisory",
                   "route_advisory", "verified_provider_execution",
                   "durable_result_reconciliation", "critique_advisory",
                   "task_result", "memory_proposal"]
    assert all(s["owner"] and s["boundary"] for s in spec["stages"])
    assert "NO_PROTECTED_EFFECT" in str(spec["stages"][3])


@pytest.mark.parametrize("required", [
    "task_id", "request_id", "accepted_task_journal_head", "source_digest",
    "provider_id", "model_id", "loaded_artifact_sha256", "process_id",
    "host_generation", "session_id", "one_use_nonce", "nonce_consumed_once",
    "receiver_owned_verifier_id", "current_trust_and_revocation",
    "route_id", "unique_durable_route_selection", "transport_response_digest",
    "response_sha256", "preactive_done_event_id", "preactive_completed_run_id",
    "task_result_readback_digest"])
def test_provider_reconcile_contract_retains_field(spec, required):
    assert required in spec["provider_receipt_required"]


@pytest.mark.parametrize("promotion", [
    "source_to_truth", "route_to_effect", "model_text_to_completion",
    "critique_to_authentication", "initiative_to_consent",
    "memory_proposal_to_memory", "project_chat_to_runtime"])
def test_no_silent_evidence_or_authority_promotion(spec, promotion):
    assert promotion in spec["forbidden_promotions"]


def test_missing_provenance_and_unknown_response_force_hold(spec):
    for key in ("ABSENT_PREDECESSOR", "UNVERIFIED_SOURCE_ORIGIN",
                "REPLAYED_NONCE", "UNKNOWN_TRANSPORT_OUTCOME",
                "UNAUTHORIZED_MEMORY_ADMISSION", "NONUNIQUE_ROUTE"):
        assert key in spec["hard_hold"]
    assert "UNVERIFIED_ADMITTED_FLAG" in spec["source_not_enough"]
    assert "SAME_ORIGIN_AGREEMENT" in spec["source_not_enough"]


def test_source_measured_dispositions_are_not_model_benchmarks(spec):
    a, b = spec["measured_source_evaluations"]
    assert (a["baseline_utility"], a["candidate_utility"],
            a["candidate_unsafe"]) == ("0/2", "1/2", "0/5")
    assert (b["baseline_utility"], b["candidate_utility"],
            b["candidate_unsafe"], b["disposition"]) == (
            "4/4", "2/4", "3/10", "REJECT_AUTO_ADMISSION")


@pytest.fixture
def real_mono(tmp_path):
    root = os.environ.get("VERA_MONO77_SOURCE_ROOT")
    if not root or not (Path(root) / "packages/vera_core/src/vera_core/local_conversation.py").is_file():
        pytest.skip("exact Mono #77 source unavailable, not passing real-Mono compatibility")
    from vera_core.task_execution import TaskExecutionLedger
    from vera_core.local_conversation import open_message, ConversationError, conversation_history
    return TaskExecutionLedger(tmp_path / "tasks.sqlite3"), open_message, ConversationError, conversation_history


def test_real_mono_exact_intake_replay_and_no_result(real_mono):
    ledger, open_message, _, history = real_mono
    first = open_message(ledger, "desktop-test", "request-1", "literal  inner")
    before = tuple(ledger.events())
    same = open_message(ledger, "desktop-test", "request-1", "literal  inner")
    assert same["task_id"] == first["task_id"]
    assert same["opened_head"] == first["opened_head"]
    assert tuple(ledger.events()) == before
    assert first["origin_authenticated"] is False
    assert ledger.read(first["task_id"]).closed is False
    assert history(ledger, "desktop-test")[0]["text"] == "literal  inner"


@pytest.mark.parametrize("changed", ["literal inner", " literal  inner", "literal  inner "])
def test_real_mono_rejects_substitution_not_normalization(real_mono, changed):
    ledger, open_message, ConversationError, _ = real_mono
    open_message(ledger, "desktop-test", "request-1", "literal  inner")
    with pytest.raises(ConversationError):
        open_message(ledger, "desktop-test", "request-1", changed)


def test_real_mono_distinct_message_requires_distinct_task(real_mono):
    ledger, open_message, _, _ = real_mono
    x = open_message(ledger, "desktop-test", "request-1", "one")
    y = open_message(ledger, "desktop-test", "request-2", "two")
    assert x["task_id"] != y["task_id"]
    assert not x["origin_authenticated"] and not y["origin_authenticated"]
