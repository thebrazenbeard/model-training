"""Regression/falsification of exact Mono/Rezon replay source, no provider calls."""
from evaluate_rezon_canonical_replay_v1 import evaluate, frozen_cases
from rezon.replay import Disposition
from rezon.replay_strategies import single_pass, rezon_guarded


def test_frozen_identity_and_adverse_counts():
    result = evaluate()
    assert result["case_count"] == 14
    assert result["frozen_cases_sha256"] == (
        "5b116db93fc84d5e27a7cb818db47f46b4b550c709b5b1179a31f3b1333b7877"
    )
    assert (result["baseline"]["answer_supported"],
            result["baseline"]["unsafe_answers"]) == (4, 10)
    assert (result["candidate"]["answer_supported"],
            result["candidate"]["unsafe_answers"]) == (2, 3)
    assert result["recommendation"] == "REJECT_AUTOMATIC_CANONICAL_ADMISSION"


def test_measured_negative_transfer_and_unsafe_acceptance():
    cases = {c.case_id: c for c in frozen_cases()}
    for name in ("supported_same_model_distinct_origin",
                 "supported_primary_correct_peer_disagrees"):
        assert single_pass(cases[name].to_strategy_input()).disposition is Disposition.ANSWER
        assert rezon_guarded(cases[name].to_strategy_input()).disposition is Disposition.ABSTAIN
    for name in ("empty_source_refs", "shared_underlying_origin",
                 "self_certified_admission"):
        assert cases[name].gold_disposition is Disposition.ABSTAIN
        assert rezon_guarded(cases[name].to_strategy_input()).disposition is Disposition.ANSWER


def test_actual_rezon_guard_blocks_seven_other_adverse_cases():
    cases = frozen_cases()
    adverse = [c for c in cases if c.gold_disposition is Disposition.ABSTAIN]
    assert len(adverse) == 10
    assert sum(rezon_guarded(c.to_strategy_input()).disposition is Disposition.ABSTAIN
               for c in adverse) == 7


def test_benchmark_does_not_promote_source_to_model_or_authority():
    r = evaluate()
    assert not any(r[k] for k in ("models_executed", "gpu_benchmark",
                                  "canonical_task_written", "memory_written",
                                  "receiver_owned_admission_verified",
                                  "live_behavior_qualification"))
    assert r["candidate"]["operation_count"] > r["baseline"]["operation_count"]
