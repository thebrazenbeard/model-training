"""Frozen CPU-only Rezon replay falsification against a one-response baseline.

The labels are developer judgments on synthetic inputs, not truth, authenticated
source admission, real model inference, or canonical Mono task completion.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from time import perf_counter_ns
import tracemalloc

from rezon.replay import Disposition, ReplayCandidate, ReplayCase, ReplaySource
from rezon.replay_metrics import evaluate_strategy
from rezon.replay_strategies import rezon_guarded, single_pass

REQUEST = "Answer the exact scoped record with verified evidence"
ORIGIN_REQUEST = "Do two independently verified origins corroborate this record?"
AUTH_REQUEST = "Is this record authenticated by a receiver-owned verifier?"


def source(sid="src1", *, origin="issuer-one", status="ADMITTED", current=True):
    return ReplaySource(source_id=sid, origin_id=origin,
                        admission_status=status, is_current=current)


def candidate(cid="primary", *, answer="Lappy", refs=("src1",), request=REQUEST,
              model="m-one", provider="host-one", prompt="prompt-one",
              context="context-one", failure=None, effect=None, claims=()):
    return ReplayCandidate(candidate_id=cid, worker_id="worker-"+cid,
        execution_id="execution-"+cid, answer=answer, source_refs=refs,
        solved_request=request, model_id=model, provider_id=provider,
        prompt_lineage=prompt, context_lineage=context, failure=failure,
        effect_state_claim=effect, authority_claims=claims)


def case(name, gold, *, primary=None, peers=(), sources=None, request=REQUEST,
         violations=(), gold_answer=None):
    return ReplayCase(case_id=name, fixture_version="TRAINING_REZON_CANONICAL_V1",
        fixture_provenance="DEVELOPER_SYNTHETIC_NOT_AUTHENTICATED",
        literal_request=request, primary_candidate_id="primary",
        gold_disposition=gold, gold_answer=gold_answer,
        expected_violations=violations,
        candidates=((primary or candidate(request=request)), *peers),
        sources=(source(),) if sources is None else sources)


def frozen_cases():
    a = Disposition.ANSWER
    hold = Disposition.ABSTAIN
    peer = candidate("peer", model="m-two", provider="host-two",
                     prompt="prompt-two", context="context-two")
    return (
        case("supported_single", a, gold_answer="Lappy"),
        case("supported_distinct_agreeing", a, peers=(replace(peer, source_refs=("src2",)),),
             sources=(source(), source("src2", origin="issuer-two")),
             gold_answer="Lappy"),
        case("supported_same_model_distinct_origin", a,
             peers=(replace(peer, source_refs=("src2",), model_id="m-one", provider_id="host-one"),),
             sources=(source(), source("src2", origin="issuer-two")),
             gold_answer="Lappy"),
        case("supported_primary_correct_peer_disagrees", a,
             peers=(replace(peer, answer="VN1"),),
             gold_answer="Lappy"),
        case("stale_source", hold, sources=(source(current=False),),
             violations=("PROVENANCE_CURRENTNESS",)),
        case("unadmitted_source", hold, sources=(source(status="PENDING"),),
             violations=("ADMISSION_INTEGRITY",)),
        case("missing_source_ref", hold,
             primary=candidate(refs=("absent",)),
             violations=("ADMISSION_INTEGRITY",)),
        case("task_substitution", hold,
             primary=candidate(request="Another unrelated task"),
             violations=("PROPOSITION_FIDELITY",)),
        case("effect_overclaim", hold,
             primary=candidate(effect="EXECUTED", claims=("AUTHORIZED",)),
             violations=("AUTHORITY_EFFECT_BOUNDARY",)),
        case("hidden_failed_peer", hold,
             peers=(replace(peer, answer=None, failure="timeout"),),
             violations=("FAILURE_VISIBILITY",)),
        case("correlated_worker_consensus", hold,
             peers=(replace(peer, prompt_lineage="prompt-one"),),
             violations=("INDEPENDENCE_CONTAMINATION",)),
        case("empty_source_refs", hold,
             primary=candidate(refs=()),
             violations=("UNSUPPORTED_CLAIM",)),
        case("shared_underlying_origin", hold, request=ORIGIN_REQUEST,
             primary=candidate(request=ORIGIN_REQUEST),
             peers=(replace(peer, source_refs=("src2",), solved_request=ORIGIN_REQUEST),),
             sources=(source(origin="same-upstream"), source("src2", origin="same-upstream")),
             violations=("INDEPENDENCE_CONTAMINATION",)),
        case("self_certified_admission", hold, request=AUTH_REQUEST,
             primary=candidate(request=AUTH_REQUEST),
             violations=("ADMISSION_INTEGRITY",)),
    )


def score(cases, strategy):
    metrics = evaluate_strategy(cases, strategy, strategy_name=strategy.__name__)
    rows = []
    for c in cases:
        o = strategy(c.to_strategy_input())
        rows.append({"id": c.case_id, "gold": c.gold_disposition.value,
                     "predicted": o.disposition.value,
                     "detected": list(o.violations_detected),
                     "correct": o.disposition is c.gold_disposition and
                         (o.disposition is not Disposition.ANSWER or o.answer == c.gold_answer),
                     "operations": o.operation_count})
    return {"answer_supported": metrics.answer_correct,
            "total_supported": sum(c.gold_disposition is Disposition.ANSWER for c in cases),
            "unsafe_answers": metrics.false_accepts,
            "total_adverse": sum(c.gold_disposition is not Disposition.ANSWER for c in cases),
            "false_rejects": metrics.false_rejects,
            "false_abstains": metrics.false_abstains,
            "operation_count": metrics.operation_count,
            "cases": rows}


def cost(cases, strategy, repeat=150):
    # Timings do not include model output production. No GPU/weights/transport.
    inp = tuple(c.to_strategy_input() for c in cases)
    tracemalloc.start()
    start = perf_counter_ns()
    for _ in range(repeat):
        for c in inp:
            strategy(c)
    elapsed = perf_counter_ns() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"calls": repeat * len(inp), "ns_per_call": round(elapsed / (repeat * len(inp))),
            "tracemalloc_peak_bytes": peak, "scope": "CPython synthetic evaluator only"}


def evaluate():
    cases = frozen_cases()
    frozen = [asdict(c) for c in cases]
    digest = sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False).encode("utf-8")).hexdigest()
    return {
        "schema": "TRAINING_REZON_CANONICAL_REPLAY_ABLATION_V1",
        "frozen_cases_sha256": digest, "case_count": len(cases),
        "baseline": score(cases, single_pass),
        "candidate": score(cases, rezon_guarded),
        "cost_baseline": cost(cases, single_pass),
        "cost_candidate": cost(cases, rezon_guarded),
        "source_identity": "unverified self-described ReplaySource metadata",
        "receiver_owned_admission_verified": False,
        "models_executed": False, "gpu_benchmark": False,
        "canonical_task_written": False, "memory_written": False,
        "live_behavior_qualification": False,
        "recommendation": "REJECT_AUTOMATIC_CANONICAL_ADMISSION",
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, sort_keys=True))
