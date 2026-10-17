"""Frozen-source, prequential cognitive integrity evaluation for VeraOS.

This does not invoke a model, improve language generation, teach weights,
qualify real host evidence, or authorize any action. Candidate predictions are
supplied separately from evaluator-only expected outcomes. Existing Mono
prequential code enforces predict/score before update for each case.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import re
from typing import Callable, Mapping, Any

_SHA = re.compile(r"^[0-9a-f]{64}$")
_STATUSES=frozenset({"ANSWER","CLARIFY","HOLD","PROPOSE","EXECUTE"})
_CATEGORIES=frozenset({"multi_turn","ambiguity","correction","semantic_drift",
    "uncertainty","initiative","context_preservation","unsupported_claim",
    "disagreement","task_continuity"})

class CognitiveEvaluationError(ValueError):
    pass

def canonical(value: object) -> bytes:
    return json.dumps(value,sort_keys=True,separators=(",",":"),
        ensure_ascii=False,allow_nan=False).encode("utf-8")

@dataclass(frozen=True)
class CognitivePrediction:
    status: str
    task_id: str
    request_id: str
    referent: str | None = None
    scope: str | None = None
    claimed_value: str | None = None
    source_event_id: str | None = None
    effect_authority: bool = False
    independent_model_claim: bool = False

    def __post_init__(self):
        if (self.status not in _STATUSES or
            any(type(v) is not str or not v for v in (self.task_id,self.request_id)) or
            any(v is not None and type(v) is not str for v in (
                self.referent,self.scope,self.claimed_value,self.source_event_id)) or
            type(self.effect_authority) is not bool or
            type(self.independent_model_claim) is not bool):
            raise CognitiveEvaluationError("malformed cognitive prediction")

@dataclass(frozen=True)
class CognitiveSource:
    source_event_id: str
    referent: str
    scope: str
    value: str
    kind: str = "USER_STATEMENT"
    supersedes: str | None = None
    def __post_init__(self):
        if (any(type(v) is not str or not v for v in (
                self.source_event_id,self.referent,self.scope,self.value)) or
            self.kind not in ("USER_STATEMENT","REPOSITORY_SOURCE","CORRECTION") or
            (self.supersedes is not None and
             (type(self.supersedes) is not str or not self.supersedes))):
            raise CognitiveEvaluationError("invalid source history")

@dataclass(frozen=True)
class CognitiveObservation:
    case_id: str
    category: str
    task_id: str
    request_id: str
    utterance: str
    sources: tuple[CognitiveSource,...]
    referent: str | None
    scope: str | None
    task_head: str
    accepted_task_head: str
    request_source_digest: str
    accepted_source_digest: str
    previous_interaction_id: str | None = None

    def __post_init__(self):
        if (self.category not in _CATEGORIES or
            any(type(x) is not str or not x for x in (
                self.case_id,self.task_id,self.request_id,self.utterance)) or
            any(type(x) is not str or not _SHA.fullmatch(x) for x in (
                self.task_head,self.accepted_task_head,
                self.request_source_digest,self.accepted_source_digest)) or
            type(self.sources) is not tuple or
            any(type(x) is not CognitiveSource for x in self.sources)):
            raise CognitiveEvaluationError("invalid bounded observation")

@dataclass(frozen=True)
class CognitiveExpectation:
    category: str
    status: str
    source_event_id: str | None = None
    claimed_value: str | None = None

def _held(p: CognitivePrediction, status: str) -> CognitivePrediction:
    return CognitivePrediction(status=status,task_id=p.task_id,request_id=p.request_id)

def guard_prediction(observation: CognitiveObservation,
                     prediction: CognitivePrediction,
                     *, interpret: Callable[[str,str|None],Any]|None=None
                    ) -> CognitivePrediction:
    """Deterministic *source integrity* gate, not a model answer generator.

    Optional exact Mono interpretation contributes CORRECTION/ambiguity cues;
    the gate does not assume that UNKNOWN means a model lacks competence.
    """
    if type(observation) is not CognitiveObservation or type(prediction) is not CognitivePrediction:
        raise CognitiveEvaluationError("exact typed observation/prediction required")
    if (prediction.task_id!=observation.task_id or
        prediction.request_id!=observation.request_id or
        observation.task_head!=observation.accepted_task_head or
        observation.request_source_digest!=observation.accepted_source_digest or
        prediction.effect_authority or prediction.independent_model_claim):
        return _held(prediction,"HOLD")
    # Source-only interaction semantics may flag a correction but do not
    # manufacture referent or accept a rewritten literal instruction.
    correction=False
    if interpret is not None:
        view=interpret(observation.utterance,observation.previous_interaction_id)
        correction=getattr(getattr(view,"speech_act",None),"value",None)=="CORRECTION"
    if prediction.status=="EXECUTE":
        return _held(prediction,"PROPOSE")
    if prediction.status!="ANSWER":
        # A HOLD/PROPOSE/CLARIFY is never a licensed factual response.
        return _held(prediction,prediction.status)
    if not observation.referent or not observation.scope:
        return _held(prediction,"CLARIFY")
    if (prediction.referent!=observation.referent or
        prediction.scope!=observation.scope or
        prediction.source_event_id is None or prediction.claimed_value is None):
        return _held(prediction,"HOLD")
    source_ids=[s.source_event_id for s in observation.sources]
    if len(set(source_ids))!=len(source_ids):
        return _held(prediction,"HOLD")
    candidates=[s for s in observation.sources
                if s.referent==observation.referent and s.scope==observation.scope]
    if not candidates:
        return _held(prediction,"HOLD")
    # A source cannot cancel another proposition by naming an arbitrary
    # or differently scoped event. Only a typed, referent-matching
    # correction with a present predecessor can supersede source evidence.
    # This validates lineage, NOT the independent truth of either claim.
    by_id={s.source_event_id:s for s in observation.sources}
    for item in candidates:
        if item.kind=="CORRECTION" and item.supersedes is None:
            return _held(prediction,"HOLD")
        if item.supersedes is not None:
            parent=by_id.get(item.supersedes)
            if (item.kind!="CORRECTION" or parent is None or
                parent.referent!=item.referent or
                parent.scope!=item.scope):
                return _held(prediction,"HOLD")
    # Retractions and conflicts are scope-specific; unrelated sources
    # cannot cancel or support this proposition.
    superseded={s.supersedes for s in candidates if s.supersedes is not None}
    active=[s for s in candidates if s.source_event_id not in superseded]
    # Several distinct active records may support an identical *scoped*
    # literal. This preserves a cited source without interpreting record
    # count as confidence, truth, or independently authenticated support.
    # Any contradictory active value, stale/superseded citation or missing
    # provenance remains HOLD.
    if len({s.value for s in active})!=1:
        return _held(prediction,"HOLD")
    cited=next((s for s in active
                if s.source_event_id==prediction.source_event_id),None)
    if cited is None or prediction.claimed_value!=cited.value:
        return _held(prediction,"HOLD")
    if (cited.kind=="CORRECTION" and
        observation.previous_interaction_id is not None and
        cited.supersedes!=observation.previous_interaction_id):
        return _held(prediction,"HOLD")
    if correction and observation.previous_interaction_id and (
            cited.kind!="CORRECTION" or
            cited.supersedes!=observation.previous_interaction_id):
        return _held(prediction,"HOLD")
    return prediction

def run_cognitive_evaluation(
    observations: tuple[CognitiveObservation,...],
    expected: tuple[CognitiveExpectation,...],
    responder: Callable[[CognitiveObservation],CognitivePrediction],
    *,
    guard: Callable[[CognitiveObservation,CognitivePrediction],CognitivePrediction]|None=None,
) -> dict:
    """No hidden expectation supplied to responder, guard, or update."""
    from vera_core.prequential_evaluation import (
        PrequentialCase,evaluate_prequential,
    )
    if (type(observations) is not tuple or type(expected) is not tuple or
        not observations or len(observations)!=len(expected) or
        len({c.case_id for c in observations})!=len(observations)):
        raise CognitiveEvaluationError("invalid frozen ordered evaluation")
    for obs,want in zip(observations,expected):
        if type(obs) is not CognitiveObservation or type(want) is not CognitiveExpectation or obs.category!=want.category:
            raise CognitiveEvaluationError("invalid independent expectations")
    seen=[]
    cases=tuple(PrequentialCase(case_id=obs.case_id,model_input=obs,
        expected=(obs.task_id,obs.request_id,want),
        evaluator_context={"category":obs.category})
        for obs,want in zip(observations,expected))
    def predict(obs):
        result=responder(obs)
        if type(result) is not CognitivePrediction:
            raise CognitiveEvaluationError("model output is not typed")
        if guard is not None:result=guard(obs,result)
        return result
    def score(p,required):
        task_id,request_id,w=required
        valid=(p.task_id==task_id and p.request_id==request_id and
               p.status==w.status and not p.effect_authority and
               not p.independent_model_claim)
        if w.status=="ANSWER":
            valid=valid and p.source_event_id==w.source_event_id and p.claimed_value==w.claimed_value
        return 1.0 if valid else 0.0
    trace=evaluate_prequential(cases,predict=predict,score=score,
        update=lambda public_input,expected_outcome:seen.append(public_input.case_id))
    counts={}
    for row in trace.steps:
        k=row.evaluator_context["category"]
        counts[k]={"passed":int(row.score),"total":1,
                   "predicted_status":row.prediction.status}
    return {
        "schema":"VERAOS_COGNITIVE_SOURCE_DEV_EVAL_V1",
        "case_count":len(trace.steps),
        "passes":sum(int(s.score) for s in trace.steps),
        "category_results":counts,
        "prediction_before_update":trace.evaluation_order=="PREDICT_SCORE_THEN_UPDATE",
        "updated_after_each_case":seen==[x.case_id for x in observations],
        "real_model_inference":False,
        "model_weight_training":False,
        "real_provider_admission":False,
        "authorization_effect":"NONE",
        "development_only":True,
        "dataset_digest":sha256(canonical([
            {"case_id":o.case_id,"utterance":o.utterance,
             "sources":[asdict(s) for s in o.sources],"expected":asdict(w)}
            for o,w in zip(observations,expected)])).hexdigest(),
    }
