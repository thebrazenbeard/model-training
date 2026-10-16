"""Vera Training research: attributed dissent and false-challenge development checks.

A heuristic *review selection* evaluator only. Does not authenticate claims,
independence, entailment, correctness, decision adoption, effect, or model skill.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import re
from typing import Any, Callable, Mapping

_IDENT=re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,127}$")
_SHA=re.compile(r"^[a-f0-9]{64}$")
_TYPES=frozenset({"REPOSITORY_SOURCE","RUNTIME_OBSERVATION","INFERENCE",
                  "HYPOTHESIS","VERIFICATION","EFFECT","UNKNOWN"})
_OBSERVATIONAL=frozenset({"REPOSITORY_SOURCE","RUNTIME_OBSERVATION","VERIFICATION"})
_STATES=frozenset({"EVIDENCE_CONFLICT_FOR_REVIEW","SAME_CLAIM","DIFFERENT_PROPOSITION",
                   "DIFFERENT_SCOPE","DIFFERENT_REVISION","UNVERIFIED_DISSENT",
                   "MIXED_CLAIM_TYPES","SELF_OR_CORRELATED_CLAIMS"})


class ChallengeError(ValueError):
    pass


def _id(v: str) -> str:
    if type(v) is not str or _IDENT.fullmatch(v) is None:
        raise ChallengeError("Invalid source identity or semantic subject")
    return v


def _safe_obj(v: Any, depth: int = 0) -> None:
    if depth>12:
        raise ChallengeError("Structured payload too deep")
    if v is None or type(v) in (bool,str):
        return
    if type(v) is int and -9007199254740991<=v<=9007199254740991:
        return
    if type(v) is list and len(v)<=100:
        for x in v: _safe_obj(x,depth+1)
        return
    if type(v) is dict and len(v)<=100:
        for k,item in v.items():
            if type(k) is not str or not k.isascii() or not k.isprintable():
                raise ChallengeError("Invalid semantic object key")
            _safe_obj(item,depth+1)
        return
    raise ChallengeError("Unsupported noncanonical value")


def canonical(value: Any) -> bytes:
    _safe_obj(value)
    data=json.dumps(value,sort_keys=True,separators=(",",":"),
                    ensure_ascii=False,allow_nan=False).encode("utf8")
    if len(data)>8192:
        raise ChallengeError("Bounded content size exceeded")
    return data


@dataclass(frozen=True, slots=True)
class AttributedClaim:
    """Node, agent, model and shared identity are deliberately independent IDs."""
    message_id: str
    shared_identity: str
    node_id: str
    agent_id: str
    model_digest: str | None
    claim_type: str
    referent: str
    predicate: str
    scope: str
    revision: int
    evidence_digest: str | None
    evidence_group: str | None
    value: Any
    _snapshot: bytes = field(init=False,repr=False,compare=False)

    def __post_init__(self):
        for v in (self.message_id,self.shared_identity,self.node_id,self.agent_id,
                  self.referent,self.predicate,self.scope):
            _id(v)
        if self.model_digest is not None and _SHA.fullmatch(self.model_digest) is None:
            raise ChallengeError("Invalid model artifact digest")
        if self.evidence_digest is not None and _SHA.fullmatch(self.evidence_digest) is None:
            raise ChallengeError("Invalid evidence digest")
        if self.evidence_group is not None:
            _id(self.evidence_group)
        if self.claim_type not in _TYPES:
            raise ChallengeError("Unknown proposition claim type")
        if type(self.revision) is not int or self.revision < 0:
            raise ChallengeError("Revision must be nonnegative integer")
        object.__setattr__(self, "_snapshot",canonical(self.value))

    def mapping(self) -> dict[str,Any]:
        result={name:getattr(self,name) for name in self.__dataclass_fields__
                if name not in ("value","_snapshot")}
        result["value"]=json.loads(self._snapshot)
        return result

    @property
    def digest(self)->str:
        return sha256(canonical(self.mapping())).hexdigest()


@dataclass(frozen=True, slots=True)
class ChallengeFinding:
    classification: str
    challenge_for_review: bool
    model_independence_proven: bool
    effect_authorized: bool
    accepted_as_fact: bool
    source_digests: tuple[str,str]
    retained_claims: tuple[dict,dict]


# Receiver-owned independent evidence validation must be external.
VerifyEvidence=Callable[[AttributedClaim], bool]
VerifyIndependence=Callable[[AttributedClaim,AttributedClaim], bool]


def _verified(callback, *args) -> bool:
    if callback is None:
        return False
    try:
        return callback(*args) is True
    except Exception:
        return False


def review_pair(
    a: AttributedClaim,
    b: AttributedClaim,
    *,
    verify_evidence: VerifyEvidence | None = None,
    verify_independence: VerifyIndependence | None = None,
)->ChallengeFinding:
    """A candidate conflict is not adoption, prohibition, or verification.

    Preserve *both* exact claims regardless of disposition; no statement
    overwrites or strengthens a proposition in a different provenance lane.
    """
    if not isinstance(a,AttributedClaim) or not isinstance(b,AttributedClaim):
        raise ChallengeError("Two typed claims required")
    if a.shared_identity!=b.shared_identity or (a.referent,a.predicate)!=(b.referent,b.predicate):
        state="DIFFERENT_PROPOSITION"
    elif a.scope!=b.scope:
        state="DIFFERENT_SCOPE"
    elif a.revision!=b.revision:
        state="DIFFERENT_REVISION"
    elif a._snapshot==b._snapshot:
        state="SAME_CLAIM"
    elif (a.agent_id==b.agent_id and a.node_id==b.node_id) or (
        a.evidence_group is not None and a.evidence_group==b.evidence_group
    ):
        state="SELF_OR_CORRELATED_CLAIMS"
    elif a.claim_type!=b.claim_type or a.claim_type not in _OBSERVATIONAL:
        state="MIXED_CLAIM_TYPES"
    elif not (a.evidence_digest and b.evidence_digest and
              a.evidence_group and b.evidence_group and
              _verified(verify_evidence,a) and _verified(verify_evidence,b) and
              _verified(verify_independence,a,b)):
        state="UNVERIFIED_DISSENT"
    else:
        state="EVIDENCE_CONFLICT_FOR_REVIEW"

    if state not in _STATES:
        raise ChallengeError("Unknown review classification")
    return ChallengeFinding(
        state,state=="EVIDENCE_CONFLICT_FOR_REVIEW",
        False,False,False,(a.digest,b.digest),
        (a.mapping(),b.mapping()),
    )


def measure_false_challenge_rate(
    examples: list[dict],
    *,
    verify_evidence: VerifyEvidence | None = None,
    verify_independence: VerifyIndependence | None = None,
)->dict:
    """Gold boolean means useful attributed conflict *for review*, never truth.

    The naive comparator highlights *any* disagreement between different
    role labels; the candidate compares exact claim/provenance boundaries.
    """
    if type(examples) is not list or not 6<=len(examples)<=100:
        raise ChallengeError("Bounded labeled fixture required")
    ids=set()
    totals={"positive":0,"negative":0,
            "naive_tp":0,"naive_fp":0,"naive_fn":0,
            "candidate_tp":0,"candidate_fp":0,"candidate_fn":0}
    records=[]
    for case in examples:
        if type(case) is not dict or set(case)!={"id","a","b","expected_challenge"}:
            raise ChallengeError("Malformed labeled example")
        key=_id(case["id"])
        if key in ids:raise ChallengeError("Duplicate fixture identity")
        ids.add(key)
        a=AttributedClaim(**case["a"])
        b=AttributedClaim(**case["b"])
        if type(case["expected_challenge"]) is not bool:
            raise ChallengeError("Gold challenge flag must be boolean")
        expected=case["expected_challenge"]
        finding=review_pair(a,b,verify_evidence=verify_evidence,
                            verify_independence=verify_independence)
        naive=(a.agent_id!=b.agent_id and a._snapshot!=b._snapshot)
        totals["positive" if expected else "negative"]+=1
        for key,prediction in (("naive",naive),("candidate",finding.challenge_for_review)):
            if prediction and expected:totals[key+"_tp"]+=1
            elif prediction and not expected:totals[key+"_fp"]+=1
            elif not prediction and expected:totals[key+"_fn"]+=1
        records.append({"id":case["id"],"classification":finding.classification,
                        "expected":expected,"naive":naive,
                        "candidate":finding.challenge_for_review,
                        "source_digests":finding.source_digests})
    return {"scope":"SYNTHETIC_DEVELOPMENT_NOT_HELD_OUT",
            "total":len(examples),"totals":totals,"records":records,
            "candidate_false_challenge_rate":totals["candidate_fp"]/max(1,totals["negative"]),
            "naive_false_challenge_rate":totals["naive_fp"]/max(1,totals["negative"]),
            "qualification":"NONE"}
