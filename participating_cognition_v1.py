"""Source-level Vera participating cognition adapter (Training research V1).

Combines existing Vera Core semantic-plan receipts, Mosaic deterministic
handoff receipts, and Cricket advisory review outcomes WITHOUT installing,
selecting or invoking a live QualifiedVeraRuntime. No learned model, brain,
external permission, global memory, or automatic authority transition.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import re
from typing import Any, Mapping

from independent_challenge_eval_v1 import (
    AttributedClaim, ChallengeFinding, ChallengeError, canonical, review_pair,
)

_SHA40=re.compile(r"^[0-9a-f]{40}$")
_IDENT=re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_FIDELITY=frozenset({"EXACT","CONSTRUCTIVE","LOSSY","UNREPRESENTABLE"})


class ParticipationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ParticipantBinding:
    """Node, actor, runtime instance and optional neural model remain distinct."""
    identity: str
    node_id: str
    agent_id: str
    runtime_instance_id: str
    model_sha256: str | None
    source_commit: str

    def __post_init__(self):
        for v in (self.identity,self.node_id,self.agent_id,self.runtime_instance_id):
            if type(v) is not str or _IDENT.fullmatch(v) is None:
                raise ParticipationError("Invalid participant referent")
        if type(self.source_commit) is not str or not _SHA40.fullmatch(self.source_commit):
            raise ParticipationError("Exact 40-character source commit required")
        if self.model_sha256 is not None and (
            type(self.model_sha256) is not str or
            re.fullmatch(r"[0-9a-f]{64}",self.model_sha256) is None
        ):
            raise ParticipationError("Invalid optional model digest")

    def as_dict(self)->dict[str,Any]:
        return {k:getattr(self,k) for k in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class Contribution:
    participant: ParticipantBinding
    claim: AttributedClaim
    task_id: str

    def __post_init__(self):
        if not isinstance(self.participant,ParticipantBinding) or not isinstance(self.claim,AttributedClaim):
            raise ParticipationError("Participant and attributed claim required")
        if type(self.task_id) is not str or _IDENT.fullmatch(self.task_id) is None:
            raise ParticipationError("Invalid task ID")
        p,c=self.participant,self.claim
        if (p.identity!=c.shared_identity or p.node_id!=c.node_id or
            p.agent_id!=c.agent_id or p.model_sha256!=c.model_digest):
            raise ParticipationError("Cross-identity or model/agent provenance substitution")

    def as_dict(self)->dict:
        return {"participant":self.participant.as_dict(),
                "claim":self.claim.mapping(),"claim_sha256":self.claim.digest,
                "task_id":self.task_id}


@dataclass(frozen=True,slots=True)
class ParticipationReceipt:
    """Provenance and advisory results, not accepted beliefs or execution."""
    schema: str
    task_id: str
    contributions: tuple[dict,dict]
    dissent: dict
    core_semantic_plan: dict
    mosaic_handoff: dict
    cricket_review: dict
    status: str = "SOURCE_EXPERIMENT_ONLY"
    identity_effect: str = "NONE"
    shared_memory_effect: str = "NONE"
    authority_effect: str = "NONE"
    executed_effect: str = "NONE"
    model_independence: str = "NOT_ESTABLISHED"
    cognitive_gain: str = "NOT_ESTABLISHED"
    _snapshot: bytes = field(init=False,repr=False,compare=False)

    def __post_init__(self):
        result={k:getattr(self,k) for k in self.__dataclass_fields__ if k!="_snapshot"}
        result["contributions"]=list(result["contributions"])
        object.__setattr__(self,"_snapshot",canonical(result))

    def as_dict(self)->dict:
        return json.loads(self._snapshot)


def _safe_receipt(label:str, raw: Mapping[str,Any], allowed:set[str]) -> dict:
    if not isinstance(raw,Mapping):
        raise ParticipationError(f"{label}: structured source receipt required")
    if set(raw)!=allowed:
        raise ParticipationError(f"{label}: unknown or missing receipt fields")
    data=dict(raw)
    canonical(data)
    return data


def compose_episode(
    first: Contribution, second: Contribution, *,
    core_plan: Mapping[str,Any],
    mosaic_result: Mapping[str,Any],
    cricket_result: Mapping[str,Any],
    verify_evidence=None,
    verify_independence=None,
    verify_participant_independence=None,
) -> ParticipationReceipt:
    """Requires reports computed *by* real source modules at the call site.

    Never infer that source 'PASS', model agreement, or a handoff is effect
    authority or same-process identity. Inputs are source-test observations.
    A participant-level verifier must independently establish whether separate
    claimed runtime instances are truly independent; a claim-only verifier is
    insufficient. Both callbacks are untrusted until receiver-owned.
    """
    if not isinstance(first,Contribution) or not isinstance(second,Contribution):
        raise ParticipationError("Two typed contributions required")
    if first.task_id!=second.task_id or first.participant.identity!=second.participant.identity:
        raise ParticipationError("Cannot combine different task/shared identity")
    if first.participant.agent_id == second.participant.agent_id:
        raise ParticipationError("Same logical agent cannot be counted twice by restarting it")
    # A claim-only callback does not see node-bound participant runtime IDs.
    # A reused declared runtime identity is correlation, NOT erasure of either
    # logical participant and NOT a second independent piece of evidence.
    if first.participant.runtime_instance_id == second.participant.runtime_instance_id:
        dissent=review_pair(first.claim,second.claim)
        if dissent.classification=="UNVERIFIED_DISSENT":
            dissent=replace(dissent,classification="SELF_OR_CORRELATED_CLAIMS")
    else:
        def _verify_pair(a,b):
            if verify_participant_independence is None or verify_independence is None:
                return False
            # Exceptions, missing verifiers, non-Boolean results fail closed
            # through review_pair's callback guard.
            return (verify_participant_independence(first,second) is True and
                    verify_independence(a,b) is True)
        dissent=review_pair(first.claim,second.claim,
            verify_evidence=verify_evidence,verify_independence=_verify_pair)
    plan=_safe_receipt("Vera Core semantic plan",core_plan,{
        "source_profile","target_profile","required_capabilities",
        "native_capabilities","missing_capabilities","unresolved_capabilities",
        "rewrites","fidelity","semantic_equivalence","authorization_effect",
        "catalog_id","catalog_digest",
    })
    if (plan["fidelity"] not in _FIDELITY or
        plan["authorization_effect"]!="NONE" or
        plan["semantic_equivalence"]!="NOT_ESTABLISHED"):
        raise ParticipationError("Semantic transfer cannot grant authority/equivalence")
    required,native,missing,unresolved=(
        plan[x] for x in ("required_capabilities","native_capabilities",
                          "missing_capabilities","unresolved_capabilities"))
    if not all(type(x) is list and all(type(y) is str for y in x)
               for x in (required,native,missing,unresolved)):
        raise ParticipationError("Invalid semantic-plan capability inventory")
    if (set(native)|set(missing))!=set(required) or set(native)&set(missing):
        raise ParticipationError("Semantic plan misstates native/missing capabilities")
    if not set(unresolved).issubset(set(missing)):
        raise ParticipationError("Unresolved capabilities not tied to missing ones")
    if bool(unresolved)!=(plan["fidelity"]=="UNREPRESENTABLE"):
        raise ParticipationError("Unresolved semantic transfer falsely declared representable")
    # An EXACT receipt with missing capabilities cannot omit their rewrites.
    # Verify the entire source-plan inventory, not just its top-level label.
    rewrites=plan["rewrites"]
    rewrite_keys={"source_capability","rule_name","fidelity","description"}
    if type(rewrites) is not list or any(
        type(r) is not dict or set(r)!=rewrite_keys or
        any(type(r[k]) is not str or not r[k] for k in rewrite_keys) or
        r["fidelity"] not in _FIDELITY for r in rewrites
    ):
        raise ParticipationError("Invalid semantic rewrite entries")
    rewrite_sources=[r["source_capability"] for r in rewrites]
    if (len(rewrite_sources)!=len(set(rewrite_sources)) or
        set(rewrite_sources)|set(unresolved)!=set(missing) or
        set(rewrite_sources)&set(unresolved)):
        raise ParticipationError("Semantic rewrite coverage contradicts missing capabilities")
    severity={"EXACT":0,"CONSTRUCTIVE":1,"LOSSY":2,"UNREPRESENTABLE":3}
    expected_fidelity=("UNREPRESENTABLE" if unresolved else
        max(("EXACT", *(r["fidelity"] for r in rewrites)),
            key=severity.__getitem__))
    if plan["fidelity"]!=expected_fidelity:
        raise ParticipationError("Semantic fidelity contradicts rewrite severity")
    handoff=_safe_receipt("Mosaic P1 result",mosaic_result,{
        "case_id","family","condition","answer","expected_value",
        "correct","stale_error","correction_burden","payload_bytes",
    })
    if handoff["condition"] not in ("RESET","MOSAIC_STATE","FULL_HISTORY","ORDINARY_RETRIEVAL"):
        raise ParticipationError("Unknown Mosaic condition")
    if type(handoff["correct"]) is not bool or type(handoff["stale_error"]) is not bool:
        raise ParticipationError("Invalid handoff measured fields")
    if (type(handoff["answer"]) is not str or
        type(handoff["expected_value"]) is not str or
        handoff["correct"] != (handoff["answer"]==handoff["expected_value"]) or
        (handoff["correct"] and handoff["stale_error"]) or
        type(handoff["correction_burden"]) is not int or
        handoff["correction_burden"] != (0 if handoff["correct"] else 1) or
        type(handoff["payload_bytes"]) is not int or handoff["payload_bytes"] < 0):
        raise ParticipationError("Contradictory or invalid Mosaic source-handoff receipt")
    review=_safe_receipt("Cricket review",cricket_result,{"disposition","findings"})
    if (review["disposition"] not in ("PASS","CHALLENGE","BLOCK") or
        type(review["findings"]) is not list):
        raise ParticipationError("Invalid Cricket advisory result")
    required_finding={"rule_id","severity","title","rationale","evidence",
                      "recommendation","source"}
    if any(type(f) is not dict or set(f)!=required_finding or
           f["severity"] not in ("NOTE","CHALLENGE","BLOCK")
           for f in review["findings"]):
        raise ParticipationError("Invalid Cricket finding content")
    # Cricket.review computes its disposition from the strongest finding.
    # Reject stronger AND weaker claimed dispositions, including BLOCK with
    # empty findings, so synthetic receipts cannot masquerade as real Cricket.
    severities={finding["severity"] for finding in review["findings"]}
    expected_disposition=("BLOCK" if "BLOCK" in severities else
                          "CHALLENGE" if "CHALLENGE" in severities else "PASS")
    if review["disposition"]!=expected_disposition:
        raise ParticipationError("Contradictory Cricket disposition and findings")
    # A blocking review must remain visible even when the semantic plan
    # separately declares the transfer unrepresentable.
    if review["disposition"]=="BLOCK":
        status="CRICKET_BLOCKED"
    elif plan["fidelity"]=="UNREPRESENTABLE":
        status="SEMANTIC_TRANSFER_UNREPRESENTABLE"
    elif review["disposition"]=="CHALLENGE":
        status="CRICKET_CHALLENGED"
    elif dissent.classification in ("DIFFERENT_PROPOSITION","DIFFERENT_SCOPE","DIFFERENT_REVISION"):
        status="UNRESOLVED_REFERENT_OR_SCOPE"
    else:
        status="SOURCE_EXPERIMENT_ONLY"
    return ParticipationReceipt(
        "VERA_PARTICIPATING_COGNITION_SOURCE_V1",
        first.task_id,(first.as_dict(),second.as_dict()),
        {"classification":dissent.classification,
         "source_digests":list(dissent.source_digests),
         "challenge_for_review":dissent.challenge_for_review,
         "accepted_as_fact":False,"independence_proven":False},
        plan,handoff,review,status=status,
    )
