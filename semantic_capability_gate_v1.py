"""Receiver-side advisory gate for machine-native cognitive handoffs.

Neither a model advertisement nor this gate authenticates a node, verifies a
certificate's issuer, launches inference, allocates GPU, or authorizes effects.
An independent adapter supplies VERIFIED qualifications to this pure function.
A successful result means *advisory compatibility only*.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Callable

_SHA = re.compile(r"^[0-9a-f]{64}$")
_IDENT = re.compile(r"^[a-z][a-z0-9_.:/-]{0,95}$")
_MODES = frozenset({"TYPED_STATE", "TOKEN_IDS", "LATENT", "KV_CACHE"})
_CATEGORIES = ("referent", "correction", "scope", "authority", "recovery")


class SemanticGateError(ValueError):
    pass


def _digest(value: str) -> str:
    if type(value) is not str or not _SHA.fullmatch(value):
        raise SemanticGateError("Expected exact lowercase SHA256")
    return value


def _id(value: str) -> str:
    if type(value) is not str or not _IDENT.fullmatch(value):
        raise SemanticGateError("Invalid typed identifier")
    return value


def _hash_struct(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelLineage:
    model_sha256: str
    tokenizer_sha256: str
    adapter_sha256: str
    prompt_template_sha256: str
    decoding_sha256: str
    runtime_layout_sha256: str
    semantic_space_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "model_sha256", "tokenizer_sha256", "adapter_sha256",
            "prompt_template_sha256", "decoding_sha256", "runtime_layout_sha256",
        ):
            _digest(getattr(self, name))
        if self.semantic_space_sha256 is not None:
            _digest(self.semantic_space_sha256)

    @property
    def exact_id(self) -> str:
        return _hash_struct({
            name: getattr(self, name) for name in self.__dataclass_fields__
        })


@dataclass(frozen=True, slots=True)
class SemanticTask:
    task_id: str
    capability_id: str
    ontology_sha256: str
    input_schema_sha256: str
    output_schema_sha256: str
    context_sha256: str
    effect_class: str = "READ_ONLY"
    privacy_class: str = "PUBLIC"

    def __post_init__(self):
        _id(self.task_id)
        _id(self.capability_id)
        for name in ("ontology_sha256", "input_schema_sha256",
                     "output_schema_sha256", "context_sha256"):
            _digest(getattr(self, name))
        if self.effect_class not in ("READ_ONLY", "REVERSIBLE_MUTATION",
                                     "PROTECTED_MUTATION", "IRREVERSIBLE"):
            raise SemanticGateError("Unknown effect class")
        if self.privacy_class not in ("PUBLIC", "RESTRICTED", "SECRET"):
            raise SemanticGateError("Unknown privacy class")


@dataclass(frozen=True, slots=True)
class CategoryScore:
    name: str
    passed: int
    total: int

    def __post_init__(self):
        if self.name not in _CATEGORIES:
            raise SemanticGateError("Unknown qualification category")
        if (type(self.passed) is not int or type(self.total) is not int
                or self.total < 5 or not 0 <= self.passed <= self.total):
            raise SemanticGateError("Invalid qualification counts")


@dataclass(frozen=True, slots=True)
class Qualification:
    """Claimed eval data. Only receiver-trusted verification can qualify it."""

    lineage_sha256: str
    capability_id: str
    ontology_sha256: str
    input_schema_sha256: str
    output_schema_sha256: str
    evaluator_sha256: str
    heldout_dataset_sha256: str
    scores: tuple[CategoryScore, ...]

    def __post_init__(self):
        for name in (
            "lineage_sha256", "ontology_sha256", "input_schema_sha256",
            "output_schema_sha256", "evaluator_sha256", "heldout_dataset_sha256",
        ):
            _digest(getattr(self, name))
        _id(self.capability_id)
        if (type(self.scores) is not tuple or
            any(not isinstance(x, CategoryScore) for x in self.scores) or
            tuple(x.name for x in self.scores) != _CATEGORIES or
            sum(x.total for x in self.scores) < 25):
            raise SemanticGateError("Five named held-out categories required")

    @property
    def evidence_id(self) -> str:
        return _hash_struct({
            "lineage": self.lineage_sha256,
            "capability": self.capability_id,
            "ontology": self.ontology_sha256,
            "input": self.input_schema_sha256,
            "output": self.output_schema_sha256,
            "evaluator": self.evaluator_sha256,
            "dataset": self.heldout_dataset_sha256,
            "scores": [(s.name, s.passed, s.total) for s in self.scores],
        })


@dataclass(frozen=True, slots=True)
class Handoff:
    task: SemanticTask
    sender: ModelLineage
    receiver: ModelLineage
    mode: str
    peer_node_id: str
    alignment_evidence_sha256: str | None = None

    def __post_init__(self):
        if (not isinstance(self.task, SemanticTask) or
            not isinstance(self.sender, ModelLineage) or
            not isinstance(self.receiver, ModelLineage)):
            raise SemanticGateError("Handoff requires typed task and model lineages")
        if self.mode not in _MODES:
            raise SemanticGateError("Unsupported neural handoff mode")
        _id(self.peer_node_id)
        if self.alignment_evidence_sha256 is not None:
            _digest(self.alignment_evidence_sha256)


@dataclass(frozen=True, slots=True)
class GateResult:
    status: str
    reasons: tuple[str, ...]
    task_id: str
    receiver_lineage_sha256: str
    qualification_evidence_sha256: str | None

    @property
    def eligible(self) -> bool:
        return self.status == "ADVISORY_COMPATIBLE"


# Returned True ONLY after verifier-owned validation against external evidence.
VerifyQualification = Callable[[Qualification], bool]
VerifyAlignment = Callable[[str, Handoff], bool]


def _verified(callback, *args) -> bool:
    """Unknown/failed third-party verification is a HOLD, never an ALLOW."""
    if callback is None:
        return False
    try:
        return callback(*args) is True
    except Exception:
        return False


def assess_handoff(
    handoff: Handoff,
    *,
    qualification: Qualification | None,
    verify_qualification: VerifyQualification | None,
    verify_alignment: VerifyAlignment | None = None,
    authenticated_peer_node_id: str | None = None,
    allow_kv_cache: bool = False,
    restricted_transport_attested: bool = False,
) -> GateResult:
    """Fail-closed compatibility screening; not authority or dispatch."""
    if not isinstance(handoff, Handoff):
        raise TypeError("Handoff required")
    task = handoff.task
    reasons: list[str] = []
    if authenticated_peer_node_id != handoff.peer_node_id:
        reasons.append("PEER_NOT_AUTHENTICATED")
    if task.effect_class != "READ_ONLY":
        reasons.append("EFFECT_AUTHORITY_NOT_IN_SCOPE")
    if task.privacy_class == "SECRET":
        reasons.append("SECRET_HANDOFF_UNSUPPORTED")
    elif task.privacy_class == "RESTRICTED" and not restricted_transport_attested:
        reasons.append("RESTRICTED_TRANSPORT_NOT_ATTESTED")

    if qualification is None or not isinstance(qualification, Qualification):
        reasons.append("MISSING_QUALIFICATION")
    else:
        if (qualification.lineage_sha256 != handoff.receiver.exact_id or
            qualification.capability_id != task.capability_id or
            qualification.ontology_sha256 != task.ontology_sha256 or
            qualification.input_schema_sha256 != task.input_schema_sha256 or
            qualification.output_schema_sha256 != task.output_schema_sha256):
            reasons.append("QUALIFICATION_BINDING_MISMATCH")
        if not _verified(verify_qualification, qualification):
            reasons.append("QUALIFICATION_NOT_VERIFIED")
        if any(score.passed * 5 < score.total * 4 for score in qualification.scores):
            reasons.append("QUALIFICATION_BELOW_THRESHOLD")

    if handoff.mode == "TOKEN_IDS":
        if handoff.sender.tokenizer_sha256 != handoff.receiver.tokenizer_sha256:
            reasons.append("TOKENIZER_MISMATCH")
    elif handoff.mode == "LATENT":
        if (handoff.sender.semantic_space_sha256 is None or
            handoff.sender.semantic_space_sha256 != handoff.receiver.semantic_space_sha256):
            reasons.append("SEMANTIC_SPACE_MISMATCH")
        if (handoff.alignment_evidence_sha256 is None or
            not _verified(verify_alignment, handoff.alignment_evidence_sha256, handoff)):
            reasons.append("LATENT_ALIGNMENT_NOT_VERIFIED")
    elif handoff.mode == "KV_CACHE":
        if not allow_kv_cache:
            reasons.append("KV_OPT_IN_REQUIRED")
        # Do not equate a same-model string with KV compatibility.
        # Context, decoding, prompt, adapter and runtime layout all matter.
        for field in ("model_sha256", "tokenizer_sha256", "adapter_sha256",
                      "prompt_template_sha256", "decoding_sha256",
                      "runtime_layout_sha256"):
            if getattr(handoff.sender, field) != getattr(handoff.receiver, field):
                reasons.append("KV_"+field.upper()+"_MISMATCH")
        if (handoff.alignment_evidence_sha256 is None or
            not _verified(verify_alignment, handoff.alignment_evidence_sha256,handoff)):
            reasons.append("KV_CONTEXT_AND_POSITION_NOT_VERIFIED")
    # TYPED_STATE is model-agnostic; ontology and output contracts
    # are still bound through the independent qualification.
    return GateResult(
        "HELD" if reasons else "ADVISORY_COMPATIBLE", tuple(reasons),
        task.task_id, handoff.receiver.exact_id,
        qualification.evidence_id if isinstance(qualification, Qualification) else None,
    )
