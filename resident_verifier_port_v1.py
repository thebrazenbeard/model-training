"""OS-composable *source contract* for one no-effect local cognition trial.

No caller-provided PASS may admit a model. A trusted VeraOS composition
would inject its installed host verifier + Vera Mono OutboundTrustRegistry
(PROVIDER role) and a durable nonce/task journal. The only executable positive
mode here is an explicitly test-only scoped verifier simulation. Real requests
HOLD until OS-owned runtime implementation/review.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import re
from collections.abc import Mapping

_SHA=re.compile(r"^[0-9a-f]{64}$")
_NONCE=re.compile(r"^[0-9a-f]{32}$")
_ID=re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_ENDPOINT="http://127.0.0.1:18766/v1"
_PROVIDER="dev:virtio-qwen4b"
_MODEL="qwen3:4b-instruct"
_MANIFEST="0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
_BLOB="85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
_BYTES=2497280480


@dataclass(frozen=True)
class ResidentModelRequest:
    provider_id: str
    model_id: str
    endpoint: str
    serving_host: str
    generation: int
    session_id: str
    nonce: str
    manifest_sha256: str
    artifact_sha256: str
    artifact_bytes: int
    task_id: str
    task_digest: str
    observed_at_epoch_s: int
    expires_at_epoch_s: int
    minimum_ram_mib: int
    route_class: str

    def __post_init__(self):
        for value in (self.provider_id,self.model_id,self.serving_host,
                      self.session_id,self.task_id):
            if type(value) is not str or _ID.fullmatch(value) is None:
                raise ValueError("invalid scoped request identity")
        for field in ("nonce","manifest_sha256","artifact_sha256","task_digest"):
            val=getattr(self,field)
            pattern=_NONCE if field=="nonce" else _SHA
            if type(val) is not str or pattern.fullmatch(val) is None:
                raise ValueError("invalid scoped request digest or nonce")
        for name in ("generation","artifact_bytes","observed_at_epoch_s",
                     "expires_at_epoch_s","minimum_ram_mib"):
            v=getattr(self,name)
            if type(v) is not int or v < (0 if name.endswith("_s") or name=="minimum_ram_mib" else 1):
                raise ValueError("invalid exact request resource/generation/time")
        if (self.provider_id!=_PROVIDER or self.model_id!=_MODEL or
            self.endpoint!=_ENDPOINT or self.route_class!="COGNITION_ONLY" or
            self.manifest_sha256!=_MANIFEST or
            self.artifact_sha256!=_BLOB or self.artifact_bytes!=_BYTES):
            raise ValueError("unrecognized development Qwen control identity")
        if self.expires_at_epoch_s<=self.observed_at_epoch_s:
            raise ValueError("invalid trial expiration")


@dataclass(frozen=True)
class SignedHostEvidence:
    issuer_id: str
    key_id: str
    key_digest: str
    signature: str

    def __post_init__(self):
        for text in (self.issuer_id,self.key_id):
            if type(text) is not str or _ID.fullmatch(text) is None:
                raise ValueError("invalid verifier identity")
        for text in (self.key_digest,self.signature):
            if type(text) is not str or _SHA.fullmatch(text) is None:
                raise ValueError("invalid evidence signature or key digest")


@dataclass(frozen=True)
class ResidentAdmissionDecision:
    status: str
    holds: tuple[str,...]
    request_sha256: str
    task_id: str
    provider_id: str
    endpoint: str
    issuer_id: str
    registry_generation: int | None
    eligible_for_test_consumer: bool=False
    normal_resident_admitted: bool=False
    production_qualified: bool=False
    effect_authority: bool=False
    identity_authority: bool=False


def canonical_request(request: ResidentModelRequest) -> bytes:
    if type(request) is not ResidentModelRequest:
        raise ValueError("typed exact request required")
    return json.dumps({
        "schema":"VERA_OS_SCOPED_MODEL_REQUEST_V1",
        **asdict(request),
        "allowed_actions":["COGNITION_TURN"],
        "identity_authority":False,
        "protected_effect_authority":False,
        "canonical_memory_authority":False,
    },sort_keys=True,separators=(",",":"),ensure_ascii=False,
        allow_nan=False).encode("utf8")


def _assert_receipt(trust,evidence,request):
    """Reuse Mono outbound provider registry API, not another trust hierarchy."""
    result=trust.assert_current(
        authority_id=evidence.issuer_id,role="PROVIDER",
        provider_id=request.provider_id,
        key_id=evidence.key_id,key_digest=evidence.key_digest)
    # Real Mono returns AuthorityCurrentnessReceipt; fixture may return
    # a same-shape mapping. Mere boolean from a claimed verifier is rejected.
    get=(result.get if isinstance(result,Mapping) else
         lambda key,default=None:getattr(result,key,default))
    if (get("authority_id")!=evidence.issuer_id or get("role")!="PROVIDER" or
        get("provider_id")!=request.provider_id or
        get("key_id")!=evidence.key_id or
        get("key_digest")!=evidence.key_digest or
        type(get("registry_generation")) is not int or
        get("registry_generation")<1 or
        type(get("receipt_digest")) is not str or
        _SHA.fullmatch(get("receipt_digest")) is None):
        raise ValueError("malformed trust currentness receipt")
    return get("registry_generation")


def evaluate_resident_admission(request, evidence, trust, verifier, *,
                                 now_epoch_s:int,mode:str):
    if type(request) is not ResidentModelRequest or type(evidence) is not SignedHostEvidence:
        raise ValueError("typed signed host evidence required")
    if type(now_epoch_s) is not int or now_epoch_s<0:
        raise ValueError("invalid time")
    digest=hashlib.sha256(canonical_request(request)).hexdigest()
    holds=[]
    if mode!="TEST_ONLY":
        holds.append("NO_OS_BOUND_TRUSTED_VERIFIER")
    if (not request.observed_at_epoch_s<=now_epoch_s<=request.expires_at_epoch_s or
        now_epoch_s-request.observed_at_epoch_s>120):
        holds.append("EXPIRED")
    generation=None
    try:
        generation=_assert_receipt(trust,evidence,request)
    except (ValueError,TypeError,KeyError,AttributeError,PermissionError):
        holds.append("UNTRUSTED_OR_REVOKED_ISSUER")
    try:
        signed_ok=verifier.verify_signature(
            message=canonical_request(request),signature=evidence.signature,
            key_id=evidence.key_id)
        if signed_ok is not True:
            holds.append("BAD_SIGNATURE")
    except (ValueError,TypeError,KeyError,AttributeError):
        holds.append("BAD_SIGNATURE")
    try:
        actual=verifier.reobserve(request)
        expected={
            "serving_host":request.serving_host,
            "generation":request.generation,
            "session_id":request.session_id,
            "nonce":request.nonce,
            "artifact_sha256":request.artifact_sha256,
            "manifest_sha256":request.manifest_sha256,
            "artifact_bytes":request.artifact_bytes,
            "endpoint":request.endpoint,
            "model_id":request.model_id,
            "task_id":request.task_id,
            "task_digest":request.task_digest,
        }
        if not isinstance(actual,Mapping) or any(actual.get(k)!=v for k,v in expected.items()):
            holds.append("HOST_OBSERVATION_MISMATCH")
        if (not isinstance(actual,Mapping) or
            type(actual.get("ram_available_mib")) is not int or
            actual["ram_available_mib"]<request.minimum_ram_mib):
            holds.append("HOST_RESOURCE_NOT_OBSERVED")
        observed_at=actual.get("observed_at_epoch_s") if isinstance(actual,Mapping) else None
        if (type(observed_at) is not int or
            observed_at != request.observed_at_epoch_s or
            now_epoch_s - observed_at not in range(121)):
            holds.append("HOST_SAMPLE_OUTSIDE_WINDOW")
    except (ValueError,TypeError,KeyError,AttributeError):
        holds.append("HOST_OBSERVATION_MISMATCH")
    eligible=not holds
    return ResidentAdmissionDecision(
        "TEST_VERIFIER_SCOPED_ACCEPT" if eligible else "HOLD",
        tuple(dict.fromkeys(holds)),digest,request.task_id,
        request.provider_id,request.endpoint,evidence.issuer_id,generation,
        eligible_for_test_consumer=eligible)


def consume_resident_admission(decision,request,evidence,trust,verifier,*,
                               now_epoch_s:int,mode:str,test_consumer):
    """Reverify at consumption and consume once before passing a HELD OS shape.

    This is a test-only route sink; it never calls actual native_cognition.
    OS must install its independent verifier/atomic ledger and separately
    authorize any real provider. Do not import this lab as runtime dependency.
    """
    if type(decision) is not ResidentAdmissionDecision or mode!="TEST_ONLY":
        raise ValueError("only synthetic source consumer exists")
    latest=evaluate_resident_admission(request,evidence,trust,verifier,
                                        now_epoch_s=now_epoch_s,mode=mode)
    if latest!=decision or not latest.eligible_for_test_consumer:
        raise ValueError("held, stale or modified verifier decision")
    if verifier.consume_once(request) is not True:
        raise ValueError("replayed or unavailable one-use trial checkpoint")
    if not callable(test_consumer):
        raise ValueError("test-only callback missing")
    held={
        "schema":"VERA_OS_COGNITION_PROVIDER_V1",
        "provider_id":request.provider_id,
        "provider_kind":"LOCAL_MODEL",
        "capabilities":["text"],
        "transport":{"kind":"OPENAI_COMPATIBLE_HTTP",
                     "attachment":request.endpoint},
        "qualification":{
            "status":"HOLD","issuer":"thebrazenbeard/vera-os",
            "subject_sha256":request.artifact_sha256,"suite_sha256":"0"*64},
        "authority":{
            "identity_authority":False,
            "canonical_memory_write_authority":False,
            "protected_effect_authority":False,
            "browser_authority":False},
        "admission":{
            "status":"HELD","route_class":"COGNITION_ONLY",
            "selector_owner":"thebrazenbeard/portal"},
    }
    return test_consumer(held)
