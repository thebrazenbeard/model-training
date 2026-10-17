"""Read-only development model preflight. Never selects or loads a provider.

A synthetic acceptance proves the source adapter can correlate one bounded
model/file sample with the intended host and existing two-participant task.
No inference, model qualification, trusted issuer or normal route is implied.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from veraos_development_evidence_v1 import (DevelopmentAssessment, DEV_MODEL, DEV_OS_HEAD, bind_participation_development_cut)

_SHA=re.compile(r"^[0-9a-f]{64}$")
_NONCE=re.compile(r"^[0-9a-f]{32}$")
_IDENT=re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


@dataclass(frozen=True)
class ServingSession:
    host_id: str
    generation: int
    session_id: str
    nonce: str

    def __post_init__(self):
        if any(type(v) is not str or _IDENT.fullmatch(v) is None for v in
               (self.host_id,self.session_id)):
            raise ValueError("serving host/session must have exact identity")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("invalid serving generation")
        if type(self.nonce) is not str or _NONCE.fullmatch(self.nonce) is None:
            raise ValueError("invalid bounded receiver nonce")


@dataclass(frozen=True)
class ControlledTrialTarget:
    os_source: str
    task_id: str
    model_name: str
    model_lineage: str
    serving_host_id: str
    artifact_sha256: str
    artifact_size_bytes: int
    minimum_free_ram_mib: int
    route_class: str
    intended_endpoint: str

    def __post_init__(self):
        if (type(self.artifact_sha256) is not str or
                _SHA.fullmatch(self.artifact_sha256) is None or
                type(self.artifact_size_bytes) is not int or
                self.artifact_size_bytes < 1 or
                type(self.minimum_free_ram_mib) is not int or
                self.minimum_free_ram_mib < 0):
            raise ValueError("expected artifact and capacity must be explicit")
        for v in (self.task_id,self.serving_host_id):
            if type(v) is not str or _IDENT.fullmatch(v) is None:
                raise ValueError("invalid task or host")


@dataclass(frozen=True)
class ArtifactResourceSample:
    session: ServingSession
    observed_at_epoch_s: int
    sample_id: str
    sha256: str
    size_bytes: int
    cpu_threads: int
    available_ram_mib: int | None

    def __post_init__(self):
        if type(self.session) is not ServingSession:
            raise ValueError("typed serving session required")
        if (type(self.sample_id) is not str or _IDENT.fullmatch(self.sample_id) is None or
                type(self.sha256) is not str or _SHA.fullmatch(self.sha256) is None or
                any(type(v) is not int or v < 0 for v in
                    (self.observed_at_epoch_s,self.size_bytes,self.cpu_threads)) or
                (self.available_ram_mib is not None and
                    (type(self.available_ram_mib) is not int or
                     self.available_ram_mib < 0))):
            raise ValueError("typed read-only sample required")


@dataclass(frozen=True)
class ControlledTrialDecision:
    status: str
    holds: tuple[str,...]
    target_host: str
    source_model_lineage: str
    source_preflight_accepted: bool = False
    model_identity_authenticated: bool = False
    normal_resident_admitted: bool = False
    inference_executed: bool = False
    effect_authority: bool = False
    trained_vera_model_proven: bool = False


def evaluate_controlled_trial(
    target: ControlledTrialTarget,
    before: ServingSession,
    sample: ArtifactResourceSample,
    after: ServingSession,
    artifact_bytes: bytes,
    development: DevelopmentAssessment,
    cognition: dict,
    *, episode: object, now_epoch_s: int, evidence_scope: str,
    fixture_nonce_ledger: set[tuple[str,str]] | None = None,
) -> ControlledTrialDecision:
    """Evaluate a single no-effect preflight fixture; source-only ACCEPT/HOLD.

    In an actual receiver implementation, bytes must be read at the serving
    host from the exact artifact used by the provider, and session readbacks
    before and after must be receiver-owned. This source adapter is not that
    receiver and deliberately NEVER accepts real-host caller claims.
    """
    if (type(target) is not ControlledTrialTarget or
        type(before) is not ServingSession or
        type(sample) is not ArtifactResourceSample or
        type(after) is not ServingSession or
        type(development) is not DevelopmentAssessment or
        type(cognition) is not dict or
        type(now_epoch_s) is not int or now_epoch_s < 0 or
        type(evidence_scope) is not str):
        raise ValueError("typed source-only trial arguments required")
    holds=[]
    if (evidence_scope!="SYNTHETIC_NO_EFFECT_FIXTURE"):
        holds.append("HOST_CAPTURE_NOT_RECEIVER_AUTHENTICATED")
    if (target.os_source!=DEV_OS_HEAD or target.model_name!=DEV_MODEL or
        target.model_lineage!="QWEN_DEVELOPMENT"):
        holds.append("MODEL_LINEAGE_NOT_QUALIFIED")
    if (target.route_class!="COGNITION_ONLY" or
        target.intended_endpoint!="http://127.0.0.1:18766/v1"):
        holds.append("UNRECOGNIZED_CONTROLLED_ROUTE")
    if target.serving_host_id != before.host_id or before!=after or sample.session!=before:
        holds.append("SERVING_SESSION_CHANGED")
    if type(artifact_bytes) is not bytes or len(artifact_bytes)>1024*1024:
        holds.append("SYNTHETIC_ARTIFACT_BYTES_REQUIRED")
    else:
        digest=hashlib.sha256(artifact_bytes).hexdigest()
        if (digest != target.artifact_sha256 or
            digest != sample.sha256 or
            len(artifact_bytes)!=target.artifact_size_bytes or
            len(artifact_bytes)!=sample.size_bytes):
            holds.append("ARTIFACT_BYTES_MISMATCH")
    if now_epoch_s - sample.observed_at_epoch_s not in range(121):
        holds.append("SAMPLE_OUTSIDE_WINDOW")
    if sample.cpu_threads < 1 or (
            sample.available_ram_mib is None or
            sample.available_ram_mib < target.minimum_free_ram_mib):
        holds.append("INSUFFICIENT_HOST_RAM")
    if (development.status!="DEVELOPMENT_METADATA_ONLY" or
            development.model_bytes_authenticated is not False or
            development.normal_resident_admitted is not False or
            development.route_authorized is not False):
        holds.append("UNTRUSTED_DEVELOPMENT_CUT")
    if cognition.get("task_id")!=target.task_id:
        holds.append("COGNITION_TASK_MISMATCH")
    try:
        independently_rebound=bind_participation_development_cut(
            episode,development,expected_task_id=target.task_id)
    except (ValueError,TypeError,KeyError):
        holds.append("COGNITION_RECEIPT_RELABELED")
    else:
        if cognition!=independently_rebound:
            holds.append("COGNITION_RECEIPT_RELABELED")
    if (cognition.get("status")!="HELD_NORMAL_RESIDENT" or
        cognition.get("effect_authority") is not False or
        cognition.get("accepted_as_task_result") is not False or
        cognition.get("cognitive_independence_proven") is not False or
        cognition.get("model_identity_authenticated") is not False or
        cognition.get("development_subject")!=DEV_OS_HEAD or
        type(cognition.get("participants")) is not list or
        len(cognition.get("participants"))!=2 or
        len(set(cognition["participants"]))!=2):
        holds.append("COGNITION_NOT_SOURCE_ONLY")
    # This in-memory cache exercises one-use semantics *only in a synthetic
    # fixture*. A controlled real experiment requires the existing receiver's
    # durable, atomic nonce/task ledger; do not substitute this set.
    if evidence_scope=="SYNTHETIC_NO_EFFECT_FIXTURE":
        if type(fixture_nonce_ledger) is not set:
            holds.append("FIXTURE_NONCE_LEDGER_MISSING")
        elif not holds:
            nonce_key=("nonce",before.host_id,before.nonce)
            sample_key=("sample",before.host_id,sample.sample_id)
            if nonce_key in fixture_nonce_ledger or sample_key in fixture_nonce_ledger:
                holds.append("REPLAYED_FIXTURE_NONCE_OR_SAMPLE")
            else:
                fixture_nonce_ledger.update((nonce_key,sample_key))
    accepted=not holds
    return ControlledTrialDecision(
        "ACCEPT_SYNTHETIC_DEV_PREFLIGHT" if accepted else "HOLD_SOURCE_PREFLIGHT",
        tuple(dict.fromkeys(holds)),target.serving_host_id,target.model_lineage,
        source_preflight_accepted=accepted)

# Source-only adapter to the EXISTING VeraOS cognition_provider schema.
# Important: VeraOS validate_provider checks shape/status, NOT the issuer.
# Therefore this adapter NEVER emits PASS/VERAOS_ROUTE_ELIGIBLE.
_REAL_TRIAL_REQUIREMENTS=(
    "actual_windows_ollama_serving_host_bound_to_receiver_registration",
    "receiver_owned_one_use_nonce_and_generation",
    "same_host_session_bracketing_and_fresh_capacity_sample",
    "actual_loaded_model_blob_manifest_and_byte_hash_verified",
    "task_identity_semantics_dissent_and_fresh_journal_head",
    "operator_scoped_development_trial_authority",
    "installed_veraos_trusted_provider_verification_and_portal_route",
    "durable_single_request_result_and_idempotent_replay",
)


def prepare_resident_admission_handoff(
    target: ControlledTrialTarget,
    decision: ControlledTrialDecision,
) -> dict:
    """Build only the OS *HELD* descriptor and a typed wiring/proof boundary.

    A synthetic green result grants source-wiring TEST eligibility only. The
    host owner must separately implement and verify real evidence at the OS
    boundary; this module cannot issue or bless a live provider qualification.
    """
    if type(target) is not ControlledTrialTarget or type(decision) is not ControlledTrialDecision:
        raise ValueError("typed source target and decision required")
    synthetic=(decision.status=="ACCEPT_SYNTHETIC_DEV_PREFLIGHT" and
               decision.source_preflight_accepted is True and
               decision.holds==() and
               decision.normal_resident_admitted is False and
               decision.model_identity_authenticated is False and
               decision.inference_executed is False and
               decision.effect_authority is False and
               decision.trained_vera_model_proven is False and
               target.model_lineage=="QWEN_DEVELOPMENT" and
               target.model_name==DEV_MODEL and
               target.os_source==DEV_OS_HEAD and
               target.route_class=="COGNITION_ONLY" and
               target.intended_endpoint=="http://127.0.0.1:18766/v1" and
               decision.target_host==target.serving_host_id and
               decision.source_model_lineage==target.model_lineage)
    return {
        "schema":"VERAOS_CONTROLLED_RESIDENT_ADMISSION_HANDOFF_V1",
        "readiness":("SYNTHETIC_WIRING_ELIGIBLE" if synthetic
                     else "HOLD_UNVERIFIED_REAL_ADMISSION"),
        "synthetic_wiring_eligible":synthetic,
        "real_resident_admitted":False,
        "provider_authenticated":False,
        "real_admission_requires":list(_REAL_TRIAL_REQUIREMENTS),
        "serving_host":target.serving_host_id,
        "task_id":target.task_id,
        "model_lineage":target.model_lineage,
        "inventory_digest_is_artifact_hash":False,
        "suite_and_issuer_independently_verified":False,
        "provider":{
            "schema":"VERA_OS_COGNITION_PROVIDER_V1",
            "provider_id":"dev:virtio-qwen4b",
            "provider_kind":"LOCAL_MODEL",
            "capabilities":["text"],
            "transport":{
                "kind":"OPENAI_COMPATIBLE_HTTP",
                "attachment":target.intended_endpoint,
            },
            # These are structural placeholders. The subject is the source
            # target digest (synthetic for fixture); issuer is schema-required
            # source text, NOT an authenticated trusted signer.
            "qualification":{
                "status":"HOLD",
                "issuer":"thebrazenbeard/vera-os",
                "subject_sha256":target.artifact_sha256,
                "suite_sha256":"0"*64,
            },
            "authority":{
                "identity_authority":False,
                "canonical_memory_write_authority":False,
                "protected_effect_authority":False,
                "browser_authority":False,
            },
            "admission":{
                "status":"HELD",
                "route_class":"COGNITION_ONLY",
                "selector_owner":"thebrazenbeard/portal",
            },
        },
    }


def validate_resident_admission_handoff(
    target: ControlledTrialTarget,
    decision: ControlledTrialDecision,
    candidate: object,
) -> bool:
    """Fail closed for changed source handoff. Never attest a runtime PASS.

    This exact comparison guards corruption between source adapter and OS
    fixture wiring. It cannot authenticate trusted runtime observations.
    """
    if type(candidate) is not dict or candidate!=prepare_resident_admission_handoff(target,decision):
        raise ValueError("unrecognized, forged, or escalated resident admission source")
    return True
