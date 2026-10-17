"""VeraOS-composable resident cognition verification before any model socket.

Uses the ACTUAL QualifiedVeraRuntime outbound trust and task ledger.
No network model adapter, Python callback token, installation or provider
permission is created here. Source-trial leases cannot enter native_cognition.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from resident_verifier_port_v1 import (
    ResidentModelRequest, SignedHostEvidence, canonical_request,
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


class ResidentSourceHold(PermissionError):
    """No OS model operation is permitted by this result."""


@dataclass(frozen=True, slots=True)
class ScopedResidentTurn:
    model_request: ResidentModelRequest
    request_id: str
    task_text: str
    source_digest: str
    expected_task_head: str
    actions: tuple[str, ...] = ("COGNITION_TURN",)

    def __post_init__(self):
        if (type(self.model_request) is not ResidentModelRequest or
            type(self.request_id) is not str or _ID.fullmatch(self.request_id) is None or
            type(self.task_text) is not str or not self.task_text.strip() or
            self.task_text != self.task_text.strip() or len(self.task_text)>2048 or
            type(self.source_digest) is not str or _SHA.fullmatch(self.source_digest) is None or
            type(self.expected_task_head) is not str or _SHA.fullmatch(self.expected_task_head) is None or
            type(self.actions) is not tuple or self.actions != ("COGNITION_TURN",) or
            self.source_digest != self.model_request.task_digest):
            raise ValueError("invalid exact cognition-only task provenance")


def canonical_turn(turn: ScopedResidentTurn) -> bytes:
    if type(turn) is not ScopedResidentTurn:
        raise ValueError("typed exact resident turn required")
    return json.dumps({
        "schema":"VERAOS_RESIDENT_BOUND_TASK_AND_HOST_V1",
        "request_sha256":hashlib.sha256(canonical_request(turn.model_request)).hexdigest(),
        "request_id":turn.request_id,
        "task_id":turn.model_request.task_id,
        "task_text":turn.task_text,
        "source_digest":turn.source_digest,
        "expected_task_head":turn.expected_task_head,
        "allowed_actions":["COGNITION_TURN"],
        "tools":[],"browser_authority":False,
        "canonical_memory_authority":False,
        "identity_authority":False,
        "protected_effect_authority":False,
    },sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf8")


@dataclass(frozen=True, slots=True)
class ScopedSourceLease:
    status: str
    task_id: str
    request_id: str
    provider_id: str
    model_id: str
    endpoint: str
    serving_host: str
    session_id: str
    generation: int
    request_sha256: str
    issued_checkpoint_head: str
    expires_at_epoch_s: int
    real_model_admitted: bool = False
    network_dispatch_authorized: bool = False
    protected_effect_authority: bool = False
    canonical_task_completed: bool = False


class QualifiedRuntimeModelGate:
    """OS composition seam: caller does not supply trust or verifier per turn.

    A trusted OS image must provide host_verifier with *actual* provisioned
    signature verification and process-loaded model observation at startup.
    A test-injected Python verifier may demonstrate only source eligibility.
    It is never a host-authentication root without OS installation evidence.
    """

    def __init__(self, runtime: Any, host_verifier: Any):
        from vera_core import QualifiedVeraRuntime
        if type(runtime) is not QualifiedVeraRuntime:
            raise TypeError("exact host-composed QualifiedVeraRuntime required")
        if not all(callable(getattr(host_verifier, method, None))
                   for method in ("verify_signature","reobserve","consume_once")):
            raise TypeError("OS provisioned verifier functions required")
        self._runtime=runtime
        self._verifier=host_verifier

    def reserve_source_trial(self, turn: ScopedResidentTurn, signed: SignedHostEvidence,
                             *, now_epoch_s: int) -> ScopedSourceLease:
        """Check exact task/provider/currentness and consume a durable task CAS.

        NEVER calls the native HTTP adapter; never returns a live-model grant.
        """
        if type(turn) is not ScopedResidentTurn or type(signed) is not SignedHostEvidence:
            raise ResidentSourceHold("MALFORMED_SOURCE_TURN")
        request=turn.model_request
        if (type(now_epoch_s) is not int or now_epoch_s < request.observed_at_epoch_s or
            now_epoch_s>request.expires_at_epoch_s or
            now_epoch_s-request.observed_at_epoch_s>120):
            raise ResidentSourceHold("STALE_HOST_OBSERVATION")
        ledger=self._runtime.tasks
        try:
            state=ledger.read(request.task_id)
            if (state.closed or state.active_dependencies or
                state.unresolved_correction_ids or state.active_delegations or
                state.packet.subject!=turn.task_text or
                state.opened_lifecycle_evidence_digest!=turn.source_digest or
                state.journal_head_digest!=turn.expected_task_head):
                raise ResidentSourceHold("TASK_CHANGED_OR_CANCELLED")
            current=self._runtime.outbound_trust.assert_current(
                authority_id=signed.issuer_id,role="PROVIDER",
                provider_id=request.provider_id,
                key_id=signed.key_id,key_digest=signed.key_digest)
            if (current.authority_id!=signed.issuer_id or
                current.role!="PROVIDER" or
                current.provider_id!=request.provider_id or
                current.key_id!=signed.key_id or
                current.key_digest!=signed.key_digest):
                raise ResidentSourceHold("PROVIDER_AUTHORITY_MISMATCH")
            if self._verifier.verify_signature(
                message=canonical_turn(turn),signature=signed.signature,
                key_id=signed.key_id) is not True:
                raise ResidentSourceHold("HOST_SIGNATURE_INVALID")
            observed=self._verifier.reobserve(request)
            if type(observed) is not dict:
                raise ResidentSourceHold("NO_HOST_PROCESS_ATTESTATION")
            expected={
                "serving_host":request.serving_host,
                "generation":request.generation,
                "session_id":request.session_id,
                "nonce":request.nonce,
                "model_id":request.model_id,
                "artifact_sha256":request.artifact_sha256,
                "manifest_sha256":request.manifest_sha256,
                "artifact_bytes":request.artifact_bytes,
                "endpoint":request.endpoint,
                "task_id":request.task_id,
                "task_digest":request.task_digest,
                "observed_at_epoch_s":request.observed_at_epoch_s,
            }
            if any(observed.get(key)!=value for key,value in expected.items()):
                raise ResidentSourceHold("HOST_MODEL_OR_SESSION_SUBSTITUTED")
            if (type(observed.get("loaded_process_pid")) is not int or
                observed["loaded_process_pid"] <= 0 or
                observed.get("model_source")!="SERVING_PROCESS_ATTESTATION" or
                type(observed.get("ram_available_mib")) is not int or
                observed["ram_available_mib"]<request.minimum_ram_mib):
                raise ResidentSourceHold("STORED_BYTES_NOT_LOADED_PROCESS_PROOF")
            if self._verifier.consume_once(request) is not True:
                raise ResidentSourceHold("NONCE_ALREADY_CONSUMED")
            # The canonical task ledger's OWN atomic expected_task_head
            # prevents a same-task correction/dependency/cancellation race.
            digest=hashlib.sha256(canonical_turn(turn)).hexdigest()
            latest=ledger.checkpoint(
                request.task_id, "model-source-trial-"+request.nonce,
                completed_evidence=("synthetic host evidence reviewed; no model executed",),
                blockers=("no installed VeraOS trusted provider admission",),
                protected_effects_still_gated=("COGNITION_TURN",),
                next_frontier="OS-owned installed host key and loaded model attestation",
                lifecycle_evidence_digest=digest,
                expected_task_head=turn.expected_task_head)
            # This lease intentionally cannot be passed as admission to a
            # network-backed native route. PORTAL must install/authorize a
            # distinct resident-owned VERIFIED development grant.
            return ScopedSourceLease(
                "SCOPED_SOURCE_TRIAL_RESERVED",request.task_id,turn.request_id,
                request.provider_id,request.model_id,request.endpoint,
                request.serving_host,request.session_id,request.generation,
                digest,latest.journal_head_digest,request.expires_at_epoch_s)
        except ResidentSourceHold:
            raise
        except (ValueError,TypeError,KeyError,AttributeError,PermissionError) as exc:
            raise ResidentSourceHold("TRUST_TASK_OR_NONCE_VERIFICATION_FAILED") from exc

def make_held_resident_conversation_handler(
    gate: QualifiedRuntimeModelGate,
    turn: ScopedResidentTurn,
    signed: SignedHostEvidence,
    *,
    now_epoch_s: int,
):
    """Direct PR37 conversation_handler(task,request_id), NEVER a model adapter.

    This source-only OS composition hook has no provider URL callback, model
    factory, socket, tool registry, or synthetic qualification PASS. It never
    calls native_cognition. The only possible HTTP outcome is BLOCKED.
    """
    if (type(gate) is not QualifiedRuntimeModelGate or
        type(turn) is not ScopedResidentTurn or
        type(signed) is not SignedHostEvidence):
        raise TypeError("exact source-only resident composition required")

    def handle(task: str, request_id: str) -> dict:
        held={"status":"BLOCKED","run_id":None,"final_text":None}
        if (type(task) is not str or type(request_id) is not str or
            task!=turn.task_text or request_id!=turn.request_id):
            return held
        try:
            decision=gate.reserve_source_trial(
                turn,signed,now_epoch_s=now_epoch_s)
        except ResidentSourceHold:
            return held
        if decision.network_dispatch_authorized or decision.real_model_admitted:
            raise ResidentSourceHold("SOURCE_ONLY_MUST_NEVER_DISPATCH")
        return held

    return handle
