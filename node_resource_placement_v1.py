"""Read-only node/resource fit for L3 Training + VN1 paired review.

No model deployment, authentication, capacity reservation, route selection or
authority. Resource identity and cognitive participant identity stay separate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import re

_IDENT=re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_SHA=re.compile(r"^[0-9a-f]{64}$")
_KINDS={"CPU_COORDINATION","CPU_MODEL","GPU_MODEL"}


class ResourceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NodeResourceObservation:
    node_id: str
    registry_generation: int
    registry_session: str
    registry_present: bool
    observed_at_epoch_s: int
    cpu_logical_threads: int
    ram_available_mib: Optional[int]
    gpu_vram_free_mib: Optional[int]
    gpu_vram_total_mib: Optional[int]

    def __post_init__(self):
        if any(type(x) is not str or not _IDENT.fullmatch(x) for x in
               (self.node_id,self.registry_session)):
            raise ResourceError("Invalid node/session identity")
        if (type(self.registry_generation) is not int or self.registry_generation < 1 or
            type(self.registry_present) is not bool or
            type(self.observed_at_epoch_s) is not int or
            self.observed_at_epoch_s < 0 or
            type(self.cpu_logical_threads) is not int or self.cpu_logical_threads < 1):
            raise ResourceError("Invalid registry/CPU observation")
        for item in (self.ram_available_mib,self.gpu_vram_free_mib,
                     self.gpu_vram_total_mib):
            if item is not None and (type(item) is not int or item < 0):
                raise ResourceError("Missing resource must be None, not a guessed value")
        if (self.gpu_vram_free_mib is not None and
            self.gpu_vram_total_mib is not None and
            self.gpu_vram_free_mib > self.gpu_vram_total_mib):
            raise ResourceError("GPU free exceeds observed total")


@dataclass(frozen=True, slots=True)
class CognitiveWorkload:
    task_id: str
    kind: str
    min_cpu_threads: int
    min_free_ram_mib: int
    min_free_vram_mib: int = 0
    model_artifact_sha256: Optional[str] = None

    def __post_init__(self):
        if type(self.task_id) is not str or not _IDENT.fullmatch(self.task_id):
            raise ResourceError("Invalid task ID")
        if self.kind not in _KINDS:
            raise ResourceError("Unknown resource workload kind")
        for n in (self.min_cpu_threads,self.min_free_ram_mib,self.min_free_vram_mib):
            if type(n) is not int or n < 0:
                raise ResourceError("Invalid capacity floor")
        if self.min_cpu_threads < 1:
            raise ResourceError("CPU floor required")
        if self.kind == "CPU_COORDINATION":
            if self.model_artifact_sha256 is not None or self.min_free_vram_mib:
                raise ResourceError("CPU coordinator is not a neural model")
        else:
            if type(self.model_artifact_sha256) is not str or not _SHA.fullmatch(self.model_artifact_sha256):
                raise ResourceError("Model-bearing tasks require exact artifact ID")
            if self.kind=="GPU_MODEL" and self.min_free_vram_mib < 1:
                raise ResourceError("GPU inference requires explicit free VRAM floor")


@dataclass(frozen=True, slots=True)
class ResourceFitReceipt:
    node_id: str
    task_id: str
    status: str
    reasons: tuple[str,...]
    registry_generation: int
    registry_session: str
    model_qualified: bool = False
    route_authorized: bool = False
    capacity_reserved: bool = False
    physical_identity_authenticated: bool = False


def advise_resource_fit(
    observed: NodeResourceObservation, task: CognitiveWorkload, *,
    expected_node_id: str, expected_generation: int, expected_session: str,
    now_epoch_s: int, freshness_seconds: int = 120,
    verify_registry_observation=None, verify_model_artifact=None,
    verify_resource_observation=None, verify_evidence_binding=None,
) -> ResourceFitReceipt:
    """Advisory resource fit only; no authentication, permit or reservation.

    Independent registry/session, model-artifact and resource measurements are
    separate receiver-owned verifier seams. Boolean fixtures do not attest
    their origin or imply actual capacity.
    """
    if not isinstance(observed,NodeResourceObservation) or not isinstance(task,CognitiveWorkload):
        raise ResourceError("Typed observation and workload required")
    if (type(expected_node_id) is not str or _IDENT.fullmatch(expected_node_id) is None or
        type(expected_session) is not str or _IDENT.fullmatch(expected_session) is None or
        type(expected_generation) is not int or expected_generation < 1):
        raise ResourceError("Invalid expected node/session reference")
    if (type(now_epoch_s) is not int or now_epoch_s < 0 or
        type(freshness_seconds) is not int or not 1<=freshness_seconds<=300):
        raise ResourceError("Invalid readback/freshness bound")
    reasons=[]
    if verify_registry_observation is None:
        reasons.append("REGISTRY_EVIDENCE_UNVERIFIED")
    else:
        try:
            verified = verify_registry_observation(observed, expected_node_id, expected_generation, expected_session)
        except Exception:
            verified = False
        if verified is not True:
            reasons.append("REGISTRY_EVIDENCE_REJECTED")
    if task.model_artifact_sha256 is not None:
        if verify_model_artifact is None:
            reasons.append("MODEL_ARTIFACT_UNVERIFIED")
        else:
            try:
                verified_model = verify_model_artifact(observed, task.model_artifact_sha256)
            except Exception:
                verified_model = False
            if verified_model is not True:
                reasons.append("MODEL_ARTIFACT_REJECTED")
    # Registry liveness is separate from independently measured resource
    # capacity. The resource verifier receives the exact workload budget.
    if verify_resource_observation is None:
        reasons.append("RESOURCE_MEASUREMENT_UNVERIFIED")
    else:
        try:
            measured = verify_resource_observation(observed,task)
        except Exception:
            measured = False
        if measured is not True:
            reasons.append("RESOURCE_MEASUREMENT_REJECTED")
    # Three independently positive callbacks do not establish a single
    # generation-scoped resource/artifact snapshot. Re-read all claims together
    # under the receiver-owned session boundary before advisory fit.
    if verify_evidence_binding is None:
        reasons.append("EVIDENCE_BINDING_UNVERIFIED")
    else:
        try:
            bound = verify_evidence_binding(observed,task,expected_node_id,
                                            expected_generation,expected_session,now_epoch_s)
        except Exception:
            bound = False
        if bound is not True:
            reasons.append("EVIDENCE_BINDING_REJECTED")
    if (observed.node_id!=expected_node_id or
        observed.registry_generation!=expected_generation or
        observed.registry_session!=expected_session):
        reasons.append("NODE_SESSION_MISMATCH")
    if not observed.registry_present:
        reasons.append("REGISTRY_NOT_PRESENT")
    age=now_epoch_s-observed.observed_at_epoch_s
    if age < 0 or age > freshness_seconds:
        reasons.append("OBSERVATION_STALE_OR_FUTURE")
    if observed.cpu_logical_threads<task.min_cpu_threads:
        reasons.append("INSUFFICIENT_CPU")
    if observed.ram_available_mib is None:
        reasons.append("AVAILABLE_RAM_UNKNOWN")
    elif observed.ram_available_mib<task.min_free_ram_mib:
        reasons.append("INSUFFICIENT_AVAILABLE_RAM")
    if task.kind=="GPU_MODEL":
        if observed.gpu_vram_free_mib is None or observed.gpu_vram_total_mib is None:
            reasons.append("GPU_TELEMETRY_UNKNOWN")
        elif observed.gpu_vram_free_mib<task.min_free_vram_mib:
            reasons.append("INSUFFICIENT_FREE_VRAM")
    return ResourceFitReceipt(
        node_id=observed.node_id,task_id=task.task_id,
        status=("RESOURCE_CLAIM_ONLY" if (
                    "REGISTRY_EVIDENCE_UNVERIFIED" in reasons and
                    set(reasons).issubset({
                        "REGISTRY_EVIDENCE_UNVERIFIED",
                        "RESOURCE_MEASUREMENT_UNVERIFIED",
                        "EVIDENCE_BINDING_UNVERIFIED"})
                ) else "HELD" if reasons else "RESOURCE_FIT_ONLY"),
        reasons=tuple(reasons),
        registry_generation=observed.registry_generation,
        registry_session=observed.registry_session,
    )
