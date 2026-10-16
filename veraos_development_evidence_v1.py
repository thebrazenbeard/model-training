"""VeraOS #32 source-only metadata contract, NEVER model admission."""
from dataclasses import dataclass
import re
import hashlib
import json
DEV_OS_HEAD="5ddac1459c3a7b06f251721a14048fc26e60bda0"
DEV_MODEL="qwen3:4b-instruct"
DEV_OLLAMA_REPORTED_DIGEST="0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
DEV_MODEL_BYTES=2497293803
_HEX64=re.compile(r"^[0-9a-f]{64}$")
_NONCE=re.compile(r"^[0-9a-f]{32}$")
_ID=re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")

@dataclass(frozen=True)
class DevelopmentInventory:
    source_head: str
    model: str
    reported_ollama_digest: str
    reported_ollama_size_bytes: int
    quantization: str

    def __post_init__(self):
        if (type(self.source_head) is not str or
            not re.fullmatch(r"[0-9a-f]{40}",self.source_head) or
            type(self.model) is not str or not self.model or
            type(self.reported_ollama_digest) is not str or
            _HEX64.fullmatch(self.reported_ollama_digest) is None or
            type(self.reported_ollama_size_bytes) is not int or
            not 0 < self.reported_ollama_size_bytes < 4_000_000_000 or
            type(self.quantization) is not str or not self.quantization):
            raise ValueError("invalid reported development model inventory")

@dataclass(frozen=True)
class DevelopmentCut:
    registry_node_id: str
    registry_generation: int
    registry_session: str
    registry_nonce: str
    telemetry_node_id: str
    telemetry_generation: int
    telemetry_session: str
    telemetry_nonce: str
    inventory_node_id: str
    inventory_generation: int
    inventory_session: str
    inventory_nonce: str
    artifact_node_id: str
    artifact_generation: int
    artifact_session: str
    artifact_nonce: str
    observed_at_epoch_s: int
    cpu_threads: int
    ram_available_mib: int | None
    gpu_free_vram_mib: int | None
    artifact_bytes_sha256: str | None
    descriptor_status: str
    route: str

    def __post_init__(self):
        for prefix in ("registry","telemetry","inventory","artifact"):
            if (type(getattr(self,prefix+"_node_id")) is not str or
                _ID.fullmatch(getattr(self,prefix+"_node_id")) is None or
                type(getattr(self,prefix+"_generation")) is not int or
                getattr(self,prefix+"_generation") < 1 or
                type(getattr(self,prefix+"_session")) is not str or
                _ID.fullmatch(getattr(self,prefix+"_session")) is None or
                type(getattr(self,prefix+"_nonce")) is not str or
                _NONCE.fullmatch(getattr(self,prefix+"_nonce")) is None):
                raise ValueError("invalid node generation/session/nonce metadata")
        if (type(self.observed_at_epoch_s) is not int or self.observed_at_epoch_s < 0 or
            type(self.cpu_threads) is not int or self.cpu_threads < 1):
            raise ValueError("invalid development resource timestamp or cpu")
        for val in (self.ram_available_mib,self.gpu_free_vram_mib):
            if val is not None and (type(val) is not int or val < 0):
                raise ValueError("invalid resource estimate")
        if self.artifact_bytes_sha256 is not None and (
            type(self.artifact_bytes_sha256) is not str or
            _HEX64.fullmatch(self.artifact_bytes_sha256) is None):
            raise ValueError("artifact readback hash claim must be lowercase sha256")
        if (type(self.descriptor_status) is not str or
            type(self.route) is not str):
            raise ValueError("invalid route descriptor")

@dataclass(frozen=True)
class DevelopmentAssessment:
    status: str
    inventory_provenance: str
    holds: tuple[str,...]
    model_bytes_authenticated: bool = False
    normal_resident_admitted: bool = False
    route_authorized: bool = False
    identity_effect: str = "NONE"

def evaluate_development_cut(cut: DevelopmentCut, inventory: DevelopmentInventory, *,
    expected_node_id: str, expected_generation: int,
    expected_session: str, expected_nonce: str, now_epoch_s: int) -> DevelopmentAssessment:
    """Structural join of PR32 developer claims, NOT a trusted attestation.

    The PR32 Ollama digest is a reported inventory identity, never promoted
    into the raw bytes SHA of a GGUF artifact or a signed provider admission.
    No callback or branch head can grant a normal-resident route here.
    """
    if type(cut) is not DevelopmentCut or type(inventory) is not DevelopmentInventory:
        raise ValueError("typed source observations required")
    if (type(expected_node_id) is not str or _ID.fullmatch(expected_node_id) is None or
        type(expected_generation) is not int or expected_generation < 1 or
        type(expected_session) is not str or _ID.fullmatch(expected_session) is None or
        type(expected_nonce) is not str or _NONCE.fullmatch(expected_nonce) is None or
        type(now_epoch_s) is not int or now_epoch_s < 0):
        raise ValueError("invalid receiver expectations")
    hard=[]
    expected=(expected_node_id,expected_generation,expected_session,expected_nonce)
    for prefix in ("registry","telemetry","inventory","artifact"):
        actual=tuple(getattr(cut,prefix+"_"+suffix)
                     for suffix in ("node_id","generation","session","nonce"))
        if actual!=expected:
            hard.append("JOINT_SESSION_PROVENANCE_MISMATCH")
            break
    if now_epoch_s-cut.observed_at_epoch_s not in range(121):
        hard.append("OBSERVATION_OUTSIDE_WINDOW")
    if (inventory.source_head!=DEV_OS_HEAD or inventory.model!=DEV_MODEL or
        inventory.reported_ollama_digest!=DEV_OLLAMA_REPORTED_DIGEST or
        inventory.reported_ollama_size_bytes!=DEV_MODEL_BYTES or
        inventory.quantization!="Q4_K_M"):
        hard.append("PINNED_PR32_MODEL_INVENTORY_MISMATCH")
    if cut.route!="DIAGNOSTIC_QEMU_ONLY":
        hard.append("DIAGNOSTIC_ROUTE_REQUIRED")
    if cut.descriptor_status!="PASS":
        hard.append("DIAGNOSTIC_TEST_DESCRIPTOR_NOT_PASS")
    holds=list(dict.fromkeys(hard))
    holds.append("DIAGNOSTIC_PASS_IS_SELF_DECLARED")
    if cut.artifact_bytes_sha256 is None:
        holds.append("MODEL_ARTIFACT_BYTES_UNVERIFIED")
    else:
        holds.append("ARTIFACT_DIGEST_SELF_REPORTED_NOT_ATTESTED")
    if cut.gpu_free_vram_mib is None:
        holds.append("GPU_TELEMETRY_UNOBSERVED")
    return DevelopmentAssessment(
        "HELD_SOURCE_PROVENANCE" if hard else "DEVELOPMENT_METADATA_ONLY",
        "OLLAMA_REPORTED_NOT_ARTIFACT_BYTES",tuple(holds))

def bind_participation_development_cut(episode, cut: DevelopmentAssessment, *,
                                       expected_task_id: str) -> dict:
    """Link an existing Training receipt to the *reported* VeraOS dev model.

    Receipt labels and model inventory bytes are source observations only.
    This binds no VeraOS provider, protected effect, normal resident, or mind.
    """
    data = episode.as_dict() if hasattr(episode,"as_dict") else episode
    if type(data) is not dict or type(cut) is not DevelopmentAssessment:
        raise ValueError("typed source cognition and development assessment required")
    if (type(expected_task_id) is not str or not expected_task_id or
        data.get("schema")!="VERA_PARTICIPATING_COGNITION_SOURCE_V1" or
        data.get("task_id")!=expected_task_id):
        raise ValueError("cognition task source mismatch")
    if (any(data.get(k)!="NONE" for k in (
            "identity_effect","shared_memory_effect","authority_effect",
            "executed_effect")) or
        data.get("model_independence")!="NOT_ESTABLISHED" or
        data.get("cognitive_gain")!="NOT_ESTABLISHED"):
        raise ValueError("unverified cognition or effects cannot join VeraOS")
    people=data.get("contributions")
    if type(people) is not list or len(people)!=2:
        raise ValueError("two attributed source participants required")
    agents=[]
    for c in people:
        if type(c) is not dict or c.get("task_id")!=expected_task_id:
            raise ValueError("participant task mismatch")
        p=c.get("participant")
        if type(p) is not dict or type(p.get("agent_id")) is not str or not p["agent_id"]:
            raise ValueError("missing logical agent identity")
        agents.append(p["agent_id"])
    if len(set(agents))!=2:
        raise ValueError("duplicate logical agent")
    if (cut.model_bytes_authenticated is not False or
        cut.normal_resident_admitted is not False or
        cut.route_authorized is not False or cut.identity_effect!="NONE"):
        raise ValueError("model-development assessment escalated authority")
    try:
        source_bytes=json.dumps(data,sort_keys=True,separators=(",",":"),
                                ensure_ascii=False,allow_nan=False).encode("utf8")
    except (TypeError,ValueError) as exc:
        raise ValueError("uncanonicalizable source cognition receipt") from exc
    return {
        "status":"HELD_NORMAL_RESIDENT",
        "task_id":expected_task_id,
        "participants":agents,
        "source_cognition_sha256":hashlib.sha256(source_bytes).hexdigest(),
        "development_subject":DEV_OS_HEAD,
        "reported_model":DEV_MODEL,
        "reported_ollama_digest":DEV_OLLAMA_REPORTED_DIGEST,
        "inventory_provenance":cut.inventory_provenance,
        "development_cut_status":cut.status,
        "model_identity_authenticated":False,
        "cognitive_independence_proven":False,
        "accepted_as_task_result":False,
        "effect_authority":False,
    }
