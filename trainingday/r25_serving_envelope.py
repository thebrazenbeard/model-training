"""R25 synthetic training-to-serving custody and rollback *candidate* gate.

No model loading, filesystem paths, CUDA, quantizer invocation, service calls,
private evaluation data or deployment APIs are used.  Every decision refuses
real deployment. Caller-supplied SHA fields establish declaration consistency,
not physical artifact identity or an independent fresh-process observation.
"""

from __future__ import annotations

import hashlib
import json
import re


SCHEMA = "VERA_TRAININGDAY_R25_SERVING_CANDIDATE_V1"
MODEL_ID = "Qwen/Qwen3.5-4B"
# The pinned Qwen3.5-4B config declares this architecture. The training
# runners' distinct Qwen3_5ForCausalLM loader request is NOT runtime verified.
MODEL_ARCH = "Qwen3_5ForConditionalGeneration"
SOURCE_FORMAT = "NF4_DOUBLE_BF16"
TARGET_FORMAT = "GGUF_Q5_K_S"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_REV = re.compile(r"[0-9a-f]{40}\Z")
_TENSOR_NAME = re.compile(r"[A-Za-z0-9_.]{1,160}\Z")
_LORA_TENSOR = re.compile(
    r"(?P<module>[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)"
    r"\.lora_(?P<family>[AB])(?P<adapter>\.[A-Za-z0-9_]+)?\.weight\Z"
)
# Synthetic host-metadata admission ceiling only; NOT measured GPU/WDDM fit.
_MAX_DECLARED_FP32_ADAPTER_BYTES = 64 * 2**20
_BOUND_FIELDS = (
    "base_model_id", "base_architecture", "base_revision",
    "base_binding_sha256", "tokenizer_binding_sha256",
    "corpus_manifest_sha256", "adapter_bytes_sha256",
    "adapter_tensor_manifest_sha256",
)
_HASH_FIELDS = _BOUND_FIELDS[3:]
_QUANT_FIELDS = (
    "source_format", "target_format", "source_base_binding_sha256",
    "source_tokenizer_binding_sha256", "source_adapter_sha256",
    "converter_binary_sha256", "converter_args_sha256",
    "target_model_sha256", "transformation_receipt_sha256",
)
_RELOAD_FIELDS = (
    "observed_base_binding_sha256", "observed_tokenizer_binding_sha256",
    "observed_adapter_bytes_sha256", "observed_tensor_manifest_sha256",
    "fresh_process_nonce",
)


def _sha(value: object) -> bool:
    return type(value) is str and _SHA.fullmatch(value) is not None


def _plain_string_keyed_dict(value: object) -> bool:
    """Require exact containers and keys before equality or canonical hashing.

    A dict may contain a str subclass whose __hash__/__eq__ aliases a required
    field while JSON serialization preserves the attacker's different text.
    """
    return type(value) is dict and all(type(key) is str for key in value)


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("ascii")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def tensor_manifest_digest(tensors: object) -> str:
    """Validate and hash declared LoRA *metadata*, never tensor bytes."""
    if type(tensors) is not list or not 2 <= len(tensors) <= 256:
        raise ValueError("tensor manifest must contain 2–256 LoRA tensors")
    names: set[str] = set()
    paired: dict[tuple[str, str], dict[str, list[int]]] = {}
    total_declared_bytes = 0
    for row in tensors:
        if not _plain_string_keyed_dict(row) or set(row) != {"name", "shape", "dtype"}:
            raise ValueError("tensor metadata schema invalid")
        name, shape, dtype = row["name"], row["shape"], row["dtype"]
        if (type(name) is not str or not _TENSOR_NAME.fullmatch(name)
                or name in names):
            raise ValueError("LoRA tensor name missing, malformed or duplicated")
        names.add(name)
        match = _LORA_TENSOR.fullmatch(name)
        if match is None:
            raise ValueError("LoRA A/B tensor module identity malformed")
        if (type(shape) is not list or len(shape) != 2
                or any(type(dim) is not int or not 1 <= dim <= 65536 for dim in shape)):
            raise ValueError("LoRA tensor matrix shape invalid")
        if type(dtype) is not str or dtype != "float32":
            raise ValueError("unqualified LoRA tensor dtype")
        total_declared_bytes += shape[0] * shape[1] * 4
        if total_declared_bytes > _MAX_DECLARED_FP32_ADAPTER_BYTES:
            raise ValueError("declared FP32 adapter bytes exceed synthetic admission ceiling")
        key = (match.group("module"), match.group("adapter") or "")
        members = paired.setdefault(key, {})
        family = match.group("family")
        if family in members:
            raise ValueError("duplicate LoRA A/B module family")
        members[family] = shape
    for module, members in paired.items():
        if set(members) != {"A", "B"}:
            raise ValueError("unpaired LoRA A/B module tensors")
        if members["A"][0] != members["B"][1]:
            raise ValueError("incompatible LoRA A/B matrix rank")
    return _digest(tensors)


def assess_serving_candidate(pinned: object, candidate: object) -> dict:
    """Bound declared source/quantizer identity; never authorize deployment.

    All field hashes may have been selected by the same caller. Physical
    quantization and reload are not proven even if a synthetic claim matches.
    """
    result = {
        "schema": SCHEMA, "status": "HOLD", "reason": "INVALID_ENVELOPE",
        "candidate_sha256": None, "source_bindings_match": False,
        "quantizer_transform_physically_verified": False,
        "fresh_process_reference_parity_verified": False,
        "runtime_loader_verified": False,
        "claimed_reload_fields_match": False,
        "release_rollback_tested_on_host": False,
        "authorizes_deployment": False, "authorizes_model_promotion": False,
    }
    if (not _plain_string_keyed_dict(pinned) or set(pinned) != set(_BOUND_FIELDS)
            or not _plain_string_keyed_dict(candidate)
            or set(candidate) != set(_BOUND_FIELDS) | {
                "tensor_manifest", "quantizer", "fresh_process_reload"}):
        return result
    if (type(pinned.get("base_model_id")) is not str
            or pinned.get("base_model_id") != MODEL_ID
            or type(pinned.get("base_architecture")) is not str
            or pinned.get("base_architecture") != MODEL_ARCH
            or type(pinned.get("base_revision")) is not str
            or _REV.fullmatch(pinned["base_revision"]) is None
            or any(not _sha(pinned.get(k)) for k in _HASH_FIELDS)):
        result["reason"] = "UNQUALIFIED_BASE_OR_PIN"
        return result
    if any(type(candidate[k]) is not str or candidate[k] != pinned[k]
           for k in _BOUND_FIELDS):
        result["reason"] = "PINNED_SOURCE_MISMATCH"
        return result
    try:
        manifest_sha = tensor_manifest_digest(candidate["tensor_manifest"])
    except (ValueError, TypeError, OverflowError):
        result["reason"] = "MALFORMED_TENSOR_MANIFEST"
        return result
    if manifest_sha != pinned["adapter_tensor_manifest_sha256"]:
        result["reason"] = "ADAPTER_TENSOR_MANIFEST_MISMATCH"
        return result
    quantizer = candidate["quantizer"]
    if not _plain_string_keyed_dict(quantizer) or set(quantizer) != set(_QUANT_FIELDS):
        result["reason"] = "MALFORMED_QUANTIZER_TRANSFORM"
        return result
    if (type(quantizer["source_format"]) is not str
            or type(quantizer["target_format"]) is not str
            or any(not _sha(quantizer[k]) for k in _QUANT_FIELDS[2:5])):
        result["reason"] = "MALFORMED_QUANTIZER_TRANSFORM"
        return result
    if (quantizer["source_format"] != SOURCE_FORMAT
            or quantizer["target_format"] != TARGET_FORMAT):
        result["reason"] = "UNQUALIFIED_QUANTIZATION_FORMAT"
        return result
    if (quantizer["source_base_binding_sha256"] != pinned["base_binding_sha256"]
            or quantizer["source_tokenizer_binding_sha256"] != pinned["tokenizer_binding_sha256"]
            or quantizer["source_adapter_sha256"] != pinned["adapter_bytes_sha256"]):
        result["reason"] = "QUANTIZATION_SOURCE_MISMATCH"
        return result
    if any(not _sha(quantizer[k]) for k in _QUANT_FIELDS[5:]):
        result["reason"] = "MALFORMED_QUANTIZATION_DIGEST"
        return result
    result["source_bindings_match"] = True
    reload_claim = candidate["fresh_process_reload"]
    if reload_claim is not None:
        if (not _plain_string_keyed_dict(reload_claim) or set(reload_claim) != set(_RELOAD_FIELDS)
                or any(not _sha(reload_claim[k]) for k in _RELOAD_FIELDS[:-1])
                or type(reload_claim["fresh_process_nonce"]) is not str
                or not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", reload_claim["fresh_process_nonce"])):
            result["reason"] = "MALFORMED_RELOAD_CLAIM"
            return result
        if (reload_claim["observed_base_binding_sha256"] != pinned["base_binding_sha256"]
                or reload_claim["observed_tokenizer_binding_sha256"] != pinned["tokenizer_binding_sha256"]
                or reload_claim["observed_adapter_bytes_sha256"] != pinned["adapter_bytes_sha256"]
                or reload_claim["observed_tensor_manifest_sha256"] != manifest_sha):
            result["reason"] = "RELOAD_CLAIM_MISMATCH"
            return result
        result["claimed_reload_fields_match"] = True
    # Canonical object hash is only a content-integrity pointer. It is not a
    # signature, producer identity, quantizer run or fresh process receipt.
    result["candidate_sha256"] = _digest(candidate)
    result["reason"] = "DECLARATIONS_CONSISTENT_NOT_INDEPENDENTLY_ATTESTED"
    return result


def simulate_reversible_cutover(previous_active_sha256: object,
                                candidate_sha256: object,
                                *, fail_at: str | None = None) -> dict:
    """Fault-injected *pure simulation*; cannot mutate a real active model."""
    if not _sha(previous_active_sha256) or not _sha(candidate_sha256):
        raise ValueError("staged and rollback artifact hashes are required")
    if previous_active_sha256 == candidate_sha256:
        raise ValueError("new candidate is identical to prior active source")
    transitions = ("STAGE", "VERIFY", "SWAP", "POSTCHECK", "COMMIT")
    # Membership must never execute a caller-defined __eq__ implementation.
    # An aliased value could otherwise forge a successful rollback rehearsal.
    if fail_at is not None and (type(fail_at) is not str or fail_at not in transitions):
        raise ValueError("unknown injected transition failure")
    trace = []
    for transition in transitions:
        trace.append(transition)
        if transition == fail_at:
            trace.extend(("ROLLBACK_REQUESTED", "RESTORE_PREVIOUS", "ROLLBACK_RECHECK"))
            return {
                "status": "SYNTHETIC_ROLLBACK_REHEARSED",
                "trace": trace, "simulated_active_sha256": previous_active_sha256,
                "previous_sha256_retained": True, "rolled_back": True,
                "production_active_untouched": True, "authorizes_deployment": False,
            }
    return {
        "status": "SYNTHETIC_CUTOVER_REHEARSED",
        "trace": trace, "simulated_active_sha256": candidate_sha256,
        "previous_sha256_retained": True, "rolled_back": False,
        "production_active_untouched": True, "authorizes_deployment": False,
    }
