"""Source-only, synthetic-fixture sparse weight edit validator.

Exp8-inspired byte-level experiment; NOT the Labyrinth #127 ALU, and NOT an
edit to the official Vera Qwen model. This module cannot open, modify, or
qualify an actual checkpoint. Its artifact SHA fields are metadata-format
checks, NOT proof of any real checkpoint, tokenizer, or config bytes. Its
random-arm seed and split IDs are declarations, not independently verified
randomization or heldout custody. A caller must supply an already synthetic
tiny tensor; no filesystem path (nor in-place edit) is accepted.

The official Qwen LoRA training lane is entirely independent. A successful
synthetic roundtrip says nothing about conversation, identity, or 4B hosts.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from typing import Any


SCHEMA = "TRAININGDAY_NO_GRADIENT_WEIGHT_EDIT_CANDIDATE_V1"
LANE = "NO_GRADIENT_WEIGHT_EDIT"
HOLD = "SOURCE_ONLY_HARD_HOLD"
FALSE_CLAIMS = {
    "qwen_weight_edit_verified": False,
    "persona_verified": False,
    "host_qualified": False,
}
MAX_ELEMENTS = 1024
MAX_CHANGED = 64
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_HEX_FLOAT = re.compile(r"[0-9a-f]{8}\Z")
_TENSOR_NAME = re.compile(r"synthetic\.layer\.(15|19)\.mlp\.out\Z")


class WeightEditCandidateError(ValueError):
    """Candidate fails an explicit research safety or integrity precondition."""


def _need(condition: bool, explanation: str) -> None:
    if not condition:
        raise WeightEditCandidateError(explanation)


def _keys(obj: Any, expected: set[str], label: str) -> None:
    _need(type(obj) is dict and set(obj) == expected, f"invalid {label} keys")


def _sha(value: Any, label: str) -> None:
    _need(type(value) is str and _SHA.fullmatch(value) is not None, f"invalid {label} SHA-256")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite_float32_le(buffer: bytes) -> bool:
    return all(math.isfinite(value) for (value,) in struct.iter_unpack("<f", buffer))


def validate_candidate(manifest: dict[str, Any], base_tensor: bytes) -> bytes:
    """Apply a synthetic candidate into new bytes after fail-closed validation.

    This verifies only the provided tiny tensor, not the full base checkpoint,
    sidecars, model identity, actual training quality, or held-out evaluation.
    """
    _keys(manifest, {"schema", "lane", "admission", "base", "tensor", "edits", "control", "feedback", "claims"}, "manifest")
    _need(manifest["schema"] == SCHEMA and manifest["lane"] == LANE, "research lane mismatch")
    _need(manifest["admission"] == HOLD, "admission must remain SOURCE_ONLY_HARD_HOLD")
    _need(
        type(manifest["claims"]) is dict
        and set(manifest["claims"]) == set(FALSE_CLAIMS)
        and all(manifest["claims"][key] is False for key in FALSE_CLAIMS),
        "unsupported qualification claims",
    )
    base = manifest["base"]
    _keys(base, {"kind", "repository", "revision", "checkpoint_sha256", "tokenizer_sha256", "config_sha256"}, "base")
    _need(base["kind"] == "synthetic_fixture" and base["repository"] == "synthetic/test-only", "real model checkpoints are unsupported")
    for key in ("revision", "checkpoint_sha256", "tokenizer_sha256", "config_sha256"):
        _sha(base[key], key)
    tensor = manifest["tensor"]
    _keys(tensor, {"name", "dtype", "shape", "before_sha256", "after_sha256"}, "tensor")
    _need(type(tensor["name"]) is str and _TENSOR_NAME.fullmatch(tensor["name"]) is not None, "tensor not allowlisted")
    _need(tensor["dtype"] == "float32_le", "only synthetic little-endian float32 supported")
    shape = tensor["shape"]
    _need(type(shape) is list and 1 <= len(shape) <= 4, "invalid tensor shape")
    count = 1
    for dim in shape:
        _need(type(dim) is int and 1 <= dim <= MAX_ELEMENTS, "invalid tensor dimension")
        count *= dim
        _need(count <= MAX_ELEMENTS, "tensor exceeds synthetic bound")
    _need(type(base_tensor) is bytes and len(base_tensor) == count * 4, "base byte length/shape mismatch")
    _need(_finite_float32_le(base_tensor), "base tensor contains non-finite values")
    _sha(tensor["before_sha256"], "before")
    _sha(tensor["after_sha256"], "after")
    _need(_digest(base_tensor) == tensor["before_sha256"], "base tensor SHA mismatch")

    control = manifest["control"]
    _keys(control, {"arm", "seed"}, "control")
    _need(control["arm"] in ("treatment", "random", "sham", "zero"), "unknown control arm")
    if control["arm"] == "random":
        _need(type(control["seed"]) is int and 0 <= control["seed"] < 2**64, "random control needs a fixed seed")
    else:
        _need(control["seed"] is None, "only random control may use a seed")
    feedback = manifest["feedback"]
    _keys(feedback, {"practice_split_id", "heldout_split_id", "heldout_data_in_edit"}, "feedback")
    _need(all(type(feedback[key]) is str and bool(feedback[key]) for key in ("practice_split_id", "heldout_split_id")), "split IDs required")
    _need(feedback["practice_split_id"] != feedback["heldout_split_id"], "practice/heldout IDs overlap")
    _need(feedback["heldout_data_in_edit"] is False, "heldout feedback cannot inform edits")

    edits = manifest["edits"]
    _need(type(edits) is list and len(edits) <= MAX_CHANGED, "changed-element budget exceeded")
    _need((len(edits) == 0) == (control["arm"] in ("sham", "zero")), "control arm/edit count mismatch")
    patched = bytearray(base_tensor)
    prior_index = -1
    for edit in edits:
        _keys(edit, {"flat_index", "before_hex", "after_hex"}, "sparse edit")
        index = edit["flat_index"]
        _need(type(index) is int and prior_index < index < count, "duplicate, unordered, or out-of-range index")
        prior_index = index
        for key in ("before_hex", "after_hex"):
            _need(type(edit[key]) is str and _HEX_FLOAT.fullmatch(edit[key]) is not None, "invalid float32 byte hex")
        before = bytes.fromhex(edit["before_hex"])
        after = bytes.fromhex(edit["after_hex"])
        _need(before != after, "non-changing sparse edit")
        _need(_finite_float32_le(after), "sparse edit introduces non-finite value")
        start = index * 4
        _need(patched[start:start + 4] == before, "sparse edit before-bytes mismatch")
        patched[start:start + 4] = after
    result = bytes(patched)
    _need(_digest(result) == tensor["after_sha256"], "patched tensor SHA mismatch")
    _need(_finite_float32_le(result), "patched tensor contains non-finite values")
    return result


def reverse_candidate(manifest: dict[str, Any], patched_tensor: bytes) -> bytes:
    """Prove exact tensor-byte reversibility; not entire checkpoint reversibility."""
    _need(type(patched_tensor) is bytes, "patched tensor must be immutable bytes")
    _need(type(manifest) is dict, "reverse manifest must be an object")
    edits = manifest.get("edits")
    _need(type(edits) is list and len(edits) <= MAX_CHANGED, "invalid reverse edits")
    reverted = bytearray(patched_tensor)
    for edit in reversed(edits):
        _keys(edit, {"flat_index", "before_hex", "after_hex"}, "reverse edit")
        index = edit["flat_index"]
        _need(type(index) is int and 0 <= index < len(reverted) // 4, "reverse index out of range")
        for key in ("before_hex", "after_hex"):
            _need(type(edit[key]) is str and _HEX_FLOAT.fullmatch(edit[key]) is not None, "invalid reverse float32 byte hex")
        before, after = bytes.fromhex(edit["before_hex"]), bytes.fromhex(edit["after_hex"])
        start = index * 4
        _need(reverted[start:start + 4] == after, "reverse after-bytes mismatch")
        reverted[start:start + 4] = before
    original = bytes(reverted)
    _need(type(manifest.get("tensor")) is dict and "before_sha256" in manifest["tensor"], "reverse tensor metadata missing")
    _need(_digest(original) == manifest["tensor"]["before_sha256"], "reverse base SHA mismatch")
    # Re-apply with all ordinary guardrails; rejects malformed or forged manifests.
    _need(validate_candidate(manifest, original) == patched_tensor, "reverse roundtrip mismatch")
    return original


def source_only_receipt(manifest: dict[str, Any], base_tensor: bytes) -> dict[str, Any]:
    """Make testable claims only about synthetic tensor bytes, never real models."""
    patched = validate_candidate(manifest, base_tensor)
    _need(reverse_candidate(manifest, patched) == base_tensor, "non-reversible synthetic candidate")
    return {
        "lane": LANE,
        "admission": HOLD,
        "synthetic_tensor_byte_roundtrip_verified": True,
        **FALSE_CLAIMS,
    }
