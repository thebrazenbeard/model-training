"""Source-only handoff guard for a real standalone Vera neural model.

This verifies local artifact bytes and rejects authority-bearing candidate metadata.
It cannot establish independent model quality, selection, runtime consumption or effect authority.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Mapping


SHA256 = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
ARTIFACT_ROLES = ("adapter_weights", "adapter_config", "gguf")
NO_AUTHORITY_FIELDS = (
    "identity_authority",
    "canonical_memory_write_authority",
    "browser_authority",
    "protected_effect_authority",
    "may_self_qualify",
)


class CandidateRejected(ValueError):
    """Invalid or unbound model handoff candidate."""


def _require_digest(value: object, field: str, pattern=SHA256) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise CandidateRejected(f"{field} must be a canonical lowercase digest")
    return value


def validate_descriptor(manifest: Mapping[str, object]) -> None:
    if not isinstance(manifest, dict):
        raise CandidateRejected("candidate must be a JSON object")
    fixed = {
        "schema": "VERA_STANDALONE_NEURAL_MODEL_CANDIDATE_V1",
        "producer": "thebrazenbeard/vera_model_training",
        "product": "STANDALONE_VERA_MODEL",
        "integration": "OPTIONAL_COGNITION_PROVIDER",
        "route_class": "COGNITION_ONLY",
        "standalone_inference_backend": "KOBOLDCPP",
        "qualification_state": "UNQUALIFIED_CANDIDATE",
    }
    for field, expected in fixed.items():
        if manifest.get(field) != expected:
            raise CandidateRejected(f"{field} must equal {expected!r}")
    name = manifest.get("model_id")
    if not isinstance(name, str) or MODEL_NAME.fullmatch(name) is None:
        raise CandidateRejected("model_id must be an unambiguous nonempty identifier")
    base = manifest.get("base_model")
    if not isinstance(base, dict) or not isinstance(base.get("id"), str) or not base["id"].strip():
        raise CandidateRejected("base model identifier is required")
    _require_digest(base.get("revision"), "base_model.revision", REVISION)
    for field in ("training_recipe_sha256", "dataset_manifest_sha256"):
        _require_digest(manifest.get(field), field)
    length = manifest.get("context_length")
    if type(length) is not int or length < 1:
        raise CandidateRejected("context_length must be a positive integer")
    if not isinstance(manifest.get("quantization"), str) or not manifest["quantization"].strip():
        raise CandidateRejected("quantization must be declared")
    if manifest.get("private_user_data_included") is not False:
        raise CandidateRejected("private user data must not be included in the public handoff")
    for field in NO_AUTHORITY_FIELDS:
        if manifest.get(field) is not False:
            raise CandidateRejected(f"{field} must explicitly be false")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_ROLES):
        raise CandidateRejected("exact adapter_weights, adapter_config and gguf artifacts required")
    paths: set[str] = set()
    for role in ARTIFACT_ROLES:
        item = artifacts[role]
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise CandidateRejected(f"{role} must have exactly path and sha256")
        path = item.get("path")
        if not isinstance(path, str) or "\\" in path or ":" in path or not path.strip():
            raise CandidateRejected(f"{role} path is invalid")
        parts = PurePosixPath(path)
        if parts.is_absolute() or any(part in (".", "..") for part in path.split("/")) or str(parts) != path:
            raise CandidateRejected(f"{role} path must be relative, canonical and traversal-free")
        if path in paths:
            raise CandidateRejected("artifact roles must not share a path")
        paths.add(path)
        _require_digest(item.get("sha256"), role + ".sha256")


def _hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_candidate(manifest_path: Path, artifact_root: Path) -> dict[str, object]:
    """Byte verification only. Does not qualify/activate a Vera model."""
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise CandidateRejected("candidate JSON is invalid") from exc
    validate_descriptor(manifest)
    root = artifact_root.resolve(strict=True)
    if not root.is_dir():
        raise CandidateRejected("artifact_root must be a directory")
    for role in ARTIFACT_ROLES:
        descriptor = manifest["artifacts"][role]
        full = (root / descriptor["path"]).resolve(strict=True)
        if not full.is_relative_to(root) or not full.is_file():
            raise CandidateRejected(f"{role} artifact escapes root or is not a file")
        if full.stat().st_size <= 4 or _hash_file(full) != descriptor["sha256"]:
            raise CandidateRejected(f"{role} artifact is empty or differs from descriptor")
        if role == "gguf":
            with full.open("rb") as stream:
                if stream.read(4) != b"GGUF":
                    raise CandidateRejected("model is not a GGUF artifact")
    return {
        "schema": "VERA_MODEL_HANDOFF_BYTE_VERIFICATION_V1",
        "status": "ARTIFACT_BYTES_VERIFIED_NOT_MODEL_QUALIFIED",
        "candidate_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "model_id": manifest["model_id"],
        "gguf_sha256": manifest["artifacts"]["gguf"]["sha256"],
        "route_class": "COGNITION_ONLY",
        "independent_model_qualification": False,
        "pre_active_target_selected": False,
        "portal_runtime_consumption": False,
        "veraos_interface_qualified": False,
        "protected_effect_authority": False,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m training.verify_vera_model_handoff CANDIDATE.json ARTIFACT_ROOT")
    try:
        print(json.dumps(verify_candidate(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
    except (CandidateRejected, OSError) as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        raise SystemExit(2)
