from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from successor.experiments.train_v10r2_dev import (
    DevTrainingHold,
    load_and_validate_dev_spec,
)


HASH_SEMANTICS = "UTF8_TEXT_LF_NORMALIZED_COMMITTED_CONTENT"
PREREG_STATUS = "PREREGISTERED_NO_EXECUTION_AUTHORITY"


class ExecutionProjectionHold(RuntimeError):
    """Fail-closed refusal for a V10R3R1 execution-spec projection."""


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ExecutionProjectionHold(f"JSON object required:{path}")
    return value


def _committed_sha256(repo_root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ExecutionProjectionHold("template must be inside repo root") from exc
    try:
        committed = subprocess.check_output(
            ["git", "-C", str(repo_root), "show", f"HEAD:{relative}"],
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        raise ExecutionProjectionHold("unable to read committed prereg template") from exc
    if b"\r\n" in committed:
        raise ExecutionProjectionHold("committed prereg template is not LF-normalized")
    return hashlib.sha256(committed).hexdigest()


def _projection(value: dict) -> dict:
    projected = dict(value)
    projected.pop("status", None)
    projected.pop("authority", None)
    return projected


def verify_execution_projection(
    repo_root: Path,
    *,
    template_path: Path,
    protocol_path: Path,
    execution_spec_path: Path,
) -> dict:
    repo_root = Path(repo_root)
    template_path = Path(template_path)
    protocol_path = Path(protocol_path)
    execution_spec_path = Path(execution_spec_path)

    if template_path.resolve() == execution_spec_path.resolve():
        raise ExecutionProjectionHold("execution spec must be distinct from prereg template")

    template = _read_json(template_path)
    protocol = _read_json(protocol_path)

    if template.get("status") != PREREG_STATUS:
        raise ExecutionProjectionHold("prereg template is not fail-closed")
    authority = protocol.get("authority")
    if not isinstance(authority, dict):
        raise ExecutionProjectionHold("protocol authority missing")
    if protocol.get("status") != PREREG_STATUS:
        raise ExecutionProjectionHold("protocol is not preregistration-only")
    if authority.get("development_training_authorized") is not False:
        raise ExecutionProjectionHold("protocol unexpectedly authorizes training")
    if authority.get("execution_authority_required") is not True:
        raise ExecutionProjectionHold("protocol execution-authority gate missing")

    replication = protocol.get("replication")
    if not isinstance(replication, dict):
        raise ExecutionProjectionHold("protocol replication binding missing")
    if replication.get("spec_hash_semantics") != HASH_SEMANTICS:
        raise ExecutionProjectionHold("template hash semantics mismatch")
    committed_sha = _committed_sha256(repo_root, template_path)
    if replication.get("spec_sha256") != committed_sha:
        raise ExecutionProjectionHold("committed prereg template hash mismatch")

    try:
        execution = load_and_validate_dev_spec(execution_spec_path)
    except DevTrainingHold as exc:
        raise ExecutionProjectionHold(
            f"execution spec lacks valid current training authority:{exc}"
        ) from exc

    if _projection(execution) != _projection(template):
        raise ExecutionProjectionHold("execution projection mismatch")
    output_namespace = execution["output"]["namespace"]
    if replication.get("output_namespace") != output_namespace:
        raise ExecutionProjectionHold("output namespace mismatch")

    return {
        "schema": "V10R3R1_EXECUTION_PROJECTION_VERIFICATION_V1",
        "status": "PASS",
        "template_committed_sha256": committed_sha,
        "template_hash_semantics": HASH_SEMANTICS,
        "recipe_projection_match": True,
        "output_namespace_match": True,
        "gpu_effect": "NONE",
        "claim_ceiling": (
            "EXECUTION_SPEC_PROJECTION_VERIFICATION_ONLY_NOT_EXECUTION_AUTHORITY_"
            "NOT_TRAINING_RESULT"
        ),
    }
