from __future__ import annotations

import hashlib
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_semantic_screen_binding(
    receipt: dict,
    *,
    target_path: Path,
    expected_threshold: float,
    expected_model_archive_sha256: str,
    expected_registry_sha256: str,
) -> dict:
    reasons: list[str] = []
    actual_target_sha = _sha256_file(target_path)
    target_sha = receipt.get("target_sha256")

    if not isinstance(target_sha, str):
        reasons.append("target_sha256_missing")
    elif target_sha != actual_target_sha:
        reasons.append(
            f"target_sha256_mismatch:{target_sha}!={actual_target_sha}"
        )

    if receipt.get("status") != "PASS":
        reasons.append(f"screen_status:{receipt.get('status')}")
    if receipt.get("threshold") != expected_threshold:
        reasons.append(
            f"threshold_mismatch:{receipt.get('threshold')}!="
            f"{expected_threshold}"
        )
    if receipt.get("model_archive_sha256") != expected_model_archive_sha256:
        reasons.append("model_archive_sha256_mismatch")
    registry_sha = receipt.get("registry", {}).get("registry_sha256")
    if registry_sha != expected_registry_sha256:
        reasons.append("registry_sha256_mismatch")

    return {
        "schema": "V10_SEMANTIC_SCREEN_BINDING_CHECK_V1",
        "status": "PASS" if not reasons else "HOLD",
        "target_sha256": actual_target_sha,
        "reasons": sorted(reasons),
    }
