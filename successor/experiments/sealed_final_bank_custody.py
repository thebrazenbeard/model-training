from __future__ import annotations

import hashlib
import json
from pathlib import Path

from successor.experiments.sealed_final_bank_commitment import (
    FORBIDDEN_PLAINTEXT_KEYS,
    REQUIRED_HASHES,
    validate_sealed_final_bank_commitment,
)


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _find_plaintext_keys(value, *, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text.casefold() in FORBIDDEN_PLAINTEXT_KEYS:
                found.append(f"{path}.{key_text}")
            found.extend(
                _find_plaintext_keys(child, path=f"{path}.{key_text}")
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(
                _find_plaintext_keys(child, path=f"{path}[{index}]")
            )
    return found


def _base_commitment_hash(commitment: dict) -> str:
    value = dict(commitment)
    value.pop("commitment_sha256", None)
    return _sha256(_canonical_bytes(value))


def build_public_commitment(custodian_manifest: dict) -> dict:
    if not isinstance(custodian_manifest, dict):
        raise RuntimeError("custodian manifest must be an object")
    plaintext_paths = _find_plaintext_keys(custodian_manifest)
    if plaintext_paths:
        raise RuntimeError(
            "plaintext fields are forbidden in custody metadata: "
            + ",".join(plaintext_paths)
        )
    if (
        custodian_manifest.get("schema")
        != "V10_FINAL_BANK_CUSTODIAN_MANIFEST_V1"
    ):
        raise RuntimeError("invalid custodian manifest schema")
    if custodian_manifest.get("status") != "FROZEN_SEALED_PRETRAINING":
        raise RuntimeError("custodian manifest is not frozen/sealed")

    artifacts = custodian_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("custodian artifact hashes missing")
    for field in REQUIRED_HASHES:
        if not _valid_sha256(artifacts.get(field)):
            raise RuntimeError(f"invalid custodian artifact hash:{field}")

    custody = custodian_manifest.get("custody")
    if not isinstance(custody, dict):
        raise RuntimeError("custody metadata missing")
    storage = custody.get("sealed_storage")
    if not isinstance(storage, dict):
        raise RuntimeError("sealed storage metadata missing")
    if storage.get("training_lane_access") is not False:
        raise RuntimeError("plaintext_exposed through sealed storage")
    if custody.get("plaintext_exposed_to_training_lane") is not False:
        raise RuntimeError("plaintext_exposed_to_training_lane")
    if custody.get("training_lane_receives_hashes_only") is not True:
        raise RuntimeError("training lane hashes-only boundary missing")

    freeze = custodian_manifest.get("freeze")
    if not isinstance(freeze, dict):
        raise RuntimeError("freeze metadata missing")
    freeze_digest = freeze.get("freeze_subject_digest")
    if not _valid_sha256(freeze_digest):
        raise RuntimeError("invalid freeze subject digest")
    if freeze.get("post_freeze_mutation") is not False:
        raise RuntimeError("post-freeze mutation is not false")

    public = {
        "schema": "V10_SEALED_FINAL_BANK_COMMITMENT_V1",
        "bank_id": custodian_manifest.get("bank_id"),
        "status": "SEALED_PRETRAINING_FINAL_BANK",
        "lane_counts": custodian_manifest.get("lane_counts"),
        "behavioral_dimension_counts": custodian_manifest.get(
            "behavioral_dimension_counts"
        ),
        "adversarial_dimension_counts": custodian_manifest.get(
            "adversarial_dimension_counts"
        ),
        "behavioral_family_counts": custodian_manifest.get(
            "behavioral_family_counts"
        ),
        "plaintext_artifacts": {
            field: artifacts.get(field) for field in REQUIRED_HASHES
        },
        "custody": {
            "plaintext_exposed_to_training_lane": custody.get(
                "plaintext_exposed_to_training_lane"
            ),
            "custodian_ids": custody.get("custodian_ids"),
            "human_reviewer_id": custody.get("human_reviewer_id"),
            "training_lane_receives_hashes_only": custody.get(
                "training_lane_receives_hashes_only"
            ),
        },
        "admission": custodian_manifest.get("admission"),
        "freeze_subject_digest": freeze_digest,
    }
    check = validate_sealed_final_bank_commitment(public)
    if check["status"] != "SEALED_BANK_GATE_PASS":
        raise RuntimeError(
            "sealed commitment validation failed: "
            + " | ".join(check["reasons"])
        )
    public["commitment_sha256"] = _base_commitment_hash(public)
    return public


def _compare_exact(
    reasons: list[str],
    *,
    label: str,
    left,
    right,
) -> None:
    if left != right:
        reasons.append(f"{label}_mismatch")


def verify_reveal_manifest(
    commitment: dict,
    reveal_manifest: dict,
) -> dict:
    reasons: list[str] = []
    if not isinstance(commitment, dict):
        return {
            "schema": "V10_FINAL_BANK_REVEAL_CHECK_V1",
            "status": "HOLD",
            "reasons": ["commitment_not_object"],
        }
    gate = validate_sealed_final_bank_commitment(commitment)
    if gate["status"] != "SEALED_BANK_GATE_PASS":
        reasons.extend(
            f"commitment:{reason}" for reason in gate["reasons"]
        )
    expected_commitment_sha = _base_commitment_hash(commitment)
    if commitment.get("commitment_sha256") != expected_commitment_sha:
        reasons.append("commitment_sha256_mismatch")

    if not isinstance(reveal_manifest, dict):
        reasons.append("reveal_manifest_not_object")
        reveal_manifest = {}
    if (
        reveal_manifest.get("schema")
        != "V10_FINAL_BANK_REVEAL_MANIFEST_V1"
    ):
        reasons.append("reveal_schema_mismatch")
    _compare_exact(
        reasons,
        label="bank_id",
        left=reveal_manifest.get("bank_id"),
        right=commitment.get("bank_id"),
    )
    for field in (
        "lane_counts",
        "behavioral_dimension_counts",
        "adversarial_dimension_counts",
        "behavioral_family_counts",
    ):
        _compare_exact(
            reasons,
            label=field,
            left=reveal_manifest.get(field),
            right=commitment.get(field),
        )

    reveal_artifacts = reveal_manifest.get("plaintext_artifacts")
    committed_artifacts = commitment.get("plaintext_artifacts")
    if not isinstance(reveal_artifacts, dict):
        reasons.append("reveal_artifacts_missing")
        reveal_artifacts = {}
    if not isinstance(committed_artifacts, dict):
        reasons.append("commitment_artifacts_missing")
        committed_artifacts = {}
    for field in REQUIRED_HASHES:
        actual = reveal_artifacts.get(field)
        expected = committed_artifacts.get(field)
        if actual != expected:
            reasons.append(f"artifact_hash_mismatch:{field}")

    if (
        reveal_manifest.get("freeze_subject_digest")
        != commitment.get("freeze_subject_digest")
    ):
        reasons.append("freeze_subject_digest_mismatch")

    plaintext_paths = _find_plaintext_keys(reveal_manifest)
    if plaintext_paths:
        reasons.extend(
            f"reveal_plaintext_field_present:{path}"
            for path in plaintext_paths
        )

    reasons = sorted(set(reasons))
    return {
        "schema": "V10_FINAL_BANK_REVEAL_CHECK_V1",
        "status": "REVEAL_MATCH" if not reasons else "HOLD",
        "bank_id": commitment.get("bank_id"),
        "commitment_sha256": commitment.get("commitment_sha256"),
        "reasons": reasons,
        "claim_ceiling": (
            "REVEAL_HASH_AND_METADATA_MATCH_ONLY / "
            "DOES_NOT SCORE OR ALTER FINAL BANK / "
            "DOES_NOT AUTHORIZE POST_REVEAL TUNING"
        ),
    }


def verify_file_sha256(path: Path, expected: str) -> str:
    if not _valid_sha256(expected):
        raise ValueError("expected SHA-256 is invalid")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"file hash mismatch for {path}: {actual} != {expected}"
        )
    return actual