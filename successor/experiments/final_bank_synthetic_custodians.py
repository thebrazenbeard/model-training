from __future__ import annotations

import hashlib
import json
import re


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def build_identity_receipt(
    *,
    actor_id: str,
    provider: str,
    model: str,
    model_family: str,
    revision: str,
    runtime: str,
    seed: int,
    sampling: dict,
    artifacts: dict,
) -> dict:
    value = {
        "schema": "V10_FINAL_BANK_SYNTHETIC_CUSTODIAN_IDENTITY_V1",
        "actor_id": actor_id,
        "provider": provider,
        "model": model,
        "model_family": model_family,
        "revision": revision,
        "runtime": runtime,
        "seed": seed,
        "sampling": sampling,
        "artifacts": artifacts,
    }
    value["identity_receipt_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return value


def _verify_receipt(receipt: dict, *, label: str) -> list[str]:
    if not isinstance(receipt, dict):
        raise ValueError(f"identity_receipt_not_object:{label}")
    claimed = receipt.get("identity_receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("identity_receipt_sha256", None)
    expected = hashlib.sha256(_canonical(unsigned)).hexdigest()
    if claimed != expected:
        raise ValueError("identity_receipt_sha256_mismatch")
    if receipt.get("schema") != "V10_FINAL_BANK_SYNTHETIC_CUSTODIAN_IDENTITY_V1":
        raise ValueError(f"identity_receipt_schema_mismatch:{label}")

    reasons: list[str] = []
    for field in ("actor_id", "provider", "model", "model_family", "revision", "runtime"):
        if not _nonempty(receipt.get(field)):
            reasons.append(f"synthetic_identity_field_missing:{label}:{field}")
    if not isinstance(receipt.get("seed"), int):
        reasons.append(f"synthetic_identity_seed_invalid:{label}")
    if not isinstance(receipt.get("sampling"), dict) or not receipt["sampling"]:
        reasons.append(f"synthetic_identity_sampling_invalid:{label}")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        reasons.append(f"synthetic_identity_artifacts_invalid:{label}")
    elif any(
        not isinstance(value, str) or not _SHA256.fullmatch(value)
        for value in artifacts.values()
    ):
        reasons.append(f"synthetic_identity_artifact_hash_invalid:{label}")
    family = str(receipt.get("model_family", "")).casefold()
    model = str(receipt.get("model", "")).casefold()
    if "qwen" in family or "qwen" in model:
        reasons.append(f"qwen_lineage_forbidden:{label}")
    return reasons


def validate_synthetic_pair(receipt_a: dict, receipt_b: dict) -> dict:
    reasons = _verify_receipt(receipt_a, label="A")
    reasons.extend(_verify_receipt(receipt_b, label="B"))

    if receipt_a.get("actor_id") == receipt_b.get("actor_id"):
        reasons.append("synthetic_actor_ids_not_distinct")
    if receipt_a.get("model") == receipt_b.get("model"):
        reasons.append("exact_models_not_distinct")
    if (
        str(receipt_a.get("model_family", "")).casefold()
        == str(receipt_b.get("model_family", "")).casefold()
    ):
        reasons.append("model_families_not_distinct")
    if (
        receipt_a.get("provider"),
        receipt_a.get("runtime"),
    ) == (
        receipt_b.get("provider"),
        receipt_b.get("runtime"),
    ):
        reasons.append("provider_runtime_identity_not_distinct")

    reasons = sorted(set(reasons))
    return {
        "schema": "V10_FINAL_BANK_SYNTHETIC_CUSTODIAN_PAIR_CHECK_V1",
        "status": "SYNTHETIC_CUSTODIANS_BOUND" if not reasons else "HOLD",
        "reasons": reasons,
        "custodians": {
            "A": {
                key: receipt_a.get(key)
                for key in (
                    "actor_id",
                    "provider",
                    "model",
                    "model_family",
                    "revision",
                    "runtime",
                    "seed",
                    "sampling",
                    "identity_receipt_sha256",
                )
            },
            "B": {
                key: receipt_b.get(key)
                for key in (
                    "actor_id",
                    "provider",
                    "model",
                    "model_family",
                    "revision",
                    "runtime",
                    "seed",
                    "sampling",
                    "identity_receipt_sha256",
                )
            },
        },
        "claim_ceiling": (
            "SYNTHETIC_CUSTODIAN_IDENTITIES_ONLY / "
            "NO HUMAN AUTHOR OR REVIEWER BOUND / "
            "NO CUSTODY SURFACE / NO FINAL PLAINTEXT / NO TRAINING"
        ),
    }
