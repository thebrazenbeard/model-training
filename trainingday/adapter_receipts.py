"""Static custody verification for zero/correct/corrupt neural LoRA arms.

This only establishes recorded tensor-digest consistency and adapter file
persistence. It does NOT reload model weights, attest optimizer execution,
prove model quality, or authorize a local-model promotion.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


class AdapterReceiptError(ValueError):
    """One or more adapter artifact and custody obligations were not met."""


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_value(value) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[a-f0-9]{64}", value))


def verify_training_triplet(
    folders: dict, *, expected_manifest_sha256: str,
    expected_selected_id_sha256: str, expected_base_revision: str,
    expected_base_binding_sha256: str
) -> dict:
    """Require three disjoint saved, matched control arms and hash their files.

    The three receipts come from the training runner, not an independent
    evaluator. Claimed tensor changes remain self-attested until reloaded and
    checked in a different process with trustworthy runtime evidence.
    """
    if not isinstance(folders, dict) or set(folders) != {"zero", "correct", "corrupt"}:
        raise AdapterReceiptError("missing zero/correct/corrupt controls")
    if not all(_sha_value(v) for v in (
        expected_manifest_sha256, expected_selected_id_sha256,
        expected_base_binding_sha256
    )) or not isinstance(expected_base_revision, str) or not re.fullmatch(
        r"[0-9a-f]{40}", expected_base_revision
    ):
        raise AdapterReceiptError("invalid pinned source identifiers")
    receipts = {}
    paths = set()
    model_hashes = {}
    for mode in ("zero", "correct", "corrupt"):
        root = Path(folders[mode])
        if not root.is_dir() or root.is_symlink():
            raise AdapterReceiptError("missing/symlinked adapter directory")
        resolved = root.resolve()
        if resolved in paths:
            raise AdapterReceiptError("duplicate control output directory")
        paths.add(resolved)
        receipt_file = root / "training_receipt.json"
        if not receipt_file.is_file() or receipt_file.is_symlink():
            raise AdapterReceiptError("training receipt missing")
        try:
            receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdapterReceiptError("training receipt malformed") from exc
        corpus = receipt.get("corpus")
        if not isinstance(corpus, dict) or corpus.get("manifest_sha256") != expected_manifest_sha256:
            raise AdapterReceiptError("manifest pin mismatch")
        if corpus.get("selected_id_sha256") != expected_selected_id_sha256:
            raise AdapterReceiptError("selected train IDs differ")
        if (receipt.get("status") != "ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED"
                or receipt.get("mode") != mode
                or receipt.get("base_revision") != expected_base_revision
                or receipt.get("base_binding_sha256") != expected_base_binding_sha256):
            raise AdapterReceiptError("control receipt source/mode mismatch")
        steps = receipt.get("new_optimizer_steps")
        if (type(steps) is not int or receipt.get("steps") != steps
                or (steps != 0 if mode == "zero" else steps < 1)):
            raise AdapterReceiptError("control optimizer steps mismatch")
        if not all(_sha_value(receipt.get(k)) for k in (
            "trainable_digest_before", "trainable_digest_after",
            "adapter_model_sha256", "adapter_config_sha256"
        )):
            raise AdapterReceiptError("invalid model/adapter digest fields")
        folder = root / "adapter"
        if folder.is_symlink():
            raise AdapterReceiptError("adapter link disallowed")
        for filename, key in (
            ("adapter_model.safetensors", "adapter_model_sha256"),
            ("adapter_config.json", "adapter_config_sha256")
        ):
            path = folder / filename
            if not path.is_file() or path.is_symlink() or _sha_file(path) != receipt[key]:
                raise AdapterReceiptError("adapter file checksum mismatch")
        model_hashes[mode] = receipt["adapter_model_sha256"]
        receipts[mode] = receipt
    before = receipts["zero"]["trainable_digest_before"]
    if any(r["trainable_digest_before"] != before for r in receipts.values()):
        raise AdapterReceiptError("initial weight digests do not match")
    if len({r.get("seed") for r in receipts.values()}) != 1:
        raise AdapterReceiptError("matched control seed mismatch")
    if (receipts["zero"]["trainable_digest_after"] != before
            or any(receipts[mode]["trainable_digest_after"] == before
                   for mode in ("correct", "corrupt"))):
        raise AdapterReceiptError("weight digest/optimizer control outcome invalid")
    if receipts["correct"]["new_optimizer_steps"] != receipts["corrupt"]["new_optimizer_steps"]:
        raise AdapterReceiptError("matched trained/corrupt optimizer steps differ")
    return {
        "status": "ADAPTER_BYTES_VERIFIED_NOT_INDEPENDENTLY_RELOADED",
        "modes": ["zero", "correct", "corrupt"],
        "optimizer_steps": {mode: receipts[mode]["new_optimizer_steps"]
                            for mode in ("zero", "correct", "corrupt")},
        "adapter_sha256": model_hashes,
        "manifest_sha256": expected_manifest_sha256,
        "selected_id_sha256": expected_selected_id_sha256,
        "base_revision": expected_base_revision,
        "new_weight_updates_verified": False,
        "model_reloaded_in_separate_process": False,
        "independent_semantic_improvement_verified": False,
    }
