"""Static adapter-custody gate: not a model load or behavioral test."""
import hashlib
import json
from pathlib import Path

import pytest

from trainingday.adapter_receipts import AdapterReceiptError, verify_training_triplet


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _triplet(root):
    base = "a" * 64
    selected = "b" * 64
    frozen = "c" * 64
    binding = "d" * 64
    revision = "e" * 40
    arms = {}
    for mode, steps in (("zero", 0), ("correct", 1), ("corrupt", 1)):
        parent = root / mode
        folder = parent / "adapter"
        folder.mkdir(parents=True)
        weights = (mode * 10).encode("utf-8")
        config = (mode + " config").encode("utf-8")
        (folder / "adapter_model.safetensors").write_bytes(weights)
        (folder / "adapter_config.json").write_bytes(config)
        receipt = {
            "status": "ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED",
            "mode": mode,
            "steps": steps,
            "new_optimizer_steps": steps,
            "seed": 41,
            "corpus": {"manifest_sha256": base, "selected_id_sha256": selected},
            "base_revision": revision,
            "base_binding_sha256": binding,
            "trainable_digest_before": frozen,
            "trainable_digest_after": frozen if mode == "zero" else ("f" if mode == "correct" else "9") * 64,
            "adapter_model_sha256": _sha(weights),
            "adapter_config_sha256": _sha(config),
        }
        (parent / "training_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        arms[mode] = parent
    return arms, {"expected_manifest_sha256": base,
                  "expected_selected_id_sha256": selected,
                  "expected_base_revision": revision,
                  "expected_base_binding_sha256": binding}


def test_triplet_proves_persistence_only_not_reload(tmp_path):
    folders, args = _triplet(tmp_path)
    result = verify_training_triplet(folders, **args)
    assert result["status"] == "ADAPTER_BYTES_VERIFIED_NOT_INDEPENDENTLY_RELOADED"
    assert result["modes"] == ["zero", "correct", "corrupt"]
    assert result["optimizer_steps"] == {"zero": 0, "correct": 1, "corrupt": 1}
    assert result["new_weight_updates_verified"] is False
    assert "evaluation_accuracy" not in result


def test_tampered_adapter_refused_even_with_valid_receipt(tmp_path):
    folders, args = _triplet(tmp_path)
    (folders["correct"] / "adapter/adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(AdapterReceiptError, match="checksum"):
        verify_training_triplet(folders, **args)


def test_mismatched_seed_or_initial_digest_refused(tmp_path):
    folders, args = _triplet(tmp_path)
    receipt = folders["corrupt"] / "training_receipt.json"
    data = json.loads(receipt.read_text())
    data["trainable_digest_before"] = "1" * 64
    receipt.write_text(json.dumps(data))
    with pytest.raises(AdapterReceiptError, match="initial"):
        verify_training_triplet(folders, **args)


def test_unchanged_trained_tensors_are_refused(tmp_path):
    folders, args = _triplet(tmp_path)
    receipt = folders["correct"] / "training_receipt.json"
    data = json.loads(receipt.read_text())
    data["trainable_digest_after"] = data["trainable_digest_before"]
    receipt.write_text(json.dumps(data))
    with pytest.raises(AdapterReceiptError, match="weight"):
        verify_training_triplet(folders, **args)


def test_wrong_bound_manifest_refused(tmp_path):
    folders, args = _triplet(tmp_path)
    args["expected_manifest_sha256"] = "0" * 64
    with pytest.raises(AdapterReceiptError, match="manifest"):
        verify_training_triplet(folders, **args)
