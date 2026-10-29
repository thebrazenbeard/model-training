import json

import pytest

from trainingday.admission import AdmissionError, _hash_file
from trainingday.reload_verify import validate_receipt


def _receipt(tmp_path):
    binding = tmp_path / "binding.json"
    binding.write_text("{}", encoding="utf-8")
    adapter_dir = tmp_path / "run" / "adapter"
    adapter_dir.mkdir(parents=True)
    model = adapter_dir / "adapter_model.safetensors"
    model.write_bytes(b"fake safe-tensor fixture")
    config = adapter_dir / "adapter_config.json"
    config.write_text("{}", encoding="utf-8")
    data = {
        "status": "ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED",
        "mode": "correct", "new_optimizer_steps": 1,
        "trainable_digest_after": "a" * 64,
        "base_binding_sha256": _hash_file(binding),
        "adapter_model_sha256": _hash_file(model),
        "adapter_config_sha256": _hash_file(config),
    }
    receipt = tmp_path / "run" / "training_receipt.json"
    receipt.write_text(json.dumps(data), encoding="utf-8")
    return binding, receipt, model, data


def test_accepts_custodied_trained_receipt(tmp_path):
    binding, receipt, _, data = _receipt(tmp_path)
    admitted, adapter_dir = validate_receipt(receipt, binding)
    assert admitted == data
    assert adapter_dir == receipt.parent / "adapter"


def test_rejects_tampered_adapter(tmp_path):
    binding, receipt, model, _ = _receipt(tmp_path)
    model.write_bytes(b"altered adapter")
    with pytest.raises(AdmissionError, match="custody mismatch"):
        validate_receipt(receipt, binding)


def test_rejects_zero_step_receipt(tmp_path):
    binding, receipt, _, data = _receipt(tmp_path)
    data["new_optimizer_steps"] = 0
    receipt.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(AdmissionError, match="no actual trained"):
        validate_receipt(receipt, binding)
