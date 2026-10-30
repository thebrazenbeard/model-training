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


R15_RECIPE = "TRAININGDAY_QWEN35_FRESH_QLORA_R15_NUMERIC_MATCHED"
R15_PROMPT = "R15_STRICT_INTEGER_V1"
V2_MANIFEST = "90b55cf8742f9222c91f88e5727dcd0113322e4721e4fd3d1bdcdf65f40d0836"
LABYRINTH_HEAD = "1d38e4f25b5663bf8a84ff5e139875d4152e3680"


def _r15_fields(data):
    data.update(recipe=R15_RECIPE, question_contract=R15_PROMPT,
                corpus={"manifest_sha256": V2_MANIFEST,
                        "producer_git_head": LABYRINTH_HEAD})


def _check_r15(receipt, binding):
    return validate_receipt(
        receipt, binding, expected_recipe=R15_RECIPE,
        expected_question_contract=R15_PROMPT,
        expected_manifest_sha256=V2_MANIFEST,
        expected_producer_git_head=LABYRINTH_HEAD,
    )


def test_accepts_exact_r15_eval_receipt_without_breaking_r14_generic(tmp_path):
    binding, receipt, _, data = _receipt(tmp_path)
    # Generic reload remains backward compatible with R14 receipts.
    validate_receipt(receipt, binding)
    _r15_fields(data)
    receipt.write_text(json.dumps(data), encoding="utf-8")
    actual, _ = _check_r15(receipt, binding)
    assert actual["question_contract"] == R15_PROMPT


@pytest.mark.parametrize("tamper,expected_error", [
    (lambda data: data.update(recipe="TRAININGDAY_QWEN35_FRESH_QLORA_R14"), "recipe"),
    (lambda data: data.update(question_contract="R14_UNALIGNED"), "question contract"),
    (lambda data: data["corpus"].update(manifest_sha256="f" * 64), "manifest"),
    (lambda data: data["corpus"].update(producer_git_head="a" * 40), "producer"),
    (lambda data: data.pop("corpus"), "manifest"),
])
def test_rejects_stale_or_misbound_r15_eval_receipt(tmp_path, tamper, expected_error):
    binding, receipt, _, data = _receipt(tmp_path)
    _r15_fields(data)
    tamper(data)
    receipt.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(AdmissionError, match=expected_error):
        _check_r15(receipt, binding)
