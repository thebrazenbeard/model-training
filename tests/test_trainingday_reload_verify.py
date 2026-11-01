import json

import pytest

from trainingday.admission import AdmissionError, _hash_file
from trainingday.checkpoint_custody import canonical_sha, create_custody
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


def _r16_receipt(tmp_path):
    """Synthetic bytes exercise the actual R16 contract/custody parser, never a GPU."""
    binding, receipt_path, _, receipt = _receipt(tmp_path)
    corpus = {
        "manifest_sha256": "90b55cf8742f9222c91f88e5727dcd0113322e4721e4fd3d1bdcdf65f40d0836",
        "producer_git_head": "1d38e4f25b5663bf8a84ff5e139875d4152e3680",
    }
    contract = {
        "schema": "VERA_TRAININGDAY_R16_RUN_CONTRACT_V1",
        "recipe": "TRAININGDAY_QWEN35_QLORA_R16_CHECKPOINTABLE_MATCHED",
        "question_contract": "R15_STRICT_INTEGER_V1",
        "manifest_sha256": corpus["manifest_sha256"],
        "producer_git_head": corpus["producer_git_head"],
        "mode": "correct", "base_model": "Qwen/Qwen3.5-4B",
        "base_revision": "b" * 40,
        "base_binding_sha256": _hash_file(binding),
        "optimizer": "adamw_bnb_8bit", "gradient_accumulation": 8,
        "lora_r": 4, "lora_alpha": 16, "max_length": 512,
        "save_strategy": "steps", "save_steps": 8,
        "cases": 64, "seed": 41, "learning_rate": 2e-5,
        "run_id": "run", "steps": 8,
        "initial_digest": "0" * 64,
        "training_input_sha256": "c" * 64,
    }
    (receipt_path.parent / "r16-run-contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    checkpoint = receipt_path.parent / "trainer-work" / "checkpoint-8"
    checkpoint.mkdir(parents=True)
    files = {
        "trainer_state.json": b'{"global_step":8}',
        "optimizer.pt": b"synthetic-optimizer-state",
        "scheduler.pt": b"synthetic-scheduler-state",
        "rng_state.pth": b"synthetic-rng-state",
        "adapter_model.safetensors": b"synthetic-step-adapter",
        "adapter_config.json": b'{}',
    }
    for filename, value in files.items():
        (checkpoint / filename).write_bytes(value)
    custody = create_custody(checkpoint, contract_sha256=canonical_sha(contract))
    custody_path = receipt_path.parent / "latest-checkpoint-custody.json"
    custody_path.write_text(json.dumps(custody), encoding="utf-8")
    receipt.update({
        "status": "R16_ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED",
        "recipe": contract["recipe"],
        "question_contract": contract["question_contract"],
        "corpus": corpus, "base_model": contract["base_model"],
        "base_revision": contract["base_revision"],
        "training_input_sha256": contract["training_input_sha256"],
        "seed": contract["seed"],
        "new_optimizer_steps": 8, "resumed_from_step": 0,
        "trainable_digest_before": "0" * 64,
        "optimizer_resume_equivalence": "UNVERIFIED_REQUIRES_UNINTERRUPTED_MATCH",
        "optimizer_identity": {"backend": "bitsandbytes"},
        "r16_run_contract_sha256": canonical_sha(contract),
        "checkpoint_step": 8,
        "checkpoint_custody_file_sha256": _hash_file(custody_path),
        "checkpoint_custody_state": custody["state"],
    })
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return binding, receipt_path, checkpoint, receipt


def test_accepts_r16_receipt_only_after_full_contract_and_checkpoint_custody(tmp_path):
    binding, receipt, _, data = _r16_receipt(tmp_path)
    admitted, adapter_dir = validate_receipt(receipt, binding)
    assert admitted == data
    assert adapter_dir == receipt.parent / "adapter"


def test_r16_missing_contract_fails_closed(tmp_path):
    binding, receipt, _, _ = _r16_receipt(tmp_path)
    (receipt.parent / "r16-run-contract.json").unlink()
    with pytest.raises(AdmissionError, match="R16 run contract absent"):
        validate_receipt(receipt, binding)


def test_r16_tampered_checkpoint_fails_closed(tmp_path):
    binding, receipt, checkpoint, _ = _r16_receipt(tmp_path)
    (checkpoint / "optimizer.pt").write_bytes(b"tampered-optimizer")
    with pytest.raises(AdmissionError, match="checkpoint bytes/step mismatch"):
        validate_receipt(receipt, binding)


def test_r16_tampered_contract_fails_closed(tmp_path):
    binding, receipt, _, _ = _r16_receipt(tmp_path)
    contract_path = receipt.parent / "r16-run-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["learning_rate"] = 0.0001
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(AdmissionError, match="R16 recipe/run/manifest/base contract mismatch"):
        validate_receipt(receipt, binding)


def test_r16_status_cannot_be_relabelled_as_legacy_recipe(tmp_path):
    binding, receipt, _, data = _r16_receipt(tmp_path)
    data["recipe"] = R15_RECIPE
    receipt.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(AdmissionError, match="R16 reload recipe differs"):
        validate_receipt(receipt, binding)


def test_r16_zero_step_never_admitted_as_trained(tmp_path):
    binding, receipt, _, data = _r16_receipt(tmp_path)
    data["new_optimizer_steps"] = 0
    receipt.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(AdmissionError, match="no actual trained adapter"):
        validate_receipt(receipt, binding)


@pytest.mark.parametrize("tamper_optimizer", [False, True])
def test_r16_legacy_status_downgrade_never_bypasses_checkpoint_custody(
    tmp_path, tamper_optimizer,
):
    binding, receipt_path, checkpoint, receipt = _r16_receipt(tmp_path)
    receipt["status"] = "ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    if tamper_optimizer:
        (checkpoint / "optimizer.pt").write_bytes(b"tampered-optimizer")
    with pytest.raises(AdmissionError, match="R16.*legacy"):
        validate_receipt(receipt_path, binding)


def test_r16_downgrade_cannot_strip_fields_while_sidecar_remains(tmp_path):
    binding, receipt_path, _, receipt = _r16_receipt(tmp_path)
    receipt["status"] = "ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED"
    receipt["recipe"] = "TRAININGDAY_QWEN35_FRESH_QLORA_R15_NUMERIC_MATCHED"
    for key in (
        "r16_run_contract_sha256", "resumed_from_step",
        "optimizer_resume_equivalence", "checkpoint_step",
        "checkpoint_custody_file_sha256", "checkpoint_custody_state",
    ):
        receipt.pop(key)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(AdmissionError, match="R16.*legacy"):
        validate_receipt(receipt_path, binding)


def test_r16_checkpoint_directory_itself_blocks_legacy_downgrade(tmp_path):
    binding, receipt_path, _, receipt = _r16_receipt(tmp_path)
    receipt["status"] = "ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED"
    receipt["recipe"] = "TRAININGDAY_QWEN35_FRESH_QLORA_R15_NUMERIC_MATCHED"
    for key in (
        "r16_run_contract_sha256", "resumed_from_step",
        "optimizer_resume_equivalence", "checkpoint_step",
        "checkpoint_custody_file_sha256", "checkpoint_custody_state",
    ):
        receipt.pop(key)
    for name in ("r16-run-contract.json", "latest-checkpoint-custody.json"):
        (receipt_path.parent / name).unlink()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(AdmissionError, match="R16.*legacy"):
        validate_receipt(receipt_path, binding)


def test_r16_recipe_marker_without_sidecars_cannot_use_legacy_status(tmp_path):
    binding, receipt_path, _, receipt = _receipt(tmp_path)
    receipt["recipe"] = "TRAININGDAY_QWEN35_QLORA_R16_CHECKPOINTABLE_MATCHED"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(AdmissionError, match="R16.*legacy"):
        validate_receipt(receipt_path, binding)


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
