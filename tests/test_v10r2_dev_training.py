from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from successor.experiments import train_v10r2_dev as trainer_module

from successor.experiments.train_v10r2_dev import (
    DevTrainingHold,
    effect_label,
    load_and_validate_dev_spec,
    summarize_optimizer,
    verify_resume_adapter,
)


def _write_spec(path: Path, *, optimizer: str = "adamw_bnb_8bit") -> Path:
    value = {
        "schema": "V10R2_QWEN35_DEV_TRAINING_SPEC_V1",
        "status": "AUTHORIZED_LOCAL_DEVELOPMENT_TRAINING",
        "authority": {
            "actor_id": "PATRICK_USER_AUTHORITY",
            "source": "EXPLICIT_CURRENT_USER_INSTRUCTION",
            "scope": "CHANGE_TRAINING_METHOD_AND_CONTINUE_LOCAL_MODEL_TRAINING",
            "paid_compute_authorized": False,
            "merge_authorized": False,
            "deploy_activate_authorized": False,
        },
        "source_subject": {
            "parent_revision": "a" * 40,
            "training_corpus_id": "corpus",
            "train_sha256": "b" * 64,
            "train_rows": 50000,
            "runtime_binding_sha256": "c" * 64,
        },
        "trainer": {
            "max_optimizer_steps": 1,
            "optimizer": optimizer,
            "gradient_accumulation_steps": 8,
            "per_device_train_batch_size": 1,
            "max_length": 512,
            "learning_rate": 2e-5,
            "lr_scheduler_type": "cosine",
            "warmup_optimizer_steps": 0,
            "completion_only_loss": True,
            "packing": False,
            "shuffle_dataset": True,
            "gradient_checkpointing": True,
            "bf16": True,
            "tf32": True,
            "gradient_checkpointing_use_reentrant": False,
            "logging_steps": 1,
            "save_strategy": "no",
            "eval_strategy": "no",
            "report_to": "none",
            "seed": 20261001,
        },
        "quantization": {
            "load_in_4bit": True,
            "type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
        },
        "lora": {
            "r": 4,
            "alpha": 16,
            "dropout": 0,
            "target_modules": "all-linear",
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "model_load": {
            "device_map": {"": 0},
            "dtype": "bfloat16",
            "use_cache": False,
            "prepare_model_for_kbit_training_use_gradient_checkpointing": True,
        },
        "output": {
            "namespace": str(path.parent / "adapter"),
            "save_final_adapter": True,
            "overwrite_forbidden": True,
        },
        "qualification": {
            "external_final_bank_required_for_qualification": True,
            "development_training_may_precede_external_qualification": True,
            "claim_ceiling": "LOCAL_DEVELOPMENT_ADAPTER_ONLY_NOT_EXTERNALLY_QUALIFIED",
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_dev_spec_accepts_nonpaged_bnb_8bit(tmp_path: Path) -> None:
    spec = load_and_validate_dev_spec(_write_spec(tmp_path / "spec.json"))
    assert spec["trainer"]["optimizer"] == "adamw_bnb_8bit"
    assert spec["trainer"]["max_optimizer_steps"] == 1


def test_dev_spec_accepts_nonpaged_torchao_8bit(tmp_path: Path) -> None:
    path = _write_spec(tmp_path / "spec.json", optimizer="adamw_torch_8bit")
    spec = load_and_validate_dev_spec(path)
    assert spec["trainer"]["optimizer"] == "adamw_torch_8bit"


def test_dev_spec_accepts_four_step_staged_window(tmp_path: Path) -> None:
    path = _write_spec(tmp_path / "spec.json")
    value = json.loads(path.read_text())
    value["trainer"]["max_optimizer_steps"] = 4
    value["development_window"] = {"start_row": 0, "row_count": 32}
    path.write_text(json.dumps(value), encoding="utf-8")
    spec = load_and_validate_dev_spec(path)
    assert spec["trainer"]["max_optimizer_steps"] == 4
    assert spec["development_window"]["row_count"] == 32


def test_dev_spec_rejects_misaligned_staged_window(tmp_path: Path) -> None:
    path = _write_spec(tmp_path / "spec.json")
    value = json.loads(path.read_text())
    value["trainer"]["max_optimizer_steps"] = 4
    value["development_window"] = {"start_row": 0, "row_count": 24}
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DevTrainingHold, match="staged optimizer window"):
        load_and_validate_dev_spec(path)


def test_ordinary_dev_spec_rejects_twenty_steps(tmp_path: Path) -> None:
    path = _write_spec(tmp_path / "spec.json")
    value = json.loads(path.read_text())
    value["trainer"]["max_optimizer_steps"] = 20
    value["development_window"] = {"start_row": 0, "row_count": 160}
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DevTrainingHold, match="1 through 8"):
        load_and_validate_dev_spec(path)


def test_recipe_semantics_spec_accepts_fresh_continuous_twenty_steps(
    tmp_path: Path,
) -> None:
    path = _write_spec(tmp_path / "spec.json")
    value = json.loads(path.read_text())
    value["experiment_class"] = "CONTINUOUS_STATE_RECIPE_SEMANTICS"
    value["trainer"]["max_optimizer_steps"] = 20
    value["trainer"]["shuffle_dataset"] = False
    value["trainer"]["train_sampling_strategy"] = "sequential"
    value["development_window"] = {"start_row": 0, "row_count": 160}
    value["comparison"] = {
        "expected_initial_trainable_parameter_digest": "1" * 64,
    }
    path.write_text(json.dumps(value), encoding="utf-8")

    spec = load_and_validate_dev_spec(path)
    assert spec["experiment_class"] == "CONTINUOUS_STATE_RECIPE_SEMANTICS"
    assert spec["trainer"]["max_optimizer_steps"] == 20
    assert spec["development_window"] == {"start_row": 0, "row_count": 160}


def test_recipe_semantics_spec_rejects_missing_initial_digest(
    tmp_path: Path,
) -> None:
    path = _write_spec(tmp_path / "spec.json")
    value = json.loads(path.read_text())
    value["experiment_class"] = "CONTINUOUS_STATE_RECIPE_SEMANTICS"
    value["trainer"]["max_optimizer_steps"] = 20
    value["trainer"]["shuffle_dataset"] = False
    value["trainer"]["train_sampling_strategy"] = "sequential"
    value["development_window"] = {"start_row": 0, "row_count": 160}
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DevTrainingHold, match="expected initial trainable digest"):
        load_and_validate_dev_spec(path)


def test_recipe_semantics_spec_rejects_random_sampling(tmp_path: Path) -> None:
    path = _write_spec(tmp_path / "spec.json")
    value = json.loads(path.read_text())
    value["experiment_class"] = "CONTINUOUS_STATE_RECIPE_SEMANTICS"
    value["trainer"]["max_optimizer_steps"] = 20
    value["trainer"]["shuffle_dataset"] = False
    value["trainer"]["train_sampling_strategy"] = "random"
    value["development_window"] = {"start_row": 0, "row_count": 160}
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DevTrainingHold, match="sequential sampling"):
        load_and_validate_dev_spec(path)


def test_dev_spec_accepts_sequential_sampling_for_bounded_stage(
    tmp_path: Path,
) -> None:
    path = _write_spec(tmp_path / "spec.json")
    value = json.loads(path.read_text())
    value["trainer"]["max_optimizer_steps"] = 4
    value["trainer"]["shuffle_dataset"] = False
    value["trainer"]["train_sampling_strategy"] = "sequential"
    value["development_window"] = {"start_row": 0, "row_count": 32}
    path.write_text(json.dumps(value), encoding="utf-8")
    spec = load_and_validate_dev_spec(path)
    assert spec["trainer"]["train_sampling_strategy"] == "sequential"


def test_recipe_semantics_spec_rejects_resume_adapter(tmp_path: Path) -> None:
    path = _write_spec(tmp_path / "spec.json")
    value = json.loads(path.read_text())
    value["experiment_class"] = "CONTINUOUS_STATE_RECIPE_SEMANTICS"
    value["trainer"]["max_optimizer_steps"] = 20
    value["trainer"]["shuffle_dataset"] = False
    value["trainer"]["train_sampling_strategy"] = "sequential"
    value["development_window"] = {"start_row": 0, "row_count": 160}
    value["resume_adapter"] = {
        "path": str(tmp_path / "prior" / "adapter"),
        "adapter_model_sha256": "1" * 64,
        "source_receipt_path": "source-receipt.json",
        "source_receipt_sha256": "2" * 64,
        "source_spec_path": "source-spec.json",
        "source_spec_sha256": "4" * 64,
        "source_weight_digest_after": "3" * 64,
        "previous_optimizer_steps": 8,
        "source_cumulative_optimizer_steps": 8,
        "next_train_row": 0,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DevTrainingHold, match="fresh adapter"):
        load_and_validate_dev_spec(path)


def test_dev_spec_rejects_paged_optimizer(tmp_path: Path) -> None:
    path = _write_spec(tmp_path / "spec.json", optimizer="paged_adamw_8bit")
    with pytest.raises(DevTrainingHold, match="paged optimizers are forbidden"):
        load_and_validate_dev_spec(path)


def test_dev_spec_rejects_plain_adamw_torch(tmp_path: Path) -> None:
    path = _write_spec(tmp_path / "spec.json", optimizer="adamw_torch")
    with pytest.raises(DevTrainingHold, match="approved non-paged 8-bit"):
        load_and_validate_dev_spec(path)


def test_dev_spec_rejects_paid_compute_or_deploy_authority(tmp_path: Path) -> None:
    path = _write_spec(tmp_path / "spec.json")
    value = json.loads(path.read_text())
    value["authority"]["paid_compute_authorized"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DevTrainingHold, match="paid compute"):
        load_and_validate_dev_spec(path)


def test_dev_spec_requires_external_bank_for_qualification(tmp_path: Path) -> None:
    path = _write_spec(tmp_path / "spec.json")
    value = json.loads(path.read_text())
    value["qualification"]["external_final_bank_required_for_qualification"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DevTrainingHold, match="external final bank"):
        load_and_validate_dev_spec(path)


def test_optimizer_summary_identifies_nonpaged_bnb_8bit_optimizer() -> None:
    class Args:
        optim_bits = 8
        min_8bit_size = 4096

    class FakeOptimizer:
        is_paged = False
        args = Args()

    FakeOptimizer.__module__ = "bitsandbytes.optim.adamw"
    FakeOptimizer.__name__ = "AdamW"

    result = summarize_optimizer(FakeOptimizer())
    assert result["class"] == "AdamW"
    assert result["module"] == "bitsandbytes.optim.adamw"
    assert result["backend"] == "bitsandbytes"
    assert result["is_paged"] is False
    assert result["optim_bits"] == 8
    assert result["min_8bit_size"] == 4096



def test_optimizer_summary_unwraps_accelerate_wrapper() -> None:
    class Args:
        optim_bits = 8
        min_8bit_size = 4096

    class Inner:
        is_paged = False
        args = Args()

    Inner.__module__ = "bitsandbytes.optim.adamw"
    Inner.__name__ = "AdamW"

    class Wrapper:
        def __init__(self):
            self.optimizer = Inner()

    result = summarize_optimizer(Wrapper())
    assert result["class"] == "AdamW"
    assert result["module"] == "bitsandbytes.optim.adamw"
    assert result["is_paged"] is False
    assert result["optim_bits"] == 8
    assert result["wrapper_chain"] == ["Wrapper"]


def test_optimizer_summary_identifies_nonpaged_torchao_8bit_optimizer() -> None:
    class FakeOptimizer:
        block_size = 256

    FakeOptimizer.__module__ = "torchao.optim.adam"
    FakeOptimizer.__name__ = "AdamW8bit"

    result = summarize_optimizer(FakeOptimizer())
    assert result["class"] == "AdamW8bit"
    assert result["module"] == "torchao.optim.adam"
    assert result["backend"] == "torchao"
    assert result["is_paged"] is False
    assert result["optim_bits"] == 8


def test_effect_label_tracks_step_count_and_backend() -> None:
    assert effect_label("adamw_bnb_8bit", 1) == "ONE_LOCAL_BNB_ADAMW_8BIT_OPTIMIZER_STEP"
    assert effect_label("adamw_bnb_8bit", 4) == "FOUR_LOCAL_BNB_ADAMW_8BIT_OPTIMIZER_STEPS"
    assert effect_label("adamw_torch_8bit", 1) == "ONE_LOCAL_TORCHAO_ADAMW_8BIT_OPTIMIZER_STEP"
    assert effect_label("adamw_bnb_8bit", 20) == "TWENTY_LOCAL_BNB_ADAMW_8BIT_OPTIMIZER_STEPS"
    assert effect_label("adamw_bnb_8bit", 32) == "THIRTY_TWO_LOCAL_BNB_ADAMW_8BIT_OPTIMIZER_STEPS"


def test_effect_label_rejects_unknown_backend_or_count() -> None:
    with pytest.raises(DevTrainingHold, match="effect label"):
        effect_label("paged_adamw_8bit", 1)
    with pytest.raises(DevTrainingHold, match="effect label"):
        effect_label("adamw_bnb_8bit", 0)
    with pytest.raises(DevTrainingHold, match="effect label"):
        effect_label("adamw_bnb_8bit", 33)



def test_effect_label_is_checked_before_dataset_or_gpu_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = load_and_validate_dev_spec(_write_spec(tmp_path / "spec.json"))
    validated["trainer"]["max_optimizer_steps"] = 33
    monkeypatch.setattr(trainer_module, "load_and_validate_dev_spec", lambda _: validated)

    def forbidden_loader():
        raise AssertionError("GPU training stack must not initialize")

    with pytest.raises(DevTrainingHold, match="effect label step count"):
        trainer_module.execute_dev_training(
            tmp_path,
            train_jsonl=tmp_path / "unread-data.jsonl",
            spec_path=tmp_path / "spec.json",
            training_stack_loader=forbidden_loader,
        )

def test_continuation_spec_binds_prior_adapter_and_next_row(tmp_path: Path) -> None:
    path = _write_spec(tmp_path / "spec.json")
    value = json.loads(path.read_text())
    value["trainer"]["max_optimizer_steps"] = 8
    value["development_window"] = {"start_row": 32, "row_count": 64}
    value["resume_adapter"] = {
        "path": str(tmp_path / "prior" / "adapter"),
        "adapter_model_sha256": "1" * 64,
        "source_receipt_path": "source-receipt.json",
        "source_receipt_sha256": "2" * 64,
        "source_spec_path": "source-spec.json",
        "source_spec_sha256": "4" * 64,
        "source_weight_digest_after": "3" * 64,
        "previous_optimizer_steps": 4,
        "source_cumulative_optimizer_steps": 4,
        "next_train_row": 32,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    spec = load_and_validate_dev_spec(path)
    assert spec["resume_adapter"]["previous_optimizer_steps"] == 4
    assert spec["development_window"]["start_row"] == 32


def test_continuation_spec_rejects_row_reuse(tmp_path: Path) -> None:
    path = _write_spec(tmp_path / "spec.json")
    value = json.loads(path.read_text())
    value["trainer"]["max_optimizer_steps"] = 8
    value["development_window"] = {"start_row": 0, "row_count": 64}
    value["resume_adapter"] = {
        "path": str(tmp_path / "prior" / "adapter"),
        "adapter_model_sha256": "1" * 64,
        "source_receipt_path": "source-receipt.json",
        "source_receipt_sha256": "2" * 64,
        "source_spec_path": "source-spec.json",
        "source_spec_sha256": "4" * 64,
        "source_weight_digest_after": "3" * 64,
        "previous_optimizer_steps": 4,
        "source_cumulative_optimizer_steps": 4,
        "next_train_row": 32,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DevTrainingHold, match="next_train_row"):
        load_and_validate_dev_spec(path)


def test_continuation_spec_rejects_invalid_adapter_hash(tmp_path: Path) -> None:
    path = _write_spec(tmp_path / "spec.json")
    value = json.loads(path.read_text())
    value["resume_adapter"] = {
        "path": str(tmp_path / "prior" / "adapter"),
        "adapter_model_sha256": "bad",
        "source_receipt_path": "source-receipt.json",
        "source_receipt_sha256": "2" * 64,
        "source_spec_path": "source-spec.json",
        "source_spec_sha256": "4" * 64,
        "source_weight_digest_after": "3" * 64,
        "previous_optimizer_steps": 4,
        "source_cumulative_optimizer_steps": 4,
        "next_train_row": 0,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DevTrainingHold, match="adapter_model_sha256"):
        load_and_validate_dev_spec(path)



def _make_resume_fixture(tmp_path: Path) -> tuple[Path, dict]:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    model = adapter_dir / "adapter_model.safetensors"
    model.write_bytes(b"adapter-bytes")
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 4,
                "lora_alpha": 16,
                "lora_dropout": 0,
                "bias": "none",
            }
        ),
        encoding="utf-8",
    )

    source_spec = _write_spec(tmp_path / "source-spec.json")
    source_value = json.loads(source_spec.read_text())
    source_value["trainer"]["max_optimizer_steps"] = 4
    source_value["development_window"] = {"start_row": 0, "row_count": 32}
    source_spec.write_text(json.dumps(source_value), encoding="utf-8")
    source_spec_sha = hashlib.sha256(source_spec.read_bytes()).hexdigest()

    receipt = {
        "spec_sha256": source_spec_sha,
        "weight_digest_after": "3" * 64,
        "weight_digest_changed": True,
        "cumulative_optimizer_steps": 4,
    }
    receipt_sha = hashlib.sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    receipt["receipt_sha256"] = receipt_sha
    receipt_path = tmp_path / "source-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    resume = {
        "path": str(adapter_dir),
        "adapter_model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "source_receipt_path": str(receipt_path),
        "source_receipt_sha256": receipt_sha,
        "source_spec_path": str(source_spec),
        "source_spec_sha256": source_spec_sha,
        "source_weight_digest_after": "3" * 64,
        "previous_optimizer_steps": 4,
        "source_cumulative_optimizer_steps": 4,
        "next_train_row": 32,
    }
    return adapter_dir, resume


def test_verify_resume_adapter_binds_adapter_spec_and_receipt(tmp_path: Path) -> None:
    _, resume = _make_resume_fixture(tmp_path)
    result = verify_resume_adapter(tmp_path, resume)
    assert result["status"] == "RESUME_ADAPTER_VERIFIED"
    assert result["next_train_row"] == 32
    assert result["previous_optimizer_steps"] == 4
    assert result["source_cumulative_optimizer_steps"] == 4


def test_verify_resume_adapter_rejects_modified_adapter(tmp_path: Path) -> None:
    adapter_dir, resume = _make_resume_fixture(tmp_path)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(DevTrainingHold, match="adapter_model_sha256 mismatch"):
        verify_resume_adapter(tmp_path, resume)


def test_verify_resume_adapter_uses_source_cumulative_steps_for_nested_continuation(tmp_path: Path) -> None:
    adapter_dir, resume = _make_resume_fixture(tmp_path)
    source_spec_path = Path(resume["source_spec_path"])
    source_spec = json.loads(source_spec_path.read_text())
    source_spec["trainer"]["max_optimizer_steps"] = 8
    source_spec["development_window"] = {"start_row": 32, "row_count": 64}
    source_spec["resume_adapter"] = {
        "path": str(tmp_path / "earlier" / "adapter"),
        "adapter_model_sha256": "a" * 64,
        "source_receipt_path": str(tmp_path / "earlier-receipt.json"),
        "source_receipt_sha256": "b" * 64,
        "source_spec_path": str(tmp_path / "earlier-spec.json"),
        "source_spec_sha256": "c" * 64,
        "source_weight_digest_after": "d" * 64,
        "previous_optimizer_steps": 4,
        "source_cumulative_optimizer_steps": 4,
        "next_train_row": 32,
    }
    source_spec_path.write_text(json.dumps(source_spec), encoding="utf-8")
    source_spec_sha = hashlib.sha256(source_spec_path.read_bytes()).hexdigest()

    receipt_path = Path(resume["source_receipt_path"])
    receipt = {
        "spec_sha256": source_spec_sha,
        "weight_digest_after": "3" * 64,
        "weight_digest_changed": True,
        "cumulative_optimizer_steps": 12,
    }
    receipt_sha = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    receipt["receipt_sha256"] = receipt_sha
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    resume["source_spec_sha256"] = source_spec_sha
    resume["source_receipt_sha256"] = receipt_sha
    resume["previous_optimizer_steps"] = 8
    resume["source_cumulative_optimizer_steps"] = 12
    resume["next_train_row"] = 96

    result = verify_resume_adapter(tmp_path, resume)
    assert result["previous_optimizer_steps"] == 8
    assert result["source_cumulative_optimizer_steps"] == 12
    assert result["next_train_row"] == 96