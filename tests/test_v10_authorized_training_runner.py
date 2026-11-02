from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import successor.experiments.train_v10_qwen35_authorized as runner


def _sha(ch: str) -> str:
    return ch * 64


def _contract() -> dict:
    return {
        "schema": "VERA_SUCCESSOR_V10_QWEN35_EXPERIMENT_CONTRACT_V2",
        "experiment_id": "exp-v2",
        "base_model": {
            "repo": "rodrigomt/Qwen3.5-4B-Uncensored-Aggressive",
            "revision": "a" * 40,
            "parent_adapter": None,
            "fresh_adapter_required": True,
        },
        "source_subject": {
            "training_corpus_id": "corpus-v10",
            "train_rows": 2,
            "validation_rows": 1,
            "train_sha256": None,
            "validation_sha256": None,
        },
        "training_recipe": {
            "method": "QLORA_SFT_ONLY",
            "seed": 20261001,
            "epochs": 1.0,
            "learning_rate": 2e-5,
            "lr_scheduler_type": "cosine",
            "warmup_optimizer_steps": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_length": 512,
            "overflow_policy": "ERROR_NO_TRUNCATION",
            "optimizer": "adamw_torch",
            "completion_only_loss": True,
            "packing": False,
            "shuffle_dataset": True,
            "gradient_checkpointing": True,
            "quantization": {
                "load_in_4bit": True,
                "type": "nf4",
                "double_quant": True,
                "compute_dtype": "bfloat16",
            },
            "lora": {
                "r": 4,
                "alpha": 16,
                "dropout": 0.0,
                "target_modules": "all-linear",
            },
            "validation_role": (
                "POST_TRAIN_DIAGNOSTIC_ONLY_NO_RECIPE_OR_CHECKPOINT_SELECTION"
            ),
        },
    }


def _authority(output_namespace: str) -> dict:
    return {
        "schema": "V10_QWEN35_TRAINING_AUTHORITY_V1",
        "status": "AUTHORIZED",
        "authority_actor_id": "PATRICK_USER_AUTHORITY",
        "effect": "TRAIN_ONE_FRESH_QLORA_ADAPTER",
        "max_training_runs": 1,
        "output_namespace": output_namespace,
        "receipt_sha256": _sha("1"),
    }


def _runtime_binding() -> dict:
    return {
        "schema": "V10_QWEN35_TRAINING_RUNTIME_BINDING_V1",
        "status": "FROZEN_TARGET_RUNTIME_SUPPLEMENT",
        "binding_sha256": _sha("2"),
        "target": {
            "python_path": "python",
            "python_version": "3.12.10",
            "base_path": "base",
            "gpu": "gpu",
            "vram_mib": 4096,
            "driver": "driver",
            "cuda_runtime": "13.0",
            "cost_class": "LOCAL_ZERO_INCREMENTAL_COMPUTE_COST",
        },
        "packages": {
            "torch": "2.14.0+cu130",
            "transformers": "5.17.0",
            "trl": "1.13.0",
            "peft": "0.21.0",
            "bitsandbytes": "0.50.2",
            "datasets": "5.0.1",
            "accelerate": "1.15.0",
            "safetensors": "0.8.0",
            "huggingface_hub": "1.33.0",
            "tokenizers": "0.23.2",
            "jinja2": "3.1.6",
            "numpy": "2.5.3",
        },
        "base_artifacts": {
            "model.safetensors-00001-of-00002.safetensors": _sha("3"),
            "model.safetensors-00002-of-00002.safetensors": _sha("4"),
            "tokenizer.json": _sha("5"),
        },
    }


def test_hold_gate_does_not_load_training_stack() -> None:
    called = False

    def loader():
        nonlocal called
        called = True
        raise AssertionError("must not load")

    with pytest.raises(runner.TrainingHold):
        runner.gated_training_stack(
            {
                "status": "HOLD",
                "training_allowed": False,
                "reasons": ["no authority"],
            },
            loader,
        )
    assert called is False


def test_ready_gate_loads_stack_once() -> None:
    calls = []

    def loader():
        calls.append("loaded")
        return {"stack": True}

    stack = runner.gated_training_stack(
        {
            "status": "READY_PRECONDITIONS",
            "training_allowed": True,
            "reasons": [],
        },
        loader,
    )
    assert stack == {"stack": True}
    assert calls == ["loaded"]


def test_verified_jsonl_binds_exact_bytes_and_row_count(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_bytes(
        b'{"prompt":"a","response":"b"}\n'
        b'{"prompt":"c","response":"d"}\n'
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = runner.load_verified_jsonl(
        path,
        expected_sha256=digest,
        expected_rows=2,
    )
    assert len(rows) == 2
    with pytest.raises(runner.TrainingHold, match="hash mismatch"):
        runner.load_verified_jsonl(
            path,
            expected_sha256="0" * 64,
            expected_rows=2,
        )
    with pytest.raises(runner.TrainingHold, match="row count"):
        runner.load_verified_jsonl(
            path,
            expected_sha256=digest,
            expected_rows=3,
        )


def test_verified_jsonl_rejects_invalid_training_shape(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_bytes(b'{"prompt":"a"}\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(runner.TrainingHold, match="prompt/response"):
        runner.load_verified_jsonl(
            path,
            expected_sha256=digest,
            expected_rows=1,
        )


def test_output_namespace_must_match_authority_and_not_exist(
    tmp_path: Path,
) -> None:
    output = tmp_path / "authorized"
    authority = _authority(str(output))
    runner.validate_output_namespace(output, authority)
    with pytest.raises(runner.TrainingHold, match="namespace mismatch"):
        runner.validate_output_namespace(
            tmp_path / "other",
            authority,
        )
    output.mkdir()
    with pytest.raises(runner.TrainingHold, match="already exists"):
        runner.validate_output_namespace(output, authority)


def test_runtime_observation_requires_exact_bound_packages() -> None:
    binding = _runtime_binding()
    observed = {
        "python_version": "3.12.10",
        "gpu": "gpu",
        "vram_mib": 4096,
        "driver": "driver",
        "cuda_runtime": "13.0",
        "packages": dict(binding["packages"]),
        "base_artifacts": dict(binding["base_artifacts"]),
    }
    assert runner.validate_runtime_observation(binding, observed)["status"] == "PASS"
    observed["packages"]["bitsandbytes"] = "0.0.0"
    with pytest.raises(
        runner.TrainingHold,
        match="package mismatch:bitsandbytes",
    ):
        runner.validate_runtime_observation(binding, observed)


def test_runtime_observation_allows_one_mib_vram_reporting_rounding() -> None:
    binding = _runtime_binding()
    observed = {
        "python_version": "3.12.10",
        "gpu": "gpu",
        "vram_mib": 4095,
        "driver": "driver",
        "cuda_runtime": "13.0",
        "packages": dict(binding["packages"]),
        "base_artifacts": dict(binding["base_artifacts"]),
    }

    result = runner.validate_runtime_observation(binding, observed)
    assert result["status"] == "PASS"
    assert result["vram_delta_mib"] == 1


def test_runtime_observation_rejects_material_vram_difference() -> None:
    binding = _runtime_binding()
    observed = {
        "python_version": "3.12.10",
        "gpu": "gpu",
        "vram_mib": 4080,
        "driver": "driver",
        "cuda_runtime": "13.0",
        "packages": dict(binding["packages"]),
        "base_artifacts": dict(binding["base_artifacts"]),
    }

    with pytest.raises(
        runner.TrainingHold,
        match="runtime target mismatch:vram_mib:4080!=4096",
    ):
        runner.validate_runtime_observation(binding, observed)


def test_execution_spec_matches_frozen_recipe() -> None:
    spec = runner.default_execution_spec()
    result = runner.validate_execution_spec(_contract(), spec)
    assert result["status"] == "PASS"
    changed = json.loads(json.dumps(spec))
    changed["trainer"]["gradient_accumulation_steps"] = 7
    with pytest.raises(
        runner.TrainingHold,
        match="gradient_accumulation_steps",
    ):
        runner.validate_execution_spec(_contract(), changed)


def test_start_receipt_is_exclusive_and_self_hashed(tmp_path: Path) -> None:
    output = tmp_path / "run"
    authority = _authority(str(output))
    receipt = runner.write_run_start_receipt(
        output,
        authority=authority,
        contract_sha256=_sha("6"),
        execution_spec_sha256=_sha("7"),
        runtime_binding_sha256=_sha("2"),
        sealed_commitment_sha256=_sha("8"),
        train_sha256=_sha("9"),
        validation_sha256=_sha("a"),
        started_at="2026-10-02T00:00:00Z",
    )
    receipt_path = output / "RUN_STARTED.json"
    stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert stored == receipt
    unsigned = dict(stored)
    claimed = unsigned.pop("receipt_sha256")
    expected = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert claimed == expected
    with pytest.raises(runner.TrainingHold, match="already exists"):
        runner.write_run_start_receipt(
            output,
            authority=authority,
            contract_sha256=_sha("6"),
            execution_spec_sha256=_sha("7"),
            runtime_binding_sha256=_sha("2"),
            sealed_commitment_sha256=_sha("8"),
            train_sha256=_sha("9"),
            validation_sha256=_sha("a"),
            started_at="2026-10-02T00:00:01Z",
        )


def test_plan_only_ready_claim_never_loads_training_stack(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        runner,
        "assess_preflight",
        lambda root: {
            "status": "READY_PRECONDITIONS",
            "training_allowed": True,
            "reasons": [],
            "training_authority": {"status": "LOCALLY_CLAIMED"},
        },
    )
    called = False

    def loader():
        nonlocal called
        called = True
        return {}

    result = runner.plan_execution(
        tmp_path,
        training_stack_loader=loader,
    )
    assert result["status"] == "HOLD"
    assert called is False



class _FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return "USER " + messages[0]["content"] + " ASSISTANT"

    def __call__(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": text.split()}


def test_prepare_sft_rows_enforces_no_truncation_budget() -> None:
    tok = _FakeTokenizer()
    rows = [{"prompt": "short", "response": "answer"}]
    prepared, report = runner.prepare_sft_rows(
        tok,
        rows,
        max_length=16,
    )
    assert len(prepared) == 1
    assert report["over_budget"] == 0

    long_rows = [
        {
            "prompt": " ".join(["p"] * 10),
            "response": " ".join(["r"] * 10),
        }
    ]
    with pytest.raises(runner.TrainingHold, match="token budget exceeded"):
        runner.prepare_sft_rows(
            tok,
            long_rows,
            max_length=8,
        )


def test_lora_target_coverage_requires_all_three_families() -> None:
    good = [
        "model.layers.0.linear_attn.in_proj_qkv",
        "model.layers.1.self_attn.q_proj",
        "model.layers.2.mlp.gate_proj",
    ]
    result = runner.validate_lora_target_coverage(good)
    assert result["status"] == "PASS"
    with pytest.raises(runner.TrainingHold, match="LoRA coverage incomplete"):
        runner.validate_lora_target_coverage(good[:2])


def test_execute_hold_never_loads_stack_or_touches_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner,
        "assess_preflight",
        lambda root: {
            "status": "READY_PRECONDITIONS",
            "training_allowed": True,
            "reasons": [],
            "training_authority": {"status": "LOCALLY_CLAIMED"},
        },
    )
    called = False

    def loader():
        nonlocal called
        called = True
        raise AssertionError("training stack must not load")

    output = tmp_path / "run"
    # Even a locally forged READY preflight could not cross the physical gate.
    with pytest.raises(runner.TrainingHold, match="HARD HOLD"):
        runner.execute_authorized_training(
            tmp_path,
            train_jsonl=tmp_path / "missing-train.jsonl",
            validation_jsonl=tmp_path / "missing-validation.jsonl",
            output_dir=output,
            training_stack_loader=loader,
        )
    assert called is False
    assert not output.exists()



def test_execution_spec_freezes_material_runtime_knobs() -> None:
    spec = runner.default_execution_spec()
    assert spec["trainer"]["bf16"] is True
    assert spec["trainer"]["tf32"] is True
    assert spec["trainer"]["gradient_checkpointing_use_reentrant"] is False
    assert spec["trainer"]["logging_steps"] == 10
    assert spec["trainer"]["save_strategy"] == "no"
    assert spec["trainer"]["eval_strategy"] == "no"
    assert spec["trainer"]["report_to"] == "none"
    assert spec["model_load"] == {
        "device_map": {"": 0},
        "dtype": "bfloat16",
        "use_cache": False,
        "prepare_model_for_kbit_training_use_gradient_checkpointing": True,
    }
    assert spec["lora"]["bias"] == "none"
    assert spec["lora"]["task_type"] == "CAUSAL_LM"

    for path, value in (
        (("trainer", "tf32"), False),
        (("trainer", "gradient_checkpointing_use_reentrant"), True),
        (("model_load", "dtype"), "float16"),
        (("lora", "bias"), "all"),
    ):
        changed = json.loads(json.dumps(spec))
        changed[path[0]][path[1]] = value
        with pytest.raises(runner.TrainingHold):
            runner.validate_execution_spec(_contract(), changed)
