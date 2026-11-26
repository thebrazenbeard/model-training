from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "successor" / "evaluate_successor.py"


def _write_json(root: Path, relative: str, value: dict) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _current_hold_fixture(root: Path) -> None:
    _write_json(
        root,
        "successor/experiments/V10_QWEN35_EXPERIMENT_CONTRACT_V1.json",
        {
            "schema": "VERA_SUCCESSOR_V10_QWEN35_EXPERIMENT_CONTRACT_V1",
            "status": "PREREGISTERED_BLOCKED",
            "evaluation_bank": {
                "behavioral": {"sha256": None},
                "adversarial": {"sha256": None},
                "retention": {"sha256": None},
            },
            "blocking_preconditions": {
                "fresh_evaluation_bank_frozen": False,
                "independent_bank_admission_verified": False,
                "contamination_screen_against_v10_and_consumed_finals_verified": False,
                "qwen_token_budget_no_overflow_verified": False,
                "zero_cost_execution_target_bound": False,
                "exact_training_runtime_versions_bound": False,
                "patrick_exact_weight_change_authority": False,
            },
        },
    )
    _write_json(
        root,
        "successor/experiments/V10_QWEN35_FINAL_BANK_ADMISSION_V1.json",
        {
            "schema": "V10_QWEN35_FINAL_BANK_ADMISSION_V1",
            "status": "SPEC_FROZEN_NO_CASES_ADMITTED",
        },
    )
    _write_json(
        root,
        "successor/experiments/V10_QWEN35_EXCLUSION_REGISTRY_V1.json",
        {
            "schema": "V10_QWEN35_EXCLUSION_REGISTRY_V1",
            "status": "FROZEN_EXCLUSION_IDENTITIES",
        },
    )


def test_v10_preflight_cli_fails_closed_on_current_hold_state(tmp_path: Path) -> None:
    _current_hold_fixture(tmp_path)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--v10-preflight-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["schema"] == "V10_QWEN35_PREFLIGHT_V1"
    assert payload["status"] == "HOLD"
    assert payload["training_allowed"] is False
    assert "fresh_evaluation_bank_frozen" in payload["reasons"]
    assert "patrick_exact_weight_change_authority" in payload["reasons"]


def test_v10_preflight_rejects_true_bank_flag_without_frozen_hashes(tmp_path: Path) -> None:
    _current_hold_fixture(tmp_path)
    contract_path = (
        tmp_path
        / "successor/experiments/V10_QWEN35_EXPERIMENT_CONTRACT_V1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    for key in contract["blocking_preconditions"]:
        contract["blocking_preconditions"][key] = True
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--v10-preflight-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "HOLD"
    assert "fresh_evaluation_bank_hashes_missing" in payload["reasons"]


def test_v10_preflight_prefers_v2_contract_when_present(tmp_path: Path) -> None:
    _current_hold_fixture(tmp_path)
    _write_json(
        tmp_path,
        "successor/experiments/V10_QWEN35_EXPERIMENT_CONTRACT_V2.json",
        {
            "schema": "VERA_SUCCESSOR_V10_QWEN35_EXPERIMENT_CONTRACT_V2",
            "status": "PREREGISTERED_BLOCKED",
            "evaluation_bank": {
                "behavioral": {"sha256": None},
                "adversarial": {"sha256": None},
                "retention": {"sha256": None},
            },
            "blocking_preconditions": {
                "fresh_evaluation_bank_frozen": False,
                "independent_bank_admission_verified": False,
                "semantic_contamination_screen_verified": False,
                "qwen_token_budget_no_overflow_verified": True,
                "zero_cost_execution_target_bound": True,
                "exact_training_runtime_versions_bound": True,
                "patrick_exact_weight_change_authority": False,
            },
        },
    )
    _write_json(
        tmp_path,
        "successor/experiments/V10_QWEN35_FINAL_BANK_ADMISSION_V2.json",
        {
            "schema": "V10_QWEN35_FINAL_BANK_ADMISSION_V2",
            "status": "SPEC_FROZEN_NO_CASES_ADMITTED",
        },
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--v10-preflight-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert "qwen_token_budget_no_overflow_verified" not in payload["reasons"]
    assert "final_bank_cases_not_admitted" in payload["reasons"]
    assert "semantic_contamination_screen_verified" in payload["reasons"]



def _v2_sealed_hold_fixture(root: Path) -> None:
    _current_hold_fixture(root)
    _write_json(
        root,
        "successor/experiments/V10_QWEN35_EXPERIMENT_CONTRACT_V2.json",
        {
            "schema": "VERA_SUCCESSOR_V10_QWEN35_EXPERIMENT_CONTRACT_V2",
            "status": "PREREGISTERED_BLOCKED",
            "experiment_id": "experiment-v2-test",
            "base_model": {
                "repo": "rodrigomt/Qwen3.5-4B-Uncensored-Aggressive",
                "revision": "a" * 40,
                "parent_adapter": None,
                "fresh_adapter_required": True,
                "mutable_revision_allowed": False,
            },
            "source_subject": {
                "training_corpus_id": "corpus-v10-qwen512",
                "train_sha256": "8" * 64,
                "validation_sha256": "9" * 64,
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
                    "POST_TRAIN_DIAGNOSTIC_ONLY_"
                    "NO_RECIPE_OR_CHECKPOINT_SELECTION"
                ),
                "runtime_target": {
                    "base_path": r"D:\\VERA\\models\\latest-trained\\base",
                    "cost_class": "LOCAL_ZERO_INCREMENTAL_COMPUTE_COST",
                    "driver": "616.92",
                    "gpu": "NVIDIA GeForce RTX 3050 Laptop GPU",
                    "vram_mib": 4096,
                },
                "runtime_versions": {
                    "python": "3.12.10",
                    "torch": "2.14.0+cu130",
                    "transformers": "5.17.0",
                    "trl": "1.13.0",
                    "peft": "0.21.0",
                },
                "token_preflight": {
                    "file_sha256": "d" * 64,
                },
            },
            "evaluation_bank": {
                "behavioral": {"sha256": None},
                "adversarial": {"sha256": None},
                "retention": {"sha256": None},
            },
            "blocking_preconditions": {
                "fresh_evaluation_bank_frozen": False,
                "independent_bank_admission_verified": False,
                "semantic_contamination_screen_verified": False,
                "qwen_token_budget_no_overflow_verified": True,
                "zero_cost_execution_target_bound": True,
                "exact_training_runtime_versions_bound": True,
                "patrick_exact_weight_change_authority": False,
            },
        },
    )
    _write_json(
        root,
        "successor/experiments/V10_QWEN35_FINAL_BANK_ADMISSION_V2.json",
        {
            "schema": "V10_QWEN35_FINAL_BANK_ADMISSION_V2",
            "status": "SPEC_FROZEN_NO_CASES_ADMITTED",
        },
    )


    _write_json(
        root,
        "successor/experiments/V10_QWEN35_TRAINING_RUNTIME_BINDING_V1.json",
        _runtime_binding(),
    )
    execution_binding, runner_raw, execution_spec = _test_execution_binding()
    runner_path = (
        root
        / "successor/experiments/train_v10_qwen35_authorized.py"
    )
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    runner_path.write_bytes(runner_raw)
    evaluator_path = root / "successor/evaluate_successor.py"
    evaluator_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator_path.write_bytes(TEST_PREFLIGHT_RAW)
    _write_json(
        root,
        "successor/experiments/V10_QWEN35_TRAINING_EXECUTION_SPEC_V1.json",
        execution_spec,
    )
    _write_json(
        root,
        "successor/experiments/V10_QWEN35_TRAINING_EXECUTION_BINDING_V1.json",
        execution_binding,
    )


def _runtime_binding() -> dict:
    value = {
        "schema": "V10_QWEN35_TRAINING_RUNTIME_BINDING_V1",
        "date": "2026-10-01",
        "status": "FROZEN_TARGET_RUNTIME_SUPPLEMENT",
        "target": {
            "python_path": r"C:\\ProgramData\\ProRun\\model-env\\Scripts\\python.exe",
            "python_version": "3.12.10",
            "base_path": r"D:\\VERA\\models\\latest-trained\\base",
            "gpu": "NVIDIA GeForce RTX 3050 Laptop GPU",
            "vram_mib": 4096,
            "driver": "616.92",
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
            "model.safetensors-00001-of-00002.safetensors": "a" * 64,
            "model.safetensors-00002-of-00002.safetensors": "b" * 64,
            "tokenizer.json": "c" * 64,
        },
        "provenance": {
            "token_preflight_path": "successor/corpus/v10_qwen512/token_preflight.json",
            "token_preflight_file_sha256": "d" * 64,
        },
        "claim_ceiling": "TEST_RUNTIME_BINDING",
    }
    value["binding_sha256"] = hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return value


TEST_PREFLIGHT_RAW = b"# bound test preflight evaluator\n"


def _execution_spec() -> dict:
    return {
        "schema": "V10_QWEN35_TRAINING_EXECUTION_SPEC_V1",
        "effect": "TRAIN_ONE_FRESH_QLORA_ADAPTER",
        "trainer": {
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
            "bf16": True,
            "tf32": True,
            "gradient_checkpointing_use_reentrant": False,
            "logging_steps": 10,
            "save_strategy": "no",
            "eval_strategy": "no",
            "report_to": "none",
            "validation_role": (
                "POST_TRAIN_DIAGNOSTIC_ONLY_NO_RECIPE_OR_CHECKPOINT_SELECTION"
            ),
        },
        "quantization": {
            "load_in_4bit": True,
            "type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
        },
        "model_load": {
            "device_map": {"": 0},
            "dtype": "bfloat16",
            "use_cache": False,
            "prepare_model_for_kbit_training_use_gradient_checkpointing": True,
        },
        "lora": {
            "r": 4,
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": "all-linear",
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "artifact_policy": {
            "fresh_adapter_only": True,
            "parent_adapter": None,
            "save_strategy": "final_adapter_only",
            "validation_checkpoint_selection": False,
        },
    }


def _git_blob_sha(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def _test_execution_binding() -> tuple[dict, bytes, dict]:
    runner_raw = b"# bound test runner\n"
    spec = _execution_spec()
    spec_sha = hashlib.sha256(
        json.dumps(
            spec,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    value = {
        "schema": "V10_QWEN35_TRAINING_EXECUTION_BINDING_V1",
        "status": "FROZEN_AUTHORIZABLE_EXECUTION_SUBJECT",
        "runner": {
            "path": (
                "successor/experiments/"
                "train_v10_qwen35_authorized.py"
            ),
            "git_blob_sha": _git_blob_sha(runner_raw),
        },
        "preflight_evaluator": {
            "path": "successor/evaluate_successor.py",
            "git_blob_sha": _git_blob_sha(TEST_PREFLIGHT_RAW),
        },
        "execution_spec": {
            "path": (
                "successor/experiments/"
                "V10_QWEN35_TRAINING_EXECUTION_SPEC_V1.json"
            ),
            "sha256": spec_sha,
        },
    }
    value["binding_sha256"] = hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return value, runner_raw, spec


def _sealed_commitment() -> dict:
    dimensions = [f"H{i:02d}" for i in range(1, 21)]
    value = {
        "schema": "V10_SEALED_FINAL_BANK_COMMITMENT_V1",
        "bank_id": "sealed-bank-v1",
        "status": "SEALED_PRETRAINING_FINAL_BANK",
        "lane_counts": {
            "behavioral": 10000,
            "adversarial": 2000,
            "retention": 1500,
        },
        "behavioral_dimension_counts": {
            dimension: 500 for dimension in dimensions
        },
        "adversarial_dimension_counts": {
            dimension: 100 for dimension in dimensions
        },
        "behavioral_family_counts": {
            dimension: 50 for dimension in dimensions
        },
        "plaintext_artifacts": {
            "bank_sha256": "1" * 64,
            "grader_bundle_sha256": "2" * 64,
            "review_bundle_sha256": "3" * 64,
            "source_manifest_sha256": "4" * 64,
            "contamination_receipt_sha256": "5" * 64,
            "sealed_archive_sha256": "6" * 64,
        },
        "custody": {
            "plaintext_exposed_to_training_lane": False,
            "custodian_ids": ["cust-a", "cust-b", "cust-h"],
            "human_reviewer_id": "review-h",
            "training_lane_receives_hashes_only": True,
        },
        "admission": {
            "exact_normalized_exclusion": "PASS",
            "semantic_contamination": "PASS",
            "independent_review": "PASS",
            "bank_frozen": True,
            "post_freeze_case_mutation": False,
        },
        "freeze_subject_digest": "7" * 64,
    }
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    value["commitment_sha256"] = hashlib.sha256(payload).hexdigest()
    return value


def test_valid_sealed_commitment_clears_only_bank_evidence_blockers(
    tmp_path: Path,
) -> None:
    _v2_sealed_hold_fixture(tmp_path)
    commitment = _sealed_commitment()
    _write_json(
        tmp_path,
        "successor/experiments/V10_SEALED_FINAL_BANK_COMMITMENT_V1.json",
        commitment,
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--v10-preflight-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "HOLD"
    assert payload["training_allowed"] is False
    assert payload["reasons"] == ["patrick_exact_weight_change_authority"]
    assert payload["sealed_final_bank"]["status"] == "VERIFIED"
    assert payload["sealed_final_bank"]["bank_id"] == "sealed-bank-v1"
    assert (
        payload["sealed_final_bank"]["commitment_sha256"]
        == commitment["commitment_sha256"]
    )
    assert payload["derived_preconditions"] == {
        "fresh_evaluation_bank_frozen": True,
        "independent_bank_admission_verified": True,
        "semantic_contamination_screen_verified": True,
    }


def test_sealed_commitment_self_hash_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    _v2_sealed_hold_fixture(tmp_path)
    commitment = _sealed_commitment()
    commitment["commitment_sha256"] = "0" * 64
    _write_json(
        tmp_path,
        "successor/experiments/V10_SEALED_FINAL_BANK_COMMITMENT_V1.json",
        commitment,
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--v10-preflight-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["status"] == "HOLD"
    assert "sealed_commitment_sha256_mismatch" in payload["reasons"]
    assert "fresh_evaluation_bank_frozen" in payload["reasons"]
    assert payload["sealed_final_bank"]["status"] == "INVALID"


def test_sealed_commitment_with_plaintext_field_fails_closed(
    tmp_path: Path,
) -> None:
    _v2_sealed_hold_fixture(tmp_path)
    commitment = _sealed_commitment()
    commitment["prompt"] = "secret final prompt"
    unsigned = dict(commitment)
    unsigned.pop("commitment_sha256")
    commitment["commitment_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    _write_json(
        tmp_path,
        "successor/experiments/V10_SEALED_FINAL_BANK_COMMITMENT_V1.json",
        commitment,
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--v10-preflight-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["status"] == "HOLD"
    assert any(
        reason.startswith("sealed_commitment:plaintext_field_present:")
        for reason in payload["reasons"]
    )
    assert "independent_bank_admission_verified" in payload["reasons"]



def _training_authority_receipt(contract: dict, commitment: dict) -> dict:
    recipe_sha = hashlib.sha256(
        json.dumps(
            contract["training_recipe"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    value = {
        "schema": "V10_QWEN35_TRAINING_AUTHORITY_V1",
        "status": "AUTHORIZED",
        "authority_actor_id": "PATRICK_USER_AUTHORITY",
        "authorization_source": "EXPLICIT_CURRENT_USER_INSTRUCTION",
        "experiment_id": contract["experiment_id"],
        "effect": "TRAIN_ONE_FRESH_QLORA_ADAPTER",
        "base_model_revision": contract["base_model"]["revision"],
        "training_corpus_id": contract["source_subject"]["training_corpus_id"],
        "train_sha256": contract["source_subject"]["train_sha256"],
        "validation_sha256": contract["source_subject"]["validation_sha256"],
        "training_recipe_sha256": recipe_sha,
        "sealed_final_bank_commitment_sha256": commitment[
            "commitment_sha256"
        ],
        "training_runtime_binding_sha256": _runtime_binding()[
            "binding_sha256"
        ],
        "training_execution_binding_sha256": _test_execution_binding()[0][
            "binding_sha256"
        ],
        "max_training_runs": 1,
        "output_namespace": "successor/artifacts/test-authorized-run-v1",
        "paid_compute_authorized": False,
        "merge_authorized": False,
        "install_activate_deploy_authorized": False,
    }
    value["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return value


def test_boolean_authority_flag_cannot_replace_exact_authority_receipt(
    tmp_path: Path,
) -> None:
    _v2_sealed_hold_fixture(tmp_path)
    contract_path = (
        tmp_path
        / "successor/experiments/V10_QWEN35_EXPERIMENT_CONTRACT_V2.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["blocking_preconditions"][
        "patrick_exact_weight_change_authority"
    ] = True
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    commitment = _sealed_commitment()
    _write_json(
        tmp_path,
        "successor/experiments/V10_SEALED_FINAL_BANK_COMMITMENT_V1.json",
        commitment,
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--v10-preflight-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["status"] == "HOLD"
    assert "patrick_exact_weight_change_authority" in payload["reasons"]
    assert payload["training_authority"]["status"] == "ABSENT"


def test_exact_authority_receipt_can_clear_only_final_authority_gate(
    tmp_path: Path,
) -> None:
    _v2_sealed_hold_fixture(tmp_path)
    contract_path = (
        tmp_path
        / "successor/experiments/V10_QWEN35_EXPERIMENT_CONTRACT_V2.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    commitment = _sealed_commitment()
    _write_json(
        tmp_path,
        "successor/experiments/V10_SEALED_FINAL_BANK_COMMITMENT_V1.json",
        commitment,
    )
    authority = _training_authority_receipt(contract, commitment)
    _write_json(
        tmp_path,
        "successor/experiments/V10_QWEN35_TRAINING_AUTHORITY_V1.json",
        authority,
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--v10-preflight-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["status"] == "READY_PRECONDITIONS"
    assert payload["training_allowed"] is True
    assert payload["reasons"] == []
    assert payload["training_authority"]["status"] == "VERIFIED"
    assert (
        payload["training_authority"]["receipt_sha256"]
        == authority["receipt_sha256"]
    )


def test_authority_receipt_for_wrong_bank_fails_closed(
    tmp_path: Path,
) -> None:
    _v2_sealed_hold_fixture(tmp_path)
    contract_path = (
        tmp_path
        / "successor/experiments/V10_QWEN35_EXPERIMENT_CONTRACT_V2.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    commitment = _sealed_commitment()
    _write_json(
        tmp_path,
        "successor/experiments/V10_SEALED_FINAL_BANK_COMMITMENT_V1.json",
        commitment,
    )
    authority = _training_authority_receipt(contract, commitment)
    authority["sealed_final_bank_commitment_sha256"] = "f" * 64
    unsigned = dict(authority)
    unsigned.pop("receipt_sha256")
    authority["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    _write_json(
        tmp_path,
        "successor/experiments/V10_QWEN35_TRAINING_AUTHORITY_V1.json",
        authority,
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--v10-preflight-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["status"] == "HOLD"
    assert "training_authority:sealed_commitment_sha256_mismatch" in payload[
        "reasons"
    ]
    assert "patrick_exact_weight_change_authority" in payload["reasons"]



def test_runtime_boolean_cannot_replace_exact_runtime_binding(
    tmp_path: Path,
) -> None:
    _v2_sealed_hold_fixture(tmp_path)
    runtime_path = (
        tmp_path
        / "successor/experiments/V10_QWEN35_TRAINING_RUNTIME_BINDING_V1.json"
    )
    runtime_path.unlink()
    commitment = _sealed_commitment()
    _write_json(
        tmp_path,
        "successor/experiments/V10_SEALED_FINAL_BANK_COMMITMENT_V1.json",
        commitment,
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--v10-preflight-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert "exact_training_runtime_versions_bound" in payload["reasons"]
    assert payload["training_runtime"]["status"] == "ABSENT"


def test_runtime_binding_core_package_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    _v2_sealed_hold_fixture(tmp_path)
    runtime_path = (
        tmp_path
        / "successor/experiments/V10_QWEN35_TRAINING_RUNTIME_BINDING_V1.json"
    )
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["packages"]["torch"] = "0.0.0"
    unsigned = dict(runtime)
    unsigned.pop("binding_sha256")
    runtime["binding_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    commitment = _sealed_commitment()
    _write_json(
        tmp_path,
        "successor/experiments/V10_SEALED_FINAL_BANK_COMMITMENT_V1.json",
        commitment,
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--v10-preflight-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert "training_runtime:package_version_mismatch:torch" in payload[
        "reasons"
    ]
    assert "exact_training_runtime_versions_bound" in payload["reasons"]
    assert payload["training_runtime"]["status"] == "INVALID"


def test_authority_receipt_for_wrong_runtime_binding_fails_closed(
    tmp_path: Path,
) -> None:
    _v2_sealed_hold_fixture(tmp_path)
    contract_path = (
        tmp_path
        / "successor/experiments/V10_QWEN35_EXPERIMENT_CONTRACT_V2.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    commitment = _sealed_commitment()
    _write_json(
        tmp_path,
        "successor/experiments/V10_SEALED_FINAL_BANK_COMMITMENT_V1.json",
        commitment,
    )
    authority = _training_authority_receipt(contract, commitment)
    authority["training_runtime_binding_sha256"] = "e" * 64
    unsigned = dict(authority)
    unsigned.pop("receipt_sha256")
    authority["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    _write_json(
        tmp_path,
        "successor/experiments/V10_QWEN35_TRAINING_AUTHORITY_V1.json",
        authority,
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--v10-preflight-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert (
        "training_authority:training_runtime_binding_sha256_mismatch"
        in payload["reasons"]
    )
    assert "patrick_exact_weight_change_authority" in payload["reasons"]



def test_execution_binding_is_required_even_with_runtime_and_bank(
    tmp_path: Path,
) -> None:
    _v2_sealed_hold_fixture(tmp_path)
    (
        tmp_path
        / "successor/experiments/"
        "V10_QWEN35_TRAINING_EXECUTION_BINDING_V1.json"
    ).unlink()
    commitment = _sealed_commitment()
    _write_json(
        tmp_path,
        "successor/experiments/V10_SEALED_FINAL_BANK_COMMITMENT_V1.json",
        commitment,
    )
    contract = json.loads(
        (
            tmp_path
            / "successor/experiments/"
            "V10_QWEN35_EXPERIMENT_CONTRACT_V2.json"
        ).read_text(encoding="utf-8")
    )
    authority = _training_authority_receipt(contract, commitment)
    _write_json(
        tmp_path,
        "successor/experiments/V10_QWEN35_TRAINING_AUTHORITY_V1.json",
        authority,
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--v10-preflight-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert "training_execution_binding_missing" in payload["reasons"]
    assert "patrick_exact_weight_change_authority" in payload["reasons"]
    assert payload["training_execution"]["status"] == "ABSENT"


def test_execution_binding_detects_runner_substitution(
    tmp_path: Path,
) -> None:
    _v2_sealed_hold_fixture(tmp_path)
    runner_path = (
        tmp_path
        / "successor/experiments/train_v10_qwen35_authorized.py"
    )
    runner_path.write_text("# substituted runner\n", encoding="utf-8")
    commitment = _sealed_commitment()
    _write_json(
        tmp_path,
        "successor/experiments/V10_SEALED_FINAL_BANK_COMMITMENT_V1.json",
        commitment,
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--v10-preflight-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert "training_execution:runner_git_blob_sha_mismatch" in payload[
        "reasons"
    ]
    assert payload["training_execution"]["status"] == "INVALID"


def test_authority_receipt_for_wrong_execution_binding_fails_closed(
    tmp_path: Path,
) -> None:
    _v2_sealed_hold_fixture(tmp_path)
    commitment = _sealed_commitment()
    _write_json(
        tmp_path,
        "successor/experiments/V10_SEALED_FINAL_BANK_COMMITMENT_V1.json",
        commitment,
    )
    contract = json.loads(
        (
            tmp_path
            / "successor/experiments/"
            "V10_QWEN35_EXPERIMENT_CONTRACT_V2.json"
        ).read_text(encoding="utf-8")
    )
    authority = _training_authority_receipt(contract, commitment)
    authority["training_execution_binding_sha256"] = "f" * 64
    unsigned = dict(authority)
    unsigned.pop("receipt_sha256")
    authority["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    _write_json(
        tmp_path,
        "successor/experiments/V10_QWEN35_TRAINING_AUTHORITY_V1.json",
        authority,
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--v10-preflight-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert (
        "training_authority:training_execution_binding_sha256_mismatch"
        in payload["reasons"]
    )
    assert "patrick_exact_weight_change_authority" in payload["reasons"]



def test_training_execution_binding_detects_preflight_evaluator_substitution(
    tmp_path: Path,
) -> None:
    _v2_sealed_hold_fixture(tmp_path)
    experiments = tmp_path / "successor" / "experiments"
    runner_path = (
        tmp_path / "successor" / "experiments" / "train_v10_qwen35_authorized.py"
    )
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    runner_path.write_text("RUNNER = True\n", encoding="utf-8")
    evaluator_path = tmp_path / "successor" / "evaluate_successor.py"
    evaluator_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator_path.write_text("EVALUATOR = True\n", encoding="utf-8")
    spec_path = experiments / "V10_QWEN35_TRAINING_EXECUTION_SPEC_V1.json"
    spec = {
        "schema": "V10_QWEN35_TRAINING_EXECUTION_SPEC_V1",
        "effect": "TRAIN_ONE_FRESH_QLORA_ADAPTER",
        "trainer": {
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
            "bf16": True,
            "tf32": True,
            "gradient_checkpointing_use_reentrant": False,
            "logging_steps": 10,
            "save_strategy": "no",
            "eval_strategy": "no",
            "report_to": "none",
            "validation_role": (
                "POST_TRAIN_DIAGNOSTIC_ONLY_NO_RECIPE_OR_CHECKPOINT_SELECTION"
            ),
        },
        "quantization": {
            "load_in_4bit": True,
            "type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
        },
        "model_load": {
            "device_map": {"": 0},
            "dtype": "bfloat16",
            "use_cache": False,
            "prepare_model_for_kbit_training_use_gradient_checkpointing": True,
        },
        "lora": {
            "r": 4,
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": "all-linear",
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "artifact_policy": {
            "fresh_adapter_only": True,
            "parent_adapter": None,
            "save_strategy": "final_adapter_only",
            "validation_checkpoint_selection": False,
        },
    }
    _write_json(tmp_path, str(spec_path.relative_to(tmp_path)), spec)

    def git_blob(raw: bytes) -> str:
        return hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()

    binding = {
        "schema": "V10_QWEN35_TRAINING_EXECUTION_BINDING_V1",
        "status": "FROZEN_AUTHORIZABLE_EXECUTION_SUBJECT",
        "runner": {
            "path": "successor/experiments/train_v10_qwen35_authorized.py",
            "git_blob_sha": git_blob(runner_path.read_bytes()),
        },
        "preflight_evaluator": {
            "path": "successor/evaluate_successor.py",
            "git_blob_sha": git_blob(evaluator_path.read_bytes()),
        },
        "execution_spec": {
            "path": "successor/experiments/V10_QWEN35_TRAINING_EXECUTION_SPEC_V1.json",
            "sha256": hashlib.sha256(
                json.dumps(
                    spec,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
        },
    }
    binding["binding_sha256"] = hashlib.sha256(
        json.dumps(
            binding,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    _write_json(
        tmp_path,
        "successor/experiments/V10_QWEN35_TRAINING_EXECUTION_BINDING_V1.json",
        binding,
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--v10-preflight-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert payload["training_execution"]["status"] == "VERIFIED"

    evaluator_path.write_text("EVALUATOR = False\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--v10-preflight-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["training_execution"]["status"] == "INVALID"
    assert "training_execution:preflight_git_blob_sha_mismatch" in payload[
        "reasons"
    ]

def test_execution_binding_accepts_git_clean_filter_identity_for_crlf_checkout(
    tmp_path: Path,
) -> None:
    _v2_sealed_hold_fixture(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "core.autocrlf", "true"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / ".gitattributes").write_text("*.py text\n", encoding="utf-8")

    paths = [
        tmp_path / "successor/evaluate_successor.py",
        tmp_path / "successor/experiments/train_v10_qwen35_authorized.py",
    ]
    for path in paths:
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b"\n", b"\r\n"))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--v10-preflight-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["training_execution"]["status"] == "VERIFIED"
    assert not any("git_blob_sha_mismatch" in reason for reason in payload["reasons"])
