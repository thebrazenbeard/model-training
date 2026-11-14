from __future__ import annotations

import hashlib
import json

import pytest

from successor.experiments.materialize_v10_training_authority import (
    build_training_authority_receipt,
)


def _contract() -> dict:
    return {
        "experiment_id": "VERA_SUCCESSOR_V10_QWEN35_FRESH_QLORA_20261001_V2",
        "base_model": {"revision": "base-rev"},
        "source_subject": {
            "training_corpus_id": "corpus-v1",
            "train_sha256": "a" * 64,
            "validation_sha256": "b" * 64,
        },
        "training_recipe": {
            "method": "QLORA_SFT_ONLY",
            "seed": 20261001,
            "epochs": 1,
        },
    }


def _intent() -> dict:
    return {
        "schema": "V10_QWEN35_TRAINING_AUTHORITY_INTENT_V1",
        "status": "USER_DIRECT_CONDITIONAL_ONE_RUN_AUTHORITY",
        "authority_actor_id": "PATRICK_USER_AUTHORITY",
        "effect": "TRAIN_ONE_FRESH_QLORA_ADAPTER",
        "max_training_runs": 1,
        "paid_compute_authorized": False,
        "merge_authorized": False,
        "install_activate_deploy_authorized": False,
    }


def _preflight(*, reasons: list[str] | None = None) -> dict:
    return {
        "schema": "V10_QWEN35_PREFLIGHT_V1",
        "status": "HOLD",
        "training_allowed": False,
        "reasons": (
            ["patrick_exact_weight_change_authority"]
            if reasons is None
            else reasons
        ),
        "sealed_final_bank": {
            "status": "VERIFIED",
            "bank_id": "bank-v1",
            "commitment_sha256": "c" * 64,
        },
        "training_runtime": {
            "status": "VERIFIED",
            "binding_sha256": "d" * 64,
        },
        "training_execution": {
            "status": "VERIFIED",
            "binding_sha256": "e" * 64,
        },
        "training_authority": {
            "status": "ABSENT",
            "authority_actor_id": None,
            "receipt_sha256": None,
        },
    }


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def test_materializes_exact_single_run_receipt_after_all_other_gates_pass() -> None:
    contract = _contract()
    receipt = build_training_authority_receipt(
        intent=_intent(),
        contract=contract,
        preflight=_preflight(),
        output_namespace=r"D:\VERA\models\adapters\v10-qwen35-bank-c-run1",
    )

    assert receipt["schema"] == "V10_QWEN35_TRAINING_AUTHORITY_V1"
    assert receipt["status"] == "AUTHORIZED"
    assert receipt["authority_actor_id"] == "PATRICK_USER_AUTHORITY"
    assert receipt["authorization_source"] == "EXPLICIT_CURRENT_USER_INSTRUCTION"
    assert receipt["effect"] == "TRAIN_ONE_FRESH_QLORA_ADAPTER"
    assert receipt["max_training_runs"] == 1
    assert receipt["sealed_final_bank_commitment_sha256"] == "c" * 64
    assert receipt["training_runtime_binding_sha256"] == "d" * 64
    assert receipt["training_execution_binding_sha256"] == "e" * 64
    assert receipt["training_recipe_sha256"] == hashlib.sha256(
        _canonical(contract["training_recipe"])
    ).hexdigest()

    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256")
    assert claimed == hashlib.sha256(_canonical(unsigned)).hexdigest()


def test_refuses_to_materialize_while_any_non_authority_gate_remains() -> None:
    with pytest.raises(
        RuntimeError,
        match="preflight has unresolved non-authority blockers",
    ):
        build_training_authority_receipt(
            intent=_intent(),
            contract=_contract(),
            preflight=_preflight(
                reasons=[
                    "final_bank_cases_not_admitted",
                    "patrick_exact_weight_change_authority",
                ]
            ),
            output_namespace=r"D:\VERA\models\adapters\blocked",
        )


def test_refuses_invalid_or_broadened_user_intent() -> None:
    intent = _intent()
    intent["max_training_runs"] = 2
    with pytest.raises(ValueError, match="intent:max_training_runs"):
        build_training_authority_receipt(
            intent=intent,
            contract=_contract(),
            preflight=_preflight(),
            output_namespace=r"D:\VERA\models\adapters\blocked",
        )
