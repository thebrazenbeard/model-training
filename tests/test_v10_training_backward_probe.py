from __future__ import annotations

import pytest

from successor.experiments.probe_v10_qwen35_backward import (
    ProbeHold,
    finalize_probe_receipt,
    validate_probe_preconditions,
)


def _held_preflight() -> dict:
    return {
        "schema": "V10_QWEN35_PREFLIGHT_V1",
        "status": "HOLD",
        "training_allowed": False,
        "reasons": [
            "final_bank_cases_not_admitted",
            "fresh_evaluation_bank_frozen",
            "independent_bank_admission_verified",
            "patrick_exact_weight_change_authority",
            "semantic_contamination_screen_verified",
        ],
        "training_runtime": {
            "status": "VERIFIED",
            "binding_sha256": "a" * 64,
        },
        "training_execution": {
            "status": "VERIFIED",
            "binding_sha256": "b" * 64,
        },
    }


def test_probe_allows_only_bank_and_authority_holds_with_verified_stack() -> None:
    result = validate_probe_preconditions(_held_preflight())

    assert result["status"] == "PROBE_ALLOWED_WITH_TRAINING_HOLD"
    assert result["weight_change_authorized"] is False
    assert result["runtime_binding_sha256"] == "a" * 64
    assert result["execution_binding_sha256"] == "b" * 64


def test_probe_refuses_unrelated_execution_or_runtime_defect() -> None:
    value = _held_preflight()
    value["training_execution"]["status"] = "INVALID"
    value["reasons"].append("training_execution:runner_git_blob_sha_mismatch")

    with pytest.raises(ProbeHold, match="training execution is not verified"):
        validate_probe_preconditions(value)


def test_probe_refuses_unexpected_preflight_reason() -> None:
    value = _held_preflight()
    value["reasons"].append("mystery_new_blocker")

    with pytest.raises(ProbeHold, match="unexpected preflight blockers"):
        validate_probe_preconditions(value)


def test_probe_pass_requires_backward_without_optimizer_or_weight_change() -> None:
    receipt = finalize_probe_receipt(
        precondition_check={
            "status": "PROBE_ALLOWED_WITH_TRAINING_HOLD",
            "runtime_binding_sha256": "a" * 64,
            "execution_binding_sha256": "b" * 64,
        },
        contract_sha256="c" * 64,
        execution_spec_sha256="d" * 64,
        train_sha256="e" * 64,
        row_index=17,
        row_case_id="train-case-17",
        trainable_parameter_count=128,
        targeted_module_count=64,
        loss=1.25,
        microbatch_losses=[1.25] * 8,
        microbatches_completed=8,
        expected_gradient_accumulation_steps=8,
        gradients_present=True,
        nonzero_gradient_parameter_count=32,
        optimizer_created=False,
        weight_digest_before="f" * 64,
        weight_digest_after="f" * 64,
        cuda_memory={
            "allocated_before_backward_mib": 3000.0,
            "peak_allocated_mib": 3500.0,
            "peak_reserved_mib": 3700.0,
        },
        output_artifacts_written=False,
    )

    assert receipt["status"] == "BACKWARD_PROBE_PASS_NO_WEIGHT_CHANGE"
    assert receipt["weight_digest_unchanged"] is True
    assert receipt["optimizer_created"] is False
    assert receipt["output_artifacts_written"] is False
    assert receipt["microbatches_completed"] == 8
    assert receipt["expected_gradient_accumulation_steps"] == 8
    assert receipt["effect"] == "EPHEMERAL_FORWARD_BACKWARD_ONLY_NO_OPTIMIZER_NO_WEIGHT_CHANGE"


def test_probe_receipt_requires_full_frozen_gradient_accumulation_window() -> None:
    with pytest.raises(ProbeHold, match="gradient accumulation window incomplete"):
        finalize_probe_receipt(
            precondition_check={
                "status": "PROBE_ALLOWED_WITH_TRAINING_HOLD",
                "runtime_binding_sha256": "a" * 64,
                "execution_binding_sha256": "b" * 64,
            },
            contract_sha256="c" * 64,
            execution_spec_sha256="d" * 64,
            train_sha256="e" * 64,
            row_index=0,
            row_case_id="train-case-0",
            trainable_parameter_count=128,
            targeted_module_count=64,
            loss=1.0,
            microbatch_losses=[1.0] * 7,
            microbatches_completed=7,
            expected_gradient_accumulation_steps=8,
            gradients_present=True,
            nonzero_gradient_parameter_count=1,
            optimizer_created=False,
            weight_digest_before="f" * 64,
            weight_digest_after="f" * 64,
            cuda_memory={},
            output_artifacts_written=False,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gradients_present", False, "no trainable gradients"),
        ("optimizer_created", True, "optimizer was created"),
        ("weight_digest_after", "0" * 64, "trainable parameter digest changed"),
        ("output_artifacts_written", True, "output artifacts were written"),
    ],
)
def test_probe_receipt_fails_closed_on_effect_boundary_violation(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs = {
        "precondition_check": {
            "status": "PROBE_ALLOWED_WITH_TRAINING_HOLD",
            "runtime_binding_sha256": "a" * 64,
            "execution_binding_sha256": "b" * 64,
        },
        "contract_sha256": "c" * 64,
        "execution_spec_sha256": "d" * 64,
        "train_sha256": "e" * 64,
        "row_index": 0,
        "row_case_id": "train-case-0",
        "trainable_parameter_count": 128,
        "targeted_module_count": 64,
        "loss": 1.0,
        "microbatch_losses": [1.0] * 8,
        "microbatches_completed": 8,
        "expected_gradient_accumulation_steps": 8,
        "gradients_present": True,
        "nonzero_gradient_parameter_count": 1,
        "optimizer_created": False,
        "weight_digest_before": "f" * 64,
        "weight_digest_after": "f" * 64,
        "cuda_memory": {},
        "output_artifacts_written": False,
    }
    kwargs[field] = value

    with pytest.raises(ProbeHold, match=message):
        finalize_probe_receipt(**kwargs)
