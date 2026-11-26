from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from successor.experiments.train_v10r2_dev import (
    DevTrainingHold,
    load_and_validate_dev_spec,
)


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "successor" / "experiments"


def _json(name: str) -> dict:
    return json.loads((EXP / name).read_text(encoding="utf-8"))


def _committed_text_sha(name: str) -> str:
    text = (EXP / name).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_recipe_semantics_order_manifest_is_exact_and_unique() -> None:
    manifest = _json(
        "V10R3_RECIPE_SEMANTICS_ORDER_ROWS0_159_20261004_V2.json"
    )
    assert manifest["schema"] == "V10R3_RECIPE_SEMANTICS_ORDER_MANIFEST_V2"
    assert manifest["record_ids_serialization"] == "NEWLINE_UTF8_FINAL_LF"
    assert manifest["row_count"] == 160
    assert manifest["row_indices"] == list(range(160))
    ids = manifest["record_ids"]
    assert len(ids) == 160
    assert len(set(ids)) == 160
    digest = hashlib.sha256(
        ("\n".join(ids) + "\n").encode("utf-8")
    ).hexdigest()
    assert digest == manifest["record_ids_sha256"]
    assert digest == (
        "904ac73c903a3878f950215d004a7d4a17af45a002d2637cba3b3a7995974142"
    )

    groups = manifest["optimizer_step_groups"]
    assert len(groups) == 20
    assert [g["optimizer_step"] for g in groups] == list(range(1, 21))
    assert [i for g in groups for i in g["row_indices"]] == list(range(160))
    assert all(len(g["row_indices"]) == 8 for g in groups)
    assert manifest["shuffle_dataset"] is False
    assert manifest["train_sampling_strategy"] == "sequential"
    assert manifest["invariants"]["each_selected_row_exactly_once"] is True
    assert manifest["invariants"]["same_effective_row_order_both_arms"] is True


def test_matched_specs_share_fresh_initialization_and_order_policy() -> None:
    stage1 = load_and_validate_dev_spec(
        EXP / "V10R3_STAGED_RESET_CONTROL_STAGE1_SPEC_20261004_V2.json"
    )
    continuous = load_and_validate_dev_spec(
        EXP / "V10R3_CONTINUOUS20_TRAINING_SPEC_20261004_V2.json"
    )

    assert "resume_adapter" not in stage1
    assert "resume_adapter" not in continuous
    for key in ("source_subject", "quantization", "lora", "model_load"):
        assert stage1[key] == continuous[key]

    common_trainer_fields = (
        "seed",
        "learning_rate",
        "lr_scheduler_type",
        "warmup_optimizer_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "optimizer",
        "completion_only_loss",
        "packing",
        "shuffle_dataset",
        "train_sampling_strategy",
        "gradient_checkpointing",
        "bf16",
        "tf32",
        "gradient_checkpointing_use_reentrant",
    )
    for key in common_trainer_fields:
        assert stage1["trainer"][key] == continuous["trainer"][key]

    assert stage1["trainer"]["shuffle_dataset"] is False
    assert stage1["trainer"]["train_sampling_strategy"] == "sequential"
    assert continuous["trainer"]["shuffle_dataset"] is False
    assert continuous["trainer"]["train_sampling_strategy"] == "sequential"
    assert stage1["development_window"] == {"start_row": 0, "row_count": 32}
    assert continuous["development_window"] == {
        "start_row": 0,
        "row_count": 160,
    }
    assert stage1["trainer"]["max_optimizer_steps"] == 4
    assert continuous["trainer"]["max_optimizer_steps"] == 20
    assert stage1["output"]["namespace"] != continuous["output"]["namespace"]
    expected_digest = (
        "134dc5a9fdbdff2a6edc7c1bae6e999fed112b81048ae9af6fd7298338079bfd"
    )
    assert (
        stage1["comparison"]["expected_initial_trainable_parameter_digest"]
        == expected_digest
    )
    assert (
        continuous["comparison"]["expected_initial_trainable_parameter_digest"]
        == expected_digest
    )
    assert (
        stage1["comparison"]["initial_digest_source"]["receipt_sha256"]
        == "8c02cad57d1ec6f89139623310d9de8a0ac13a484701e5a371a868a0baa3bc38"
    )


def test_protocol_and_specs_bind_same_order_manifest() -> None:
    manifest_name = "V10R3_RECIPE_SEMANTICS_ORDER_ROWS0_159_20261004_V2.json"
    manifest = _json(manifest_name)
    manifest_sha = _committed_text_sha(manifest_name)
    protocol = _json("V10R3_STAGED_VS_CONTINUOUS_PROTOCOL_20261004_V2.json")
    stage1 = _json("V10R3_STAGED_RESET_CONTROL_STAGE1_SPEC_20261004_V2.json")
    continuous = _json("V10R3_CONTINUOUS20_TRAINING_SPEC_20261004_V2.json")

    assert (
        protocol["matched_training_subject"]["record_ids_serialization"]
        == "NEWLINE_UTF8_FINAL_LF"
    )
    assert protocol["matched_training_subject"]["order_manifest_sha256"] == manifest_sha
    assert protocol["matched_training_subject"]["ordered_record_ids_sha256"] == (
        manifest["record_ids_sha256"]
    )
    for spec in (stage1, continuous):
        assert spec["comparison"]["order_manifest_sha256"] == manifest_sha
        assert spec["comparison"]["full_matched_order_record_ids_sha256"] == (
            manifest["record_ids_sha256"]
        )

    namespaces = {
        stage1["output"]["namespace"],
        continuous["output"]["namespace"],
        protocol["control"]["final_namespace"],
    }
    assert len(namespaces) == 3





def test_protocol_binds_exact_committed_text_hashes() -> None:
    protocol = _json("V10R3_STAGED_VS_CONTINUOUS_PROTOCOL_20261004_V2.json")
    bindings = protocol["committed_file_bindings"]
    expected = {
        "control_stage1_spec": (
            "V10R3_STAGED_RESET_CONTROL_STAGE1_SPEC_20261004_V2.json"
        ),
        "continuous20_spec": "V10R3_CONTINUOUS20_TRAINING_SPEC_20261004_V2.json",
        "order_manifest": (
            "V10R3_RECIPE_SEMANTICS_ORDER_ROWS0_159_20261004_V2.json"
        ),
        "fresh_eval_panel": "V10R3_RECIPE_SEMANTICS_PANEL_ROWS448_511_V1.json",
    }
    for key, name in expected.items():
        assert bindings[key]["sha256"] == _committed_text_sha(name)
        assert (
            bindings[key]["hash_semantics"]
            == "UTF8_TEXT_LF_NORMALIZED_COMMITTED_CONTENT"
        )

    panel = _json("V10R3_RECIPE_SEMANTICS_PANEL_ROWS448_511_V1.json")
    assert (
        bindings["fresh_eval_panel"]["record_ids_sha256"]
        == panel["record_ids_sha256"]
    )


def test_protocol_requires_observed_fresh_initial_digest_before_training() -> None:
    protocol = _json("V10R3_STAGED_VS_CONTINUOUS_PROTOCOL_20261004_V2.json")
    gate = protocol["initialization_equivalence_gate"]
    expected = "134dc5a9fdbdff2a6edc7c1bae6e999fed112b81048ae9af6fd7298338079bfd"
    assert gate["required"] is True
    assert gate["expected_initial_trainable_parameter_digest"] == expected
    assert gate["runner_receipt_field"] == "initial_trainable_parameter_digest"
    assert gate["control_stage1_must_match_before_first_optimizer_update"] is True
    assert gate["continuous20_must_match_before_first_optimizer_update"] is True
    assert gate["failure_disposition"] == "HOLD_BEFORE_TRAINER_TRAIN"


def test_superseded_v1_global_shuffle_candidate_is_not_executable() -> None:
    with pytest.raises(
        DevTrainingHold,
        match="shuffle_dataset=false|sequential sampling",
    ):
        load_and_validate_dev_spec(
            EXP / "V10R3_CONTINUOUS20_TRAINING_SPEC_20261004_V1.json"
        )


def test_protocol_binds_preexecution_verifier_receipt() -> None:
    protocol = _json("V10R3_STAGED_VS_CONTINUOUS_PROTOCOL_20261004_V2.json")
    verifier = protocol["preexecution_evidence"]["local_verifier"]
    receipt = ROOT / verifier["path"]
    assert verifier["status"] == "PASS"
    assert (
        verifier["hash_semantics"]
        == "UTF8_TEXT_LF_NORMALIZED_COMMITTED_CONTENT"
    )
    text = receipt.read_text(encoding="utf-8-sig").replace("\\r\\n", "\\n")
    assert verifier["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()