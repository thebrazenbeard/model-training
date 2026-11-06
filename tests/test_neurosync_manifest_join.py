"""Negative cross-manifest tests: standalone, offline, zero model/host effects."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
if Path(__file__).resolve().parent.name == "tests":
    # Published model-training/tests layout: always exercise the repository
    # implementation, and fail loudly if the production module is absent.
    if not (ROOT / "trainingday" / "neurosync_experiment_design.py").is_file():
        raise RuntimeError("missing required trainingday/neurosync_experiment_design.py")
    sys.path.insert(0, str(ROOT))
    from trainingday.neurosync_experiment_design import example_source_plan
else:
    # Isolated contributor-workspace layout used before the draft PR upload.
    sys.path.insert(0, str(ROOT / "ns_bridge_contract"))
    sys.path.insert(0, str(ROOT / "ns_experiment_design"))
    from neurosync_experiment_design import example_source_plan
from trainingday.neurosync_manifest_join import MAPPING_SCHEMA, validate_manifest_join
from trainingday.neurosync_transfer_preflight import SCHEMA


def fixtures():
    study = example_source_plan()
    model = {
        "architecture": "Qwen3.5-4B",
        "base_revision": "a" * 40,
        "base_sha256": study["models"]["shared_base_sha256"],
        "tokenizer_sha256": study["models"]["tokenizer_sha256"],
        "chat_template_sha256": "a" * 64,
        "adapter_targets": ["q_proj", "v_proj"],
        "adapter_rank": 8,
        "adapter_alpha": 16.0,
        "adapter_scaling_rule": "alpha_over_rank",
        "adapter_scaling": 2.0,
        "format": "peft_adapter",
    }
    pre = {
        "schema": SCHEMA,
        "source_revisions": {
            "model_training_git_sha": study["source_pins"]["model_training_head"],
            "neuro_sync_git_sha": study["source_pins"]["neuro_sync_head"],
        },
        "pathway": "delta_transfer",
        "donor": model,
        "recipient": deepcopy(model),
        "dataset": {
            "train_split_sha256": "3" * 64,
            "eval_split_sha256": "4" * 64,
            "rights_receipt_sha256": "5" * 64,
        },
        "evaluation": {
            "donor_disconnected": True,
            "recipient_cold_restart": True,
            "control_arms": ["untouched", "context_only", "shuffled_donor"],
            "training_evidence_class": "PROPOSED",
        },
    }
    mapping = {
        "schema": MAPPING_SCHEMA,
        "study_id": study["study_id"],
        "preflight_train_split_sha256": pre["dataset"]["train_split_sha256"],
        "study_train_manifest_sha256": study["dataset"]["approved_train_manifest_sha256"],
        "preflight_eval_split_sha256": pre["dataset"]["eval_split_sha256"],
        "study_heldout_manifest_sha256": study["dataset"]["sealed_heldout_manifest_sha256"],
        "preflight_rights_receipt_sha256": pre["dataset"]["rights_receipt_sha256"],
        "study_rights_reference_sha256": study["dataset"]["data_rights_reference_sha256"],
        "mapping_receipt_sha256": "6" * 64,
    }
    return pre, study, mapping


class ManifestJoinTests(unittest.TestCase):
    def test_no_mapping_never_pretends_same_experiment(self):
        pre, study, _ = fixtures()
        result = validate_manifest_join(pre, study)
        self.assertFalse(result.structural_matches)
        self.assertIn("mapping:INDEPENDENTLY_VERIFIED_RELATION_REQUIRED", result.blockers)
        self.assertEqual(result.disposition, "UNJOINED_HOLD")
        self.assertEqual(len(result.preflight_canonical_sha256), 64)
        self.assertEqual(len(result.study_canonical_sha256), 64)

    def test_matching_claim_structural_only_and_permanent_hold(self):
        pre, study, mapping = fixtures()
        result = validate_manifest_join(pre, study, dataset_mapping_claim=mapping)
        self.assertTrue(result.structural_matches, result.blockers)
        self.assertEqual(result.disposition, "STRUCTURALLY_MATCHED_UNVERIFIED_HOLD")
        self.assertFalse(result.same_experiment_qualified)
        self.assertFalse(result.physical_launch_authorized)
        self.assertFalse(result.state_commit_authorized)
        self.assertIn("DATASET_MAPPING_NOT_INDEPENDENTLY_VERIFIED", result.warnings)
        self.assertEqual(len(result.mapping_claim_canonical_sha256), 64)
        self.assertFalse(result.as_dict()["same_experiment_qualified"])

    def test_source_head_mismatch_blocks_join(self):
        pre, study, mapping = fixtures()
        pre["source_revisions"]["neuro_sync_git_sha"] = "0" * 40
        result = validate_manifest_join(pre, study, dataset_mapping_claim=mapping)
        self.assertIn("cross_manifest.neuro_sync_head:MISMATCH", result.blockers)

    def test_model_base_mismatch_blocks_join(self):
        pre, study, mapping = fixtures()
        study["models"]["shared_base_sha256"] = "9" * 64
        result = validate_manifest_join(pre, study, dataset_mapping_claim=mapping)
        self.assertIn("cross_manifest.shared_base_sha256:MISMATCH", result.blockers)

    def test_tokenizer_mismatch_blocks_join(self):
        pre, study, mapping = fixtures()
        study["models"]["tokenizer_sha256"] = "7" * 64
        result = validate_manifest_join(pre, study, dataset_mapping_claim=mapping)
        self.assertIn("cross_manifest.tokenizer_sha256:MISMATCH", result.blockers)

    def test_mapping_claim_does_not_solve_wrong_split(self):
        pre, study, mapping = fixtures()
        pre["dataset"]["train_split_sha256"] = "7" * 64
        result = validate_manifest_join(pre, study, dataset_mapping_claim=mapping)
        self.assertIn("mapping.preflight_train_split_sha256:REFERENCE_MISMATCH", result.blockers)

    def test_mapping_cannot_claim_unverified_authority(self):
        pre, study, mapping = fixtures()
        mapping["independently_verified"] = True
        result = validate_manifest_join(pre, study, dataset_mapping_claim=mapping)
        self.assertIn("mapping:SCHEMA_FIELDS_MISMATCH", result.blockers)
        self.assertFalse(result.same_experiment_qualified)

    def test_rights_claim_reference_mismatch(self):
        pre, study, mapping = fixtures()
        mapping["study_rights_reference_sha256"] = "7" * 64
        result = validate_manifest_join(pre, study, dataset_mapping_claim=mapping)
        self.assertIn("mapping.study_rights_reference_sha256:REFERENCE_MISMATCH", result.blockers)

    def test_study_independent_validation_required(self):
        pre, study, mapping = fixtures()
        study["arms"][0]["disposition"] = "EXECUTED"
        result = validate_manifest_join(pre, study, dataset_mapping_claim=mapping)
        self.assertTrue(any(b.startswith("study:VALIDATOR_REJECTED") for b in result.blockers))

    def test_invalid_duplicate_json_key_rejected(self):
        pre, study, mapping = fixtures()
        result = validate_manifest_join('{"schema": "x", "schema": "y"}', study,
                                        dataset_mapping_claim=mapping)
        self.assertTrue(any(b.startswith("preflight:INVALID_CANONICAL_JSON") for b in result.blockers))

    def test_result_constructor_has_no_authority_kwargs(self):
        from trainingday.neurosync_manifest_join import ManifestJoinResult
        with self.assertRaises(TypeError):
            ManifestJoinResult(None, None, None, False, (), (),
                               physical_launch_authorized=True)


if __name__ == "__main__":
    unittest.main()
