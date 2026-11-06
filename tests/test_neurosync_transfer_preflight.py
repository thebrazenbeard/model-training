"""Standalone, offline contract tests for TrainingDay's Neuro-Sync bridge.

These tests intentionally exercise the *public* preflight API and its fail-closed
boundary. They never create a training job, mutate weights, or contact a host.

Run from this directory with::

    python -m unittest -v test_neurosync_transfer_preflight.py

When checked into ``model-training/tests/``, the suite imports that repo's
``trainingday/neurosync_transfer_preflight.py``. The scratch companion fallback
is permitted only when the test directory is explicitly ``ns_bridge_tests``.
Set NS_PREFLIGHT_MODULE_PATH to test an explicitly selected checkout.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


def _default_module_path(test_dir: Path) -> Path:
    root = test_dir.parent
    if test_dir.name == "ns_bridge_tests":
        return root / "ns_bridge_contract" / "trainingday" / "neurosync_transfer_preflight.py"
    return root / "trainingday" / "neurosync_transfer_preflight.py"


DEFAULT_MODULE = _default_module_path(Path(__file__).resolve().parent)
MODULE_PATH = Path(os.environ.get("NS_PREFLIGHT_MODULE_PATH", DEFAULT_MODULE))


def _load_module():
    if not MODULE_PATH.is_file():
        raise RuntimeError(f"Production preflight module not found: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("neurosync_transfer_preflight_under_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module at {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


preflight = _load_module()


def valid_manifest() -> dict:
    """Independent fixture for a matching PEFT transfer; never a launch grant."""
    model = {
        "architecture": "qwen3.5-4b",
        "base_revision": "a" * 40,
        "base_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
        "chat_template_sha256": "d" * 64,
        "adapter_targets": ["q_proj", "v_proj"],
        "format": "peft_adapter",
        "adapter_rank": 8,
        "adapter_alpha": 16.0,
        "adapter_scaling_rule": "alpha_over_rank",
        "adapter_scaling": 2.0,
    }
    return {
        "schema": "trainingday.neurosync.transfer-preflight.v1",
        "source_revisions": {
            "neuro_sync_git_sha": "1" * 40,
            "model_training_git_sha": "2" * 40,
        },
        "pathway": "delta_transfer",
        "donor": deepcopy(model),
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
            "training_evidence_class": "TEST",
        },
    }


class PreflightContractTests(unittest.TestCase):
    def test_checked_in_test_tree_resolves_sibling_production_module(self):
        target = _default_module_path(Path("/example/model-training/tests"))
        self.assertEqual(
            target,
            Path("/example/model-training/trainingday/neurosync_transfer_preflight.py"),
        )

    def test_scratch_fallback_is_explicitly_scoped(self):
        self.assertEqual(
            _default_module_path(Path("/example/ns_bridge_tests")),
            Path("/example/ns_bridge_contract/trainingday/neurosync_transfer_preflight.py"),
        )
        # An unrelated directory must never silently load scratch production code.
        self.assertEqual(
            _default_module_path(Path("/example/another_tests")),
            Path("/example/trainingday/neurosync_transfer_preflight.py"),
        )

    def test_missing_selected_production_module_is_not_skipped(self):
        missing = Path("/nonexistent-model-training/trainingday/neurosync_transfer_preflight.py")
        with patch.dict(globals(), {"MODULE_PATH": missing}):
            with self.assertRaisesRegex(RuntimeError, "not found"):
                _load_module()

    def assert_permanent_hold(self, result):
        self.assertFalse(result.physical_launch_authorized, "Preflight must never launch training")
        self.assertFalse(result.state_commit_authorized, "Preflight must never commit state")
        self.assertIn(result.status, {"SOURCE_COMPATIBLE_HOLD", "INCOMPATIBLE_HOLD"})

    def assert_incompatible(self, manifest, reason):
        result = preflight.validate_transfer_manifest(manifest)
        self.assert_permanent_hold(result)
        self.assertFalse(result.compatible, reason)
        self.assertEqual(result.status, "INCOMPATIBLE_HOLD", reason)
        self.assertTrue(result.blockers, f"Missing blocker receipt: {reason}")
        return result

    def test_matching_adapter_is_source_compatible_but_never_authorized(self):
        result = preflight.validate_transfer_manifest(valid_manifest())
        self.assertTrue(result.compatible, result.blockers)
        self.assertEqual(result.status, "SOURCE_COMPATIBLE_HOLD")
        self.assertFalse(result.blockers)
        self.assert_permanent_hold(result)

    def test_result_receipt_does_not_upgrade_assertions_to_runtime_proof(self):
        result = preflight.validate_transfer_manifest(valid_manifest())
        receipt = result.as_dict()
        self.assertEqual(receipt["status"], "SOURCE_COMPATIBLE_HOLD")
        self.assertFalse(receipt["physical_launch_authorized"])
        self.assertFalse(receipt["state_commit_authorized"])
        self.assertFalse(receipt["rights_verified"])
        self.assertFalse(receipt["artifact_bytes_verified"])
        self.assertFalse(receipt["evaluation_executed_or_verified"])
        self.assertEqual(receipt["evidence_class"], "SOURCE_INSPECTION")
        self.assertTrue(receipt["warnings"], "Source-only attestation limitations must be explicit")

    def test_result_constructor_cannot_override_authority_flags(self):
        result = preflight.PreflightResult(True, (), ())
        self.assertFalse(result.physical_launch_authorized)
        self.assertFalse(result.state_commit_authorized)
        with self.assertRaises(TypeError):
            preflight.PreflightResult(True, (), (), physical_launch_authorized=True)
        with self.assertRaises(TypeError):
            preflight.PreflightResult(True, (), (), state_commit_authorized=True)

    def test_validation_does_not_mutate_input_manifest(self):
        manifest = valid_manifest()
        baseline = deepcopy(manifest)
        preflight.validate_transfer_manifest(manifest)
        self.assertEqual(manifest, baseline)

    def test_alternate_documented_scaling_rule_when_consistent(self):
        manifest = valid_manifest()
        for role in ("donor", "recipient"):
            manifest[role]["adapter_rank"] = 16
            manifest[role]["adapter_alpha"] = 8.0
            manifest[role]["adapter_scaling_rule"] = "alpha_over_sqrt_rank"
            manifest[role]["adapter_scaling"] = 2.0
        result = preflight.validate_transfer_manifest(manifest)
        self.assertTrue(result.compatible, result.blockers)
        self.assert_permanent_hold(result)

    def test_all_pathways_always_hold_launch_and_state(self):
        for pathway in ("delta_transfer", "adapter_copy", "context_only"):
            with self.subTest(pathway=pathway):
                manifest = valid_manifest()
                manifest["pathway"] = pathway
                self.assert_permanent_hold(preflight.validate_transfer_manifest(manifest))

    def test_unknown_or_missing_provenance_fails_closed(self):
        variants = [
            ("missing_source_revisions", lambda m: m.pop("source_revisions")),
            ("unknown_git_revision", lambda m: m["source_revisions"].update(neuro_sync_git_sha="unknown")),
            ("empty_git_revision", lambda m: m["source_revisions"].update(model_training_git_sha="")),
            ("missing_rights_receipt", lambda m: m["dataset"].pop("rights_receipt_sha256")),
            ("unknown_source_field", lambda m: m["source_revisions"].update(branch="main")),
        ]
        for name, mutate in variants:
            with self.subTest(case=name):
                manifest = valid_manifest()
                mutate(manifest)
                self.assert_incompatible(manifest, name)

    def test_schema_and_pathway_are_closed_enums(self):
        for field, bad_value in (
            ("schema", "trainingday.neurosync.transfer-preflight.v0"),
            ("schema", None),
            ("pathway", "inject_weights"),
            ("pathway", ""),
        ):
            with self.subTest(field=field, bad_value=bad_value):
                manifest = valid_manifest()
                manifest[field] = bad_value
                self.assert_incompatible(manifest, f"{field}={bad_value!r}")

    def test_base_revision_and_fingerprint_mismatch_blocks_weight_paths(self):
        for pathway in ("delta_transfer", "adapter_copy"):
            for field, other in (
                ("architecture", "not-qwen"),
                ("base_revision", "e" * 40),
                ("base_sha256", "e" * 64),
            ):
                with self.subTest(pathway=pathway, field=field):
                    manifest = valid_manifest()
                    manifest["pathway"] = pathway
                    manifest["recipient"][field] = other
                    self.assert_incompatible(manifest, f"{pathway}: {field} mismatch")

    def test_tokenizer_or_chat_template_mismatch_blocks_weight_paths(self):
        for field in ("tokenizer_sha256", "chat_template_sha256"):
            with self.subTest(field=field):
                manifest = valid_manifest()
                manifest["recipient"][field] = "f" * 64
                self.assert_incompatible(manifest, f"{field} mismatch")

    def test_adapter_target_order_or_inventory_mismatch_blocks_weight_paths(self):
        for targets in (["v_proj", "q_proj"], ["q_proj"], ["q_proj", "o_proj"], []):
            with self.subTest(targets=targets):
                manifest = valid_manifest()
                manifest["recipient"]["adapter_targets"] = targets
                self.assert_incompatible(manifest, "adapter target mapping mismatch")

    def test_adapter_scaling_rank_alpha_or_rule_mismatch_blocks_weight_paths(self):
        overrides = {
            "adapter_rank": 16,
            "adapter_alpha": 8.0,
            "adapter_scaling_rule": "alpha_over_sqrt_rank",
            "adapter_scaling": 0.5,
        }
        for field, replacement in overrides.items():
            with self.subTest(field=field):
                manifest = valid_manifest()
                manifest["recipient"][field] = replacement
                self.assert_incompatible(manifest, f"adapter scaling identity mismatch: {field}")

    def test_scaling_value_must_match_claimed_rule_even_when_models_agree(self):
        manifest = valid_manifest()
        manifest["donor"]["adapter_scaling"] = 1.7
        manifest["recipient"]["adapter_scaling"] = 1.7
        self.assert_incompatible(manifest, "identical but dishonest alpha/rank scaling")

    def test_unsupported_weight_format_fails_closed(self):
        for role in ("donor", "recipient"):
            with self.subTest(role=role):
                manifest = valid_manifest()
                manifest[role]["format"] = "gguf"
                self.assert_incompatible(manifest, "weight transfer must not assume GGUF is PEFT")

    def test_context_only_pathway_may_be_heterogeneous_but_never_launches(self):
        manifest = valid_manifest()
        manifest["pathway"] = "context_only"
        manifest["recipient"]["architecture"] = "small-other-model"
        manifest["recipient"]["base_revision"] = "e" * 40
        manifest["recipient"]["tokenizer_sha256"] = "e" * 64
        manifest["recipient"]["chat_template_sha256"] = "f" * 64
        manifest["recipient"]["format"] = "gguf"
        result = preflight.validate_transfer_manifest(manifest)
        self.assertTrue(result.compatible, result.blockers)
        self.assert_permanent_hold(result)

    def test_train_and_heldout_overlap_fails_closed(self):
        manifest = valid_manifest()
        manifest["dataset"]["eval_split_sha256"] = manifest["dataset"]["train_split_sha256"]
        self.assert_incompatible(manifest, "evaluation/test leakage")

    def test_missing_or_malformed_evidence_class_fails_closed(self):
        for value in ("independent model review", "green CI", "VERIFIED", "", 1, True, None):
            with self.subTest(value=value):
                manifest = valid_manifest()
                manifest["evaluation"]["training_evidence_class"] = value
                self.assert_incompatible(manifest, "unsupported evidence class")

    def test_low_evidence_cannot_infer_external_authorization(self):
        manifest = valid_manifest()
        manifest["evaluation"]["training_evidence_class"] = "PROPOSED"
        result = preflight.validate_transfer_manifest(manifest)
        self.assert_permanent_hold(result)

    def test_donor_must_be_disconnected_during_evaluation(self):
        manifest = valid_manifest()
        manifest["evaluation"]["donor_disconnected"] = False
        self.assert_incompatible(manifest, "donor-connected contamination")

    def test_recipient_cold_restart_required(self):
        manifest = valid_manifest()
        manifest["evaluation"]["recipient_cold_restart"] = False
        self.assert_incompatible(manifest, "in-memory carryover is not model transfer")

    def test_all_required_control_arms_present(self):
        for missing in ("untouched", "context_only", "shuffled_donor"):
            with self.subTest(missing=missing):
                manifest = valid_manifest()
                manifest["evaluation"]["control_arms"].remove(missing)
                self.assert_incompatible(manifest, f"missing {missing} control arm")

    def test_duplicate_control_arms_do_not_count_as_independent_controls(self):
        manifest = valid_manifest()
        manifest["evaluation"]["control_arms"].append("untouched")
        self.assert_incompatible(manifest, "duplicate control arm")

    def test_control_arms_must_be_a_nonempty_string_collection(self):
        for value in ("untouched,context_only,shuffled_donor", [], ["untouched", 1, "shuffled_donor"], None):
            with self.subTest(value=value):
                manifest = valid_manifest()
                manifest["evaluation"]["control_arms"] = value
                self.assert_incompatible(manifest, "invalid controls type")

    def test_boolean_impostors_are_rejected(self):
        for field in ("donor_disconnected", "recipient_cold_restart"):
            for value in (1, 0, "true", "false", None):
                with self.subTest(field=field, value=value):
                    manifest = valid_manifest()
                    manifest["evaluation"][field] = value
                    self.assert_incompatible(manifest, f"{field} must be a genuine bool")

    def test_rank_must_not_accept_boolean_float_or_nonpositive_values(self):
        for value in (True, False, 0, -1, 2.0, "8", None):
            with self.subTest(value=value):
                manifest = valid_manifest()
                manifest["donor"]["adapter_rank"] = value
                self.assert_incompatible(manifest, "invalid rank")

    def test_duplicate_or_unsorted_adapter_targets_not_canonical(self):
        for value in (["q_proj", "q_proj"], ["v_proj", "q_proj"]):
            with self.subTest(value=value):
                manifest = valid_manifest()
                for role in ("donor", "recipient"):
                    manifest[role]["adapter_targets"] = value
                self.assert_incompatible(manifest, "target list must be stable and unambiguous")

    def test_nonfinite_and_nonpositive_scaling_values_rejected(self):
        for field in ("adapter_alpha", "adapter_scaling"):
            for value in (0, -2, math.inf, -math.inf, math.nan, True, "2.0"):
                with self.subTest(field=field, value=value):
                    manifest = valid_manifest()
                    manifest["donor"][field] = value
                    self.assert_incompatible(manifest, f"invalid {field}")

    def test_malformed_sha256_and_git_hash_inputs_rejected(self):
        cases = [
            ("source_revisions", "neuro_sync_git_sha", "g" * 40),
            ("source_revisions", "model_training_git_sha", "1" * 39),
            ("donor", "base_revision", "/tmp/model"),
            ("donor", "base_sha256", "../../../../../"),
            ("recipient", "tokenizer_sha256", "SHA256:abc"),
            ("recipient", "chat_template_sha256", "F" * 63),
            ("dataset", "train_split_sha256", 0),
            ("dataset", "eval_split_sha256", True),
            ("dataset", "rights_receipt_sha256", "unknown"),
        ]
        for group, key, value in cases:
            with self.subTest(group=group, key=key):
                manifest = valid_manifest()
                manifest[group][key] = value
                self.assert_incompatible(manifest, f"malformed {group}.{key}")

    def test_path_and_command_fields_are_not_execution_capabilities(self):
        for group, key, value in (
            (None, "launch_command", "python train.py --force"),
            (None, "weights_path", "/etc/passwd"),
            ("donor", "checkpoint_path", "../../../../../private.bin"),
            ("recipient", "adapter_file", "C:\\VERA\\models\\adapter.gguf"),
            ("dataset", "rights_path", "/tmp/rights.json"),
            ("evaluation", "launch", True),
        ):
            with self.subTest(group=group, key=key):
                manifest = valid_manifest()
                target = manifest if group is None else manifest[group]
                target[key] = value
                self.assert_incompatible(manifest, "closed manifest must reject arbitrary paths/commands")

    def test_self_reported_formal_verification_still_cannot_authorize(self):
        manifest = valid_manifest()
        manifest["evaluation"]["training_evidence_class"] = "FORMAL_VERIFY"
        result = preflight.validate_transfer_manifest(manifest)
        self.assert_permanent_hold(result)
        receipt = result.as_dict()
        self.assertFalse(receipt["evaluation_executed_or_verified"])
        self.assertFalse(receipt["artifact_bytes_verified"])

    def test_nested_unknown_fields_fail_closed(self):
        for group in ("source_revisions", "donor", "recipient", "dataset", "evaluation"):
            with self.subTest(group=group):
                manifest = valid_manifest()
                manifest[group]["unreviewed_extension"] = "surprise"
                self.assert_incompatible(manifest, "unknown nested extension")

    def test_missing_top_level_structures_fail_closed_without_exception(self):
        for field in ("schema", "source_revisions", "pathway", "donor", "recipient", "dataset", "evaluation"):
            with self.subTest(field=field):
                manifest = valid_manifest()
                del manifest[field]
                self.assert_incompatible(manifest, f"missing {field}")

    def test_malformed_container_inputs_fail_closed_without_exception(self):
        for value in (None, [], "train now", 17, True):
            with self.subTest(value=value):
                self.assert_incompatible(value, "manifest must be mapping")
        for group in ("source_revisions", "donor", "recipient", "dataset", "evaluation"):
            for value in (None, [], "bad"):
                with self.subTest(group=group, value=value):
                    manifest = valid_manifest()
                    manifest[group] = value
                    self.assert_incompatible(manifest, "nested record must be mapping")


if __name__ == "__main__":
    unittest.main()
