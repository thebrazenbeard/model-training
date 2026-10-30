"""R22 source-only adapter-lineage negatives. No model, CUDA, or private bank."""

import copy
import json
import unittest
from pathlib import Path

from tests import test_r18_eval as eval_fixture
from trainingday.admission import AdmissionError, _hash_file
from trainingday.r18_eval import _preunseal_inventory


class R22AdapterLineageTests(unittest.TestCase):
    def setUp(self):
        # Reuse the R20 test-only eight-arm filesystem/receipt construction.
        self.fixture = eval_fixture.R18EvaluatorTests(
            "test_complete_unscored_eight_arm_freeze_is_hash_bound")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.plan, self.inventory, self.path = self.fixture.make_freeze_inventory()
        self.args = self.fixture.change(
            freeze_inventory=self.path,
            expected_freeze_inventory_sha256=_hash_file(self.path),
        )
        self.plan_sha = self.inventory["plan_sha256"]
        # Choose a trained control with saved bytes and a nonzero step count.
        self.entry = next(
            entry for entry in self.inventory["arms"]
            if entry["arm"] == "mezo" and entry["control"] == "correct"
        )
        self.receipt_path = Path(self.entry["receipt_path"])
        self.original = json.loads(self.receipt_path.read_text(encoding="utf-8"))

    def assert_unmodified_inventory_accepted(self):
        _, receipts, files = _preunseal_inventory(
            self.args, self.plan, self.plan_sha)
        self.assertEqual(len(receipts), 8)
        self.assertEqual(len(files), 12)

    def repin_mutated_receipt(self, changed):
        self.receipt_path.write_text(json.dumps(changed), encoding="utf-8")
        self.entry["receipt_sha256"] = _hash_file(self.receipt_path)
        self.path.write_text(json.dumps(self.inventory), encoding="utf-8")
        self.args.expected_freeze_inventory_sha256 = _hash_file(self.path)

    def test_repinning_cannot_erase_declared_base_corpus_or_seed_mismatch(self):
        self.assert_unmodified_inventory_accepted()
        for field, invalid in (
            ("base_model", "other/model"),
            ("base_revision", "0" * 40),
            ("base_binding_sha256", "0" * 64),
            ("tokenizer_binding_sha256", "0" * 64),
            ("corpus_manifest_sha256", "0" * 64),
            ("selected_id_sha256", "0" * 64),
            ("train_content_sha256", "0" * 64),
            ("seed", 999),
            ("plan_sha256", "0" * 64),
        ):
            with self.subTest(field=field):
                receipt = copy.deepcopy(self.original)
                receipt[field] = invalid
                # Re-sign every SHA visible to this caller; a plan mismatch
                # must still be rejected by the independent source contract.
                self.repin_mutated_receipt(receipt)
                with self.assertRaisesRegex(
                        AdmissionError, "pre-unseal arm receipt failed"):
                    _preunseal_inventory(self.args, self.plan, self.plan_sha)

    def test_incomplete_training_or_invalid_budget_cannot_be_relabelled(self):
        for field, invalid in (
            ("optimizer_steps", 0),
            ("example_presentations", 0),
            ("supervised_target_tokens", 0),
            ("trainable_digest_after", self.original["trainable_digest_before"]),
            ("status", "TIME_BUDGET_OVERRUN_INVALID"),
            ("train_stop_monotonic_ns", 300_000_000_000),
        ):
            with self.subTest(field=field):
                receipt = copy.deepcopy(self.original)
                receipt[field] = invalid
                self.repin_mutated_receipt(receipt)
                with self.assertRaises(AdmissionError):
                    _preunseal_inventory(self.args, self.plan, self.plan_sha)

    def test_saved_adapter_model_and_config_byte_mutation_rejected(self):
        self.assert_unmodified_inventory_accepted()
        adapter_dir = self.receipt_path.parent / "adapter"
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            with self.subTest(name=name):
                path = adapter_dir / name
                original = path.read_bytes()
                path.write_bytes(original + b"tampered")
                with self.assertRaisesRegex(
                        AdmissionError, "pre-unseal adapter bytes"):
                    _preunseal_inventory(self.args, self.plan, self.plan_sha)
                path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
