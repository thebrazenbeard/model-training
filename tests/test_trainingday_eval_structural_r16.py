"""Offline qualification checks for the R16 model evaluation handoff."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trainingday.admission import AdmissionError
from trainingday.eval_structural_r16 import (
    BANK_ID_SHA256,
    CORPUS_SHA256,
    OLD_EVAL_SHA256,
    _old_ids,
    _zero_adapter,
    evaluate,
    preflight,
)


GOOD_SELECTION = {
    "selected_count": 64,
    "excluded_scored_signature_count": 13,
    "selected_id_sha256": BANK_ID_SHA256,
    "cases": [],
}


class StructuralR16Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text("{}", encoding="utf-8")
        self.old = self.root / "old.json"
        self.old.write_text(json.dumps({"base_revision": "a" * 40}), encoding="utf-8")

    def args(self, **changes):
        obj = dict(mode="frozen", output=self.root / "output.json",
                   manifest=self.manifest, old_base_eval=self.old,
                   base_binding=self.root / "base.json", adapter_receipt=None,
                   producer_git_head="b" * 40, preflight_only=True,
                   teacher_force_nll=False)
        obj.update(changes)
        return SimpleNamespace(**obj)

    @patch("trainingday.eval_structural_r16._hash_file", return_value=CORPUS_SHA256)
    @patch("trainingday.eval_structural_r16._old_ids", return_value=["id"] * 16)
    @patch("trainingday.eval_structural_r16.preregister_r15_fresh_structural64",
           return_value=GOOD_SELECTION)
    def test_preflight_uses_pinned_bank(self, prereg, oldids, filehash):
        result = preflight(self.manifest, self.old)
        self.assertEqual(result["selected_id_sha256"], BANK_ID_SHA256)
        prereg.assert_called_once_with(self.manifest, previously_scored_16_ids=["id"] * 16)

    @patch("trainingday.eval_structural_r16._hash_file", return_value=CORPUS_SHA256)
    @patch("trainingday.eval_structural_r16._old_ids", return_value=["id"] * 16)
    @patch("trainingday.eval_structural_r16.preregister_r15_fresh_structural64")
    def test_selection_mismatch_rejected(self, prereg, oldids, filehash):
        for key, invalid in (("selected_count", 63),
                             ("excluded_scored_signature_count", 12),
                             ("selected_id_sha256", "0" * 64)):
            bad = dict(GOOD_SELECTION)
            bad[key] = invalid
            prereg.return_value = bad
            with self.subTest(key=key), self.assertRaises(AdmissionError):
                preflight(self.manifest, self.old)

    def test_unpinned_old_evaluation_rejected_before_parse(self):
        with self.assertRaises(AdmissionError):
            _old_ids(self.old)
        self.assertNotEqual(OLD_EVAL_SHA256,
                            __import__("hashlib").sha256(self.old.read_bytes()).hexdigest())

    @patch("trainingday.eval_structural_r16.preflight", return_value=GOOD_SELECTION)
    @patch("trainingday.eval_structural_r16._required_base")
    def test_preflight_receipt_no_cuda(self, base, selected):
        base.return_value = ({"revision": "a" * 40,
                              "model_id": "Qwen/Qwen3.5-4B"}, self.root)
        result = evaluate(self.args())
        self.assertEqual(result["status"], "R16_SOURCE_PREFLIGHT_NO_MODEL_EXECUTION")
        self.assertEqual(result["bank_id_sha256"], BANK_ID_SHA256)
        self.assertEqual(result["old_base_evaluation_sha256"], OLD_EVAL_SHA256)
        self.assertTrue((self.root / "output.json").is_file())
        self.assertNotIn("cases", result)
        self.assertNotIn("expected", json.dumps(result))

    @patch("trainingday.eval_structural_r16.preflight", return_value=GOOD_SELECTION)
    @patch("trainingday.eval_structural_r16._required_base")
    def test_reject_wrong_arm_or_missing_adapter(self, base, selected):
        base.return_value = ({"revision": "a" * 40,
                              "model_id": "Qwen/Qwen3.5-4B"}, self.root)
        with self.assertRaises(AdmissionError):
            evaluate(self.args(mode="correct"))
        with self.assertRaises(AdmissionError):
            evaluate(self.args(mode="frozen", adapter_receipt=self.old))
        with self.assertRaises(AdmissionError):
            evaluate(self.args(mode="wrong"))

    @patch("trainingday.eval_structural_r16._hash_file", return_value="f" * 64)
    def test_zero_control_is_hash_and_step_bound(self, hashes):
        parent = self.root / "zero"
        adapter = parent / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"dummy")
        (adapter / "adapter_config.json").write_text("{}")
        receipt = parent / "training_receipt.json"
        base = self.root / "base.json"
        base.write_text("{}")
        data = {
            "status": "ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED",
            "mode": "zero", "new_optimizer_steps": 0,
            "trainable_digest_before": "1" * 64,
            "trainable_digest_after": "1" * 64,
            "recipe": __import__("trainingday.train_neural", fromlist=["RECIPE"]).RECIPE,
            "question_contract": __import__("trainingday.numeric_prompt", fromlist=[
                "NUMERIC_QUESTION_CONTRACT"]).NUMERIC_QUESTION_CONTRACT,
            "corpus": {"manifest_sha256": CORPUS_SHA256, "producer_git_head": "b" * 40},
            "base_binding_sha256": "f" * 64,
            "adapter_model_sha256": "f" * 64,
            "adapter_config_sha256": "f" * 64,
        }
        receipt.write_text(json.dumps(data))
        item, target = _zero_adapter(receipt, base, "b" * 40)
        self.assertEqual(item["new_optimizer_steps"], 0)
        self.assertEqual(target, adapter)
        data["new_optimizer_steps"] = 1
        receipt.write_text(json.dumps(data))
        with self.assertRaises(AdmissionError):
            _zero_adapter(receipt, base, "b" * 40)


if __name__ == "__main__":
    unittest.main()
