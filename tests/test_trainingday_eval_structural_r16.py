"""Offline qualification checks for the R16 model evaluation handoff."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trainingday.admission import AdmissionError, _hash_file
from trainingday.checkpoint_custody import canonical_sha, create_custody
from trainingday.eval_structural_r16 import (
    BANK_ID_SHA256,
    CORPUS_SHA256,
    OLD_EVAL_SHA256,
    R16_PRODUCER_HEAD,
    R16_RECIPE,
    R16_STATUS,
    _old_ids,
    _r16_adapter,
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

    def r16_fixture(self, mode="correct"):
        parent = self.root / f"r16-{mode}"
        adapter = parent / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"saved adapter")
        (adapter / "adapter_config.json").write_text("{}")
        base = self.root / "base.json"
        base.write_text('{"pinned":"base"}')
        steps = 0 if mode == "zero" else 8
        before, after = "a" * 64, ("a" if mode == "zero" else "b") * 64
        contract = {
            "schema": "VERA_TRAININGDAY_R16_RUN_CONTRACT_V1",
            "recipe": R16_RECIPE,
            "question_contract": __import__("trainingday.numeric_prompt", fromlist=[
                "NUMERIC_QUESTION_CONTRACT"]).NUMERIC_QUESTION_CONTRACT,
            "manifest_sha256": CORPUS_SHA256,
            "producer_git_head": R16_PRODUCER_HEAD,
            "base_model": "Qwen/Qwen3.5-4B", "base_revision": "a" * 40,
            "base_binding_sha256": _hash_file(base),
            "mode": mode, "steps": steps, "cases": 64,
            "seed": 1234, "learning_rate": 0.0001, "run_id": parent.name,
            "initial_digest": None if mode == "zero" else before,
            "training_input_sha256": "1" * 64,
            "optimizer": "adamw_bnb_8bit", "gradient_accumulation": 8,
            "lora_r": 4, "lora_alpha": 16, "max_length": 512,
            "save_strategy": "steps", "save_steps": 8,
        }
        (parent / "r16-run-contract.json").write_text(json.dumps(contract))
        data = {
            "recipe": R16_RECIPE, "status": R16_STATUS,
            "question_contract": contract["question_contract"], "mode": mode,
            "base_model": contract["base_model"],
            "base_revision": contract["base_revision"],
            "corpus": {"manifest_sha256": CORPUS_SHA256,
                       "producer_git_head": R16_PRODUCER_HEAD},
            "training_input_sha256": contract["training_input_sha256"],
            "seed": contract["seed"],
            "base_binding_sha256": _hash_file(base),
            "r16_run_contract_sha256": canonical_sha(contract),
            "trainable_digest_before": before, "trainable_digest_after": after,
            "adapter_model_sha256": _hash_file(adapter / "adapter_model.safetensors"),
            "adapter_config_sha256": _hash_file(adapter / "adapter_config.json"),
            "new_optimizer_steps": steps, "resumed_from_step": 0,
            "optimizer_resume_equivalence": "UNVERIFIED_REQUIRES_UNINTERRUPTED_MATCH",
            "optimizer_identity": None if mode == "zero" else {
                "backend": "bitsandbytes", "is_paged": False, "optim_bits": 8},
        }
        if mode != "zero":
            checkpoint = parent / "trainer-work" / f"checkpoint-{steps}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": steps}))
            for name in ("optimizer.pt", "scheduler.pt", "adapter_model.safetensors",
                         "adapter_config.json", "rng_state.pth"):
                (checkpoint / name).write_bytes(name.encode("ascii"))
            custody = create_custody(checkpoint, contract_sha256=canonical_sha(contract))
            custody_file = parent / "latest-checkpoint-custody.json"
            custody_file.write_text(json.dumps(custody))
            data.update(checkpoint_step=steps,
                        checkpoint_custody_file_sha256=_hash_file(custody_file),
                        checkpoint_custody_state=custody["state"])
        receipt = parent / "training_receipt.json"
        receipt.write_text(json.dumps(data))
        return receipt, base, data, contract

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

    def test_r16_zero_null_contract_initial_digest_admitted(self):
        receipt, base, data, contract = self.r16_fixture("zero")
        admitted, adapter = _r16_adapter(
            receipt, base, producer_head=R16_PRODUCER_HEAD, required_mode="zero")
        self.assertEqual(admitted["new_optimizer_steps"], 0)
        self.assertEqual(adapter, receipt.parent / "adapter")
        data["new_optimizer_steps"] = 1
        receipt.write_text(json.dumps(data))
        with self.assertRaises(AdmissionError):
            _r16_adapter(receipt, base, producer_head=R16_PRODUCER_HEAD,
                         required_mode="zero")

    def test_r16_trained_arm_admits_pinned_checkpoint(self):
        receipt, base, data, contract = self.r16_fixture("correct")
        admitted, adapter = _r16_adapter(
            receipt, base, producer_head=R16_PRODUCER_HEAD, required_mode="correct")
        self.assertEqual(admitted["checkpoint_step"], 8)
        self.assertEqual(adapter, receipt.parent / "adapter")
        with self.assertRaises(AdmissionError):
            _r16_adapter(receipt, base, producer_head="f" * 40,
                         required_mode="correct")
        with self.assertRaises(AdmissionError):
            _r16_adapter(receipt, base, producer_head=R16_PRODUCER_HEAD,
                         required_mode="corrupt")

    def test_r16_receipt_contract_step_and_custody_tamper_fail_closed(self):
        receipt, base, data, contract = self.r16_fixture("corrupt")
        for key, value in (("status", "ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED"),
                           ("recipe", "TRAININGDAY_QWEN35_FRESH_QLORA_R15_NUMERIC_MATCHED"),
                           ("new_optimizer_steps", 1),
                           ("checkpoint_step", 16),
                           ("trainable_digest_after", data["trainable_digest_before"]),
                           ("checkpoint_custody_file_sha256", "f" * 64)):
            with self.subTest(receipt_field=key):
                modified = dict(data, **{key: value})
                receipt.write_text(json.dumps(modified))
                with self.assertRaises(AdmissionError):
                    _r16_adapter(receipt, base, producer_head=R16_PRODUCER_HEAD,
                                 required_mode="corrupt")
        receipt.write_text(json.dumps(data))
        contract_file = receipt.parent / "r16-run-contract.json"
        contract_file.write_text(json.dumps(dict(contract, lora_r=8)))
        with self.assertRaises(AdmissionError):
            _r16_adapter(receipt, base, producer_head=R16_PRODUCER_HEAD,
                         required_mode="corrupt")
        contract_file.write_text(json.dumps(contract))
        checkpoint_optimizer = receipt.parent / "trainer-work/checkpoint-8/optimizer.pt"
        checkpoint_optimizer.write_bytes(b"tampered optimizer")
        with self.assertRaises(AdmissionError):
            _r16_adapter(receipt, base, producer_head=R16_PRODUCER_HEAD,
                         required_mode="corrupt")

    @patch("trainingday.eval_structural_r16.preflight", return_value=GOOD_SELECTION)
    @patch("trainingday.eval_structural_r16._required_base")
    def test_r16_receipt_routes_to_source_only_evaluator(self, base_reader, selected):
        receipt, base, data, contract = self.r16_fixture("correct")
        base_reader.return_value = ({"revision": "a" * 40,
                                    "model_id": "Qwen/Qwen3.5-4B"}, self.root)
        result = evaluate(self.args(mode="correct", adapter_receipt=receipt,
                                    base_binding=base,
                                    producer_git_head=R16_PRODUCER_HEAD))
        self.assertEqual(result["status"], "R16_SOURCE_PREFLIGHT_NO_MODEL_EXECUTION")
        self.assertEqual(result["mode"], "correct")
        self.assertEqual(result["adapter_receipt_sha256"], _hash_file(receipt))


if __name__ == "__main__":
    unittest.main()
