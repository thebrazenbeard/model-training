"""R18 independent source-only evaluator tests; no CUDA or model required."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trainingday.admission import AdmissionError, _hash_file
from trainingday.eval_structural_r16 import BANK_ID_SHA256, CORPUS_SHA256
from trainingday.numeric_prompt import NUMERIC_QUESTION_CONTRACT
from trainingday.r18_eval import (
    BASE_MODEL, BASE_REVISION, EVAL_SOURCE, PRIVATE_BANK_SCHEMA,
    PROTOCOL, RECIPES, _adapter, _bank, _exact, _json, _paired, _plan,
    canonical_sha, evaluate,
)


class R18EvaluatorTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        if os.name == "nt":
            from trainingday.r18_benchmark import _protect_private_path
            _protect_private_path(self.root, directory=True)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text("{}", encoding="utf-8")
        self.binding = self.root / "base.json"
        self.binding.write_text("{}", encoding="utf-8")
        self.old = self.root / "r15-eval.json"
        self.old.write_text("{}", encoding="utf-8")
        self.exclusions = self.root / "r18-exclusions-private.json"
        self.sealed = self.root / "r18-evaluator-sealed-private.json"
        self.plan_path = self.root / "r18-plan.json"
        self.base = {"model_id": BASE_MODEL, "revision": BASE_REVISION,
                     "files": {"tokenizer_config.json": "a" * 64}}
        self.plan = {
            "schema": "VERA_TRAININGDAY_R18_BENCHMARK_PLAN_V1",
            "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
            "base_binding_sha256": _hash_file(self.binding),
            "tokenizer_binding_sha256": "a" * 64,
            "corpus_manifest_sha256": CORPUS_SHA256,
            "eval_bank_source": EVAL_SOURCE,
            "no_eval_access_before_arm_freeze": True,
            "preregistered_checkpoint_selection": "END_OF_TIME_BUDGET_ONLY",
            "promotion_policy": "NEVER_AUTOPROMOTE_R18_REQUIRES_INDEPENDENT_REPLICATION",
            "eval_examples": 64, "selected_id_sha256": "b" * 64,
            "selected_count": 1024,
            "train_content_sha256": "c" * 64,
            "eval_index_sha256": "d" * 64, "eval_custody_sha256": "e" * 64,
            "r16_eval_custody_sha256": "f" * 64,
            "r16_holdout_id_sha256": "1" * 64,
            "source_admission_status": "ADMITTED_LABYRINTH_TRAIN_ONLY_NO_WEIGHT_UPDATE",
            "seed": 17, "training_budget_seconds": 120,
            "probe_budget_seconds": 10,
            "fit_policy": "ASCENDING_LENGTH_BATCH1_THEN_BATCH2_UNTIL_FIRST_OOM",
            "fit_candidates": [{"micro_batch_size": b,
                                "actual_padded_sequence_length": n,
                                "effective_batch_size": 8, "offload": False,
                                "truncation_allowed": False}
                               for b in (1, 2) for n in (64, 128, 256, 384,
                                                      512, 768, 1024)],
            "primary_training_config": {"actual_padded_sequence_length": 256,
                                        "micro_batch_size": 1,
                                        "effective_batch_size": 8,
                                        "offload": False, "truncation_allowed": False},
        }
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        self.args = SimpleNamespace(
            manifest=self.manifest, old_base_eval=self.old,
            exclusions=self.exclusions, sealed_bank=self.sealed,
            plan=self.plan_path, base_binding=self.binding,
            adapter_receipt=None, expected_adapter_receipt_sha256=None,
            source="frozen", control="frozen",
            output=self.root / "score-public.json",
            private_output=self.root / "score-private.json", peer_private=None,
            preflight_only=True,
        )

    def change(self, **kwargs):
        return SimpleNamespace(**{**vars(self.args), **kwargs})

    def make_r18_receipt(self, source="r18-mezo", control="correct"):
        home = self.root / (source + "-" + control)
        adapter = home / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter-saved")
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        receipt = {
            "schema": "VERA_TRAININGDAY_R18_ARM_RECEIPT_V1",
            "arm": source.removeprefix("r18-"), "control": control,
            "recipe": RECIPES[source], "status": "TRAINED_UNSCORED",
            "fit_status": "FIT", "plan_sha256": canonical_sha(self.plan),
            "question_contract": NUMERIC_QUESTION_CONTRACT,
            "eval_custody_sha256": self.plan["eval_custody_sha256"],
            "eval_index_sha256": self.plan["eval_index_sha256"],
            "training_budget_seconds": self.plan["training_budget_seconds"],
            "eval_accuracy": None, "eval_nll": None, "eval_nll_sum": None,
            "eval_case_receipt_sha256": None,
            "eval_examples": 0, "eval_target_tokens": 0,
            "selected_id_sha256": self.plan["selected_id_sha256"],
            "selected_count": self.plan["selected_count"],
            "seed": self.plan["seed"],
            "corpus_manifest_sha256": CORPUS_SHA256,
            "train_content_sha256": self.plan["train_content_sha256"],
            "source_admission_status": self.plan["source_admission_status"],
            "tokenizer_binding_sha256": self.plan["tokenizer_binding_sha256"],
            "selected_fit_config": self.plan["primary_training_config"],
            "training_input_sha256": "2" * 64, "fit_probe_sha256": "3" * 64,
            "optimizer_steps": 0 if control == "zero" else 8,
            "trainable_digest_before": "4" * 64,
            "trainable_digest_after": "4" * 64 if control == "zero" else "5" * 64,
            "trainable_parameter_count": 10,
            "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
            "base_binding_sha256": _hash_file(self.binding),
            "adapter_model_sha256": _hash_file(adapter / "adapter_model.safetensors"),
            "adapter_config_sha256": _hash_file(adapter / "adapter_config.json"),
            "source_head": "9a6c3b72ef8ad425fa4df371ee10cf700972c8cd",
            "lora_target_modules": "all-linear", "lora_r": 4,
            "lora_alpha": 16, "quantization": "NF4_DOUBLE_BF16",
            "cpu_offload": False, "disk_offload": False,
            "truncation_allowed": False,
            "model_method": "MeZO-style LoRA-only ZO; frozen NF4 base",
            "zero_update": control == "zero",
        }
        path = home / "r18-receipt.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        return path, receipt

    def test_frozen_source_only_preflight_never_opens_bank_or_cuda(self):
        with patch("trainingday.r18_eval._required_base", return_value=(self.base, self.root)):
            with patch("trainingday.r18_eval._hash_file", side_effect=lambda p:
                       CORPUS_SHA256 if p == self.manifest else _hash_file(p)):
                with patch("trainingday.r18_eval._bank", side_effect=AssertionError(
                        "bank must stay sealed")):
                    report = evaluate(self.args)
        self.assertEqual(report["status"], "R18_ARM_PREFLIGHT_NO_BANK_OPENED_NO_MODEL_EXECUTION")
        self.assertFalse(self.args.output.exists())
        self.assertFalse(self.args.private_output.exists())

    def test_pinned_plan_detects_hash_or_policy_change(self):
        validated, sha = _plan(self.plan_path, self.binding, self.base)
        self.assertEqual(sha, canonical_sha(validated))
        for field, bad in (("eval_bank_source", "R16_OLD"),
                           ("no_eval_access_before_arm_freeze", False),
                           ("eval_examples", 63),
                           ("base_revision", "0" * 40),
                           ("base_binding_sha256", "7" * 64)):
            change = {**self.plan, field: bad}
            self.plan_path.write_text(json.dumps(change), encoding="utf-8")
            with self.subTest(field=field), self.assertRaises(AdmissionError):
                _plan(self.plan_path, self.binding, self.base)
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")

    def test_strict_r18_adapter_custody_positive_and_failures(self):
        for source in RECIPES:
            for control in ("zero", "correct", "corrupt"):
                path, raw = self.make_r18_receipt(source, control)
                args = self.change(source=source, control=control,
                                   adapter_receipt=path,
                                   expected_adapter_receipt_sha256=_hash_file(path))
                value, directory, digest = _adapter(args, self.plan, canonical_sha(self.plan))
                self.assertEqual(value, raw)
                self.assertTrue(directory.is_dir())
                self.assertEqual(digest, _hash_file(path))
                tampered = {**raw, "eval_accuracy": .75}
                path.write_text(json.dumps(tampered), encoding="utf-8")
                with self.assertRaises(AdmissionError):
                    _adapter(self.change(**{**vars(args),
                                           "expected_adapter_receipt_sha256": _hash_file(path)}),
                             self.plan, canonical_sha(self.plan))

    def test_adapter_hash_tamper_or_wrong_control_rejected(self):
        path, raw = self.make_r18_receipt()
        args = self.change(source="r18-mezo", control="correct",
                           adapter_receipt=path,
                           expected_adapter_receipt_sha256=_hash_file(path))
        (path.parent / "adapter" / "adapter_model.safetensors").write_bytes(b"tamper")
        with self.assertRaises(AdmissionError):
            _adapter(args, self.plan, canonical_sha(self.plan))
        with self.assertRaises(AdmissionError):
            _adapter(self.change(source="frozen", control="correct"),
                     self.plan, canonical_sha(self.plan))

    def test_r16_saved_qlora_receipt_requires_pinned_source_gate(self):
        home = self.root / "r16-run"
        home.mkdir()
        receipt_path = home / "training_receipt.json"
        record = {"recipe": "TRAININGDAY_QWEN35_QLORA_R16_CHECKPOINTABLE_MATCHED",
                  "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
                  "base_binding_sha256": _hash_file(self.binding)}
        receipt_path.write_text(json.dumps(record), encoding="utf-8")
        args = self.change(source="r16-qlora", control="correct",
                           adapter_receipt=receipt_path,
                           expected_adapter_receipt_sha256=_hash_file(receipt_path))
        with patch("trainingday.r18_eval._r16_adapter",
                   return_value=(record, home / "adapter")) as gate:
            _, adapter, _ = _adapter(args, self.plan, canonical_sha(self.plan))
        self.assertEqual(adapter, home / "adapter")
        self.assertEqual(gate.call_args.kwargs["required_mode"], "correct")
        with self.assertRaises(AdmissionError):
            _adapter(self.change(**{**vars(args),
                                    "expected_adapter_receipt_sha256": "0" * 64}),
                     self.plan, canonical_sha(self.plan))

    def test_private_custody_requires_owner_only_permissions(self):
        path = self.root / "private-seal.json"
        path.write_text("{}", encoding="utf-8")
        if os.name == "nt":
            # A child of a protected directory may already have a safe DACL.
            # Simulate an independent ACL-verifier rejection to prove that
            # private reads fail closed, then secure and verify the real file.
            from trainingday.r18_benchmark import BenchmarkError, _protect_private_path
            with patch("trainingday.r18_benchmark._windows_verify_owner_only",
                       side_effect=BenchmarkError("unsafe ACL")):
                with self.assertRaises(AdmissionError):
                    _json(path, private=True)
            _protect_private_path(path, directory=False)
        else:
            path.chmod(0o644)
            with self.assertRaises(AdmissionError):
                _json(path, private=True)
            path.chmod(0o600)
        self.assertEqual(_json(path, private=True), {})

    def test_selection_rederives_old16_r16_64_and_r18_64(self):
        old_ids = [f"reasoning-{i:064x}" for i in range(16)]
        r16_cases = [{"id": f"reasoning-{i+16:064x}", "prompt": f"old {i}",
                      "expected": str(i)} for i in range(64)]
        cases = [{"id": f"reasoning-{i+80:064x}", "prompt": f"new {i}",
                  "expected": str(i)} for i in range(64)]
        r16_ids = [r["id"] for r in r16_cases]
        pinned_plan = {**self.plan,
                       "r16_holdout_id_sha256": canonical_sha(sorted(r16_ids)),
                       "r16_eval_custody_sha256": canonical_sha({
                           "manifest": CORPUS_SHA256, "cases": r16_cases}),
                       "eval_custody_sha256": canonical_sha({
                           "manifest": CORPUS_SHA256, "cases": cases}),
                       "eval_index_sha256": canonical_sha(sorted(
                           [{"id": r["id"], "prompt_sha256": hashlib.sha256(
                               r["prompt"].encode()).hexdigest()} for r in cases],
                           key=lambda r: r["id"]))}
        self.exclusions.write_text(json.dumps({
            "schema": "VERA_TRAININGDAY_R18_PRIVATE_EVAL_EXCLUSIONS_V1",
            "old_r15_scored16_ids": old_ids, "r16_fresh64_ids": r16_ids,
            "r16_selected_id_sha256": BANK_ID_SHA256,
            "excluded_signature_count": 77}), encoding="utf-8")
        self.exclusions.chmod(0o600)
        self.sealed.write_text(json.dumps({
            "schema": PRIVATE_BANK_SCHEMA, "manifest_sha256": CORPUS_SHA256,
            "eval_custody_sha256": pinned_plan["eval_custody_sha256"],
            "eval_index_sha256": pinned_plan["eval_index_sha256"],
            "cases": cases}), encoding="utf-8")
        self.sealed.chmod(0o600)
        if os.name == "nt":
            from trainingday.r18_benchmark import _protect_private_path
            _protect_private_path(self.exclusions, directory=False)
            _protect_private_path(self.sealed, directory=False)
        r16_selection = {"selected_count": 64, "selected_id_sha256": BANK_ID_SHA256,
                         "cases": r16_cases}
        r18_selection = {"selected_count": 64, "cases": cases}
        with patch("trainingday.r18_eval._hash_file", side_effect=lambda p:
                   CORPUS_SHA256 if p == self.manifest else _hash_file(p)), \
             patch("trainingday.r18_eval._old_ids", return_value=old_ids), \
             patch("trainingday.r18_eval.preregister_r15_fresh_structural64",
                   return_value=r16_selection) as r16, \
             patch("trainingday.r18_eval.select_structural_cases",
                   return_value=r18_selection) as r18:
            self.assertEqual(_bank(self.manifest, self.old, self.exclusions,
                                   self.sealed, pinned_plan), cases)
            r16.assert_called_once_with(self.manifest, previously_scored_16_ids=old_ids)
            self.assertEqual(r18.call_args.kwargs["expected_scored_signature_count"], 77)
            self.assertEqual(r18.call_args.kwargs["previously_scored_ids"], old_ids + r16_ids)
            # A single re-used previously scored ID invalidates the new bank.
            r18_selection["cases"] = [r16_cases[0]] + cases[1:]
            with self.assertRaises(AdmissionError):
                _bank(self.manifest, self.old, self.exclusions, self.sealed, pinned_plan)

    def test_public_private_aggregation_and_blinded_pairwise(self):
        rows = [{"blind_case_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                 "exact": i % 3 == 0, "nll_sum": 1.5 + i / 100,
                 "target_tokens": 3} for i in range(64)]
        with patch("trainingday.r18_eval._required_base", return_value=(self.base, self.root)), \
             patch("trainingday.r18_eval._hash_file", side_effect=lambda p:
                   CORPUS_SHA256 if p == self.manifest else _hash_file(p)), \
             patch("trainingday.r18_eval._bank", return_value=[{}] * 64), \
             patch("trainingday.r18_eval._run_model", return_value=rows) as run:
            report = evaluate(self.change(preflight_only=False))
        self.assertEqual(report["greedy_exact_match"], 22)
        self.assertEqual(report["answer_eos_target_tokens"], 192)
        self.assertAlmostEqual(report["answer_eos_nll_per_token"],
                               sum(r["nll_sum"] for r in rows) / 192)
        self.assertEqual(report["protocol"], PROTOCOL)
        self.assertFalse(report["automatic_promotion"])
        if os.name == "nt":
            from trainingday.r18_benchmark import _windows_owner_sid, _windows_verify_owner_only
            _windows_verify_owner_only(self.args.private_output, _windows_owner_sid(),
                                       directory=False)
        else:
            self.assertEqual(oct(self.args.private_output.stat().st_mode & 0o777), "0o600")
        public_contents = self.args.output.read_text()
        for word in ("prompt", "expected", "prediction", "blind_case_sha256",
                     '"outcomes"', '"id"'):
            self.assertNotIn(word, public_contents)
        private = json.loads(self.args.private_output.read_text())
        self.assertEqual(private["outcome_sha256"], canonical_sha(rows))
        # Pairwise digest retained privately; never emits case IDs or answers.
        comparison = _paired(rows, self.args.private_output, canonical_sha(self.plan))
        self.assertEqual(comparison["this_only_correct"], 0)
        self.assertEqual(comparison["both_correct"], 22)
        self.assertEqual(comparison["both_incorrect"], 42)
        run.assert_called_once()

    def test_canonical_exact_answer_requires_greedy_integer_only(self):
        for prediction, answer, is_exact in (("12", "12", True), ("+12", "12", False),
                                             ("012", "12", False), ("12.", "12", False),
                                             ("12  ", "12", False), ("-4", "-4", True),
                                             ("12\n13", "12", False)):
            with self.subTest(prediction=prediction):
                self.assertEqual(_exact(prediction, answer), is_exact)


if __name__ == "__main__":
    unittest.main()
