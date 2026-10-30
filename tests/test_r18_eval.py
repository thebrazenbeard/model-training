"""R18 independent source-only evaluator tests; no CUDA or model required."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from trainingday.admission import AdmissionError, _hash_file
from trainingday.eval_structural_r16 import BANK_ID_SHA256, CORPUS_SHA256
from trainingday.numeric_prompt import NUMERIC_QUESTION_CONTRACT
from trainingday.r18_eval import (
    BASE_MODEL, BASE_REVISION, EVAL_SOURCE, FREEZE_SCHEMA, PRIVATE_BANK_SCHEMA,
    PROTOCOL, RECIPES, _adapter, _bank, _exact, _json, _paired, _plan,
    _integer_semantic, _preunseal_inventory, _prompt_sha, _replay_rendered_chain6,
    _source_manifest,
    canonical_sha, evaluate,
    generate_novel_r18_cases,
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
        self.generator_source = self.root / "r18_eval_source.py"
        self.generator_source.write_text("# pinned test generator source\n", encoding="utf-8")
        self.source_manifest_path = self.root / "r18-external-chain6-source-manifest.json"
        self.private_nonce = "ab" * 32
        self.generated = generate_novel_r18_cases(private_nonce_hex=self.private_nonce)
        self.source_manifest = {
            "schema": "VERA_TRAININGDAY_R18_NOVEL_CHAIN6_EVAL_SOURCE_V1",
            "protocol": PROTOCOL,
            "evaluation_distribution": "EXTERNAL_OOD_SIX_OPERATOR_COMPOSITION_NOT_V2_HOLDOUT",
            "algorithm": "R18_SHA256_PRIORITY_SIX_STEP_PRIVATE_NONCE_DUAL_ORACLE_V2",
            "private_nonce_sha256": hashlib.sha256(
                bytes.fromhex(self.private_nonce)).hexdigest(),
            "generator_code_sha256": _hash_file(self.generator_source),
            "producer_git_head": "1" * 40,
            "seed": 18041, "count": 64, "chain_length": 6,
            "operator_vocabulary": ["add", "subtract", "multiply"],
            "start_values": [-20, 20], "add_sub_values": [-9, 9],
            "multiply_values": [-3, -2, 2, 3],
            "max_abs_intermediate": 100000,
            "max_prompt_chars": 220, "max_answer_chars": 8,
            "older_v2_corpus_manifest_sha256": CORPUS_SHA256,
            "older_v2_chain_length_range": [2, 5],
            "previously_scored_v2_signature_count": 77,
            "older_v2_unseen_signatures_after_r16": 0,
            "selected_unique_signature_count": 64,
            "selected_id_sha256": hashlib.sha256(
                "\n".join(r["id"] for r in self.generated).encode()).hexdigest(),
            "selected_signature_sha256": hashlib.sha256(
                "\n".join(r["signature_sha256"] for r in self.generated).encode()).hexdigest(),
            "private_case_content_sha256": canonical_sha(self.generated),
        }
        self.source_manifest_path.write_text(json.dumps(self.source_manifest), encoding="utf-8")
        self.source_sha = canonical_sha(self.source_manifest)
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
            "eval_source_manifest_sha256": self.source_sha,
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
            plan=self.plan_path, eval_source_manifest=self.source_manifest_path,
            generator_source=self.generator_source, base_binding=self.binding,
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
                        "bank must stay sealed")), \
                     patch("trainingday.r18_eval.generate_novel_r18_cases",
                           side_effect=AssertionError("answer-bearing generation before freeze")):
                    report = evaluate(self.args)
        self.assertEqual(report["status"], "R18_ARM_PREFLIGHT_NO_BANK_OPENED_NO_MODEL_EXECUTION")
        self.assertFalse(self.args.output.exists())
        self.assertFalse(self.args.private_output.exists())

    def test_scoring_without_all_arm_freeze_inventory_cannot_unseal(self):
        # A single frozen-control request used to reach _bank() directly.
        with patch("trainingday.r18_eval._required_base", return_value=(self.base, self.root)), \
             patch("trainingday.r18_eval._hash_file", side_effect=lambda p:
                   CORPUS_SHA256 if p == self.manifest else _hash_file(p)), \
             patch("trainingday.r18_eval._bank", side_effect=AssertionError(
                   "private bank opened before complete freeze")), \
             patch("trainingday.r18_eval._run_model", side_effect=AssertionError(
                   "model scored without complete freeze")):
            with self.assertRaisesRegex(AdmissionError, "pre-unseal"):
                evaluate(self.change(preflight_only=False))

    def make_freeze_inventory(self):
        from trainingday.r18_benchmark import make_plan, selected_id_digest, validate_plan
        rows = [{"id": f"train-{i}", "split": "train", "training_use": "ADMITTED",
                 "prompt": f"What is {i} plus one?", "answer": str(i+1)}
                for i in range(8)]
        plan = make_plan(
            selected_rows=rows, admission_selected_id_sha256=selected_id_digest(rows),
            source_admission_status="ADMITTED_SOURCE_ONLY_NOT_TRAINED",
            eval_index=[{"id": f"r18-holdout-{i}",
                         "prompt_sha256": hashlib.sha256(
                             f"r18-prompt-{i}".encode()).hexdigest()}
                        for i in range(64)],
            r16_holdout_ids=[f"r16-fresh64-{i}" for i in range(64)],
            base_revision="1" * 40, base_binding_sha256="a" * 64,
            tokenizer_binding_sha256="b" * 64, corpus_manifest_sha256="c" * 64,
            eval_custody_sha256="d" * 64, eval_source_manifest_sha256="f" * 64,
            r16_eval_custody_sha256="e" * 64,
            seed=41, training_budget_seconds=100.0, probe_budget_seconds=5.0)
        receipts = []
        for arm in ("mezo", "qlora"):
            for control in ("frozen", "zero", "correct", "corrupt"):
                trained = control in ("correct", "corrupt")
                seconds = 96.0 if trained else 0.0
                receipt = {
                    "schema": "VERA_TRAININGDAY_R18_ARM_RECEIPT_V1",
                    "arm": arm, "control": control, "run_id": f"{arm}-{control}",
                    "plan_sha256": validate_plan(plan),
                    **{key: plan[key] for key in (
                        "base_model", "base_revision", "base_binding_sha256",
                        "tokenizer_binding_sha256", "corpus_manifest_sha256",
                        "selected_id_sha256", "train_content_sha256",
                        "source_admission_status", "eval_custody_sha256",
                        "eval_index_sha256", "seed")},
                    "training_budget_seconds": 100.0,
                    "training_input_sha256": "c" * 64 if control == "corrupt" else "b" * 64,
                    "fit_probe_sha256": "f" * 64,
                    "selected_fit_config": plan["primary_training_config"],
                    "max_prepared_example_tokens": 96,
                    "optimizer_steps": 8 if trained else 0,
                    "unique_examples_seen": 8 if trained else 0,
                    "example_presentations": 64 if trained else 0,
                    "input_tokens_presented": 6400 if trained else 0,
                    "forward_sample_tokens": (12800 if arm == "mezo" else 6400) if trained else 0,
                    "backward_sample_tokens": (0 if arm == "mezo" else 6400) if trained else 0,
                    "padded_forward_tokens": (16384 if arm == "mezo" else 8192) if trained else 0,
                    "padded_backward_tokens": (0 if arm == "mezo" else 8192) if trained else 0,
                    "supervised_target_tokens": 320 if trained else 0,
                    "peak_cuda_allocated_bytes": 500000,
                    "peak_cuda_reserved_bytes": 600000,
                    "peak_nvml_used_bytes": 700000, "peak_cpu_rss_bytes": 800000,
                    "trainable_digest_before": "a" * 64,
                    "trainable_digest_after": "b" * 64 if trained else "a" * 64,
                    "train_start_monotonic_ns": 100_000_000_000,
                    "train_stop_monotonic_ns": 100_000_000_000 + round(seconds * 1e9),
                    "train_deadline_monotonic_ns": 200_000_000_000,
                    "optimizer_window_seconds": seconds,
                    "stage_seconds": {"load": 2.0, "setup": 3.0, "train": seconds,
                                      "save": 1.0, "eval": 2.0},
                    "external_process_wall_seconds": 10.0 + seconds,
                    "status": "TRAINED_UNSCORED", "eval_examples": 0,
                    "eval_correct": None, "eval_accuracy": None,
                    "eval_nll": None, "eval_nll_sum": None,
                    "eval_target_tokens": 0, "eval_case_receipt_sha256": None,
                }
                receipts.append(receipt)
        entries = []
        for rec in receipts:
            if rec["control"] != "frozen":
                adapter_dir = self.root / f"{rec['arm']}-{rec['control']}" / "adapter"
                adapter_dir.mkdir(parents=True)
                model = adapter_dir / "adapter_model.safetensors"
                config = adapter_dir / "adapter_config.json"
                model.write_bytes(b"test-only adapter bytes" + rec["run_id"].encode())
                config.write_text("{}", encoding="utf-8")
                rec["adapter_model_sha256"] = _hash_file(model)
                rec["adapter_config_sha256"] = _hash_file(config)
                receipt_dir = adapter_dir.parent
            else:
                receipt_dir = self.root
            path = receipt_dir / f"{rec['arm']}-{rec['control']}-receipt.json"
            path.write_text(json.dumps(rec), encoding="utf-8")
            entries.append({"arm": rec["arm"], "control": rec["control"],
                            "receipt_path": str(path), "receipt_sha256": _hash_file(path)})
        inventory = {"schema": FREEZE_SCHEMA, "plan_sha256": validate_plan(plan),
                     "arms": entries}
        path = self.root / "freeze-inventory.json"
        path.write_text(json.dumps(inventory), encoding="utf-8")
        return plan, inventory, path

    def test_complete_unscored_eight_arm_freeze_is_hash_bound(self):
        plan, inventory, path = self.make_freeze_inventory()
        args = self.change(freeze_inventory=path,
                           expected_freeze_inventory_sha256=_hash_file(path))
        digest, pinned, saved = _preunseal_inventory(
            args, plan, inventory["plan_sha256"])
        self.assertEqual(digest, _hash_file(path))
        self.assertEqual(len(pinned), 8)
        self.assertEqual(len(saved), 12)
        # Repeated/partial controls, timing-invalid arms, and altered receipts
        # must be denied before a single sealed answer can be read.
        for mutation in (lambda x: x["arms"].pop(),
                         lambda x: x["arms"].__setitem__(1, x["arms"][0]),
                         lambda x: x.update(plan_sha256="0" * 64)):
            altered = json.loads(json.dumps(inventory))
            mutation(altered)
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    AdmissionError, "pre-unseal"):
                _preunseal_inventory(self.change(
                    freeze_inventory=path,
                    expected_freeze_inventory_sha256=_hash_file(path)),
                    plan, inventory["plan_sha256"])
        path.write_text(json.dumps(inventory), encoding="utf-8")
        rec_path = Path(inventory["arms"][0]["receipt_path"])
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        original_receipt_bytes = rec_path.read_bytes()
        rec["train_stop_monotonic_ns"] += 600_000_000_000
        rec_path.write_text(json.dumps(rec), encoding="utf-8")
        with self.assertRaisesRegex(AdmissionError, "pre-unseal"):
            _preunseal_inventory(args, plan, inventory["plan_sha256"])
        rec_path.write_bytes(original_receipt_bytes)
        saved = Path(next(item["receipt_path"] for item in inventory["arms"]
                          if item["control"] == "correct" and
                          item["arm"] == "mezo")).parent / "adapter" / "adapter_model.safetensors"
        saved.write_bytes(b"fake updated adapter")
        with self.assertRaisesRegex(AdmissionError, "adapter bytes"):
            _preunseal_inventory(args, plan, inventory["plan_sha256"])

    def test_cuda_rejection_happens_before_private_bank_unseal(self):
        with patch("trainingday.r18_eval._required_base", return_value=(self.base, self.root)), \
             patch("trainingday.r18_eval._hash_file", side_effect=lambda p:
                   CORPUS_SHA256 if p == self.manifest else _hash_file(p)), \
             patch("trainingday.r18_eval._preunseal_inventory",
                   return_value=("a" * 64, {}, {})), \
             patch("trainingday.r18_eval._cuda_preflight",
                   side_effect=RuntimeError("CUDA unavailable")), \
             patch("trainingday.r18_eval._bank", side_effect=AssertionError(
                   "bank must remain sealed if CUDA preflight fails")):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                evaluate(self.change(preflight_only=False))

    def test_full_model_load_failure_must_not_open_private_bank(self):
        with patch("trainingday.r18_eval._required_base", return_value=(self.base, self.root)), \
             patch("trainingday.r18_eval._hash_file", side_effect=lambda p:
                   CORPUS_SHA256 if p == self.manifest else _hash_file(p)), \
             patch("trainingday.r18_eval._preunseal_inventory",
                   return_value=("a" * 64, {}, {})), \
             patch("trainingday.r18_eval._cuda_preflight"), \
             patch("trainingday.r18_eval._load_model", create=True,
                   side_effect=RuntimeError("full model OOM")), \
             patch("trainingday.r18_eval._bank", side_effect=AssertionError(
                   "private bank opened before full-model load")):
            with self.assertRaisesRegex(RuntimeError, "full model OOM"):
                evaluate(self.change(preflight_only=False))

    def test_loaded_model_identity_and_cleanup_on_scoring_failure(self):
        order = []
        loaded = Mock()
        loaded.close.side_effect = lambda: order.append("close")

        def preload(*_args):
            order.append("preload")
            return loaded

        def bank(*_args):
            order.append("bank")
            return [{}] * 64

        def score(_cases, session, _plan_sha):
            self.assertIs(session, loaded)
            order.append("score")
            raise RuntimeError("synthetic scoring failure")

        with patch("trainingday.r18_eval._required_base", return_value=(self.base, self.root)), \
             patch("trainingday.r18_eval._hash_file", side_effect=lambda p:
                   CORPUS_SHA256 if p == self.manifest else _hash_file(p)), \
             patch("trainingday.r18_eval._preunseal_inventory",
                   return_value=("a" * 64, {}, {})), \
             patch("trainingday.r18_eval._cuda_preflight"), \
             patch("trainingday.r18_eval._load_model", create=True, side_effect=preload), \
             patch("trainingday.r18_eval._bank", side_effect=bank), \
             patch("trainingday.r18_eval._run_model", side_effect=score):
            with self.assertRaisesRegex(RuntimeError, "synthetic scoring failure"):
                evaluate(self.change(preflight_only=False))
        self.assertEqual(order, ["preload", "bank", "score", "close"])

    def test_loaded_model_released_if_private_bank_integrity_fails(self):
        order = []
        loaded = Mock()
        loaded.close.side_effect = lambda: order.append("close")

        def bank(*_args):
            order.append("bank")
            raise AdmissionError("sealed bank integrity failure")

        with patch("trainingday.r18_eval._required_base", return_value=(self.base, self.root)), \
             patch("trainingday.r18_eval._hash_file", side_effect=lambda p:
                   CORPUS_SHA256 if p == self.manifest else _hash_file(p)), \
             patch("trainingday.r18_eval._preunseal_inventory",
                   return_value=("a" * 64, {}, {})), \
             patch("trainingday.r18_eval._cuda_preflight"), \
             patch("trainingday.r18_eval._load_model", return_value=loaded), \
             patch("trainingday.r18_eval._bank", side_effect=bank), \
             patch("trainingday.r18_eval._run_model", side_effect=AssertionError(
                   "scoring must not begin when bank validation fails")):
            with self.assertRaisesRegex(AdmissionError, "sealed bank integrity"):
                evaluate(self.change(preflight_only=False))
        self.assertEqual(order, ["bank", "close"])

    def test_pinned_plan_detects_hash_or_policy_change(self):
        validated, sha = _plan(self.plan_path, self.binding, self.base, self.source_sha)
        self.assertEqual(sha, canonical_sha(validated))
        for field, bad in (("eval_bank_source", "R16_OLD"),
                           ("no_eval_access_before_arm_freeze", False),
                           ("eval_examples", 63),
                           ("base_revision", "0" * 40),
                           ("base_binding_sha256", "7" * 64)):
            change = {**self.plan, field: bad}
            self.plan_path.write_text(json.dumps(change), encoding="utf-8")
            with self.subTest(field=field), self.assertRaises(AdmissionError):
                _plan(self.plan_path, self.binding, self.base, self.source_sha)
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

    def test_selection_rederives_old16_r16_64_and_external_long_chain64(self):
        old_ids = [f"reasoning-{i:064x}" for i in range(16)]
        r16_cases = [{"id": f"reasoning-{i+16:064x}", "prompt": f"old {i}",
                      "expected": str(i)} for i in range(64)]
        cases = self.generated
        r16_ids = [r["id"] for r in r16_cases]
        pinned_plan = {**self.plan,
                       "r16_holdout_id_sha256": canonical_sha(sorted(r16_ids)),
                       "r16_eval_custody_sha256": canonical_sha({
                           "manifest": CORPUS_SHA256, "cases": r16_cases}),
                       "eval_custody_sha256": canonical_sha({
                           "source_manifest_sha256": self.source_sha, "cases": cases}),
                       "eval_index_sha256": canonical_sha(sorted(
                           [{"id": r["id"], "prompt_sha256": _prompt_sha(r["prompt"])}
                            for r in cases],
                           key=lambda r: r["id"]))}
        self.exclusions.write_text(json.dumps({
            "schema": "VERA_TRAININGDAY_R18_PRIVATE_EVAL_EXCLUSIONS_V1",
            "old_r15_scored16_ids": old_ids, "r16_fresh64_ids": r16_ids,
            "r16_selected_id_sha256": BANK_ID_SHA256,
            "excluded_signature_count": 77}), encoding="utf-8")
        self.exclusions.chmod(0o600)
        self.sealed.write_text(json.dumps({
            "schema": PRIVATE_BANK_SCHEMA, "manifest_sha256": CORPUS_SHA256,
            "private_nonce_hex": self.private_nonce,
            "eval_source_manifest_sha256": self.source_sha,
            "eval_custody_sha256": pinned_plan["eval_custody_sha256"],
            "eval_index_sha256": pinned_plan["eval_index_sha256"],
            "cases": cases}), encoding="utf-8")
        self.sealed.chmod(0o600)
        if os.name == "nt":
            from trainingday.r18_benchmark import _protect_private_path
            _protect_private_path(self.exclusions, directory=False)
            _protect_private_path(self.sealed, directory=False)
        r16_selection = {"selected_count": 64, "signature_count": 64,
                         "selected_id_sha256": BANK_ID_SHA256,
                         "cases": r16_cases}
        with patch("trainingday.r18_eval._hash_file", side_effect=lambda p:
                   CORPUS_SHA256 if p == self.manifest else _hash_file(p)), \
             patch("trainingday.r18_eval._old_ids", return_value=old_ids), \
             patch("trainingday.r18_eval.preregister_r15_fresh_structural64",
                   return_value=r16_selection) as r16:
            self.assertEqual(_bank(self.manifest, self.old, self.exclusions,
                                   self.sealed, pinned_plan, self.source_manifest), cases)
            r16.assert_called_once_with(self.manifest, previously_scored_16_ids=old_ids)
            # A single re-used previously scored ID invalidates sealed data.
            modified = json.loads(self.sealed.read_text())
            modified["cases"][0]["id"] = r16_ids[0]
            self.sealed.write_text(json.dumps(modified), encoding="utf-8")
            if os.name == "nt":
                from trainingday.r18_benchmark import _protect_private_path
                _protect_private_path(self.sealed, directory=False)
            with self.assertRaises(AdmissionError):
                _bank(self.manifest, self.old, self.exclusions, self.sealed,
                      pinned_plan, self.source_manifest)

    def test_external_generator_is_deterministic_rendered_and_source_pinned(self):
        actual = generate_novel_r18_cases(private_nonce_hex=self.private_nonce)
        self.assertEqual(actual, self.generated)
        self.assertEqual(canonical_sha(actual),
                         "3d3355c03ab267c56197fd8bf6d07c8518c00c146cf72204030186630b834d19")
        self.assertEqual(len({r["signature_sha256"] for r in actual}), 64)
        for case in actual:
            answer, parsed, peak = _replay_rendered_chain6(case["prompt"])
            self.assertEqual(str(answer), case["expected"])
            self.assertEqual(len(parsed), 6)
            self.assertLessEqual(peak, 100000)
            self.assertLessEqual(len(case["prompt"]), 220)
        manifest, source_sha = _source_manifest(self.source_manifest_path, self.generator_source)
        self.assertEqual(source_sha, self.source_sha)
        self.assertEqual(manifest, self.source_manifest)
        self.generator_source.write_text("# changed generator bytes\n", encoding="utf-8")
        with self.assertRaises(AdmissionError):
            _source_manifest(self.source_manifest_path, self.generator_source)

    def test_private_nonce_is_never_exported_in_public_plan_or_source(self):
        self.assertNotIn(self.private_nonce, self.plan_path.read_text(encoding="utf-8"))
        self.assertNotIn(self.private_nonce,
                         self.source_manifest_path.read_text(encoding="utf-8"))
        with self.assertRaises(AdmissionError):
            generate_novel_r18_cases(private_nonce_hex="bad")

    def test_rendered_prompt_parser_rejects_semantic_corruption(self):
        example = self.generated[0]["prompt"]
        for prompt in (example.replace("Start with ", "Beginning with "),
                       example.replace("add ", "subtract ", 1) + " trailing",
                       example.replace("multiply ", "multiply 0. add ", 1)):
            with self.subTest(prompt=prompt), self.assertRaises(AdmissionError):
                _replay_rendered_chain6(prompt)

    def test_public_private_aggregation_and_blinded_pairwise(self):
        rows = [{"blind_case_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                 "exact": i % 3 == 0, "canonical_format_exact": i % 3 == 0,
                 "prompt_tokens": 40, "multiply_count": i % 7,
                 "answer_digits": 3, "peak_abs_intermediate": 200,
                 "nll_sum": 1.5 + i / 100,
                 "target_tokens": 3} for i in range(64)]
        loaded = Mock()
        with patch("trainingday.r18_eval._required_base", return_value=(self.base, self.root)), \
             patch("trainingday.r18_eval._hash_file", side_effect=lambda p:
                   CORPUS_SHA256 if p == self.manifest else _hash_file(p)), \
             patch("trainingday.r18_eval._preunseal_inventory",
                   return_value=(_hash_file(self.plan_path), {}, {})), \
             patch("trainingday.r18_eval._cuda_preflight"), \
             patch("trainingday.r18_eval._load_model", return_value=loaded), \
             patch("trainingday.r18_eval._bank", return_value=[{}] * 64), \
             patch("trainingday.r18_eval._run_model", return_value=rows) as run:
            report = evaluate(self.change(preflight_only=False,
                                          freeze_inventory=self.plan_path))
        self.assertEqual(report["greedy_exact_match"], 22)
        self.assertEqual(report["greedy_integer_semantic_correct"], 22)
        self.assertEqual(report["greedy_canonical_format_correct"], 22)
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
        for word in ('"prompt"', '"expected"', '"prediction"', "blind_case_sha256",
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
        self.assertIs(run.call_args.args[1], loaded)
        loaded.close.assert_called_once()

    def test_canonical_exact_answer_requires_greedy_integer_only(self):
        for prediction, answer, is_exact in (("12", "12", True), ("+12", "12", False),
                                             ("012", "12", False), ("12.", "12", False),
                                             ("12  ", "12", False), ("-4", "-4", True),
                                             ("12\n13", "12", False)):
            with self.subTest(prediction=prediction):
                self.assertEqual(_exact(prediction, answer), is_exact)
        self.assertTrue(_integer_semantic("+012", "12"))
        self.assertFalse(_integer_semantic("Answer: 12", "12"))


if __name__ == "__main__":
    unittest.main()
