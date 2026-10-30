"""CPU-only contract tests; absolutely no model/GPU/held-out access."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trainingday.admission import AdmissionError
from trainingday import r18_qlora as q


class QloraContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="r18-qlora-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "outputs").mkdir()
        self.rows = [
            {"id": f"train-{i:02d}", "family": "integer", "prompt": f"add {i}",
             "answer": str(i)} for i in range(8)
        ]
        self.base = {"model_id": q.BASE_ID, "revision": q.BASE_REVISION,
                     "files": {"tokenizer_config.json": "a" * 64}}
        self.plan = {
            "schema": q.PLAN_SCHEMA, "base_model": q.BASE_ID,
            "base_revision": q.BASE_REVISION, "base_binding_sha256": "b" * 64,
            "tokenizer_binding_sha256": "a" * 64,
            "corpus_manifest_sha256": q.R16_V2_MANIFEST_SHA256,
            "selected_id_sha256": q.selected_id_digest(self.rows),
            "train_content_sha256": q.train_content_digest(self.rows),
            "selected_count": 8, "seed": 41, "fit_policy": q.FIT_POLICY,
            "fit_candidates": q.candidate_schedule(),
            "primary_training_config": {
                "actual_padded_sequence_length": 256, "micro_batch_size": 1,
                "effective_batch_size": 8, "offload": False,
                "truncation_allowed": False},
            "eval_bank_source": q.EVAL_SOURCE, "eval_custody_sha256": "c" * 64,
            "eval_index_sha256": "f" * 64,
            "source_admission_status": "ADMITTED_LABYRINTH_TRAIN_ONLY_NO_WEIGHT_UPDATE",
            "training_budget_seconds": 60, "probe_budget_seconds": 120,
        }
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        self.args = argparse.Namespace(
            control="correct", run_id="r18-qlora-correct", output_root=self.root / "outputs",
            manifest=self.root / "manifest.json", producer_git_head=q.R16_PRODUCER_HEAD,
            base_binding=self.root / "base.json", plan=plan_path,
            cases=8, seed=41, learning_rate=0.00002, work_root=self.root,
            preflight_only=True, expected_initial_digest="d" * 64,
            initial_adapter=None, initial_adapter_model_sha256=None,
            initial_adapter_config_sha256=None,
        )

    def admitted(self):
        return (patch.object(q, "_required_base", return_value=(self.base, self.root)),
                patch.object(q, "_hash_file", side_effect=lambda p:
                             q.R16_V2_MANIFEST_SHA256 if p == self.args.manifest else "b" * 64),
                patch.object(q, "admit_labyrinth", return_value=({"selected_count": 8}, self.rows)))

    def test_selected_id_digest_preserves_admission_order_and_forbids_duplicates(self):
        self.assertEqual(q.selected_id_digest(self.rows), hashlib.sha256(
            "\n".join(r["id"] for r in self.rows).encode()).hexdigest())
        self.assertNotEqual(q.selected_id_digest(self.rows),
                            q.selected_id_digest(list(reversed(self.rows))))
        with self.assertRaisesRegex(AdmissionError, "duplicated"):
            q.selected_id_digest(self.rows + [self.rows[0]])

    def test_plan_accepts_only_ascending_full_tensor_sweep_without_offload(self):
        self.assertEqual([x["actual_padded_sequence_length"] for x in q.candidate_schedule()],
                         [64, 128, 256, 384, 512, 768, 1024] * 2)
        q._validate_fit_plan(self.plan)
        self.plan["fit_candidates"][0]["offload"] = True
        with self.assertRaisesRegex(AdmissionError, "fit policy"):
            q._validate_fit_plan(self.plan)

    def test_cpu_preflight_pins_base_manifest_ids_and_clean_split(self):
        p1, p2, p3 = self.admitted()
        with p1, p2, p3:
            report, cases, _, _ = q.preflight(self.args)
        self.assertEqual(report["status"], "SOURCE_ONLY_PREFLIGHT")
        self.assertEqual(report["base_revision"], q.BASE_REVISION)
        self.assertEqual(report["eval_examples"], 0)
        self.assertIsNone(report["eval_accuracy"])
        self.assertEqual([case["id"] for case in cases], [row["id"] for row in self.rows])
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(report["train_content_sha256"], self.plan["train_content_sha256"])

    def test_pin_rejects_other_revision_even_valid_binding(self):
        p1, p2, p3 = self.admitted()
        self.base["revision"] = "d" * 40
        with p1, p2, p3, self.assertRaisesRegex(AdmissionError, "exact"):
            q.preflight(self.args)

    def test_changed_selected_training_content_rejected(self):
        p1, p2, p3 = self.admitted()
        self.rows[0]["answer"] = "tampered"
        with p1, p2, p3, self.assertRaisesRegex(AdmissionError, "train content"):
            q.preflight(self.args)

    def test_trained_control_requires_physical_or_digest_initialization_proof(self):
        self.args.expected_initial_digest = None
        with self.assertRaisesRegex(AdmissionError, "paired zero"):
            q._initial_adapter_custody(self.args)
        self.args.expected_initial_digest = "d" * 64
        adapter = self.root / "initial"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"test-model")
        (adapter / "adapter_config.json").write_bytes(b"test-config")
        self.args.initial_adapter = adapter
        self.args.initial_adapter_model_sha256 = hashlib.sha256(b"test-model").hexdigest()
        self.args.initial_adapter_config_sha256 = hashlib.sha256(b"test-config").hexdigest()
        verified = q._initial_adapter_custody(self.args)
        self.assertEqual(verified["initialization_method"],
                         "PHYSICAL_R16_ZERO_ADAPTER_FP32_DIGEST")
        (adapter / "adapter_model.safetensors").write_bytes(b"tamper")
        with self.assertRaisesRegex(AdmissionError, "physical SHA"):
            q._initial_adapter_custody(self.args)

    def test_corrupt_arm_changes_only_labels_not_ids_or_prompts(self):
        p1, p2, p3 = self.admitted()
        with p1, p2, p3:
            normal, correct, _, _ = q.preflight(self.args)
        self.args.control = "corrupt"
        p1, p2, p3 = self.admitted()
        with p1, p2, p3:
            corrupt_report, corrupt, _, _ = q.preflight(self.args)
        self.assertEqual([x["id"] for x in correct], [x["id"] for x in corrupt])
        self.assertEqual([x["prompt"] for x in correct], [x["prompt"] for x in corrupt])
        self.assertNotEqual(normal["raw_training_input_sha256"],
                            corrupt_report["raw_training_input_sha256"])

    def test_strict_completion_only_mask_and_actual_padding_shape(self):
        class Tokenizer:
            eos_token = "!"
            pad_token_id = 0

            def apply_chat_template(self, messages, **kwargs):
                return "USER:" + messages[0]["content"] + "\nASSISTANT:"

            def __call__(self, text, **kwargs):
                return {"input_ids": [ord(ch) for ch in text]}

        rows = q._tensorize(Tokenizer(), [{"id": "one", "prompt": "2+2", "response": "4"}])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target_tokens"], 2)
        self.assertGreater(rows[0]["labels"].count(-100), 0)
        self.assertEqual(rows[0]["labels"][-2:], [ord("4"), ord("!")])
        self.assertEqual(rows[0]["labels"].count(-100),
                         len(rows[0]["input_ids"]) - 2)

        class TensorStub:
            long = "long"

            @staticmethod
            def tensor(items, *, dtype, device):
                assert dtype == "long" and device == "cuda"
                return items

        seq = len(rows[0]["input_ids"]) + 5
        batch = q._pad_batch(rows, seq, 0, TensorStub)
        self.assertEqual(len(batch["input_ids"][0]), seq)
        self.assertEqual(batch["attention_mask"][0][-5:], [0] * 5)
        self.assertEqual(batch["labels"][0][-5:], [-100] * 5)
        with self.assertRaisesRegex(RuntimeError, "no-truncation"):
            q._pad_batch(rows, seq - 6, 0, TensorStub)

    def test_synthetic_fit_probes_stop_at_first_oom_then_batch2(self):
        # This exercises process ordering without importing torch or using GPU.
        observed = []

        def fake_process(command, **kwargs):
            seq = int(command[command.index("--probe-seq") + 1])
            batch = int(command[command.index("--probe-batch") + 1])
            result = Path(command[command.index("--probe-result") + 1])
            observed.append((seq, batch))
            status = "FIT" if batch == 1 and seq <= 384 else "OOM"
            result.write_text(json.dumps({"max_seq_length": seq,
                                         "micro_batch_size": batch,
                                         "probe_is_synthetic": True,
                                         "actual_tensor_shape": [batch, seq],
                                         "status": status}))
            return SimpleNamespace(returncode=0)

        with patch.object(q.subprocess, "run", side_effect=fake_process):
            probes, selected, status = q.probe_sweep(self.args, self.root, self.plan)
        self.assertEqual(observed, [(64, 1), (128, 1), (256, 1),
                                    (384, 1), (512, 1), (64, 2)])
        self.assertEqual(status, "FIT")
        self.assertEqual(selected["max_seq_length"], 384)
        self.assertEqual(selected["micro_batch_size"], 1)
        self.assertEqual(probes[-1]["status"], "OOM")

    def test_4b_oom_is_reported_without_smaller_base(self):
        def fake_process(command, **kwargs):
            result = Path(command[command.index("--probe-result") + 1])
            result.write_text(json.dumps({"max_seq_length": 64,
                                         "micro_batch_size": 1,
                                         "actual_tensor_shape": [1, 64],
                                         "probe_is_synthetic": True,
                                         "status": "OOM"}))
            return SimpleNamespace(returncode=0)

        with patch.object(q.subprocess, "run", side_effect=fake_process):
            probes, choice, status = q.probe_sweep(self.args, self.root, self.plan)
        self.assertEqual(len(probes), 1)
        self.assertIsNone(choice)
        self.assertEqual(status, "BASE_4B_QLORA_OOM")

    def test_primary_training_never_uses_method_specific_max_fit(self):
        # A 1024/B2 frontier win is diagnostic; both arms must train B1/256.
        self.args.preflight_only = False
        summary = {"plan_sha256": "e" * 64, "training_budget_seconds": 60,
                   "peak_cuda_allocated_bytes": 0, "peak_cuda_reserved_bytes": 0,
                   "peak_cpu_rss_bytes": 0, "status": "SOURCE_ONLY_PREFLIGHT"}
        probes = [{"max_seq_length": 256, "micro_batch_size": 1,
                   "status": "FIT", "probe_is_synthetic": True,
                   "peak_cuda_allocated_bytes": 900,
                   "peak_cuda_reserved_bytes": 1000,
                   "peak_cpu_rss_bytes": 1200}]
        max_fit = {"max_seq_length": 1024, "micro_batch_size": 2,
                   "effective_batch_size": 8, "gradient_accumulation": 4}
        selected = []

        def fake_training(args, output, examples, chosen, receipt):
            selected.append(chosen)
            # Synthetic max-fit probes are separate processes. Their peaks
            # must not become the primary training run's peak counters.
            self.assertEqual(receipt["peak_cuda_allocated_bytes"], 0)
            self.assertEqual(receipt["peak_cuda_reserved_bytes"], 0)
            self.assertEqual(receipt["fit_probe_peak_cuda_allocated_bytes"], 900)
            self.assertEqual(receipt["fit_probe_peak_cuda_reserved_bytes"], 1000)
            self.assertEqual(receipt["fit_probe_peak_cpu_rss_bytes"], 1200)
            receipt["peak_cuda_allocated_bytes"] = 400
            receipt["peak_cuda_reserved_bytes"] = 600
            receipt["status"] = "TRAINED_UNSCORED"
            return receipt

        with patch.object(q, "preflight", return_value=(summary, [], self.root, self.plan)), \
             patch.object(q, "probe_sweep", return_value=(probes, max_fit, "FIT")), \
             patch.object(q, "_training_loop", side_effect=fake_training):
            result = q.run(self.args)
        self.assertEqual(selected, [{"max_seq_length": 256, "micro_batch_size": 1,
                                     "effective_batch_size": 8, "gradient_accumulation": 8}])
        self.assertEqual(result["secondary_max_fit"], max_fit)
        self.assertEqual(result["selected_fit_config"], self.plan["primary_training_config"])
        self.assertEqual(result["peak_cuda_reserved_bytes"], 600)
        self.assertEqual(result["fit_probe_peak_cuda_reserved_bytes"], 1000)

    def test_primary_oom_is_never_mislabeled_as_success(self):
        self.args.preflight_only = False
        summary = {"plan_sha256": "e" * 64, "training_budget_seconds": 60,
                   "peak_cuda_allocated_bytes": 0, "peak_cuda_reserved_bytes": 0,
                   "peak_cpu_rss_bytes": 0, "status": "SOURCE_ONLY_PREFLIGHT"}
        probes = [{"max_seq_length": 64, "micro_batch_size": 1,
                   "status": "OOM", "probe_is_synthetic": True}]
        with patch.object(q, "preflight", return_value=(summary, [], self.root, self.plan)), \
             patch.object(q, "probe_sweep", return_value=(probes, None, "BASE_4B_QLORA_OOM")), \
             patch.object(q, "_training_loop") as train:
            result = q.run(self.args)
        train.assert_not_called()
        self.assertEqual(result["status"], "OOM_NO_TRAINING")

    def test_conservative_guard_never_begins_an_unbudgeted_full_ga8_step(self):
        self.assertEqual(q.next_step_guard_seconds([]), 120.0)
        self.assertFalse(q.may_start_full_step(119.999, []))
        self.assertTrue(q.may_start_full_step(120.0, []))
        # The complete step, not one microbatch, sets the next reserve.
        self.assertEqual(q.next_step_guard_seconds([65.0, 72.0, 61.0]), 110.0)
        self.assertFalse(q.may_start_full_step(109.999, [65.0, 72.0, 61.0]))
        self.assertTrue(q.may_start_full_step(110.0, [65.0, 72.0, 61.0]))
        self.assertEqual(q.next_step_guard_seconds([1.0]), 30.0)
        self.assertFalse(q.may_start_full_step(29.0, [1.0]))
        with self.assertRaisesRegex(ValueError, "positive"):
            q.next_step_guard_seconds([0.0])

    def test_effective_batch_token_objective_matches_mezo_for_variable_targets(self):
        # MeZO's reference is token-weighted across the entire eight-example
        # update. A flat 1/GA8 scale is wrong when completion lengths differ.
        counts = [2, 3, 4, 5, 6, 7, 8, 2]
        example_losses = [0.2, 2.3, 0.9, 1.4, 0.3, 1.7, 0.5, 3.1]
        examples = [{"target_tokens": count} for count in counts]
        expected = sum(n * loss for n, loss in zip(counts, example_losses)) / sum(counts)
        unweighted = sum(example_losses) / len(example_losses)
        self.assertNotAlmostEqual(expected, unweighted, places=4)

        for batch_size in (1, 2, 4, 8):
            actual = 0.0
            scale_sum = 0.0
            for start in range(0, 8, batch_size):
                rows = examples[start:start + batch_size]
                tokens = counts[start:start + batch_size]
                losses = example_losses[start:start + batch_size]
                model_mean = sum(n * loss for n, loss in zip(tokens, losses)) / sum(tokens)
                scale = q.effective_batch_token_loss_scale(rows, examples)
                actual += model_mean * scale
                scale_sum += scale
            self.assertAlmostEqual(scale_sum, 1.0, places=14)
            self.assertAlmostEqual(actual, expected, places=14)

    def test_effective_batch_scale_fails_closed_on_invalid_counts(self):
        with self.assertRaisesRegex(ValueError, "target token"):
            q.effective_batch_token_loss_scale([{"target_tokens": 0}],
                                               [{"target_tokens": 0}])
        with self.assertRaisesRegex(ValueError, "target token"):
            q.effective_batch_token_loss_scale([{"target_tokens": 2}],
                                               [{"target_tokens": 2}, {"target_tokens": -1}])
        with self.assertRaisesRegex(ValueError, "target token"):
            q.effective_batch_token_loss_scale([{"target_tokens": 3}],
                                               [{"target_tokens": 2}])


if __name__ == "__main__":
    unittest.main()
