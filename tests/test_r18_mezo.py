"""CPU-only source, custody, and finite-difference tests; no model/GPU import."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trainingday.admission import AdmissionError
from trainingday.r18_mezo import (
    BASE_REVISION, canonical_hash, direction_seed, finite_difference,
    fit_sweep, paired_zo_step, prepare_run, selected_id_digest, selected_rows,
    tokenize_cases, validate_source,
)


ROWS = [
    {"id": "case-1", "family": "arithmetic", "prompt": "1 + 1?", "answer": "2"},
    {"id": "case-2", "family": "arithmetic", "prompt": "2 + 3?", "answer": "5"},
]


class R18SourceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text('{"schema":"VERA_TRAININGDAY_ADMITTED_SHARDS_V1"}')
        self.base_binding = self.root / "binding.json"
        self.base_binding.write_text("source-only-test")
        self.output_root = self.root / "output"
        self.output_root.mkdir()
        self.args = argparse.Namespace(
            manifest=self.manifest, base_binding=self.base_binding,
            producer_git_head=None, work_root=self.root,
            output_root=self.output_root, run_id="r18-correct", mode="correct",
            steps=8, cases=2, seed=41, learning_rate=1e-3, epsilon=1e-3,
            micro_batch_size=1, examples_per_update=8, max_seq_length=512,
            budget_seconds=531.0, checkpoint_every=8,
            expected_initial_digest="a" * 64, fit_probe_sha256=None,
            preflight_only=True, fit_sweep_only=False,
            sweep_batches=[1], sweep_lengths=[64, 128, 256, 384, 512, 768, 1024],
            initial_adapter=None, initial_adapter_sha256=None,
            resume_checkpoint=None, resume_receipt_sha256=None,
        )
        self.summary = {"selected_id_sha256": selected_id_digest(ROWS),
                        "manifest_sha256": "b" * 64}
        self.admit = patch("trainingday.r18_mezo.admit",
                           return_value=(self.summary, ROWS)).start()
        self.base = patch("trainingday.r18_mezo._required_base",
                          return_value=({"revision": BASE_REVISION}, self.root)).start()
        self.addCleanup(patch.stopall)

    def test_source_preflight_is_read_only_and_matches_r16_selection_order(self):
        preflight, cases, path = validate_source(self.args)
        self.assertEqual(preflight["selected_id_sha256"], selected_id_digest(ROWS))
        self.assertEqual(path, self.root)
        self.assertEqual([x["id"] for x in cases], ["case-1", "case-2"])
        self.assertEqual(cases[0]["prompt"],
                         "1 + 1?\nReply with only the final integer. No steps or other words.")
        self.assertEqual(preflight["arm"], "mezo")
        self.assertEqual(preflight["training_input_sha256"], hashlib.sha256(
            json.dumps([{"prompt": c["prompt"], "response": c["response"]}
                        for c in cases], sort_keys=True, ensure_ascii=False)
            .encode("utf-8")).hexdigest())
        self.assertFalse((self.output_root / self.args.run_id).exists())

    def test_corrupt_control_uses_same_ids_but_changes_labels(self):
        correct, _, _ = validate_source(self.args)
        self.args.mode = "corrupt"
        corrupt, cases, _ = validate_source(self.args)
        self.assertEqual(corrupt["selected_id_sha256"], correct["selected_id_sha256"])
        self.assertNotEqual(corrupt["training_input_sha256"], correct["training_input_sha256"])
        self.assertEqual([c["response"] for c in cases], ["5", "2"])

    def test_rejects_non_update_and_missing_zero_digest(self):
        self.args.steps = 0
        with self.assertRaisesRegex(AdmissionError, "zero arm"):
            validate_source(self.args)
        self.args.steps = 8
        self.args.expected_initial_digest = None
        with self.assertRaisesRegex(AdmissionError, "zero-arm initial"):
            validate_source(self.args)

    def test_rejects_misaligned_source_selection(self):
        self.summary["selected_id_sha256"] = "0" * 64
        with self.assertRaisesRegex(AdmissionError, "selected ID"):
            validate_source(self.args)

    def test_requires_exact_pinned_zero_adapter_bytes(self):
        adapter = self.root / "r16-zero"
        adapter.mkdir()
        weights = adapter / "adapter_model.safetensors"
        weights.write_bytes(b"known-test-adapter")
        (adapter / "adapter_config.json").write_text('{"peft_type":"LORA"}')
        self.args.initial_adapter = adapter
        self.args.initial_adapter_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
        validate_source(self.args)
        weights.write_bytes(b"tampered")
        with self.assertRaisesRegex(AdmissionError, "tampered"):
            validate_source(self.args)

    def test_zero_arm_has_no_expected_init_digest_requirement(self):
        self.args.mode = "zero"
        self.args.steps = 0
        self.args.expected_initial_digest = None
        zero, cases, _ = validate_source(self.args)
        self.assertEqual(zero["steps_cap"], 0)
        self.assertEqual(len(cases), 2)


class R18MathTests(unittest.TestCase):
    def test_sweep_uses_actual_padded_shapes_and_stops_after_oom(self):
        class FakeCuda:
            class OutOfMemoryError(RuntimeError):
                pass

            def empty_cache(self):
                pass

            def reset_peak_memory_stats(self, device):
                pass

            def max_memory_allocated(self, device):
                return 1000

            def max_memory_reserved(self, device):
                return 2000

            def synchronize(self, device):
                pass

        fake_torch = argparse.Namespace(cuda=FakeCuda())
        args = argparse.Namespace(sweep_batches=[1], sweep_lengths=[64, 128, 256, 512],
                                  seed=41, epsilon=0.001)
        case = {"id": "x", "input_ids": [1] * 80,
                "labels": [-100] * 78 + [2, 3], "target_tokens": 2,
                "input_tokens": 80}

        def loss(_model, _torch, _tok, _samples, _batch, length):
            if length == 256:
                raise FakeCuda.OutOfMemoryError("CUDA out of memory")
            return 1.0

        with patch("trainingday.r18_mezo._noise_delta"), \
             patch("trainingday.r18_mezo._snapshot_trainables", return_value={}), \
             patch("trainingday.r18_mezo._restore_trainables"), \
             patch("trainingday.r18_mezo.paired_batch_loss", side_effect=loss):
            result = fit_sweep(None, [], fake_torch, None, [case], args)
        self.assertEqual([p["fit_status"] for p in result["probes"]],
                         ["infeasible", "fits", "oom"])
        self.assertEqual(result["probes"][1]["actual_tensor_shape"], [1, 128])
        self.assertEqual(result["probes"][1]["actual_nonpad_tokens"], 80)
        self.assertTrue(result["stopped_after_oom_requires_fresh_process"])

    def test_completion_only_labels_exactly_match_tokenized_chat_prefix(self):
        class DummyTokenizer:
            eos_token = "!"
            pad_token_id = 0

            def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                                    enable_thinking):
                self_calls = messages[0]["content"]
                return f"<U>{self_calls}<A>"

            def __call__(self, text, add_special_tokens):
                return {"input_ids": [ord(char) for char in text]}

        prepared, report = tokenize_cases(DummyTokenizer(), [
            {"id": "one", "prompt": "7?", "response": "7"}], 100)
        prefix = "<U>7?<A>"
        self.assertEqual(prepared[0]["input_ids"], [ord(c) for c in prefix + "7!"])
        self.assertEqual(prepared[0]["labels"],
                         [-100] * len(prefix) + [ord("7"), ord("!")])
        self.assertEqual(prepared[0]["target_tokens"], 2)
        self.assertEqual(report["max_input_tokens"], len(prefix) + 2)
        with self.assertRaisesRegex(RuntimeError, "token budget"):
            tokenize_cases(DummyTokenizer(), [
                {"id": "one", "prompt": "7?", "response": "7"}], 4)

    def test_direction_replay_is_stable_and_step_specific(self):
        self.assertEqual(direction_seed(41, 1), direction_seed(41, 1))
        self.assertNotEqual(direction_seed(41, 1), direction_seed(41, 2))

    def test_two_sided_estimate_and_actual_parameter_update(self):
        # Replace only RNG perturbation with a fixed scalar Gaussian direction
        # z=2, thereby testing the real paired_zo_step control flow and signs.
        p = {"value": 1.0}

        def noise(_tensors, _torch, seed, scale):
            self.assertEqual(seed, 99)
            p["value"] += 2.0 * scale

        def snapshot(_):
            return {"lora_A": p["value"]}

        def restore(_, _torch, saved):
            p["value"] = saved["lora_A"]

        with patch("trainingday.r18_mezo._noise_delta", side_effect=noise), \
             patch("trainingday.r18_mezo._snapshot_trainables", side_effect=snapshot), \
             patch("trainingday.r18_mezo._restore_trainables", side_effect=restore):
            plus, minus, derivative = paired_zo_step(
                [("lora_A", p)], None, seed=99, epsilon=0.01,
                learning_rate=0.1, loss_fn=lambda: p["value"] ** 2)
        self.assertAlmostEqual(plus, 1.02**2)
        self.assertAlmostEqual(minus, 0.98**2)
        self.assertAlmostEqual(derivative, 4.0)
        self.assertAlmostEqual(p["value"], 0.2)

    def test_forward_exception_restores_unperturbed_tensor(self):
        p = {"value": 1.0}

        def noise(_tensors, _torch, seed, scale):
            p["value"] += 3.0 * scale

        with patch("trainingday.r18_mezo._noise_delta", side_effect=noise), \
             patch("trainingday.r18_mezo._snapshot_trainables", side_effect=lambda _: {"lora_A": p["value"]}), \
             patch("trainingday.r18_mezo._restore_trainables",
                   side_effect=lambda _, _torch, saved: p.update(value=saved["lora_A"])):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                paired_zo_step([("lora_A", p)], None, seed=99, epsilon=0.01,
                               learning_rate=0.1,
                               loss_fn=lambda: (_ for _ in ()).throw(RuntimeError("forward failed")))
        self.assertAlmostEqual(p["value"], 1.0)

    def test_mid_perturbation_failure_restores_original(self):
        p = {"value": 1.0}
        calls = 0

        def failing_noise(_tensors, _torch, seed, scale):
            nonlocal calls
            calls += 1
            p["value"] += 3.0 * scale
            if calls == 2:
                raise RuntimeError("CUDA out of memory mid-perturbation")

        with patch("trainingday.r18_mezo._noise_delta", side_effect=failing_noise), \
             patch("trainingday.r18_mezo._snapshot_trainables", side_effect=lambda _: {"lora_A": p["value"]}), \
             patch("trainingday.r18_mezo._restore_trainables",
                   side_effect=lambda _, _torch, saved: p.update(value=saved["lora_A"])):
            with self.assertRaisesRegex(RuntimeError, "mid-perturbation"):
                paired_zo_step([("lora_A", p)], None, seed=99, epsilon=0.01,
                               learning_rate=0.1, loss_fn=lambda: p["value"] ** 2)
        self.assertEqual(p["value"], 1.0)

    def test_rejects_nonfinite_or_zero_direction(self):
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            finite_difference(float("nan"), 1.0, 0.001)
        p = {"value": 1.0}

        def noise(_tensors, _torch, seed, scale):
            p["value"] += scale

        with patch("trainingday.r18_mezo._noise_delta", side_effect=noise), \
             patch("trainingday.r18_mezo._snapshot_trainables", side_effect=lambda _: {"lora_A": p["value"]}), \
             patch("trainingday.r18_mezo._restore_trainables",
                   side_effect=lambda _, _torch, saved: p.update(value=saved["lora_A"])):
            with self.assertRaisesRegex(RuntimeError, "zero directional"):
                paired_zo_step([("lora_A", p)], None, seed=99, epsilon=0.01,
                               learning_rate=0.1, loss_fn=lambda: 10.0)
        self.assertAlmostEqual(p["value"], 1.0)


class R18ResumeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.args = argparse.Namespace(output_root=self.root, run_id="mezo-resume",
                                       checkpoint_every=8, steps=16,
                                       expected_initial_digest="a" * 64,
                                       resume_checkpoint=None,
                                       resume_receipt_sha256=None,
                                       budget_seconds=531)
        self.preflight = {"recipe": "mezo", "selected_id_sha256": "b" * 64}

    def checkpoint(self):
        output, contract, _, _, _ = prepare_run(self.args, self.preflight)
        directory = output / "checkpoint-8"
        directory.mkdir()
        data = b"safetensors-file"
        weights = directory / "lora-trainables.safetensors"
        weights.write_bytes(data)
        state = {"schema": "R18_MEZO_STATE_V1",
                 "status": "CHECKPOINT_BYTES_PINNED_NOT_RELOADED",
                 "run_contract_sha256": canonical_hash(contract),
                 "completed_steps": 8, "optimizer_window_seconds": 321,
                 "weights_sha256": hashlib.sha256(data).hexdigest(),
                 "counters": {"example_presentations": 64}}
        state_path = directory / "state.json"
        state_path.write_text(json.dumps(state))
        self.args.resume_checkpoint = directory
        self.args.resume_receipt_sha256 = hashlib.sha256(state_path.read_bytes()).hexdigest()
        return directory

    def test_external_state_sha_and_weights_gate_resume(self):
        directory = self.checkpoint()
        _, _, step, elapsed, state = prepare_run(self.args, self.preflight)
        self.assertEqual((step, elapsed), (8, 321))
        self.assertEqual(state["completed_steps"], 8)
        (directory / "lora-trainables.safetensors").write_bytes(b"tamper")
        with self.assertRaisesRegex(AdmissionError, "custody mismatch"):
            prepare_run(self.args, self.preflight)

    def test_contract_mismatch_blocks_resume(self):
        self.checkpoint()
        self.preflight["selected_id_sha256"] = "c" * 64
        with self.assertRaisesRegex(AdmissionError, "contract unsafe"):
            prepare_run(self.args, self.preflight)


if __name__ == "__main__":
    unittest.main()
