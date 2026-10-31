"""CPU-only source, custody, and finite-difference tests; no model/GPU import."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from trainingday.admission import AdmissionError
from trainingday.r34_r18_mezo_selection_v2 import make_frozen_selection_v2
from trainingday.r18_mezo import (
    BASE_REVISION, canonical_hash, direction_seed, finite_difference,
    run_fit_sweep, paired_zo_step, prepare_run, selected_id_digest, selected_rows,
    save_checkpoint,
    tokenize_cases, validate_source, next_step_guard_seconds,
    may_begin_paired_update, training_budget_status, step_timing_integrity_failure,
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
            manifest=self.manifest, plan=self.root / "preregistered-plan.json",
            base_binding=self.base_binding,
            producer_git_head=None, work_root=self.root,
            output_root=self.output_root, run_id="r18-correct", mode="correct",
            steps=8, cases=2, seed=41, learning_rate=1e-3, epsilon=1e-3,
            micro_batch_size=1, examples_per_update=8, max_seq_length=512,
            budget_seconds=531.0, checkpoint_every=0,
            expected_initial_digest="a" * 64, fit_probe_sha256=None,
            preflight_only=True, fit_sweep_only=False,
            sweep_batches=[1], sweep_lengths=[64, 128, 256, 384, 512, 768, 1024],
            initial_adapter=None, initial_adapter_sha256=None,
            initial_adapter_config_sha256=None,
            resume_checkpoint=None, resume_receipt_sha256=None,
        )
        self.summary = {"selected_id_sha256": selected_id_digest(ROWS),
                        "manifest_sha256": "b" * 64,
                        "selected_count": len(ROWS),
                        "status": "ADMITTED_LABYRINTH_TRAIN_ONLY_NO_WEIGHT_UPDATE"}
        self.args.selection_v2_binding = self.root / "selection-v2-frozen.json"
        self.args.selection_v2_binding.write_text(json.dumps(make_frozen_selection_v2(
            plan_sha256="c" * 64, summary=self.summary, selected_rows=ROWS)),
            encoding="utf-8")
        self.admit = patch("trainingday.r18_mezo.admit",
                           return_value=(self.summary, ROWS)).start()
        self.base = patch("trainingday.r18_mezo._required_base",
                          return_value=({"revision": BASE_REVISION}, self.root)).start()
        patch("trainingday.r18_mezo.R16_V2_MANIFEST_SHA256",
              hashlib.sha256(self.manifest.read_bytes()).hexdigest()).start()
        self.args.producer_git_head = "1" * 40
        patch("trainingday.r18_mezo.R16_PRODUCER_HEAD", "1" * 40).start()
        patch("trainingday.r18_mezo._validate_prereg_plan", return_value={
            "_validated_plan_sha256": "c" * 64,
            "eval_custody_sha256": "d" * 64, "eval_index_sha256": "e" * 64,
            "training_budget_seconds": self.args.budget_seconds,
        }).start()
        self.addCleanup(patch.stopall)

    def test_source_preflight_is_read_only_and_matches_r16_selection_order(self):
        preflight, cases, path = validate_source(self.args)
        self.assertEqual(preflight["selected_id_sha256"], selected_id_digest(ROWS))
        self.assertEqual(len(preflight["selected_id_v2_sha256"]), 64)
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

    def test_mezo_preflight_rejects_missing_frozen_v2_binding(self):
        self.args.selection_v2_binding = self.root / "not-created.json"
        with self.assertRaisesRegex(AdmissionError, "frozen V2 selection gate"):
            validate_source(self.args)

    def test_mezo_preflight_rejects_swapped_order_that_matches_legacy_sha(self):
        # Two different selections produce the same newline-delimited digest.
        source = [{"id": "a\nb", "family": "arithmetic", "prompt": "1 + 2?", "answer": "3"},
                  {"id": "c", "family": "arithmetic", "prompt": "2 + 4?", "answer": "6"}]
        mutated = [{**source[0], "id": "a"}, {**source[1], "id": "b\nc"}]
        self.assertEqual(selected_id_digest(source), selected_id_digest(mutated))
        self.summary = {**self.summary, "selected_id_sha256": selected_id_digest(source)}
        self.args.selection_v2_binding.write_text(json.dumps(make_frozen_selection_v2(
            plan_sha256="c" * 64, summary=self.summary, selected_rows=source)),
            encoding="utf-8")
        self.admit.stop()
        self.admit = patch("trainingday.r18_mezo.admit",
                           return_value=(self.summary, mutated)).start()
        with self.assertRaisesRegex(AdmissionError, "V2 selected IDs"):
            validate_source(self.args)

    def test_periodic_checkpoint_option_rejected_before_run_output(self):
        # The R18 loop only saves a final checkpoint after stopping. An
        # interval of eight must not be accepted as a recoverability promise.
        self.args.checkpoint_every = 8
        with self.assertRaisesRegex(AdmissionError, "periodic.*unsupported"):
            validate_source(self.args)
        self.assertFalse((self.output_root / self.args.run_id).exists())

    def test_resume_option_rejected_during_source_preflight(self):
        # A resume flag must not pass preflight and then be rejected only once
        # the GPU run has already progressed to the expensive setup stage.
        self.args.checkpoint_every = 0
        self.args.resume_checkpoint = self.root / "old-checkpoint"
        self.args.resume_receipt_sha256 = "c" * 64
        with self.assertRaisesRegex(AdmissionError, "resume.*unsupported"):
            validate_source(self.args)

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
        config = adapter / "adapter_config.json"
        config.write_text('{"peft_type":"LORA"}')
        self.args.initial_adapter = adapter
        self.args.initial_adapter_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
        self.args.initial_adapter_config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
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
    def test_step_timing_fails_closed_on_suspension_like_delay_before_deadline(self):
        # An unusually slow completed step must invalidate the arm even if the
        # full run's deadline has not yet elapsed. It is not proof of sleep.
        self.assertEqual(step_timing_integrity_failure(19_000., 19_000., 32.),
                         "STEP_EXCEEDED_ADMITTED_RESERVE")
        self.assertEqual(training_budget_status("correct", 1, 0, 0.,
                                                timing_integrity_failure=True),
                         "TIME_BUDGET_INTEGRITY_INVALID")

    def test_step_timing_detects_independent_clock_disagreement(self):
        self.assertIsNone(step_timing_integrity_failure(20., 20.1, 32.))
        self.assertEqual(step_timing_integrity_failure(5., 30., 32.),
                         "CLOCK_DISCONTINUITY_SUSPECTED")
        self.assertEqual(step_timing_integrity_failure(5., -0.1, 32.),
                         "CLOCK_DISCONTINUITY_SUSPECTED")
        with self.assertRaisesRegex(AdmissionError, "invalid step timing"):
            step_timing_integrity_failure(float("nan"), 1., 32.)

    def test_conservative_guard_checks_full_paired_step_before_start(self):
        deadline_ns = 531 * 10**9
        # The physical first-step probe measured 40s, requiring 62s reserved.
        allowed, reserve, remaining = may_begin_paired_update(
            deadline_ns - 62 * 10**9, deadline_ns, 40.0, [])
        self.assertTrue(allowed)
        self.assertEqual((reserve, remaining), (62.0, 62.0))
        allowed, reserve, remaining = may_begin_paired_update(
            deadline_ns - 61 * 10**9, deadline_ns, 40.0, [])
        self.assertFalse(allowed)
        # After physical training updates, use the SLOWEST actual training
        # duration, not a faster average or the original synthetic probe.
        self.assertEqual(next_step_guard_seconds(40.0, [20.0, 50.0, 30.0]), 77.0)
        allowed, reserve, remaining = may_begin_paired_update(
            deadline_ns - 76 * 10**9, deadline_ns, 40.0, [20, 50, 30])
        self.assertFalse(allowed)
        self.assertEqual(reserve, 77.0)

    def test_guarded_sequence_leaves_unused_budget_and_cannot_claim_overrun(self):
        deadline = 531 * 10**9
        clock = 0
        observed = []
        for _ in range(20):
            allowed, _, remaining = may_begin_paired_update(clock, deadline, 55., observed)
            if not allowed:
                break
            # Simulated actual completed 50s step (no GPU, no partial step).
            clock += 50 * 10**9
            observed.append(50.)
        self.assertEqual(len(observed), 10)
        self.assertEqual(clock, 500 * 10**9)
        self.assertEqual((deadline - clock) / 1e9, 31.0)
        self.assertEqual(training_budget_status("correct", len(observed), 0, 0.),
                         "TRAINED_UNSCORED")
        self.assertEqual(training_budget_status("correct", len(observed), 0, 0.01),
                         "TIME_BUDGET_OVERRUN_INVALID")
        self.assertEqual(training_budget_status("correct", 0, 0, 0.),
                         "NOT_TRAINED_INSUFFICIENT_SAFE_WINDOW")
        self.assertEqual(training_budget_status("zero", 0, 0, 0.), "TRAINED_UNSCORED")
        with self.assertRaisesRegex(AdmissionError, "no truthful positive"):
            next_step_guard_seconds(float("nan"), [])

    def test_fit_sweep_new_process_per_preregistered_synthetic_candidate(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            plan_path = root / "plan.json"
            args = argparse.Namespace(output_root=root, run_id="fit-only",
                                      manifest=root / "manifest", base_binding=root / "binding",
                                      plan=plan_path, selection_v2_binding=root / "v2.json",
                                      work_root=root, seed=41,
                                      budget_seconds=531, learning_rate=.001, epsilon=.001,
                                      initial_adapter=None, initial_adapter_sha256=None,
                                      initial_adapter_config_sha256=None)
            configs = [{"micro_batch_size": batch,
                        "actual_padded_sequence_length": length,
                        "effective_batch_size": 8,
                        "offload": False, "truncation_allowed": False}
                       for batch in (1, 2)
                       for length in (64, 128, 256, 384, 512, 768, 1024)]
            plan = {"probe_budget_seconds": 30, "primary_training_config": configs[1]}
            plan_path.write_text(json.dumps(plan))
            plan_sha = canonical_hash(plan)
            called = []

            def fake_subprocess(command, **_kwargs):
                index = int(command[command.index("--probe-index") + 1])
                self.assertEqual(command[command.index("--selection-v2-binding") + 1],
                                 str(args.selection_v2_binding))
                output = Path(command[command.index("--probe-result") + 1])
                called.append((index, command.count("--probe-child")))
                item = {"schema": "VERA_TRAININGDAY_R18_FIT_PROBE_V1",
                        "arm": "mezo", "plan_sha256": plan_sha,
                        "candidate_index": index, "config": configs[index],
                        "outcome": "OOM" if index in (2, 8) else "FIT",
                        "fresh_process_id": f"pid:{index}",
                        "actual_padded_sequence_length": configs[index]["actual_padded_sequence_length"],
                        "data_partition": "TRAIN_ONLY", "eval_access": False,
                        "checkpoint_selection": False,
                        "optimizer_step_completed": index not in (2, 8),
                        "elapsed_seconds": 2.0, "peak_cuda_allocated_bytes": 500,
                        "peak_cuda_reserved_bytes": 600, "peak_cpu_rss_bytes": 100,
                        "peak_nvml_used_bytes": 700,
                        "failure_artifact_sha256": "b" * 64 if index in (2, 8) else None}
                output.write_text(json.dumps(item))
                return argparse.Namespace(returncode=0)

            with patch("trainingday.r18_mezo._fit_configurations", return_value=configs), \
                 patch("trainingday.r18_mezo.subprocess.run", side_effect=fake_subprocess):
                result = run_fit_sweep(args, {"plan_sha256": plan_sha})
            self.assertEqual([x[0] for x in called], [0, 1, 2, 7, 8])
            self.assertTrue(all(child == 1 for _, child in called))
            self.assertEqual(len(result["probes"]), 5)
            self.assertEqual(result["probes"][-1]["outcome"], "OOM")
            self.assertEqual(result["fit_probe_sha256"], canonical_hash(result["probes"]))

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
                                       checkpoint_every=0, steps=16,
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

    def test_interrupted_final_checkpoint_cannot_be_reused_as_complete_run(self):
        output, contract, _, _, _ = prepare_run(self.args, self.preflight)

        def crash_mid_serialization(_data, path):
            Path(path).write_bytes(b"incomplete-safetensors")
            raise InterruptedError("simulated disk interruption")

        fake_torch_submodule = types.ModuleType("safetensors.torch")
        fake_torch_submodule.save_file = crash_mid_serialization
        fake_torch_package = types.ModuleType("safetensors")
        fake_torch_package.torch = fake_torch_submodule
        with patch.dict(sys.modules, {"safetensors": fake_torch_package,
                                      "safetensors.torch": fake_torch_submodule}):
            with self.assertRaisesRegex(InterruptedError, "disk interruption"):
                save_checkpoint(output, [], object(), contract, 1, 10.0, {})

        partial = output / "checkpoint-1"
        self.assertTrue((partial / "lora-trainables.safetensors").exists())
        self.assertFalse((partial / "state.json").exists())
        self.assertFalse((output / "r18-mezo-receipt.json").exists())
        with self.assertRaisesRegex(AdmissionError, "already occupied"):
            prepare_run(self.args, self.preflight)


if __name__ == "__main__":
    unittest.main()
