"""CPU-only R15 control and trainer-observable contracts; no GPU training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from trainingday.admission import AdmissionError
from trainingday.train_neural import (BASE_REVISION, _parameter_digest,
                                      _required_base, train, validate_run)
from test_r39_reload_proof import _Tensor, _Model, _fake_runtime


_SUMMARY = {"manifest_sha256": "a" * 64, "selected_id_sha256": "b" * 64,
            "selected_count": 8, "producer_git_head": "c" * 40}
_CASES = [{"prompt": f"case-{i}", "response": str(i)} for i in range(8)]


def _args(root: Path, *, mode: str = "zero", steps: int = 0) -> argparse.Namespace:
    output_root = root / "outputs"
    output_root.mkdir(exist_ok=True)
    binding = root / "base.json"
    binding.write_text("synthetic-base-binding", encoding="utf-8")
    return argparse.Namespace(
        base_binding=binding, output_root=output_root, run_id=f"{mode}-{steps}",
        mode=mode, steps=steps, cases=8, seed=41, learning_rate=2e-5,
        producer_git_head="c" * 40, expected_initial_digest="d" * 64 if steps else None,
        preflight_only=True, r16_checkpointable=False,
        expected_control_source_sha256=None,
    )


class R41ControlSourceTests(unittest.TestCase):
    def test_boolean_numeric_api_inputs_cannot_masquerade_as_integer_controls(self):
        # argparse's CLI type=int protects the command line, not direct callers
        # of validate_run()/train() through the Python API.
        for key, bad_value in (("steps", True), ("cases", True), ("seed", True)):
            with self.subTest(key=key):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    args = (_args(root, mode="zero", steps=0) if key == "cases"
                            else _args(root, mode="correct", steps=1))
                    args.manifest = root / "manifest.json"
                    args.manifest.write_text(json.dumps({"schema": "synthetic"}), encoding="utf-8")
                    args.work_root = root
                    setattr(args, key, bad_value)
                    selected = [{"id": "case-1", "family": "arithmetic",
                                 "prompt": "Compute 1+1", "answer": "2"}]
                    with patch("trainingday.train_neural.admit", return_value=(
                        _SUMMARY, selected,
                    )), patch("trainingday.train_neural._required_base", return_value=(
                        {"model_id": "Qwen/Qwen3.5-4B", "revision": BASE_REVISION}, root,
                    )):
                        with self.assertRaisesRegex(AdmissionError, "integer"):
                            validate_run(args)

    def test_zero_preflight_exposes_frozen_control_source_commitment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root)
            with patch("trainingday.train_neural.validate_run", return_value=(
                _SUMMARY, _CASES, {"model_id": "Qwen/Qwen3.5-4B", "revision": "c" * 40}, root
            )):
                result = train(args)
            self.assertIn("control_source_sha256", result)
            self.assertRegex(result["control_source_sha256"], r"^[a-f0-9]{64}$")

    def test_trained_arm_fails_without_zero_arm_source_commitment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root, mode="correct", steps=1)
            with patch("trainingday.train_neural.validate_run", return_value=(
                _SUMMARY, _CASES, {"model_id": "Qwen/Qwen3.5-4B", "revision": "c" * 40}, root
            )):
                with self.assertRaisesRegex(AdmissionError, "control source"):
                    train(args)

    def test_trained_arm_rejects_mismatched_zero_arm_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            correct = _args(root, mode="correct", steps=1)
            correct.expected_control_source_sha256 = "f" * 64
            altered = dict(_SUMMARY, manifest_sha256="f" * 64)
            with patch("trainingday.train_neural.validate_run", return_value=(
                altered, _CASES, {"model_id": "Qwen/Qwen3.5-4B", "revision": "c" * 40}, root
            )):
                with self.assertRaisesRegex(AdmissionError, "control source"):
                    train(correct)

    def test_corrupt_control_changes_labels_but_preserves_source_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zero = _args(root)
            base = {"model_id": "Qwen/Qwen3.5-4B", "revision": "c" * 40}
            with patch("trainingday.train_neural.validate_run", return_value=(
                _SUMMARY, _CASES, base, root
            )):
                zero_report = train(zero)
            corrupt = _args(root, mode="corrupt", steps=1)
            corrupt.expected_control_source_sha256 = zero_report["control_source_sha256"]
            altered_labels = [dict(row, response="incorrect") for row in _CASES]
            with patch("trainingday.train_neural.validate_run", return_value=(
                _SUMMARY, altered_labels, base, root
            )):
                corrupt_report = train(corrupt)
            self.assertEqual(corrupt_report["control_source_sha256"],
                             zero_report["control_source_sha256"])
            self.assertNotEqual(corrupt_report["training_input_sha256"],
                                zero_report["training_input_sha256"])


class R41TrainerObservableTests(unittest.TestCase):
    def test_nonfinite_trainable_tensor_must_not_be_hashed_as_valid_update(self):
        fake_torch, _ = _fake_runtime(_Model(_Tensor(float("nan"), "float32")))
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            _parameter_digest(_Model(_Tensor(float("nan"), "float32")), fake_torch)

    def _run_fake_training(self, *, step_value, loss_value,
                           mutate_base_during_load=False,
                           mutate_base_during_train=False, trace=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root, mode="correct", steps=1)
            args.preflight_only = False
            snapshot = root / "official-base"
            snapshot.mkdir()
            files = {}
            for name, contents in (("config.json", b"{}"),
                                   ("tokenizer_config.json", b"{}"),
                                   ("model.safetensors", b"synthetic weights")):
                (snapshot / name).write_bytes(contents)
                files[name] = hashlib.sha256(contents).hexdigest()
            bound = {"schema": "TRAININGDAY_PINNED_BASE_V1",
                     "model_id": "Qwen/Qwen3.5-4B", "revision": BASE_REVISION,
                     "loader_class": "Qwen3_5ForCausalLM",
                     "quantization": "NF4_DOUBLE_BF16",
                     "snapshot_path": str(snapshot), "files": files}
            args.base_binding.write_text(json.dumps(bound), encoding="utf-8")
            self.assertEqual(_required_base(args.base_binding)[0], bound)
            tensor = _Tensor(0.5, "float32")
            fake_torch, _ = _fake_runtime(_Model(tensor))
            fake_torch.bfloat16 = "bfloat16"
            fake_torch.cuda.get_device_properties = lambda _i: types.SimpleNamespace(
                total_memory=8 * 2**30
            )

            class Model(_Model):
                targeted_module_names = ("q_proj",)

                def __init__(self):
                    super().__init__(tensor)
                    self.config = types.SimpleNamespace(use_cache=True)

                def save_pretrained(self, path, safe_serialization):
                    if trace is not None:
                        trace["saved"] = True
                    path.mkdir()
                    (path / "adapter_model.safetensors").write_bytes(b"synthetic")
                    (path / "adapter_config.json").write_bytes(b"{}")

            model = Model()
            args.expected_initial_digest, _ = _parameter_digest(model, fake_torch)

            class Trainer:
                def __init__(self, **_kw):
                    self.model = model
                    self.state = types.SimpleNamespace(global_step=step_value)

                def create_optimizer(self):
                    return object()

                def train(self, **_kw):
                    if trace is not None:
                        trace["trained"] = True
                    if mutate_base_during_train:
                        (snapshot / "config.json").write_bytes(b'{"mutated":"during-train"}')
                    tensor.value = 1.0
                    return types.SimpleNamespace(training_loss=loss_value)

            tokenizer = types.SimpleNamespace(chat_template="template", pad_token_id=0)
            def load_base(*_args, **_kwargs):
                if mutate_base_during_load:
                    (snapshot / "config.json").write_bytes(b'{"mutated":"during-loader"}')
                return model

            stack = {
                "transformers": types.SimpleNamespace(set_seed=lambda _: None, __version__="fake"),
                "AutoTokenizer": types.SimpleNamespace(from_pretrained=lambda _: tokenizer),
                "BitsAndBytesConfig": lambda **_kw: object(),
                "Qwen3_5ForCausalLM": types.SimpleNamespace(from_pretrained=load_base),
                "prepare_model_for_kbit_training": lambda item, **_kw: item,
                "LoraConfig": lambda **_kw: object(), "SFTConfig": lambda **_kw: object(),
                "SFTTrainer": Trainer,
                "Dataset": types.SimpleNamespace(from_list=lambda x: x),
            }
            precision = types.ModuleType("successor.experiments.r12_adapter_precision")
            precision.verify_trainable = lambda *_a, **_kw: {"trainable_dtype": "float32"}
            training_stack = types.ModuleType("successor.experiments.train_v10_qwen35_authorized")
            training_stack.load_training_stack = lambda: stack
            training_stack.prepare_sft_rows = lambda *_a, **_kw: (_CASES, 20)
            training_stack.validate_lora_target_coverage = lambda *_a: {"ok": True}
            optimizer = types.ModuleType("successor.experiments.train_v10r2_dev")
            optimizer.summarize_optimizer = lambda _a: {
                "backend": "bitsandbytes", "is_paged": False, "optim_bits": 8,
            }
            psutil = types.ModuleType("psutil")
            psutil.virtual_memory = lambda: types.SimpleNamespace(available=20 * 2**30)
            modules = {"torch": fake_torch, "psutil": psutil,
                       "successor.experiments.r12_adapter_precision": precision,
                       "successor.experiments.train_v10_qwen35_authorized": training_stack,
                       "successor.experiments.train_v10r2_dev": optimizer}
            fake_summary = dict(_SUMMARY)
            base = bound
            with patch.dict(sys.modules, modules), patch(
                "trainingday.train_neural.validate_run",
                return_value=(fake_summary, _CASES, base, snapshot),
            ):
                args.mode, args.steps, args.preflight_only = "zero", 0, True
                zero = train(args)
                args.expected_control_source_sha256 = zero["control_source_sha256"]
                args.mode, args.steps, args.preflight_only = "correct", 1, False
                return train(args)

    def test_boolean_step_counter_cannot_prove_one_optimizer_step(self):
        with self.assertRaisesRegex(RuntimeError, "optimizer step"):
            self._run_fake_training(step_value=True, loss_value=0.5)

    def test_nonfinite_training_loss_cannot_be_recorded_as_success(self):
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            self._run_fake_training(step_value=1, loss_value=float("nan"))

    def test_exact_step_and_finite_loss_still_write_source_bound_receipt(self):
        receipt = self._run_fake_training(step_value=1, loss_value=0.25)
        self.assertEqual(receipt["new_optimizer_steps"], 1)
        self.assertEqual(receipt["training_loss"], 0.25)
        self.assertRegex(receipt["control_source_sha256"], r"^[0-9a-f]{64}$")

    def test_model_load_persistent_base_mutation_prevents_training_and_adapter_save(self):
        trace = {}
        with self.assertRaisesRegex(AdmissionError, "custody"):
            self._run_fake_training(step_value=1, loss_value=0.25,
                                    mutate_base_during_load=True, trace=trace)
        self.assertNotIn("trained", trace)
        self.assertNotIn("saved", trace)

    def test_training_persistent_base_mutation_prevents_adapter_save(self):
        trace = {}
        with self.assertRaisesRegex(AdmissionError, "custody"):
            self._run_fake_training(step_value=1, loss_value=0.25,
                                    mutate_base_during_train=True, trace=trace)
        self.assertTrue(trace.get("trained"))
        self.assertNotIn("saved", trace)


if __name__ == "__main__":
    unittest.main()
