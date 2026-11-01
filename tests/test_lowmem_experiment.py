"""Source-only tests; no torch, CUDA, model download, or host invocation."""

import json
import struct
import subprocess
import sys
import unittest

from trainingday.lowmem_experiment import (
    BASE_ID, BASE_REVISION, LowMemoryExperimentError, evaluate_optimizer_witness,
    one_step_kernel, proposed_zero3_config, source_plan, validate_declaration,
)
from trainingday.physical_launch_authority import PhysicalLaunchAuthorityUnavailable


def declaration():
    files = {name: "a" * 64 for name in (
        "config.json", "tokenizer_config.json", "tokenizer.json",
        "model.safetensors.index.json", "model.safetensors-00001-of-00002.safetensors")}
    return dict(model_id=BASE_ID, revision=BASE_REVISION,
                loader_class="Qwen3_5ForCausalLM", quantization="NF4_DOUBLE_BF16",
                files=files)


class LowMemSourceTests(unittest.TestCase):
    def test_source_plan_denies_and_preserves_offload_distinction(self):
        p = source_plan()
        self.assertFalse(p["authorizes_training"])
        self.assertEqual(p["optimizer_steps_observed"], 0)
        self.assertFalse(p["eval_access"])
        self.assertEqual(p["candidates"]["nf4_minimal_gpu"]["device_map"], 0)
        self.assertEqual(p["candidates"]["bf16_zero3_cpu_offload"]["implementation"],
                         "CONFIG_HYPOTHESIS_NOT_RUNNABLE")
        zero = proposed_zero3_config()
        self.assertEqual(zero["zero_optimization"]["stage"], 3)
        self.assertEqual(zero["zero_optimization"]["offload_param"]["device"], "cpu")
        self.assertNotIn("device_map", zero)

    def test_import_has_no_torch_or_model_side_effect(self):
        statement = "import sys; import trainingday.lowmem_experiment; assert 'torch' not in sys.modules"
        completed = subprocess.run([sys.executable, "-c", statement],
                                   capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")

    def test_hard_hold_before_torch_or_binding_read(self):
        with self.assertRaisesRegex(PhysicalLaunchAuthorityUnavailable, "HARD HOLD"):
            one_step_kernel("/path/does/not/exist")

    def test_declaration_wrong_model_revision_or_tokenizer_pin_rejected(self):
        validate_declaration(declaration())  # syntax only; never a real pin check
        for key, bad in (("model_id", "other/model"), ("revision", "0" * 40),
                         ("loader_class", "Qwen3_5ForConditionalGeneration")):
            data = declaration()
            data[key] = bad
            with self.subTest(key=key), self.assertRaises(LowMemoryExperimentError):
                validate_declaration(data)
        data = declaration()
        data["files"]["tokenizer.json"] = "bad"
        with self.assertRaises(LowMemoryExperimentError):
            validate_declaration(data)
        data = declaration()
        data["files"]["model.safetensors-00001-of-00002.safetensors"] = "bad"
        with self.assertRaises(LowMemoryExperimentError):
            validate_declaration(data)
        with self.assertRaises(LowMemoryExperimentError):
            validate_declaration(dict(declaration(), files=dict))
        Spoof = type("Spoof", (str,), {"__eq__": lambda _self, _other: True})
        data = declaration()
        data["model_id"] = Spoof("wrong/model")
        with self.assertRaises(LowMemoryExperimentError):
            validate_declaration(data)

    def test_fake_adapter_optimizer_delta_witness(self):
        before = {"model.layers.3.q_proj.lora_B": struct.pack("<f", 0.0)}
        after = {"model.layers.3.q_proj.lora_B": struct.pack("<f", 0.001)}
        witness = evaluate_optimizer_witness(
            before=before, after=after, loss=3.2, gradient_norm=0.2,
            optimizer_step_returned=True)
        self.assertEqual(witness["adapter_tensors_changed"], 1)
        self.assertNotEqual(witness["before_sha256"], witness["after_sha256"])
        self.assertFalse(witness["physical_training_authorized"])

    def test_fake_step_with_no_delta_or_no_gradient_is_not_witness(self):
        before = {"lora_B": struct.pack("<f", 0.0)}
        for after, grad, stepped in (
            (before, 0.2, True),
            ({"lora_B": struct.pack("<f", 0.1)}, 0.0, True),
            ({"lora_B": struct.pack("<f", 0.1)}, 0.2, False),
        ):
            with self.subTest(grad=grad, stepped=stepped), self.assertRaises(
                    LowMemoryExperimentError):
                evaluate_optimizer_witness(before=before, after=after, loss=3.2,
                                           gradient_norm=grad,
                                           optimizer_step_returned=stepped)

    def test_cli_no_execution_switch(self):
        completed = subprocess.run(
            [sys.executable, "-m", "trainingday.lowmem_experiment", "--execute"],
            capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertIn("HARD HOLD", completed.stderr)
        normal = subprocess.run(
            [sys.executable, "-m", "trainingday.lowmem_experiment"],
            capture_output=True, text=True, check=False)
        self.assertEqual(normal.returncode, 0)
        output = json.loads(normal.stdout)
        self.assertFalse(output["authorizes_training"])
        self.assertEqual(output["optimizer_steps_observed"], 0)


if __name__ == "__main__":
    unittest.main()
