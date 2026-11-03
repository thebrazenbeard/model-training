"""Synthetic CPU name-fixtures only; not a real Qwen skeleton or PEFT runtime."""

import inspect
import subprocess
import sys
import unittest

from trainingday.qwen35_adapter_coverage import (
    AdapterCoverageError, audit_loaded_minimal_lora_model,
    audit_minimal_lora_coverage,
)
from trainingday.lowmem_experiment import one_step_kernel


class _Param:
    def __init__(self, requires_grad):
        self.requires_grad = requires_grad


class _FakeNamedModel:
    def __init__(self, modules, trainables):
        self.modules = modules
        self.trainables = trainables

    def named_modules(self):
        return ((name, object()) for name in self.modules)

    def named_parameters(self):
        return ((name, _Param(name in self.trainables))
                for name in self.trainables | {"model.embed_tokens.weight"})


class AdapterNameCoverageTests(unittest.TestCase):
    def setUp(self):
        self.layer_types = tuple(
            kind for _ in range(8)
            for kind in ("linear_attention", "linear_attention",
                         "linear_attention", "full_attention")
        )
        self.prefix = "base_model.model.model.layers."
        self.modules = set()
        self.full_targets = set()
        for idx, kind in enumerate(self.layer_types):
            names = (
                ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                 "linear_attn.out_proj")
                if kind == "linear_attention"
                else ("self_attn.q_proj", "self_attn.v_proj")
            )
            for name in names:
                full_name = f"{self.prefix}{idx}.{name}"
                self.modules.add(full_name)
                if kind == "full_attention":
                    self.full_targets.add(full_name)
        self.trainables = {
            f"{name}.lora_{side}.default.weight"
            for name in self.full_targets for side in ("A", "B")
        }

    def audit(self, **overrides):
        args = {
            "layer_types": self.layer_types,
            "module_names": self.modules,
            "trainable_parameter_names": self.trainables,
        }
        args.update(overrides)
        return audit_minimal_lora_coverage(**args)

    def test_exact_pilot_shape_reports_only_eight_full_layers(self):
        evidence = self.audit()
        self.assertEqual(evidence["target_projection_modules"], 16)
        self.assertEqual(evidence["trainable_lora_a_b_tensors"], 32)
        self.assertEqual(evidence["gdn_layers_with_adapters"], 0)
        self.assertFalse(evidence["architecture_full_coverage"])
        self.assertFalse(evidence["official_weights_authenticated"])
        self.assertFalse(evidence["physical_training_authorized"])
        self.assertEqual(evidence["optimizer_steps_observed"], 0)

    def test_missing_qv_adapter_rejected_not_accepted_as_partial_pass(self):
        incomplete = self.trainables - {sorted(self.trainables)[0]}
        with self.assertRaisesRegex(AdapterCoverageError, "trainable"):
            self.audit(trainable_parameter_names=incomplete)

    def test_bad_or_missing_gdn_module_and_wrong_decoder_shape_rejected(self):
        gdn_module = next(name for name in self.modules if ".linear_attn." in name)
        with self.assertRaisesRegex(AdapterCoverageError, "decoder modules"):
            self.audit(module_names=self.modules - {gdn_module})
        with self.assertRaisesRegex(AdapterCoverageError, "24 GDN"):
            self.audit(layer_types=("full_attention",) * 32)

    def test_claiming_full_decoder_coverage_is_rejected(self):
        with self.assertRaisesRegex(AdapterCoverageError, "cannot claim"):
            self.audit(claimed_scope="FULL_DECODER")
        with self.assertRaisesRegex(AdapterCoverageError, "outside q/v"):
            self.audit(trainable_parameter_names=self.trainables | {
                f"{self.prefix}0.linear_attn.in_proj_qkv.lora_A.default.weight"
            })

    def test_unexpected_trainable_base_weight_and_duplicate_names_rejected(self):
        with self.assertRaisesRegex(AdapterCoverageError, "outside q/v"):
            self.audit(trainable_parameter_names=self.trainables | {
                "base_model.model.model.embed_tokens.weight"
            })
        with self.assertRaisesRegex(AdapterCoverageError, "duplicate"):
            self.audit(module_names=list(self.modules) + [next(iter(self.modules))])
        with self.assertRaisesRegex(AdapterCoverageError, "missing"):
            self.audit(trainable_parameter_names=list(self.trainables) * 2)

    def test_loaded_model_wrapper_uses_named_module_and_parameter_probes(self):
        fake = _FakeNamedModel(self.modules, self.trainables)
        evidence = audit_loaded_minimal_lora_model(fake, self.layer_types)
        self.assertEqual(evidence["target_projection_modules"], 16)
        fake.trainables = self.trainables - {sorted(self.trainables)[0]}
        with self.assertRaises(AdapterCoverageError):
            audit_loaded_minimal_lora_model(fake, self.layer_types)

    def test_import_is_cpu_stdlib_only(self):
        process = subprocess.run([
            sys.executable, "-c",
            "import sys; import trainingday.qwen35_adapter_coverage; "
            "assert 'torch' not in sys.modules and 'peft' not in sys.modules"
        ], capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_protected_lowmem_kernel_checks_adapter_names_before_training(self):
        source = inspect.getsource(one_step_kernel)
        self.assertLess(source.index("get_peft_model(model, LoraConfig("),
                        source.index("coverage = audit_loaded_minimal_lora_model("))
        self.assertLess(source.index("coverage = audit_loaded_minimal_lora_model("),
                        source.index("optimizer = torch.optim.AdamW("))


if __name__ == "__main__":
    unittest.main()
