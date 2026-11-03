"""Fast source guards; hosted workflow runs the actual expensive CPU experiment."""

from __future__ import annotations

import ast
import inspect
import unittest

from trainingday import qwen35_full32_trl_integrated_cpu as integrated


class TinyIntegratedTrlTests(unittest.TestCase):
    def test_each_trl_target_gradient_and_delta_has_negative_controls(self):
        # Exercise both production fail-closed helpers without heavy Torch
        # dependencies; the hosted dynamic run supplies the 152 real tensors.
        class Grad:
            def __init__(self, magnitude, finite=True):
                self.magnitude = magnitude
                self.finite = finite

            def abs(self):
                return self

            def sum(self):
                return self.magnitude

        class Param:
            def __init__(self, grad, value):
                self.grad = grad
                self.value = value

            def detach(self):
                return self.value

        class Finite:
            def __init__(self, result):
                self.result = result

            def all(self):
                return self.result

        is_finite = lambda grad: Finite(grad.finite)
        by_target = {
            "gdn": Param(Grad(0.000001), 11),
            "attention": Param(Grad(0.5), 12),
        }
        result = integrated._require_nonzero_target_gradients(
            by_target, is_finite=is_finite)
        self.assertEqual(set(result), set(by_target))
        self.assertEqual(integrated._require_target_updates(
            by_target, {"gdn": 10, "attention": 10},
            equal=lambda before, after: before == after), 2)
        by_target["gdn"].grad = Grad(0.0)
        with self.assertRaisesRegex(AssertionError, "zero=\\['gdn'\\]"):
            integrated._require_nonzero_target_gradients(
                by_target, is_finite=is_finite)
        by_target["gdn"].grad = None
        with self.assertRaisesRegex(AssertionError, "missing_or_nonfinite"):
            integrated._require_nonzero_target_gradients(
                by_target, is_finite=is_finite)
        by_target["gdn"].grad = Grad(2.0, finite=False)
        with self.assertRaisesRegex(AssertionError, "missing_or_nonfinite"):
            integrated._require_nonzero_target_gradients(
                by_target, is_finite=is_finite)
        by_target["gdn"].grad = Grad(2.0)
        with self.assertRaisesRegex(AssertionError, "attention"):
            integrated._require_target_updates(
                by_target, {"gdn": 10, "attention": 12},
                equal=lambda before, after: before == after)
        with self.assertRaisesRegex(AssertionError, "set mismatch"):
            integrated._require_target_updates(
                by_target, {"gdn": 10},
                equal=lambda before, after: before == after)

    def test_152_named_trl_peft_targets_not_only_global_or_layer_sums(self):
        source = inspect.getsource(integrated.run)
        for required in (
            "peft_modules = dict(model.named_modules())",
            "if set(injected) != set(targets):",
            "len({id(p) for p in b_by_target.values()}) != 152",
            "_require_nonzero_target_gradients(",
            "_require_target_updates(",
            '"trl_masked_lora_b_targets_with_nonzero_gradients"',
            '"trl_masked_lora_b_targets_changed_after_step"',
        ):
            self.assertIn(required, source)

    def test_repro_diagnostics_bind_exact_masked_training_batch(self):
        hasher = inspect.getsource(integrated._sha_integer_tensor)
        source = inspect.getsource(integrated.run)
        self.assertIn('dtype=torch.int64', hasher)
        self.assertNotIn('.float()', hasher)
        for required in (
            '"input_ids_sha256": _sha_integer_tensor(first_batch["input_ids"])',
            '"labels_sha256": _sha_integer_tensor(first_batch["labels"])',
            '"attention_mask_sha256": _sha_integer_tensor(first_batch["attention_mask"])',
            '"input_ids_shape": list(first_batch["input_ids"].shape)',
            '"initial_lora_a_sha256": before_a_sha',
            '"initial_frozen_embedding_sha256": base_embed_sha',
            '"initial_frozen_layer0_qkv_sha256": base_qkv_sha',
            '"numerical_environment": numerical_environment',
            '"repro_diagnostics_v1": {',
        ):
            self.assertIn(required, source)

    def test_integer_batch_fingerprint_detects_label_and_order_mutations(self):
        import torch

        original_ids = torch.tensor([[10, 11], [20, 21]], dtype=torch.long)
        original_labels = torch.tensor([[-100, 11], [-100, 21]], dtype=torch.long)
        changed_prompt_label = torch.tensor([[10, 11], [-100, 21]], dtype=torch.long)
        self.assertNotEqual(integrated._sha_integer_tensor(original_labels),
                            integrated._sha_integer_tensor(changed_prompt_label))
        self.assertNotEqual(integrated._sha_integer_tensor(original_ids),
                            integrated._sha_integer_tensor(original_ids.flip(0)))
        # Old float-cast hashing would erase the distinction at this boundary.
        self.assertNotEqual(
            integrated._sha_integer_tensor(torch.tensor([2**24], dtype=torch.long)),
            integrated._sha_integer_tensor(torch.tensor([2**24 + 1], dtype=torch.long)),
        )

    def test_official_token_ids_and_all_eight_synthetic_examples(self):
        self.assertEqual(integrated.TOKENIZER_VOCAB, 248320)
        self.assertEqual((integrated.EOS_ID, integrated.PAD_ID), (248046, 248044))
        self.assertEqual(len(integrated.ARITHMETIC), 8)
        sums = [a + b for a, b in integrated.ARITHMETIC]
        self.assertEqual(sum(n < 0 for n in sums), 4)
        self.assertEqual(sum(n >= 0 for n in sums), 4)
        self.assertEqual(len(integrated.smoke.LAYER_TYPES), 32)
        self.assertEqual(len(set(integrated.smoke.expected_target_modules())), 152)

    def test_real_trl_generates_labels_and_masked_batch_enters_optimizer(self):
        source = inspect.getsource(integrated.run)
        for required in (
            "SFTTrainer(", "Dataset.from_list(training_rows)",
            "trainer.get_train_dataloader()", "observed.sort()",
            "if observed != expected:", 'labels=first_batch["labels"]',
            "loss.backward()", "optimizer.step()", "after_sha == before_sha",
            "len(observed)",
        ):
            self.assertIn(required, source)
        self.assertNotIn("labels=ids", source)
        self.assertNotIn("labels=batch[\"input_ids\"]", source)

    def test_real_new_interpreter_reload_and_negative_control(self):
        source = inspect.getsource(integrated.run)
        child = inspect.getsource(integrated._child)
        for expected in (
            "model.save_pretrained", "safe_serialization=True", "adapter_model.safetensors",
            "subprocess.run", "receipt[\"pid\"] == os.getpid()", "missing_dir",
            "if rejected.returncode == 0:",
        ):
            self.assertIn(expected, source)
        self.assertIn("PeftModel.from_pretrained", child)
        self.assertIn("loaded_logits_sha == base_logits_sha", child)

    def test_no_pretrained_or_protected_dataset_or_cuda_launch(self):
        source = inspect.getsource(integrated)
        tree = ast.parse(source)
        self.assertTrue(any(isinstance(node, ast.FunctionDef) and node.name == "run"
                            for node in tree.body))
        self.assertNotIn("Qwen3_5ForCausalLM.from_pretrained", source)
        self.assertNotIn(".cuda()", source)
        self.assertNotIn('to("cuda")', source)
        self.assertIn("GENERATED_ARITHMETIC_ONLY_NO_R201_OR_DEV", source)
        self.assertIn("training_authorized\": False", source)


if __name__ == "__main__":
    unittest.main()
