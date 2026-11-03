"""Fast source guards; hosted workflow runs the actual expensive CPU experiment."""

from __future__ import annotations

import ast
import inspect
import unittest

from trainingday import qwen35_full32_trl_integrated_cpu as integrated


class TinyIntegratedTrlTests(unittest.TestCase):
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
