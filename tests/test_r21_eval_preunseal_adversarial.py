"""Adversarial CPU-only checks of the R18 private-evaluation trust boundary.

These tests create synthetic local receipts and fake imports. They never read a
private evaluation bank, load a model, or execute CUDA.
"""

import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import test_r18_eval as fixtures

from trainingday.admission import AdmissionError, _hash_file
from trainingday.eval_structural_r16 import CORPUS_SHA256
from trainingday.r18_eval import _load_model, evaluate


class PreunsealAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.R18EvaluatorTests(
            "test_complete_unscored_eight_arm_freeze_is_hash_bound")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_self_asserted_attestation_environment_cannot_open_bank(self):
        plan, inventory, path = self.fixture.make_freeze_inventory()
        args = self.fixture.change(
            preflight_only=False,
            freeze_inventory=path,
            expected_freeze_inventory_sha256=_hash_file(path),
            # An untrusted invoker may supply any such optional attributes.
            attestation_status="INDEPENDENT_PASS",
            evaluator_principal="trusted-evaluator",
        )
        arbitrary_flags = {
            "R18_INDEPENDENT_ATTESTATION": "PASS",
            "R18_PRIVATE_UNSEAL_AUTHORIZED": "1",
            "VERA_EVAL_BANK_ACCESS_APPROVED": "true",
        }
        with patch.dict(os.environ, arbitrary_flags), \
             patch("trainingday.r18_eval._required_base", return_value=(
                   self.fixture.base, self.fixture.root)), \
             patch("trainingday.r18_eval._hash_file", side_effect=lambda p:
                   CORPUS_SHA256 if p == self.fixture.manifest else _hash_file(p)), \
             patch("trainingday.r18_eval._source_manifest", return_value=(
                   self.fixture.source_manifest, self.fixture.source_sha)), \
             patch("trainingday.r18_eval._plan", return_value=(
                   plan, inventory["plan_sha256"])), \
             patch("trainingday.r18_eval._cuda_preflight") as cuda, \
             patch("trainingday.r18_eval._load_model") as load, \
             patch("trainingday.r18_eval._bank") as bank:
            with self.assertRaisesRegex(AdmissionError,
                                        "independent pre-unseal attestation unavailable"):
                evaluate(args)
            cuda.assert_not_called()
            load.assert_not_called()
            bank.assert_not_called()
        self.assertFalse(args.output.exists())
        self.assertFalse(args.private_output.exists())

    def test_gc_cleanup_failure_must_not_mask_primary_preload_error(self):
        """RED for a genuine cleanup-order defect; no real ML imports needed."""
        torch = ModuleType("torch")
        torch.bfloat16 = object()
        torch.cuda = Mock()
        transformers = ModuleType("transformers")

        def fail_tokenizer(*_args, **_kwargs):
            raise ValueError("synthetic primary tokenizer failure")

        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=fail_tokenizer)
        transformers.BitsAndBytesConfig = lambda **_kwargs: object()
        transformers.Qwen3_5ForCausalLM = SimpleNamespace(from_pretrained=Mock())
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), \
             patch("trainingday.r18_eval._cuda_preflight"), \
             patch("trainingday.r18_eval.gc.collect",
                   side_effect=RuntimeError("secondary gc cleanup failure")):
            # The primary model-load exception must survive best-effort cleanup.
            with self.assertRaisesRegex(ValueError,
                                        "synthetic primary tokenizer failure"):
                _load_model(self.fixture.root, None, None)


if __name__ == "__main__":
    unittest.main()
