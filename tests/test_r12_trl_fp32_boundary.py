"""TRL quantized model constructor can recast trainable LoRA back to BF16."""
import json
from pathlib import Path
import unittest

from successor.experiments.r12_adapter_precision import verify_trainable, AdapterPrecisionHold

ROOT = Path(__file__).resolve().parents[1]


class TorchLike:
    float32 = "fp32"


class Param:
    def __init__(self, dtype):
        self.dtype = dtype
        self.requires_grad = True
    @property
    def data(self):
        return self
    @data.setter
    def data(self, other):
        self.dtype = other.dtype
    def is_floating_point(self):
        return True
    def to(self, *, dtype):
        return Param(dtype)


class LoRA:
    def __init__(self):
        self.param = Param("bf16")
    def named_parameters(self):
        yield "base.layers.0.lora_A.default.weight", self.param


class R12TRLBoundaryTests(unittest.TestCase):
    def test_recast_requires_repair_after_trainer_constructor(self):
        model = LoRA()
        verify_trainable(model, TorchLike, permit_cast=True)
        self.assertEqual(model.param.dtype, "fp32")
        # Equivalent to TRL quantized SFTTrainer 1.13 constructor lines 1153-56.
        model.param.data = model.param.data.to(dtype="bf16")
        with self.assertRaisesRegex(AdapterPrecisionHold, "not FP32"):
            verify_trainable(model, TorchLike, permit_cast=False)
        repaired = verify_trainable(model, TorchLike, permit_cast=True)
        self.assertEqual(repaired["converted_from_lower_precision"], 1)
        self.assertEqual(model.param.dtype, "fp32")

    def test_precision_after_constructor_precedes_optimizer(self):
        script = (ROOT / "successor/experiments/train_v10r2_dev.py").read_text(encoding="utf8")
        trainer = script.index('trainer = stack["SFTTrainer"](')
        after = script.index('precision_after_trainer = (')
        optimizer = script.index('optimizer = trainer.create_optimizer()')
        train = script.index('train_result = trainer.train()')
        self.assertLess(trainer, after)
        self.assertLess(after, optimizer)
        self.assertLess(optimizer, train)

    def test_retry2_distinct_output_and_parent_unmodified(self):
        retry = json.loads((ROOT/"successor/experiments/V10R3R12_FP32_VERA_VOICE_EXECUTION_SPEC_20261015_TRL_RETRY2.json").read_text())
        prior = json.loads((ROOT/"successor/experiments/V10R3R12_FP32_VERA_VOICE_EXECUTION_SPEC_20261015_V1.json").read_text())
        self.assertNotEqual(retry["output"]["namespace"], prior["output"]["namespace"])
        self.assertEqual(retry["resume_adapter"]["source_cumulative_optimizer_steps"], 44)
        self.assertEqual(retry["source_subject"]["train_sha256"], prior["source_subject"]["train_sha256"])
        self.assertEqual(retry["correction_attempt"]["previous_run_optimizer_steps_observed"],8)
        self.assertFalse(retry["correction_attempt"]["previous_run_completed_adapter"])
