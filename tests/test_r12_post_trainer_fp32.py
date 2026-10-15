"""Prove FP32 cast occurs after TRL's forced BF16 cast, before optimizer creation."""
from pathlib import Path
import ast
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "successor/experiments/train_v10r2_dev.py"

class R12PostTrainerPrecisionOrderTests(unittest.TestCase):
    def test_cast_is_after_sft_trainer_constructor_before_optimizer(self):
        source = SOURCE.read_text(encoding="utf8")
        begin = source.index('    trainer = stack["SFTTrainer"](')
        optimizer = source.index("    optimizer = trainer.create_optimizer()", begin)
        # Exact post-initialization FP32 cast must be in the same local trainer block.
        between = source[begin:optimizer]
        self.assertIn("precision_before = verify_trainable(trainer.model, torch, permit_cast=True)", between)
        # No pre-trainer cast that gets silently overridden by SFTTrainer.
        self.assertNotIn("verify_trainable(model, torch, permit_cast=True)", source[source.index("    resume_check = None"):begin])

    def test_posttrain_precision_must_fail_if_trainer_downcasts_again(self):
        source = SOURCE.read_text(encoding="utf8")
        train = source.index("    train_result = trainer.train()")
        save = source.index("    trainer.model.save_pretrained(", train)
        self.assertIn("precision_after = verify_trainable(trainer.model, torch, permit_cast=False)", source[train:save])
