"""CPU-only regressions for nonfinite post-update LoRA weights."""

import inspect
import math
import unittest
from unittest.mock import patch

from trainingday import r18_mezo, r18_qlora


class Scalar:
    def __init__(self, value):
        self.value = value
        self.requires_grad = True

    def detach(self):
        return self


class Boolean:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


class TorchMock:
    @staticmethod
    def isfinite(tensor):
        return Boolean(math.isfinite(tensor.value))


class ModelMock:
    def __init__(self, tensor):
        self.tensor = tensor

    def named_parameters(self):
        return [("base.lora_A.weight", self.tensor)]


class R33FiniteGuardTests(unittest.TestCase):
    def test_mezo_final_update_inf_rolls_back_and_blocks_success(self):
        for nonfinite in (float("inf"), float("nan")):
            with self.subTest(nonfinite=nonfinite):
                tensor = Scalar(1.0)

                def noise(_tensors, _torch, _seed, scale):
                    if scale < -0.1:
                        tensor.value = nonfinite
                    else:
                        tensor.value += 2.0 * scale

                with patch.object(r18_mezo, "_noise_delta", side_effect=noise), \
                     patch.object(r18_mezo, "_snapshot_trainables", return_value={"lora_A": 1.0}), \
                     patch.object(r18_mezo, "_restore_trainables",
                                  side_effect=lambda ts, torch, saved: setattr(tensor, "value", saved["lora_A"])):
                    with self.assertRaisesRegex(RuntimeError, "nonfinite.*trainable"):
                        r18_mezo.paired_zo_step([("lora_A", tensor)], TorchMock,
                                                seed=99, epsilon=0.01, learning_rate=0.1,
                                                loss_fn=lambda: tensor.value**2)
                self.assertEqual(tensor.value, 1.0)

    def test_qlora_nonfinite_after_optimizer_step_rejected(self):
        for value in (float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "nonfinite.*trainable"):
                    r18_qlora._require_finite_trainables(ModelMock(Scalar(value)), TorchMock)

    def test_mezo_attestation_rejects_nonfinite_value(self):
        with self.assertRaisesRegex(RuntimeError, "nonfinite.*trainable"):
            r18_mezo.parameter_attestation([("lora_A", Scalar(float("nan")))], TorchMock)

    def test_qlora_source_orders_finite_check_before_step_credit_and_fit(self):
        """CPU source assertion, not a physical optimizer integration test."""
        training = inspect.getsource(r18_qlora._training_loop)
        probe = inspect.getsource(r18_qlora.probe_worker)
        self.assertLess(training.index("optimizer.step()"),
                        training.index("_require_finite_trainables(model, torch)",
                                       training.index("optimizer.step()")))
        self.assertLess(training.index("_require_finite_trainables(model, torch)",
                                       training.index("optimizer.step()")),
                        training.index("steps += 1"))
        self.assertLess(probe.index("optimizer.step()"),
                        probe.index("_require_finite_trainables(model, torch)"))
        self.assertLess(probe.index("_require_finite_trainables(model, torch)"),
                        probe.index('result.update(status="FIT"'))


if __name__ == "__main__":
    unittest.main()
