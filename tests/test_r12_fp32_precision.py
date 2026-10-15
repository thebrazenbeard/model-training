"""R12 FP32 adapter and bounded corpus regression tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from successor.experiments import r12_adapter_precision as prec
from successor.experiments import build_r12_voice_corrective as builder
from successor.experiments.train_v10r2_dev import load_and_validate_dev_spec, DevTrainingHold

ROOT = Path(__file__).resolve().parents[1]


class FakeTorch:
    float32 = "fp32"


class Tensor:
    def __init__(self, dtype="bf16", trainable=True):
        self.dtype = dtype
        self.requires_grad = trainable
    @property
    def data(self):
        return self
    @data.setter
    def data(self, value):
        self.dtype = value.dtype
    def is_floating_point(self):
        return True
    def to(self, *, dtype):
        return Tensor(dtype, self.requires_grad)


class Model:
    def __init__(self, name="layer.0.lora_A.weight"):
        self.name = name
        self.weight = Tensor()
    def named_parameters(self):
        yield self.name, self.weight


class R12PrecisionTests(unittest.TestCase):
    def test_lower_precision_upgraded_and_held_in_fp32(self):
        model = Model()
        with self.assertRaisesRegex(prec.AdapterPrecisionHold, "not FP32"):
            prec.verify_trainable(model, FakeTorch, permit_cast=False)
        proof = prec.verify_trainable(model, FakeTorch, permit_cast=True)
        self.assertEqual(proof["converted_from_lower_precision"], 1)
        self.assertEqual(model.weight.dtype, "fp32")
        self.assertEqual(prec.verify_trainable(model, FakeTorch, permit_cast=False)["converted_from_lower_precision"], 0)
    def test_non_lora_trainable_is_rejected(self):
        with self.assertRaisesRegex(prec.AdapterPrecisionHold, "non-LoRA"):
            prec.verify_trainable(Model(name="base.weight"), FakeTorch, permit_cast=True)

    def test_spec_enforces_precision_policy(self):
        source = json.loads((ROOT / builder.R11_SPEC_REL).read_text(encoding="utf8"))
        for policy, expected in [("unrecognized", "unrecognized adapter precision policy"),
                                 (prec.POLICY, "requires an exact parent")]:
            bad = copy.deepcopy(source)
            bad["adapter_precision_policy"] = policy
            if policy == prec.POLICY:
                bad.pop("resume_adapter", None)
                bad.pop("development_window", None)
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "spec.json"
                path.write_text(json.dumps(bad))
                with self.assertRaisesRegex(DevTrainingHold, expected):
                    load_and_validate_dev_spec(path)

    def test_r12_source_is_new_and_no_heldout_prompt_overlap(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.multiple(builder,
                OUT_DIR=Path(td) / "corpus", NEW_ADAPTER=Path(td) / "adapter",
                SPEC_PATH=Path(td) / "spec.json", BINDING_PATH=Path(td) / "binding.json"):
                rows, voice, rehearsal = builder.prepare()
        self.assertEqual(len(voice), 32)
        self.assertEqual(len(rehearsal), 32)
        self.assertEqual(len(rows), 50000)
        self.assertEqual(rows[352]["record_id"], voice[0]["id"])
        self.assertEqual(rows[353]["record_id"], rehearsal[0]["record_id"])
        self.assertEqual(rows[415]["record_id"], rehearsal[-1]["record_id"])
        self.assertTrue(all(r["source_class"] == "curated_nonpersonal_voice_new_r12"
                            for r in rows[352:416:2]))


if __name__ == "__main__":
    unittest.main()
