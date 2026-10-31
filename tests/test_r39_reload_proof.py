"""CPU-only adversarial checks for independent adapter-reload receipts."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from trainingday.admission import AdmissionError, _hash_file
from trainingday.reload_verify import validate_receipt, verify
from trainingday.r18_eval import BASE_REVISION
from trainingday.train_neural import _parameter_digest


class _Truth:
    def __init__(self, value: bool):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


class _Tensor:
    requires_grad = True
    shape = (1,)

    def __init__(self, value: float, dtype: str):
        self.value = value
        self.dtype = dtype

    def numel(self):
        return 1

    def detach(self):
        return self

    def to(self, *, device, dtype):
        return _Tensor(self.value, dtype)

    def contiguous(self):
        return self

    def numpy(self):
        return self

    def tobytes(self):
        return struct.pack("<f", self.value)


class _Model:
    def __init__(self, tensor):
        self.tensor = tensor

    def named_parameters(self):
        return [("lora_A.weight", self.tensor)]


def _fake_runtime(model):
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
        mem_get_info=lambda device: (6 * 2**30, 8 * 2**30),
    )
    torch = types.ModuleType("torch")
    torch.cuda = cuda
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.isfinite = lambda tensor: _Truth(math.isfinite(tensor.value))
    torch.__version__ = "fake-runtime-cpu"

    peft = types.ModuleType("peft")
    peft.PeftModel = types.SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: model,
    )
    transformers = types.ModuleType("transformers")
    transformers.BitsAndBytesConfig = lambda **_kwargs: object()
    transformers.Qwen3_5ForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: object(),
    )
    return torch, {"torch": torch, "peft": peft, "transformers": transformers}


def _prepare_fixture(base: Path, tensor):
    snapshot = base / "official-base"
    snapshot.mkdir()
    frozen = {}
    for name, data in (
        ("config.json", b"{}"),
        ("tokenizer_config.json", b"{}"),
        ("model-00001.safetensors", b"synthetic model fixture"),
    ):
        (snapshot / name).write_bytes(data)
        frozen[name] = hashlib.sha256(data).hexdigest()
    binding = base / "base-binding.json"
    binding.write_text(json.dumps({
        "schema": "TRAININGDAY_PINNED_BASE_V1",
        "model_id": "Qwen/Qwen3.5-4B",
        "revision": BASE_REVISION,
        "loader_class": "Qwen3_5ForCausalLM",
        "quantization": "NF4_DOUBLE_BF16",
        "snapshot_path": str(snapshot),
        "files": frozen,
    }), encoding="utf-8")
    run = base / "run"
    adapter = run / "adapter"
    adapter.mkdir(parents=True)
    model_file = adapter / "adapter_model.safetensors"
    model_file.write_bytes(b"synthetic adapter fixture")
    config = adapter / "adapter_config.json"
    config.write_text("{}", encoding="utf-8")
    fake_torch, modules = _fake_runtime(_Model(tensor))
    if not math.isfinite(tensor.value):
        # A deliberately forged legacy receipt can contain a nonfinite digest.
        # The training-side R41 guard must not be invoked to manufacture it.
        digest = hashlib.sha256()
        digest.update(b"lora_A.weight")
        digest.update(b"(1,)")
        digest.update(struct.pack("<f", tensor.value))
        expected, count = digest.hexdigest(), 1
    else:
        expected, count = _parameter_digest(_Model(tensor), fake_torch)
    receipt = run / "training_receipt.json"
    receipt.write_text(json.dumps({
        "status": "ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED",
        "mode": "correct",
        "new_optimizer_steps": 1,
        "trainable_digest_after": expected,
        "trainable_parameter_count": count,
        "base_revision": BASE_REVISION,
        "base_binding_sha256": _hash_file(binding),
        "adapter_model_sha256": _hash_file(model_file),
        "adapter_config_sha256": _hash_file(config),
    }), encoding="utf-8")
    return binding, receipt, base / "reload.json", modules


class ReloadProofTests(unittest.TestCase):
    def test_reloaded_native_fp16_cannot_pass_equal_fp32_normalized_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            binding, receipt, output, modules = _prepare_fixture(
                Path(temp), _Tensor(0.5, "float16")
            )
            with patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(RuntimeError, "FP32"):
                    verify(receipt, binding, output)
            self.assertFalse(output.exists())

    def test_nonfinite_reloaded_parameter_cannot_pass_digest_equality(self):
        with tempfile.TemporaryDirectory() as temp:
            binding, receipt, output, modules = _prepare_fixture(
                Path(temp), _Tensor(float("nan"), "float32")
            )
            with patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                    verify(receipt, binding, output)
            self.assertFalse(output.exists())

    def test_finite_native_fp32_adapter_still_reloads(self):
        with tempfile.TemporaryDirectory() as temp:
            binding, receipt, output, modules = _prepare_fixture(
                Path(temp), _Tensor(0.5, "float32")
            )
            with patch.dict(sys.modules, modules):
                report = verify(receipt, binding, output)
            self.assertEqual(report["status"], "INDEPENDENT_ADAPTER_RELOAD_VERIFIED")
            self.assertTrue(output.is_file())

    def test_boolean_optimizer_steps_cannot_be_accepted_as_integer(self):
        with tempfile.TemporaryDirectory() as temp:
            binding, receipt, _output, _modules = _prepare_fixture(
                Path(temp), _Tensor(0.5, "float32")
            )
            data = json.loads(receipt.read_text(encoding="utf-8"))
            data["new_optimizer_steps"] = True
            receipt.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(AdmissionError, "actual trained"):
                validate_receipt(receipt, binding)

    def test_nonpositive_or_noninteger_trainable_count_rejected_preload(self):
        for bad_count in (True, False, 0, -1, 1.0, "1"):
            with self.subTest(bad_count=bad_count):
                with tempfile.TemporaryDirectory() as temp:
                    binding, receipt, output, _modules = _prepare_fixture(
                        Path(temp), _Tensor(0.5, "float32")
                    )
                    data = json.loads(receipt.read_text(encoding="utf-8"))
                    data["trainable_parameter_count"] = bad_count
                    receipt.write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaisesRegex(AdmissionError, "trainable parameter count"):
                        validate_receipt(receipt, binding)
                    self.assertFalse(output.exists())

    def test_base_mutation_during_model_load_rejected_before_success_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            binding, receipt, output, modules = _prepare_fixture(
                Path(temp), _Tensor(0.5, "float32")
            )
            pinned = json.loads(binding.read_text(encoding="utf-8"))
            config = Path(pinned["snapshot_path"]) / "config.json"

            def loading_mutates_base(*_args, **_kwargs):
                # First _required_base passes; the model loader alters the
                # trusted snapshot before independent reload is attested.
                config.write_bytes(b'{"altered":"during-loader"}')
                return object()

            modules["transformers"].Qwen3_5ForCausalLM.from_pretrained = loading_mutates_base
            with patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(AdmissionError, "base file custody mismatch"):
                    verify(receipt, binding, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
