"""R12 inference evidence must bind exact finished adapter and refuse forged receipts."""
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "successor/experiments/probe_vera_r12_heldout_generation.py"
SPEC = importlib.util.spec_from_file_location("r12_evaluation", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def receipt_for(data):
    raw = {
        "status": "DEVELOPMENT_OPTIMIZER_STEP_COMPLETE",
        "cumulative_optimizer_steps": 52,
        "adapter_precision_policy": "FP32_RESUMED_LORA_TRAIN_AND_SAVE_V1",
        "deployment_status": "NOT_DEPLOYED",
        "resume_adapter": {"source_cumulative_optimizer_steps": 44},
        "adapter_artifacts": {"files": {
            "adapter_model.safetensors": hashlib.sha256(data).hexdigest()
        }},
    }
    raw["receipt_sha256"] = hashlib.sha256(json.dumps(raw,
        sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf8")).hexdigest()
    return raw


class R12EvaluationProofTests(unittest.TestCase):
    def test_valid_receipt_binds_adapter(self):
        with TemporaryDirectory() as td:
            adapter = Path(td) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"fake-weights-for-unit-test")
            (adapter.parent / "DEV_TRAINING_COMPLETE.json").write_text(
                json.dumps(receipt_for(b"fake-weights-for-unit-test")))
            with patch.object(probe, "R12", adapter):
                self.assertEqual(probe.verified_r12_sha(),
                    hashlib.sha256(b"fake-weights-for-unit-test").hexdigest())

    def test_reject_wrong_step_count_even_with_valid_self_hash(self):
        with TemporaryDirectory() as td:
            adapter = Path(td) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"fake")
            receipt = receipt_for(b"fake")
            receipt["cumulative_optimizer_steps"] = 44
            content = dict(receipt)
            content.pop("receipt_sha256")
            receipt["receipt_sha256"] = hashlib.sha256(json.dumps(content,
                sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf8")).hexdigest()
            (adapter.parent / "DEV_TRAINING_COMPLETE.json").write_text(json.dumps(receipt))
            with patch.object(probe, "R12", adapter):
                with self.assertRaisesRegex(RuntimeError, "lineage"):
                    probe.verified_r12_sha()

    def test_reject_existing_evaluation_output_before_model_load(self):
        with TemporaryDirectory() as td:
            output = Path(td) / "already-run.json"
            output.write_text("{}")
            with patch.object(probe, "OUT", output):
                with self.assertRaisesRegex(RuntimeError, "already attempted"):
                    probe.ensure_preflight()
