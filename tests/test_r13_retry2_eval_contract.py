"""R13 evaluation cannot accept fabricated lineage or incomplete training."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from successor.experiments import probe_vera_r13_heldout_retry2 as probe
from successor.experiments import audit_r13_fp32_tensor_delta_retry2 as audit

def make_receipt(data):
    r={
        "status":"DEVELOPMENT_OPTIMIZER_STEP_COMPLETE",
        "cumulative_optimizer_steps":84,
        "adapter_precision_policy":"FP32_RESUMED_LORA_TRAIN_AND_SAVE_V1",
        "adapter_precision_final":{"tensor_dtype":"float32"},
        "deployment_status":"NOT_DEPLOYED",
        "resume_adapter":{"source_cumulative_optimizer_steps":52,
                          "adapter_model_sha256":probe.PARENT_R12_SHA},
        "spec_sha256":probe.SPEC_SHA,
        "train_sha256":probe.CORPUS_SHA,
        "exact_resume_rehydration":{"restored_tensors":496,
                                    "adapter_model_sha256":probe.PARENT_R12_SHA,
                                    "dtype":"float32"},
        "adapter_artifacts":{"files":{"adapter_model.safetensors":hashlib.sha256(data).hexdigest()}},
    }
    r["receipt_sha256"]=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf8")).hexdigest()
    return r


class R13EvalTests(unittest.TestCase):
    def test_completed_and_rehydrated_receipt_is_required(self):
        with tempfile.TemporaryDirectory() as td:
            adapter=Path(td)/"adapter"
            adapter.mkdir()
            raw=b"synthetic-eval-tensor-bytes"
            (adapter/"adapter_model.safetensors").write_bytes(raw)
            receipt=make_receipt(raw)
            (adapter.parent/"DEV_TRAINING_COMPLETE.json").write_text(json.dumps(receipt))
            with patch.object(probe,"R13",adapter):
                self.assertEqual(probe.verified_r13_sha(),hashlib.sha256(raw).hexdigest())
                broken=dict(receipt)
                broken["cumulative_optimizer_steps"]=52
                data=dict(broken)
                data.pop("receipt_sha256")
                broken["receipt_sha256"]=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf8")).hexdigest()
                (adapter.parent/"DEV_TRAINING_COMPLETE.json").write_text(json.dumps(broken))
                with self.assertRaisesRegex(RuntimeError,"lineage"):
                    probe.verified_r13_sha()

    def test_eval_never_overwrites_existing_case_output(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"existing.json"
            out.write_text("{}")
            with patch.object(probe,"OUT",out):
                with self.assertRaisesRegex(RuntimeError,"already attempted"):
                    probe.ensure_preflight()

    def test_evaluation_is_no_system_and_multi_axis(self):
        self.assertEqual(len(probe.CASES),10)
        self.assertIn("VERA_VOICE_HELDOUT_001",probe.CASES)
        self.assertIn("VERA_VOICE_HELDOUT_007",probe.CASES)
        self.assertIn("VERA_VOICE_HELDOUT_011",probe.CASES)
        source=Path(probe.__file__).read_text(encoding="utf8")
        self.assertIn("enable_thinking=False",source)
        self.assertIn("max_new_tokens=72",source)
        self.assertNotIn("trainer.train(",source)

    def test_audit_requires_fp32_both_sides_and_does_not_overwrite(self):
        source=Path(audit.__file__).read_text(encoding="utf8")
        self.assertIn("a.dtype != torch.float32",source)
        self.assertIn("b.dtype != torch.float32",source)
        self.assertIn('receipt["exact_resume_rehydration"]["restored_tensors"] != 496',source)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"existing.json"
            out.write_text("{}")
            with patch.object(audit,"OUT",out):
                with self.assertRaisesRegex(RuntimeError,"exists"):
                    audit.audit()


if __name__=="__main__":
    unittest.main()
