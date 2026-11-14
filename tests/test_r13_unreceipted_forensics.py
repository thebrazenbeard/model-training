import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from successor.experiments.r13_unreceipted_forensics import audit_paths


def saved(path, tensors):
    data = b""
    head = {}
    for name, values in tensors.items():
        blob = struct.pack("<" + "f" * len(values), *values)
        head[name] = {"dtype": "F32", "shape": [len(values)],
                      "data_offsets": [len(data), len(data) + len(blob)]}
        data += blob
    header = json.dumps(head, sort_keys=True).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonfile(path, contents):
    path.write_text(json.dumps(contents, sort_keys=True), encoding="utf8")


class R13UnreceiptedForensicsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.parent = root/"r12"/"adapter"/"adapter_model.safetensors"
        self.child = root/"r13"/"adapter"/"adapter_model.safetensors"
        saved(self.parent, {"a": [0, 1, 2], "b": [0.1, -0.2]})
        saved(self.child, {"a": [0, 1.5, 2], "b": [0.1, -0.2]})
        self.r12 = root/"DEV_TRAINING_COMPLETE.json"
        self.spec = root/"R13_RETRY2_SPEC.json"
        self.controller = root/"CONTROLLER_V4.json"
        self.stdout = root/"TRAINER_V4_STDOUT.log"
        receipt = {"cumulative_optimizer_steps": 52,
                   "adapter_artifacts": {"files": {
                       "adapter_model.safetensors": digest(self.parent)}}}
        receipt["receipt_sha256"] = hashlib.sha256(json.dumps(
            receipt, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False).encode()).hexdigest()
        jsonfile(self.r12, receipt)
        spec = {"resume_adapter": {
                    "adapter_model_sha256": digest(self.parent),
                    "source_receipt_sha256": receipt["receipt_sha256"],
                    "source_cumulative_optimizer_steps": 52},
                "trainer": {"max_optimizer_steps": 32},
                "source_subject": {"train_sha256": "a"*64}}
        jsonfile(self.spec, spec)
        ctrl = {"status": "R13_FAILED_OR_INTERRUPTED_NOT_VERIFIED",
                "reason": "Adapter/receipt missing despite trainer exit 0",
                "source_spec_sha256": digest(self.spec),
                "parent_r12_sha256": digest(self.parent),
                "source_corpus_sha256": "a"*64,
                "target_cumulative_optimizer_steps": 84,
                "trainer_exit_code": 0}
        jsonfile(self.controller, ctrl)
        self.stdout.write_text("".join(
            f"{{'loss': 2.0, 'epoch': {i}/32}}\n" for i in range(1, 33))
            + '{"reason":"effect label step count is unsupported",'
              '"schema":"V10R2_QWEN35_DEV_TRAINING_RECEIPT_V1",'
              '"status":"HOLD"}\n')

    def run_audit(self):
        return audit_paths(self.parent, self.child, self.r12,
                           self.spec, self.controller, self.stdout,
                           expected_tensors=2)

    def test_changed_fp32_saved_adapter_reports_unverified_completion(self):
        result = self.run_audit()
        self.assertEqual("R13_FP32_DELTA_OBSERVED_COMPLETION_UNVERIFIED",
                         result["status"])
        self.assertEqual(2, result["tensor_count"])
        self.assertEqual(1, result["changed_tensors"])
        self.assertEqual(1, result["changed_elements"])
        self.assertEqual(32, result["reported_loss_steps"])
        self.assertTrue(result["saved_weight_delta_observed"])
        self.assertFalse(result["training_completion_verified"])
        self.assertFalse(result["behavior_qualified"])
        self.assertFalse(result["runtime_qualified"])
        self.assertEqual("NONE", result["authority_effect"])

    def test_missing_final_receipt_is_required(self):
        (self.child.parent.parent/"DEV_TRAINING_COMPLETE.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "unreceipted"):
            self.run_audit()

    def test_nonfinite_child_tensor_refused(self):
        saved(self.child, {"a": [0, float("nan"), 2], "b": [0.1, -0.2]})
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.run_audit()

    def test_controller_must_report_failed_verification(self):
        ctrl = json.loads(self.controller.read_text())
        ctrl["status"] = "R13_COMPLETE"
        jsonfile(self.controller, ctrl)
        with self.assertRaisesRegex(ValueError, "controller"):
            self.run_audit()

    def test_wrong_parent_receipt_binding_refused(self):
        receipt = json.loads(self.r12.read_text())
        receipt["adapter_artifacts"]["files"]["adapter_model.safetensors"] = "0"*64
        jsonfile(self.r12, receipt)
        with self.assertRaisesRegex(ValueError, "R12"):
            self.run_audit()

    def test_wrong_step_count_refused(self):
        self.stdout.write_text(self.stdout.read_text().replace(
            "{'loss': 2.0, 'epoch': 32/32}\n", ""))
        with self.assertRaisesRegex(ValueError, "32 loss"):
            self.run_audit()


    def test_equal_saved_weights_cannot_be_presented_as_delta(self):
        self.child.write_bytes(self.parent.read_bytes())
        with self.assertRaisesRegex(ValueError, "no observed saved FP32 tensor delta"):
            self.run_audit()

    def test_non_fp32_saved_tensor_refused(self):
        data = self.child.read_bytes()
        self.child.write_bytes(data.replace(b'"F32"', b'"F16"', 1))
        with self.assertRaisesRegex(ValueError, "non-FP32"):
            self.run_audit()

    def test_forensic_run_does_not_write_or_modify_original_artifacts(self):
        before = tuple(digest(p) for p in (
            self.parent, self.child, self.r12, self.spec,
            self.controller, self.stdout))
        self.run_audit()
        after = tuple(digest(p) for p in (
            self.parent, self.child, self.r12, self.spec,
            self.controller, self.stdout))
        self.assertEqual(before, after)
        self.assertFalse((self.child.parent.parent/
                          "DEV_TRAINING_COMPLETE.json").exists())

if __name__ == "__main__":
    unittest.main()
