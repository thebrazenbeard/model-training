import unittest
from training.classify_model_evidence import classify

H = "a" * 64
R = "b" * 40


class EvidenceTests(unittest.TestCase):
    def receipt(self, level=0):
        return {"schema": "VERA_NEURAL_EVIDENCE_V1",
                "training_kind": "adapter_finetune" if level else "none",
                "claims": {"weight_update": level >= 1, "artifact": level >= 2, "evaluation": level >= 3},
                "evidence": {"base_revision": R, "dataset_sha256": H, "recipe_sha256": H,
                             "run_log_sha256": H, "weights_sha256": H, "eval_manifest_sha256": H,
                             "eval_results_sha256": H, "baseline_results_sha256": H},
                "held_out": level == 3, "independent_evaluator": level == 3}

    def test_each_level_never_qualifies(self):
        for level in range(4):
            result = classify(self.receipt(level))
            self.assertEqual(result["classification"],
                             ("CONFIGURED_ONLY", "WEIGHT_UPDATE_REPORTED",
                              "ARTIFACT_REPORTED", "EVALUATION_REPORTED")[level])
            self.assertFalse(result["model_qualified"])
            self.assertFalse(result["hashes_verified"])

    def test_rejects_unbacked_training_claim(self):
        r = self.receipt(1)
        del r["evidence"]["run_log_sha256"]
        with self.assertRaises(ValueError): classify(r)

    def test_rejects_evaluation_without_independence(self):
        r = self.receipt(3)
        r["independent_evaluator"] = False
        with self.assertRaises(ValueError): classify(r)

    def test_rejects_skipped_stage(self):
        r = self.receipt(0)
        r["claims"]["evaluation"] = True
        with self.assertRaises(ValueError): classify(r)

    def test_rejects_boolean_strings(self):
        r = self.receipt(1)
        r["claims"]["weight_update"] = "true"
        with self.assertRaises(ValueError): classify(r)


if __name__ == "__main__":
    unittest.main()
