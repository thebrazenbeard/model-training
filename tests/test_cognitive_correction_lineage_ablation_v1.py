"""Exact prechange Git module comparison; source-only reproducible metric."""
import unittest
from evaluate_correction_lineage_v1 import evaluate, prior_exact_module

class FrozenCorrectionLineageAblation(unittest.TestCase):
    def test_previous_revision_is_actual_git_source(self):
        prior,body_hash=prior_exact_module()
        self.assertEqual(len(body_hash),64)
        self.assertIn("guard_prediction",prior.__dict__)

    def test_frozen_development_cases_preserve_legitimate_answers(self):
        result=evaluate()
        self.assertEqual(result["frozen_developer_fixture_sha256"],
            "3e1bd910dd44705e3d76e8b96e3c4f21f6fdcbc36cb99e312c20b872aef4b517")
        self.assertEqual(result["baseline"]["supported_total"],6)
        self.assertEqual(result["candidate"]["supported_total"],6)
        self.assertEqual(result["baseline"]["supported_preserved"],6)
        self.assertEqual(result["candidate"]["supported_preserved"],5)
        self.assertEqual(result["candidate"]["supported_false_holds"],1)

    def test_adverse_correction_acceptances_reduced_without_other_drift(self):
        result=evaluate()
        self.assertEqual(result["baseline"]["adverse_total"],12)
        self.assertEqual(result["baseline"]["unsafe_answer_acceptances"],6)
        self.assertEqual(result["candidate"]["unsafe_answer_acceptances"],0)
        self.assertEqual(result["candidate"]["adverse_correct_holds"],12)
        repaired={x["case"] for x in result["baseline"]["rows"]
                  if x["observed_status"]=="ANSWER" and not x["should_answer"]}
        self.assertEqual(repaired,{
            "orphan_predecessor","untyped_overwrite","cross_scope_parent",
            "cross_referent_parent","wrong_previous_id","no_named_predecessor"})

    def test_no_real_model_or_memory_assertions(self):
        report=evaluate()
        for name in ("model_inference","weight_training","independent_fact_review"):
            self.assertFalse(report[name])
        self.assertTrue(report["source_only"])
