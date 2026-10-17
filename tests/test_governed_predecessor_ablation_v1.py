"""Frozen exact-ledger read-only retrieval experiment, no provider."""
import unittest
from evaluate_governed_predecessor_v1 import run

class GovernedSourceComparison(unittest.TestCase):
    def test_governed_retrieval_beats_hold_only_without_unsafe_heuristic(self):
        result=run()
        self.assertEqual(result["frozen_scenario_digest"],
            "e6d5c65ff456f2a486da33d9ca2a692974fd764566b2c23cd391cc439d60426b")
        self.assertEqual(result["baseline"]["supported_answered"],0)
        self.assertEqual(result["unsafe_heuristic"]["supported_answered"],2)
        self.assertEqual(result["unsafe_heuristic"]["unsafe_acceptances"],5)
        self.assertEqual(result["governed_retrieval"]["supported_answered"],1)
        self.assertEqual(result["governed_retrieval"]["unsafe_acceptances"],0)
        self.assertEqual(result["governed_retrieval"]["false_holds"],1)

    def test_source_result_never_claims_model_or_memory_admission(self):
        result=run()
        for k in ("real_model_inference","provider_admitted",
                  "independent_semantic_verification","model_weight_training",
                  "canonical_memory_write"):
            self.assertFalse(result[k])
