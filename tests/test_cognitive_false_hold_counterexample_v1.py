"""Developer counterexample: source guard may withhold a supported answer.

This is NOT a held-out cognition score; tests measure an explicit utility
failure alongside warranted HOLD outcomes, without calling a model.
"""
from dataclasses import replace
from pathlib import Path
import unittest

from cognitive_continuity_eval_v1 import CognitiveSource, guard_prediction
from run_cognitive_continuity_dev_v1 import load_development_fixture


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "cognitive_continuity_devset_v1.json"


class ConservativeGuardUtilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        observed, expected, candidates = load_development_fixture(FIXTURE)
        cls.observed = {x.case_id: x for x in observed}
        cls.candidates = candidates

    def test_single_supported_answer_is_preserved(self):
        actual = guard_prediction(self.observed["t01"], self.candidates["t01"])
        self.assertEqual(actual.status, "ANSWER")
        self.assertEqual(actual.claimed_value, "Lappy")

    def test_two_agreeing_records_preserve_a_supported_answer(self):
        original = self.observed["t01"]
        source = original.sources[0]
        corroborated = replace(original, sources=(
            source, CognitiveSource("ev:host:corroboration", "machine", "veraos", "Lappy"),
        ))
        candidate = self.candidates["t01"]
        result = guard_prediction(corroborated, candidate)
        # Same literal supported by distinct active source event IDs.
        # This is a provenance decision, not independent truth evidence.
        self.assertEqual(candidate.status, "ANSWER")
        self.assertEqual(result.status, "ANSWER")
        self.assertEqual(result.source_event_id,"ev:host")

    def test_genuine_contradiction_requires_hold(self):
        result = guard_prediction(self.observed["t09"], self.candidates["t09"])
        self.assertEqual(result.status, "HOLD")
        self.assertIsNone(result.claimed_value)

    def test_corrected_source_preserves_a_useful_answer(self):
        result = guard_prediction(self.observed["t11"], self.candidates["t11"])
        self.assertEqual(result.status, "ANSWER")
        self.assertEqual(result.claimed_value, "role")

    def test_declared_development_tradeoff_not_trained_reasoning(self):
        supported = ["t01", "t11"]
        unsupported = ["t05", "t09", "t10"]
        preserved = sum(
            guard_prediction(self.observed[k], self.candidates[k]).status == "ANSWER"
            for k in supported
        )
        warranted_holds = sum(
            guard_prediction(self.observed[k], self.candidates[k]).status == "HOLD"
            for k in unsupported
        )
        self.assertEqual(preserved, 2)
        self.assertEqual(warranted_holds, 3)
        # Additional agreeing-supported case above: false HOLD 0/3 on
        # these synthetic developer inputs; no blind/live-model evidence.
