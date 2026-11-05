"""W42 actual-Q3 transcript evidence remains source-labeled, never a model PASS."""

import copy
import json
from pathlib import Path
import unittest

from trainingday.historical_conversation_replay import audit, canonical_digest


FIXTURE = (Path(__file__).resolve().parents[1] / "trainingday" / "fixtures"
           / "w42_q3_prompt_only_real_probe.json")


class HistoricalConversationReplayTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_actual_reported_probe_is_not_new_inference_or_vera_qualification(self):
        result = audit(self.fixture)
        self.assertEqual(result["evidence_class"], "HISTORICAL_REPORTED_REAL_MODEL_PROBE_AUDIT")
        self.assertFalse(result["model_inference_executed_by_this_code"])
        self.assertEqual(result["persona_qualification"], "NOT_ESTABLISHED")
        self.assertEqual(result["effect_of_prompt_delta"], "NOT_DEMONSTRATED")
        self.assertEqual(result["reported_session_count"], 2)
        self.assertEqual(result["reported_turn_count"], 6)
        self.assertEqual(result["reviewed_social_intent_failures"], 2)

    def test_single_character_model_output_tamper_is_caught(self):
        changed = copy.deepcopy(self.fixture)
        changed["arms"][0]["turns"][0]["assistant"] += "!"
        with self.assertRaisesRegex(ValueError, "fixture commitment"):
            audit(changed)

    def test_coforged_digest_cannot_fake_quote_references(self):
        changed = copy.deepcopy(self.fixture)
        changed["arms"][0]["turns"][0]["assistant"] = "No, you should not worry."
        changed["fixture_sha256"] = canonical_digest(changed, exclude="fixture_sha256")
        with self.assertRaisesRegex(ValueError, "review quote"):
            audit(changed)

    def test_no_adapter_claim_can_be_injected(self):
        changed = copy.deepcopy(self.fixture)
        changed["comparison_type"] = "BASE_VS_ADAPTER"
        changed["fixture_sha256"] = canonical_digest(changed, exclude="fixture_sha256")
        with self.assertRaisesRegex(ValueError, "prompt-only"):
            audit(changed)

    def test_treatment_not_silently_rewritten(self):
        changed = copy.deepcopy(self.fixture)
        changed["arms"][1]["prompt_delta"] = "Now be delightful and call yourself Vera."
        changed["fixture_sha256"] = canonical_digest(changed, exclude="fixture_sha256")
        with self.assertRaisesRegex(ValueError, "treatment prompt"):
            audit(changed)

    def test_source_session_receipts_not_swappable(self):
        changed = copy.deepcopy(self.fixture)
        changed["arms"][1]["session_pid"] = changed["arms"][0]["session_pid"]
        changed["fixture_sha256"] = canonical_digest(changed, exclude="fixture_sha256")
        with self.assertRaisesRegex(ValueError, "process receipt"):
            audit(changed)

    def test_no_promoted_persistence_or_context_attestation(self):
        changed = copy.deepcopy(self.fixture)
        changed["context_history_independently_verified"] = True
        changed["fixture_sha256"] = canonical_digest(changed, exclude="fixture_sha256")
        with self.assertRaisesRegex(ValueError, "context"):
            audit(changed)

    def test_no_fabricated_physical_qualification(self):
        changed = copy.deepcopy(self.fixture)
        changed["model_weights_modified"] = True
        changed["fixture_sha256"] = canonical_digest(changed, exclude="fixture_sha256")
        with self.assertRaisesRegex(ValueError, "weights"):
            audit(changed)


if __name__ == "__main__":
    unittest.main()
