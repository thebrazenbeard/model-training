"""Mono PR77 exact canonical dedup and terminal source requirements."""
import json
import unittest

from test_resident_model_turn_readback_v1 import ResidentModelTurnReadbackTests
from resident_model_turn_readback_v1 import verify_resident_model_turn


class Mono77ReadbackAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.case=ResidentModelTurnReadbackTests(methodName="test_synthetic_persisted_result_accurate_but_not_real_model_attribution")
        self.case.setUp()

    def tearDown(self):
        self.case.tearDown()

    def verify(self):
        return self.case.assess()

    def test_monocanonical_task_dedup_can_recover_source(self):
        self.case.db.execute(
            "UPDATE events SET dedup_key=? WHERE id='event-1'",
            ("veraos-text:"+self.case.expected.task_id,))
        result=self.verify()
        self.assertEqual(result.status,"SYNTHETIC_STORE_LINK_VERIFIED")
        self.assertFalse(result.real_model_consumption_verified)

    def test_legacy_client_request_dedup_does_not_verify_as_mono77(self):
        self.case.db.execute(
            "UPDATE events SET dedup_key='veraos-text:turn-one' WHERE id='event-1'")
        result=self.verify()
        self.assertEqual(result.status,"HOLD")
        self.assertIn("PERSISTED_TASK_CONTINUITY_MISMATCH",result.holds)

    def test_claimed_not_done_is_a_terminal_mismatch(self):
        self.case.db.execute(
            "UPDATE events SET status='CLAIMED' WHERE id='event-1'")
        self.assertEqual(self.verify().status,"HOLD")

    def test_duplicate_matching_route_journal_must_not_pass(self):
        self.case.db.execute(
            "INSERT INTO journal VALUES (2,?,?,?)",(
                "VERAOS_COGNITION_ROUTE_SELECTED","event-1",
                json.dumps({"provider_id":"dev:virtio-qwen4b",
                 "model":"qwen3:4b-instruct","route_id":"dev:virtio-qwen4b",
                 "effect_class":"COGNITION_ONLY"})))
        self.assertEqual(self.verify().status,"HOLD")
