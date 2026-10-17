"""Adversarial PR37 HTTP completion envelope; no network/model execution."""
from unittest import TestCase

from resident_http_completion_gate_v1 import normalize_http_result, HttpResultHold

class ResidentHttpCompletionGateTests(TestCase):
    def test_synthetic_persisted_complete_has_full_exact_source_tuple(self):
        raw=dict(status="COMPLETED",run_id="run:one",event_id="event:one",
            final_text="Local synthetic response",route_id="dev:virtio-qwen4b",
            provider_id="dev:virtio-qwen4b")
        result=normalize_http_result(raw, request_id="msg:one",
            expected_provider="dev:virtio-qwen4b",fixture_mode=True)
        self.assertEqual(result["status"],"COMPLETED")
        self.assertEqual(result["request_id"],"msg:one")
        self.assertEqual(result["final_text"],"Local synthetic response")
        self.assertEqual(result["run_id"],"run:one")
        self.assertFalse(result["model_authenticated"])
        self.assertFalse(result["display_authorized"])
        self.assertFalse(result["canonical_task_completed"])
        self.assertFalse(result["effect_authority"])

    def test_false_completed_missing_run_event_or_response_holds(self):
        valid=dict(status="COMPLETED",run_id="run:one",event_id="event:one",
            final_text="Synthetic response",route_id="dev:virtio-qwen4b",
            provider_id="dev:virtio-qwen4b")
        for invalid in (
            dict(run_id=None),dict(event_id=None),dict(final_text=None),
            dict(final_text=""),dict(final_text=" "*3),
            dict(final_text="a"*4097),dict(run_id=""),dict(event_id=""),
            dict(final_text=42),dict(status=True),
            dict(provider_id="other"),dict(route_id="other"),
            dict(status="PENDING"),dict(status="OUTCOME_UNKNOWN"),
            dict(run_id="wrong/id"),
            dict(model_authenticated=True),dict(effect_authority=True)):
            with self.subTest(invalid=invalid), self.assertRaises(HttpResultHold):
                normalize_http_result({**valid,**invalid},request_id="msg:one",
                    expected_provider="dev:virtio-qwen4b",fixture_mode=True)

    def test_pending_is_not_completed_and_must_have_no_result_claims(self):
        pending=dict(status="PENDING",run_id=None,event_id="event:one",
            final_text=None,route_id="dev:virtio-qwen4b",provider_id="dev:virtio-qwen4b")
        r=normalize_http_result(pending,request_id="msg:one",
            expected_provider="dev:virtio-qwen4b",fixture_mode=True)
        self.assertEqual(r["status"],"PENDING")
        self.assertIsNone(r["final_text"])
        self.assertFalse(r["display_authorized"])
        with self.assertRaises(HttpResultHold):
            normalize_http_result({**pending,"final_text":"manufactured"},
                request_id="msg:one",expected_provider="dev:virtio-qwen4b",
                fixture_mode=True)

    def test_real_mode_is_always_held_until_os_verifier_installed(self):
        answer=dict(status="COMPLETED",run_id="run:one",event_id="event:one",
            final_text="looks plausible",route_id="dev:virtio-qwen4b",
            provider_id="dev:virtio-qwen4b")
        for mode in (None,False,"VERIFIED_REAL","REAL_HOST",0):
            with self.subTest(mode=mode), self.assertRaises(HttpResultHold):
                normalize_http_result(answer,request_id="msg:one",
                    expected_provider="dev:virtio-qwen4b",fixture_mode=mode)
