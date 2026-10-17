"""Executable no-effect handoff to an OS-owned local model verifier.

The fixture HMAC is NOT a production credential or an installed trust root.
"""
from dataclasses import replace
from hashlib import sha256
import hmac
import tempfile
import unittest

from resident_verifier_port_v1 import (
    ResidentModelRequest, SignedHostEvidence, evaluate_resident_admission,
    consume_resident_admission, canonical_request,
)

class LocalTrust:
    def __init__(self):
        self.enabled=True
        self.generation=3
    def assert_current(self, *,authority_id,role,key_id,key_digest,provider_id):
        if not self.enabled or self.generation!=3 or (
           authority_id,role,key_id,key_digest,provider_id
        )!=("os-owned:local-verifier","PROVIDER","fixture-key","2"*64,
            "dev:virtio-qwen4b"):
            raise ValueError("revoked, stale, or unknown verifier")
        return {
            "registry_generation":self.generation,"receipt_digest":"5"*64,
            "authority_id":authority_id,"role":role,"provider_id":provider_id,
            "key_id":key_id,"key_digest":key_digest}

class LocalPort:
    """Only OS injection could make such a verifier trustworthy in real life."""
    def __init__(self):
        self.secret=b"test-only-key-not-a-deployment-credential"
        self.actual_host="Lappy"
        self.actual_generation=1
        self.actual_session="session:1"
        self.actual_nonce="a"*32
        self.actual_blob="85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
        self.actual_manifest="0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
        self.actual_ram=8192
        self.actual_bytes=2497280480
        self.actual_sample_time=1800000000
        self.actual_endpoint="http://127.0.0.1:18766/v1"
        self.actual_model="qwen3:4b-instruct"
        self.actual_task_id="task:one"
        self.actual_task_digest="3"*64
        self.used=set()
    def verify_signature(self, *,message,signature,key_id):
        return key_id=="fixture-key" and hmac.compare_digest(
            signature,hmac.new(self.secret,message,"sha256").hexdigest())
    def reobserve(self, request):
        return {
            "serving_host":self.actual_host,
            "generation":self.actual_generation,
            "session_id":self.actual_session,
            "nonce":self.actual_nonce,
            "artifact_sha256":self.actual_blob,
            "manifest_sha256":self.actual_manifest,
            "ram_available_mib":self.actual_ram,
            "artifact_bytes":self.actual_bytes,
            "observed_at_epoch_s":self.actual_sample_time,
            "endpoint":self.actual_endpoint,
            "model_id":self.actual_model,
            "task_id":self.actual_task_id,
            "task_digest":self.actual_task_digest,
        }
    def consume_once(self,request):
        key=(request.serving_host,request.session_id,request.nonce,request.task_id)
        if key in self.used:
            return False
        self.used.add(key)
        return True

class ResidentVerifierPortTests(unittest.TestCase):
    def setUp(self):
        self.trust=LocalTrust()
        self.port=LocalPort()
        self.request=ResidentModelRequest(
            provider_id="dev:virtio-qwen4b",model_id="qwen3:4b-instruct",
            endpoint="http://127.0.0.1:18766/v1",serving_host="Lappy",
            generation=1,session_id="session:1",nonce="a"*32,
            manifest_sha256=self.port.actual_manifest,
            artifact_sha256=self.port.actual_blob,
            artifact_bytes=2497280480,task_id="task:one",
            task_digest="3"*64,observed_at_epoch_s=1800000000,
            expires_at_epoch_s=1800000060,minimum_ram_mib=512,
            route_class="COGNITION_ONLY",
        )
        self.evidence=self.sign(self.request)
    def sign(self,req):
        return SignedHostEvidence(
            issuer_id="os-owned:local-verifier",key_id="fixture-key",
            key_digest="2"*64,signature=hmac.new(
                self.port.secret,canonical_request(req),"sha256").hexdigest())
    def evaluate(self,req=None,evidence=None,now=1800000010,mode="TEST_ONLY"):
        return evaluate_resident_admission(
            req or self.request,evidence or self.evidence,
            self.trust,self.port,now_epoch_s=now,mode=mode)
    def test_synthetic_verifier_accepts_scoped_trial_but_never_runtime_admission(self):
        decision=self.evaluate()
        self.assertEqual(decision.status,"TEST_VERIFIER_SCOPED_ACCEPT")
        self.assertTrue(decision.eligible_for_test_consumer)
        self.assertFalse(decision.normal_resident_admitted)
        self.assertFalse(decision.production_qualified)
        self.assertFalse(decision.effect_authority)
        self.assertFalse(decision.identity_authority)
        self.assertEqual(decision.task_id,"task:one")

    def test_unverified_live_request_always_held(self):
        v=self.evaluate(mode="UNVERIFIED_REAL_HOST")
        self.assertEqual(v.status,"HOLD")
        self.assertIn("NO_OS_BOUND_TRUSTED_VERIFIER",v.holds)
        self.assertFalse(v.normal_resident_admitted)

    def test_unsigned_forged_issuer_and_pass_have_no_access(self):
        for evidence in (
            replace(self.evidence,issuer_id="someone-else"),
            replace(self.evidence,signature="f"*64),
            replace(self.evidence,key_digest="4"*64),
        ):
            with self.subTest(evidence=evidence):
                self.assertEqual(self.evaluate(evidence=evidence).status,"HOLD")
        self.trust.enabled=False
        self.assertEqual(self.evaluate().status,"HOLD")

    def test_stale_changed_task_endpoint_model_and_artifact_are_held(self):
        self.assertIn("EXPIRED",self.evaluate(now=1800000061).holds)
        for mutation in (dict(task_id="task:other"),
                         dict(task_digest="f"*64)):
            with self.subTest(mutation=mutation):
                self.assertEqual(
                    self.evaluate(req=replace(self.request,**mutation)).status,"HOLD")
        for mutation in (
            dict(endpoint="http://127.0.0.1:9999/v1"),
            dict(model_id="vera-trained"),dict(artifact_sha256="f"*64),
            dict(manifest_sha256="e"*64),
        ):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                replace(self.request,**mutation)
        self.port.actual_session="session:2"
        self.assertIn("HOST_OBSERVATION_MISMATCH",self.evaluate().holds)

    def test_consumer_rechecks_and_consumes_only_one_test_turn(self):
        emitted=[]
        decision=self.evaluate()
        first=consume_resident_admission(
            decision,self.request,self.evidence,self.trust,self.port,
            now_epoch_s=1800000011,mode="TEST_ONLY",
            test_consumer=lambda target: emitted.append(target) or "fixture-ack")
        self.assertEqual(first,"fixture-ack")
        self.assertEqual(emitted[0]["admission"]["status"],"HELD")
        self.assertEqual(emitted[0]["qualification"]["status"],"HOLD")
        with self.assertRaises(ValueError):
            consume_resident_admission(
                decision,self.request,self.evidence,self.trust,self.port,
                now_epoch_s=1800000012,mode="TEST_ONLY",
                test_consumer=lambda target:"replayed")
        self.assertEqual(len(emitted),1)

    def test_validly_signed_wrong_task_still_fails_independent_task_readback(self):
        swapped=replace(self.request,task_id="task:other")
        self.assertIn("HOST_OBSERVATION_MISMATCH",
                      self.evaluate(req=swapped,evidence=self.sign(swapped)).holds)
        swapped=replace(self.request,task_digest="e"*64)
        self.assertIn("HOST_OBSERVATION_MISMATCH",
                      self.evaluate(req=swapped,evidence=self.sign(swapped)).holds)

    def test_boolean_trust_ack_cannot_replace_currentness_receipt(self):
        class FakeBooleanTrust:
            def assert_current(self,**kwargs):
                return True
        outcome=evaluate_resident_admission(
            self.request,self.evidence,FakeBooleanTrust(),self.port,
            now_epoch_s=1800000010,mode="TEST_ONLY")
        self.assertEqual(outcome.status,"HOLD")
        self.assertIn("UNTRUSTED_OR_REVOKED_ISSUER",outcome.holds)

    def test_missing_actual_bytes_stale_measurement_or_changed_endpoint_hold(self):
        for field,value in (
            ("actual_bytes",None),("actual_bytes",2497280481),
            ("actual_sample_time",1799999700),
            ("actual_endpoint","http://127.0.0.1:9999/v1"),
            ("actual_model","vera-trained"),
        ):
            with self.subTest(field=field):
                original=getattr(self.port,field)
                setattr(self.port,field,value)
                try:
                    result=self.evaluate()
                    self.assertEqual(result.status,"HOLD")
                    self.assertTrue(any(x in result.holds for x in (
                        "HOST_OBSERVATION_MISMATCH",
                        "HOST_SAMPLE_OUTSIDE_WINDOW")))
                finally:
                    setattr(self.port,field,original)

    def test_cannot_consume_stale_decision_or_substitute_verifier(self):
        decision=self.evaluate()
        self.port.actual_generation=2
        with self.assertRaises(ValueError):
            consume_resident_admission(
                decision,self.request,self.evidence,self.trust,self.port,
                now_epoch_s=1800000011,mode="TEST_ONLY",
                test_consumer=lambda target:None)
        self.port.actual_generation=1
        self.trust.generation=4
        with self.assertRaises(ValueError):
            consume_resident_admission(
                decision,self.request,self.evidence,self.trust,self.port,
                now_epoch_s=1800000011,mode="TEST_ONLY",
                test_consumer=lambda target:None)

if __name__=="__main__":
    unittest.main()
