"""Exact Mono provider-role trust registry + Training port; isolated fixtures.

No existing registry, credential or runtime trust state is changed.
"""
from pathlib import Path
import os
import sys
import tempfile
import unittest

from resident_verifier_port_v1 import evaluate_resident_admission
from test_resident_verifier_port_v1 import ResidentVerifierPortTests

class ActualMonoProviderCurrentnessTests(unittest.TestCase):
    def test_actual_registry_provider_revoke_rejects_source_trial(self):
        mono=os.environ.get("VERA_MONO_SOURCE_ROOT")
        if not mono or not (
            Path(mono)/"packages"/"vera_core"/"src"/"vera_core"/"outbound_trust.py"
        ).is_file():
            self.skipTest("pinned Mono source not installed in test sandbox")
        try:
            from jsonschema import Draft202012Validator
        except ModuleNotFoundError:
            self.skipTest("exact Mono full-package dependency jsonschema absent")
        for package_src in sorted((Path(mono)/"packages").glob("*/src")):
            sys.path.insert(0,str(package_src))
        from vera_core.outbound_trust import OutboundTrustRegistry
        sample=ResidentVerifierPortTests()
        sample.setUp()
        with tempfile.TemporaryDirectory(prefix="TRAINING-SYNTHETIC-MONO-REG-") as td:
            reg=OutboundTrustRegistry(Path(td)/"outbound-trust-test.sqlite3")
            receipt=reg.register(
                authority_id=sample.evidence.issuer_id,role="PROVIDER",
                provider_id=sample.request.provider_id,
                key_id=sample.evidence.key_id,
                key_digest=sample.evidence.key_digest,
                expected_registry_generation=0)
            self.assertEqual(receipt.schema,"VERA_MONO_AUTHORITY_CURRENTNESS_RECEIPT_V1")
            allowed=evaluate_resident_admission(
                sample.request,sample.evidence,reg,sample.port,
                now_epoch_s=1800000010,mode="TEST_ONLY")
            self.assertEqual(allowed.status,"TEST_VERIFIER_SCOPED_ACCEPT")
            self.assertFalse(allowed.normal_resident_admitted)
            self.assertEqual(allowed.registry_generation,1)
            self.assertTrue(reg.verify_chain())
            reg.revoke(
                authority_id=sample.evidence.issuer_id,role="PROVIDER",
                provider_id=sample.request.provider_id,
                expected_registry_generation=1,expected_revocation_epoch=0)
            denied=evaluate_resident_admission(
                sample.request,sample.evidence,reg,sample.port,
                now_epoch_s=1800000011,mode="TEST_ONLY")
            self.assertEqual(denied.status,"HOLD")
            self.assertIn("UNTRUSTED_OR_REVOKED_ISSUER",denied.holds)
            real=evaluate_resident_admission(
                sample.request,sample.evidence,reg,sample.port,
                now_epoch_s=1800000011,mode="UNVERIFIED_REAL_HOST")
            self.assertEqual(real.status,"HOLD")
            self.assertIn("NO_OS_BOUND_TRUSTED_VERIFIER",real.holds)
            # Mono's SQLite context closes the transaction; on Windows,
            # collect temporary connection wrappers before removing DB.
            import gc
            reg=None
            gc.collect()
