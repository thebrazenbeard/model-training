"""Synthetic-only adversarial contract tests; no bank files or real keys."""

import hashlib
import hmac
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from trainingday.r24_custody_contract import (
    CustodyContext,
    GrantEnvelope,
    GrantRequest,
    SyntheticNonceJournal,
    assess_grant,
)


def sha(label):
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class SyntheticCustodyContractTests(unittest.TestCase):
    def test_missing_or_wrong_envelope_types_fail_closed(self):
        for grant, request, context in (
            (None, self.request, self.context),
            (self.grant, None, self.context),
            (self.grant, self.request, None),
            ({"issuer": "fake"}, self.request, self.context),
            (self.grant, object(), self.context),
            (self.grant, self.request, {"audit_acked": True}),
        ):
            with self.subTest(grant=type(grant), request=type(request), context=type(context)):
                decision = assess_grant(grant, request, context, now=1700000030,
                                        verify_signature=None, nonce_journal=self.journal)
                self.assertEqual(decision.reason, "MALFORMED_ENVELOPE")
                self.assertFalse(decision.bank_access_authorized)

    def setUp(self):
        self.secret = b"synthetic-test-secret-not-a-deployment-key"
        self.request = GrantRequest(
            run_id="r24-synthetic-001",
            model_sha256=sha("model"),
            adapter_sha256=sha("adapter"),
            corpus_sha256=sha("corpus"),
            bank_object_id="synthetic-bank-object-1",
            evaluator_principal="eval-user",
        )
        self.context = CustodyContext(
            trusted_issuer="attester-user",
            bank_owner_principal="bank-owner",
            trainer_principal="trainer-user",
            audit_principal="audit-writer",
            audit_acked=True,
            raw_bank_path_exposed=False,
            bank_object_identity_stable=True,
        )
        self.grant = GrantEnvelope(
            issuer="attester-user", subject="eval-user", nonce="nonce-unique-001",
            issued_at=1700000000, expires_at=1700000120,
            run_id=self.request.run_id,
            model_sha256=self.request.model_sha256,
            adapter_sha256=self.request.adapter_sha256,
            corpus_sha256=self.request.corpus_sha256,
            bank_object_id=self.request.bank_object_id,
            signature="",
        )
        self.journal = SyntheticNonceJournal()

    def sign(self, grant):
        return replace(grant, signature=hmac.new(
            self.secret, grant.canonical_payload(), hashlib.sha256
        ).hexdigest())

    def verify(self, payload, signature, issuer):
        return issuer == "attester-user" and hmac.compare_digest(
            hmac.new(self.secret, payload, hashlib.sha256).hexdigest(), signature
        )

    def assess(self, grant=None, *, context=None, request=None, verifier=None):
        return assess_grant(
            self.sign(self.grant) if grant is None else grant,
            self.request if request is None else request,
            self.context if context is None else context,
            now=1700000030,
            verify_signature=self.verify if verifier is None else verifier,
            nonce_journal=self.journal,
        )

    def test_valid_synthetic_contract_still_denies_physical_bank_access(self):
        result = self.assess()
        self.assertTrue(result.contract_match)
        self.assertFalse(result.bank_access_authorized)
        self.assertEqual(result.reason, "SYNTHETIC_MATCH_NO_HOST_AUTHORITY")

    def test_grant_replay_denied_after_first_synthetic_acceptance(self):
        self.assertTrue(self.assess().contract_match)
        self.assertEqual(self.assess().reason, "NONCE_REPLAY")

    def test_concurrent_replay_has_at_most_one_synthetic_match(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.assess(), range(32)))
        self.assertEqual(sum(result.contract_match for result in results), 1)
        self.assertEqual(sum(result.reason == "NONCE_REPLAY" for result in results), 31)
        self.assertTrue(all(not result.bank_access_authorized for result in results))

    def test_forged_signature_fails_without_nonce_poisoning(self):
        forged = replace(self.sign(self.grant), adapter_sha256=sha("substitute"))
        self.assertEqual(self.assess(forged).reason, "SIGNATURE_INVALID")
        self.assertTrue(self.assess().contract_match)

    def test_hash_and_bank_object_mismatch_are_rejected(self):
        for field in ("model_sha256", "adapter_sha256", "corpus_sha256", "bank_object_id", "run_id"):
            with self.subTest(field=field):
                value = sha("wrong") if field.endswith("sha256") else "wrong-id"
                self.assertEqual(self.assess(self.sign(replace(self.grant, **{field: value}))).reason,
                                 "REQUEST_BINDING_MISMATCH")

    def test_expired_future_and_overlong_grants_rejected(self):
        for fields, expected in (
            ({"expires_at": 1700000029}, "EXPIRED"),
            ({"issued_at": 1700000031}, "NOT_YET_VALID"),
            ({"expires_at": 1700000500}, "GRANT_LIFETIME_EXCEEDED"),
        ):
            with self.subTest(fields=fields):
                self.assertEqual(self.assess(self.sign(replace(self.grant, **fields))).reason, expected)

    def test_missing_and_forged_issuer_and_same_principal_denied(self):
        for context in (
            replace(self.context, trusted_issuer=""),
            replace(self.context, trusted_issuer="trainer-user"),
            replace(self.context, bank_owner_principal="eval-user"),
            replace(self.context, bank_owner_principal="trainer-user"),
            replace(self.context, audit_principal="trainer-user"),
        ):
            with self.subTest(context=context):
                self.assertFalse(self.assess(context=context).contract_match)
        signed_wrong_issuer = self.sign(replace(self.grant, issuer="forger-1"))
        self.assertEqual(self.assess(signed_wrong_issuer).reason, "ISSUER_MISMATCH")

    def test_no_external_signature_verifier_or_crashing_verifier_fails_closed(self):
        result = assess_grant(
            self.sign(self.grant), self.request, self.context, now=1700000030,
            verify_signature=None, nonce_journal=self.journal,
        )
        self.assertEqual(result.reason, "NO_TRUSTED_VERIFIER")
        def crashes(*_):
            raise RuntimeError("synthetic verifier failure")
        self.assertEqual(self.assess(verifier=crashes).reason, "SIGNATURE_VERIFICATION_FAILED")

    def test_broker_path_substitution_and_missing_audit_fail_closed(self):
        for fields in (
            {"raw_bank_path_exposed": True},
            {"bank_object_identity_stable": False},
            {"audit_acked": False},
            {"audit_principal": ""},
            {"audit_principal": "bank-owner"},
        ):
            with self.subTest(fields=fields):
                self.assertFalse(self.assess(context=replace(self.context, **fields)).contract_match)

    def test_invalid_digest_integer_time_bool_and_nonce_are_rejected(self):
        for fields in (
            {"corpus_sha256": "A" * 64},
            {"model_sha256": "0" * 63},
            {"issued_at": True},
            {"expires_at": float("nan")},
            {"nonce": "../escape"},
        ):
            with self.subTest(fields=fields):
                self.assertFalse(self.assess(replace(self.sign(self.grant), **fields)).contract_match)

    def test_caller_claim_of_grant_is_not_bank_authority(self):
        result = self.assess()
        self.assertTrue(result.requires_external_broker)
        self.assertFalse(result.bank_access_authorized)
        self.assertNotIn("private_bank_path", GrantRequest.__dataclass_fields__)
        self.assertNotIn("private_bank_path", GrantEnvelope.__dataclass_fields__)


if __name__ == "__main__":
    unittest.main()
