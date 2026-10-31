"""CPU-only contract tests: no private bank, principal or signer is used."""

from dataclasses import replace
import unittest

from trainingday.r35_bank_grant_contract import (
    BankGrantRequest,
    ExpectedBankBinding,
    assess_synthetic_grant,
)


HEX64_A = "a" * 64
HEX64_B = "b" * 64
HEX64_C = "c" * 64
HEX40_A = "d" * 40


class BankGrantContractTests(unittest.TestCase):
    def setUp(self):
        self.expected = ExpectedBankBinding(
            bank_id="sealed-r18-eval-v1",
            principal_id="synthetic-evaluator-01",
            plan_sha256=HEX64_A,
            evaluator_head=HEX40_A,
            attestation_sha256=HEX64_B,
            scope="R18_PRIVATE_SCORE_READ_ONLY",
        )
        self.request = BankGrantRequest(
            schema="VERA_R35_SYNTHETIC_BANK_GRANT_V1",
            bank_id=self.expected.bank_id,
            principal_id=self.expected.principal_id,
            plan_sha256=self.expected.plan_sha256,
            evaluator_head=self.expected.evaluator_head,
            attestation_sha256=self.expected.attestation_sha256,
            scope=self.expected.scope,
            nonce="r35-synthetic-once-000001",
            issued_at_unix=1_000.0,
            expires_at_unix=1_040.0,
        )

    def evaluate(self, request=None, *, spent=frozenset(), now=1_010.0):
        return assess_synthetic_grant(
            self.request if request is None else request,
            self.expected,
            spent_nonces=spent,
            now_unix=now,
        )

    def test_matching_synthetic_request_still_never_opens_bank(self):
        report = self.evaluate()
        self.assertEqual(report.status, "INTERNALLY_CONSISTENT_ONLY_CUSTODIAN_REQUIRED")
        self.assertEqual(report.blockers, ())
        self.assertFalse(report.authorizes_bank_read)
        self.assertFalse(report.consumes_nonce)
        self.assertFalse(report.attestation_independently_verified)

    def test_each_security_binding_is_exact(self):
        variants = {
            "bank_id": "wrong-bank",
            "principal_id": "other-evaluator",
            "plan_sha256": HEX64_C,
            "evaluator_head": "f" * 40,
            "attestation_sha256": HEX64_C,
            "scope": "R18_PRIVATE_SCORE_AND_EXPORT",
        }
        for key, value in variants.items():
            with self.subTest(field=key):
                report = self.evaluate(replace(self.request, **{key: value}))
                self.assertIn("BINDING_MISMATCH", report.blockers)
                self.assertFalse(report.authorizes_bank_read)

    def test_replay_nonce_is_rejected_even_with_matching_hashes(self):
        report = self.evaluate(spent=frozenset({self.request.nonce}))
        self.assertIn("NONCE_ALREADY_SPENT_IN_DECLARED_SET", report.blockers)
        self.assertFalse(report.consumes_nonce)

    def test_malformed_spent_set_never_invokes_attacker_equality(self):
        class MaliciousNonce:
            def __hash__(self):
                return hash("r35-synthetic-once-000001")

            def __eq__(self, other):
                raise AssertionError("untrusted nonce equality invoked")

        report = self.evaluate(spent=frozenset({MaliciousNonce()}))
        self.assertIn("MALFORMED_CONTRACT", report.blockers)
        self.assertFalse(report.authorizes_bank_read)

    def test_ttl_expiry_future_issue_and_nonfinite_are_rejected(self):
        cases = (
            (self.request, 1_040.0),
            (replace(self.request, issued_at_unix=1_020.0), 1_010.0),
            (replace(self.request, expires_at_unix=1_500.0), 1_010.0),
            (replace(self.request, issued_at_unix=float("nan")), 1_010.0),
            (replace(self.request, expires_at_unix=True), 1_010.0),
        )
        for request, now in cases:
            with self.subTest(request=request, now=now):
                self.assertIn("INVALID_TIME_WINDOW", self.evaluate(request, now=now).blockers)

    def test_malformed_digest_nonce_schema_and_inputs_rejected(self):
        cases = (
            replace(self.request, plan_sha256="A" * 64),
            replace(self.request, evaluator_head="short"),
            replace(self.request, nonce=""),
            replace(self.request, schema="UNKNOWN"),
        )
        for request in cases:
            with self.subTest(request=request):
                self.assertNotEqual(self.evaluate(request).blockers, ())
        self.assertIn("MALFORMED_CONTRACT", self.evaluate({"fake": "dict"}).blockers)

    def test_untrusted_caller_evidence_is_not_authentication(self):
        # Matching caller input is not a signature, bank ACL, or atomic grant.
        report = self.evaluate()
        self.assertEqual(report.authorization_source, "NONE_SYNTHETIC_ONLY")
        self.assertFalse(report.host_custodian_installed)

    def test_hostile_schema_and_scope_equality_are_never_invoked(self):
        class EqualityTrap:
            def __eq__(self, other):
                raise AssertionError("hostile equality was invoked")

            def __ne__(self, other):
                raise AssertionError("hostile inequality was invoked")

        cases = (
            (replace(self.request, schema=EqualityTrap()), self.expected),
            (replace(self.request, scope=EqualityTrap()), self.expected),
            (self.request, replace(self.expected, scope=EqualityTrap())),
            (replace(self.request, principal_id=EqualityTrap()), self.expected),
        )
        for request, expected in cases:
            with self.subTest(request=request, expected=expected):
                report = assess_synthetic_grant(
                    request, expected, spent_nonces=frozenset(), now_unix=1_010.0)
                self.assertIn("MALFORMED_CONTRACT", report.blockers)
                self.assertFalse(report.authorizes_bank_read)

    def test_huge_integer_time_fails_closed_without_overflow(self):
        huge = 10 ** 400
        self.assertIn("MALFORMED_CONTRACT", self.evaluate(now=huge).blockers)
        for request in (
            replace(self.request, issued_at_unix=huge),
            replace(self.request, expires_at_unix=huge),
        ):
            with self.subTest(request=request):
                report = self.evaluate(request)
                self.assertIn("INVALID_TIME_WINDOW", report.blockers)
                self.assertFalse(report.authorizes_bank_read)


if __name__ == "__main__":
    unittest.main()
