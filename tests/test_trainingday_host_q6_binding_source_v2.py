"""Unqualified, additive host-to-Q6 signed-binding contract vectors.

NO installed issuer, external witness, durable antirollback ledger, GPU lease,
or authorization is represented here. This test-only V2 wrapper deliberately
leaves both the HostLaunch V1 and Labyrinth R121 exact wire formats unchanged.
Two generated private keys stand in for independent issuers only within tests.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import unittest

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trainingday.host_launch_request import assess_launch_request, canonical_request_bytes
from trainingday.physical_launch_authority import (
    PhysicalLaunchAuthorityUnavailable, require_physical_launch_authority,
)
from tests.test_trainingday_q6_host_source_vector import (
    VECTOR, _canonical_q6, _host_request, _signed_q6_ack, _verify_fixture_ack,
)


SCHEMA = "TRAININGDAY_HOST_Q6_BINDING_SOURCE_V2"
DOMAIN = b"trainingday.host-q6-binding.source-v2\0"
NO_AUTHORITY = "SOURCE_ONLY_NO_AUTHORITY"


@dataclass(frozen=True)
class SourceOnlyBindingAssessment:
    """Even fully matching toy signatures cannot yield a training grant."""

    status: str = NO_AUTHORITY
    training_authorized: bool = False
    host_identity_verified: bool = False
    gpu_lease_verified: bool = False
    durable_replay_prevention_verified: bool = False
    trusted_clock_verified: bool = False

    def __bool__(self) -> bool:
        # Defensive: never permit `if verify_binding(...): launch()`.
        return False


def _canonical_binding(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _payload(host, q6_request: dict, q6_ack: bytes, token_id: str, operation: str) -> dict:
    lease = host.lease
    expected, new = q6_request["expected"], q6_request["new"]
    return {
        "schema": SCHEMA,
        "operation": operation,
        "token_id": token_id,
        "host_launch_sha256": sha256(canonical_request_bytes(host)).hexdigest(),
        "host_instance": lease.host_instance,
        "caller_principal": lease.caller_principal,
        "run_id": host.plan.run_id,
        "gpu_lease_id": lease.gpu_lease_id,
        "gpu_lease_epoch": lease.gpu_lease_epoch,
        "host_request_nonce": lease.request_nonce,
        "issued_at_unix": lease.issued_at_unix,
        "expires_at_unix": lease.expires_at_unix,
        "q6_request_sha256": sha256(_canonical_q6(q6_request)).hexdigest(),
        "q6_ack_sha256": sha256(q6_ack).hexdigest(),
        "q6_expected_index": expected["commit_index"],
        "q6_expected_head_sha256": expected["head_sha256"],
        "q6_new_index": new["commit_index"],
        "q6_new_head_sha256": new["head_sha256"],
    }


def _signed_binding(payload: dict, lease_issuer: Ed25519PrivateKey) -> bytes:
    envelope = {"payload": payload,
                "signature_hex": lease_issuer.sign(DOMAIN + _canonical_binding(payload)).hex()}
    return _canonical_binding(envelope)


def _source_verify_binding(*, host, q6_request: dict, q6_ack: bytes,
                           binding: bytes, witness_signer, lease_issuer,
                           now: int,
                           used_tokens: set[str] | None = None) -> SourceOnlyBindingAssessment:
    """Toy source checks ONLY; caller-supplied `now` is NOT trusted host time."""
    # Both signatures are toy-generated and even different public keys do NOT
    # prove physically separated operators/custody or an installed GPU issuer.
    if (witness_signer.public_bytes_raw() == lease_issuer.public_bytes_raw()):
        raise ValueError("host issuer and Q6 witness must use distinct keys")
    _verify_fixture_ack(q6_request, q6_ack, witness_signer)
    if (q6_request["operation"] != "CAS"
            or q6_request["expected"]["commit_index"] !=
               host.lease.expected_witness_epoch
            or q6_request["expected"]["head_sha256"] !=
               host.lease.expected_witness_head_sha256
            or q6_request["new"]["commit_index"] !=
               q6_request["expected"]["commit_index"] + 1):
        raise ValueError("host/Q6 expected-head or CAS advance mismatch")
    if type(binding) is not bytes:
        raise ValueError("binding bytes required")
    try:
        envelope = json.loads(binding.decode("ascii"))
    except (ValueError, UnicodeError, TypeError) as exc:
        raise ValueError("noncanonical binding") from exc
    if (type(envelope) is not dict or set(envelope) != {"payload", "signature_hex"}
            or type(envelope["payload"]) is not dict
            or type(envelope["signature_hex"]) is not str
            or _canonical_binding(envelope) != binding):
        raise ValueError("ambiguous binding wire format")
    payload = envelope["payload"]
    if (payload.get("schema") != SCHEMA
            or payload.get("operation") not in ("START", "RESUME")
            or type(payload.get("token_id")) is not str
            or len(payload["token_id"]) != 64
            or any(c not in "0123456789abcdef" for c in payload["token_id"])
            or type(now) is not int
            or not host.lease.issued_at_unix <= now < host.lease.expires_at_unix
            or payload != _payload(host, q6_request, q6_ack,
                                   payload["token_id"], payload["operation"])):
        raise ValueError("binding subject, lease, Q6, clock or token mismatch")
    try:
        lease_issuer.verify(bytes.fromhex(envelope["signature_hex"]),
                            DOMAIN + _canonical_binding(payload))
    except (ValueError, InvalidSignature) as exc:
        raise ValueError("host lease issuer signature invalid") from exc
    if used_tokens is not None:
        # This process-local set is intentionally NOT a durable replay witness.
        if payload["token_id"] in used_tokens:
            raise ValueError("toy in-process token replay")
        used_tokens.add(payload["token_id"])
    return SourceOnlyBindingAssessment()


class HostQ6BindingSourceV2Tests(unittest.TestCase):
    def setUp(self):
        vector = json.loads(Path(VECTOR).read_text(encoding="utf-8"))
        self.host = _host_request(vector["host_launch"])
        self.cas = vector["q6_cas_request"]
        self.witness = Ed25519PrivateKey.generate()
        self.issuer = Ed25519PrivateKey.generate()
        self.assertNotEqual(self.witness.public_key().public_bytes_raw(),
                            self.issuer.public_key().public_bytes_raw())
        self.ack = _signed_q6_ack(self.cas, self.witness)
        self.token = "9" * 64
        self.payload = _payload(self.host, self.cas, self.ack, self.token, "START")
        self.binding = _signed_binding(self.payload, self.issuer)
        self.now = self.host.lease.issued_at_unix + 1

    def _verify(self, *, host=None, cas=None, ack=None, binding=None,
                issuer=None, witness=None, now=None, used=None):
        return _source_verify_binding(
            host=self.host if host is None else host,
            q6_request=self.cas if cas is None else cas,
            q6_ack=self.ack if ack is None else ack,
            binding=self.binding if binding is None else binding,
            witness_signer=self.witness.public_key() if witness is None else witness,
            lease_issuer=self.issuer.public_key() if issuer is None else issuer,
            now=self.now if now is None else now, used_tokens=used,
        )

    def test_additive_valid_toy_wire_still_never_authorizes(self):
        result = self._verify()
        self.assertEqual(result.status, NO_AUTHORITY)
        self.assertFalse(result)
        self.assertIs(result.training_authorized, False)
        self.assertIs(result.host_identity_verified, False)
        self.assertIs(result.gpu_lease_verified, False)
        self.assertIs(result.durable_replay_prevention_verified, False)
        self.assertIs(result.trusted_clock_verified, False)
        self.assertFalse(assess_launch_request(self.host).authorizes_training)
        with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
            require_physical_launch_authority(operation="toy host-Q6 v2")

    def test_host_field_and_epoch_mutations_are_signed_subject_bound(self):
        variants = (
            replace(self.host, source=replace(self.host.source,
                                             corpus_manifest_sha256="0" * 64),
                    rights=replace(self.host.rights, covered_manifest_sha256="0" * 64)),
            replace(self.host, plan=replace(self.host.plan, seed=42)),
            replace(self.host, lease=replace(self.host.lease, gpu_lease_epoch=8)),
            replace(self.host, lease=replace(self.host.lease, host_instance="altered-host")),
            replace(self.host, lease=replace(self.host.lease, request_nonce="8" * 64)),
            replace(self.host, lease=replace(self.host.lease, expires_at_unix=
                                             self.host.lease.expires_at_unix + 1)),
            replace(self.host, lease=replace(self.host.lease,
                                             expected_witness_head_sha256="4" * 64)),
        )
        for changed in variants:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self._verify(host=changed)

    def test_q6_mutations_and_ack_splice_are_rejected(self):
        variants = (
            dict(self.cas, nonce="4" * 32),
            dict(self.cas, expected={"commit_index": 11, "head_sha256": "1" * 64}),
            dict(self.cas, new={"commit_index": 14, "head_sha256": "3" * 64}),
        )
        for cas in variants:
            with self.subTest(cas=cas), self.assertRaises(ValueError):
                self._verify(cas=cas)
        ack2 = _signed_q6_ack(self.cas, self.witness)
        # Ed25519 deterministic signing means identical ACK bytes are reusable.
        self.assertEqual(ack2, self.ack)
        self.assertFalse(self._verify(ack=ack2))

    def test_same_key_signed_toy_q6_and_host_receipts_are_rejected(self):
        shared = Ed25519PrivateKey.generate()
        ack = _signed_q6_ack(self.cas, shared)
        payload = _payload(self.host, self.cas, ack, self.token, "START")
        binding = _signed_binding(payload, shared)
        with self.assertRaisesRegex(ValueError, "distinct keys"):
            self._verify(ack=ack, binding=binding,
                         witness=shared.public_key(), issuer=shared.public_key())

    def test_issuer_key_and_payload_tampering_are_rejected(self):
        with self.assertRaises(ValueError):
            self._verify(issuer=Ed25519PrivateKey.generate().public_key())
        with self.assertRaises((ValueError, InvalidSignature)):
            self._verify(witness=Ed25519PrivateKey.generate().public_key())
        for changed in (
            dict(self.payload, gpu_lease_epoch=11),
            dict(self.payload, operation="RESUME"),
            dict(self.payload, q6_ack_sha256="0" * 64),
            dict(self.payload, host_launch_sha256="0" * 64),
            dict(self.payload, token_id="8" * 64),
        ):
            forged = _canonical_binding({"payload": changed,
                                         "signature_hex": json.loads(self.binding)["signature_hex"]})
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self._verify(binding=forged)

    def test_time_and_replay_remain_source_only_not_durable(self):
        with self.assertRaises(ValueError):
            self._verify(now=self.host.lease.expires_at_unix)
        used: set[str] = set()
        self.assertFalse(self._verify(used=used))
        with self.assertRaises(ValueError):
            self._verify(used=used)
        # A fresh ephemeral set/session accepts exactly the same signature:
        # this test does NOT run another process or prove durable antirollback.
        self.assertFalse(self._verify(used=set()))
        self.assertFalse(self._verify())

    def test_caller_supplied_clock_can_not_attest_real_expiration(self):
        # The actual wall clock is outside this source fixture: caller can
        # simply claim an in-window timestamp regardless of real host time.
        claimed_now = self.host.lease.issued_at_unix + 1
        assessment = self._verify(now=claimed_now)
        self.assertIs(assessment.trusted_clock_verified, False)
        self.assertIs(assessment.training_authorized, False)


if __name__ == "__main__":
    unittest.main()
