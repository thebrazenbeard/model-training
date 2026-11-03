"""Cross-repo host/Q6 protocol vectors; NEVER a physical lease or witness.

The host declaration is read using model-training's real source dataclasses.
The Q6 CAS JSON, newline-canonicalization and Ed25519 signature domain are
copied from the immutable Labyrinth #107 schema for independent test-vector
comparison; this does NOT import, operate, or attest its remote witness.

Most importantly, the signed Q6 ACK has no host-request or GPU-lease subject.
The tests demonstrate that gap rather than confusing a valid signature with
physical training authorization.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import unittest

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trainingday.host_launch_request import (
    HostLeaseDeclaration, LaunchRequest, RightsDeclaration, RunPlan,
    SourceBinding, assess_launch_request, canonical_request_bytes,
)
from trainingday.physical_launch_authority import (
    PhysicalLaunchAuthorityUnavailable, require_physical_launch_authority,
)


VECTOR = Path(__file__).parent / "fixtures" / "trainingday_q6_host_source_vector_v1.json"
Q6_SCHEMA = "labyrinth.q6-independent-witness.source-r121.v1"
SIGNATURE_DOMAIN = b"labyrinth.q6-independent-witness-reply.source-r121.v1\0"


def _canonical_q6(value: object) -> bytes:
    # Labyrinth R120/R121: JSON ASCII, sorted compact keys, and one final LF.
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _host_request(value: dict) -> LaunchRequest:
    return LaunchRequest(
        schema=value["schema"],
        source=SourceBinding(**value["source"]),
        rights=RightsDeclaration(**value["rights"]),
        plan=RunPlan(**value["plan"]),
        lease=HostLeaseDeclaration(**value["lease"]),
    )


def _signed_q6_ack(request: dict, signer: Ed25519PrivateKey) -> bytes:
    fields = {
        "schema": Q6_SCHEMA,
        "operation": request["operation"],
        "nonce": request["nonce"],
        "request_sha256": sha256(_canonical_q6(request)).hexdigest(),
        "ok": True,
        "head": request["new"],
    }
    fields["signature_hex"] = signer.sign(
        SIGNATURE_DOMAIN + _canonical_q6(fields)).hex()
    return _canonical_q6(fields)


def _verify_fixture_ack(request: dict, reply_bytes: bytes, signer_public_key) -> None:
    """Conservative wire-vector verification; no authorization return value.

    This independent fixture check is NOT the actual Labyrinth verifier and
    does NOT authenticate an external witness, peer or service.
    """
    if reply_bytes is None or type(reply_bytes) is not bytes:
        raise ValueError("missing ACK: outcome UNKNOWN, no replay")
    try:
        fields = json.loads(reply_bytes.decode("ascii"))
    except (ValueError, UnicodeError, TypeError) as exc:
        raise ValueError("invalid ACK JSON") from exc
    if (type(fields) is not dict
            or set(fields) != {"schema", "operation", "nonce", "request_sha256",
                               "ok", "head", "signature_hex"}
            or _canonical_q6(fields) != reply_bytes
            or fields["schema"] != Q6_SCHEMA
            or fields["operation"] != request["operation"]
            or fields["nonce"] != request["nonce"]
            or fields["request_sha256"] != sha256(_canonical_q6(request)).hexdigest()
            or fields["ok"] is not True
            or fields["head"] != request["new"]):
        raise ValueError("ACK not bound to exact Q6 request")
    signature = bytes.fromhex(fields["signature_hex"])
    payload = {k: v for k, v in fields.items() if k != "signature_hex"}
    signer_public_key.verify(signature, SIGNATURE_DOMAIN + _canonical_q6(payload))


class Q6HostSourceVectorTests(unittest.TestCase):
    def setUp(self):
        vector = json.loads(VECTOR.read_text(encoding="utf-8"))
        self.assertEqual(vector["schema"], "TRAININGDAY_Q6_HOST_SOURCE_VECTOR_V1")
        self.assertEqual(vector["source"]["model_training_commit"],
                         "752a3e80df9be868926d18c574e4b9739e6f7581")
        self.assertEqual(vector["source"]["labyrinth_commit"],
                         "80bd918be22d3618d45fcb8689cc8e43e85464af")
        self.assertEqual(vector["authority"], {
            "source_only_unqualified": True,
            "authorizes_training": False,
            "host_gpu_lease_issued": False,
            "independent_q6_witness_installed": False,
            "model_execution_authenticated": False,
        })
        self.golden = vector["golden_request_digests"]
        self.host = _host_request(vector["host_launch"])
        self.cas = vector["q6_cas_request"]
        self.signer = Ed25519PrivateKey.generate()  # ephemeral, never an issuer
        self.public = self.signer.public_key()

    def test_host_launch_is_only_a_declaration(self):
        digest = sha256(canonical_request_bytes(self.host)).hexdigest()
        self.assertEqual(digest, self.golden["host_launch_sha256"])
        assessed = assess_launch_request(self.host)
        self.assertEqual(assessed.status, "SOURCE_DECLARATION_ONLY_HOST_UNVERIFIED")
        self.assertEqual(assessed.request_sha256, digest)
        for field in ("authorizes_training", "authorizes_resume",
                      "host_identity_verified", "dataset_rights_verified",
                      "gpu_lease_verified", "replay_prevention_verified",
                      "physical_training_observed"):
            self.assertIs(getattr(assessed, field), False, field)

    def test_signed_cas_ack_checks_q6_fields_but_not_host_authority(self):
        self.assertEqual(self.cas["operation"], "CAS")
        self.assertEqual(self.cas["schema"], Q6_SCHEMA)
        self.assertEqual(sha256(_canonical_q6(self.cas)).hexdigest(),
                         self.golden["q6_cas_sha256"])
        self.assertEqual(self.cas["new"]["commit_index"],
                         self.cas["expected"]["commit_index"] + 1)
        ack = _signed_q6_ack(self.cas, self.signer)
        _verify_fixture_ack(self.cas, ack, self.public)
        # Labyrinth witness does not sign the host LaunchRequest or GPU epoch:
        # a cryptographic Q6 ACK alone never upgrades the host source envelope.
        self.assertIs(assess_launch_request(self.host).authorizes_training, False)

    def test_host_gpu_epoch_or_expected_head_swap_is_not_q6_signature_bound(self):
        ack = _signed_q6_ack(self.cas, self.signer)
        original = sha256(canonical_request_bytes(self.host)).hexdigest()
        swapped_host_leases = (
            replace(self.host.lease, gpu_lease_epoch=8),
            replace(self.host.lease, expected_witness_head_sha256="4" * 64),
        )
        for lease in swapped_host_leases:
            with self.subTest(lease=lease):
                changed = replace(self.host, lease=lease)
                self.assertNotEqual(
                    original, sha256(canonical_request_bytes(changed)).hexdigest())
                # This STILL verifies: R121 signs the Q6 CAS request, not
                # the host request, claimed GPU epoch, or claimed witness head.
                _verify_fixture_ack(self.cas, ack, self.public)
                assessed = assess_launch_request(changed)
                self.assertFalse(assessed.gpu_lease_verified)
                self.assertFalse(assessed.authorizes_training)

    def test_cross_nonce_ack_reuse_and_cas_tampering_are_rejected(self):
        ack = _signed_q6_ack(self.cas, self.signer)
        changes = (
            dict(self.cas, nonce="4" * 32),
            dict(self.cas, expected={"commit_index": 12, "head_sha256": "4" * 64}),
            dict(self.cas, new={"commit_index": 13, "head_sha256": "4" * 64}),
        )
        for altered in changes:
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                _verify_fixture_ack(altered, ack, self.public)

    def test_identical_signed_ack_replay_is_not_a_durable_replay_guard(self):
        ack = _signed_q6_ack(self.cas, self.signer)
        # The stateless wire check accepts the same bytes twice. An actual
        # witness must enforce durable CAS/high-water and request uniqueness;
        # local ACK verification alone provides neither.
        _verify_fixture_ack(self.cas, ack, self.public)
        _verify_fixture_ack(self.cas, ack, self.public)
        assessed = assess_launch_request(self.host)
        self.assertFalse(assessed.replay_prevention_verified)
        self.assertFalse(assessed.authorizes_training)

    def test_bad_signer_mutated_reply_and_lost_ack_hold(self):
        ack = _signed_q6_ack(self.cas, self.signer)
        wrong_signer = Ed25519PrivateKey.generate().public_key()
        with self.assertRaises(InvalidSignature):
            _verify_fixture_ack(self.cas, ack, wrong_signer)
        altered = ack.replace(b'"ok":true', b'"ok":false')
        with self.assertRaises((ValueError, InvalidSignature)):
            _verify_fixture_ack(self.cas, altered, self.public)
        with self.assertRaises(ValueError):
            _verify_fixture_ack(self.cas, None, self.public)

    def test_unconditional_physical_hold_survives_signed_synthetic_ack(self):
        ack = _signed_q6_ack(self.cas, self.signer)
        _verify_fixture_ack(self.cas, ack, self.public)
        with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
            require_physical_launch_authority(operation="q6 interop vector")


if __name__ == "__main__":
    unittest.main()
