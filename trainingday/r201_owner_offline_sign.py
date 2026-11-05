"""Operator-invoked R201 owner claim signing; NEVER enroll or grant authority.

This offline utility requires a *separately selected* existing Ed25519 PEM key.
It neither generates keys nor uploads claims. A caller-supplied public key and
evaluator hashes remain untrusted until independently enrolled/read back by a
host-owned authority and evaluator custodian.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import re

from trainingday.calc_curriculum import (
    CALC_REPOSITORY, CALC_REVISION, CALC_SYMBOLIC_SHA256,
    SCHEMA as TRAIN_SCHEMA, TRAIN_COUNT,
)
from trainingday.r201_owner_claim_candidate import (
    CORPUS_ID, DECISIONS, DOMAIN, MODEL_ID, MODEL_REVISION, PURPOSE,
    RIGHTS_BASIS, SCHEMA, TRAIN_SHA256,
    ClaimEvidenceError, inspect_r201_owner_claim_candidate,
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_UTC = re.compile(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_NON_AUTHORITY_FLAGS = (
    "owner_key_enrollment_authenticated",
    "owner_source_rights_independently_reviewed",
    "protected_evaluator_custody_authenticated",
    "host_epoch_and_clock_authenticated",
    "corpus_rights_admitted",
    "gpu_lease_granted",
    "physical_training_authorized",
    "protected_evaluation_authorized",
)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ClaimEvidenceError(reason)


def _utc(text: str) -> datetime:
    _require(type(text) is str and _UTC.fullmatch(text) is not None,
             "R201 operator UTC timestamp malformed")
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ClaimEvidenceError("R201 operator UTC timestamp invalid") from exc


def sign_owner_claim_offline(
    *, private_key: object, owner_identity: str, dev_sha256: str,
    exclusion_sha256: str, epoch: int, not_before_utc: str,
    expires_utc: str, decision: str, now_utc: datetime,
) -> dict[str, bytes | str]:
    """Sign exact public R201 bytes, then source-verify; return NO AUTHORITY.

    The independent DEV/exclusion hashes must come from the actual evaluator;
    this function deliberately cannot verify their custody. It does not load
    private examples or host credentials and cannot change training state.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    except ImportError as exc:
        raise ClaimEvidenceError("Ed25519 cryptography dependency absent") from exc
    _require(isinstance(private_key, Ed25519PrivateKey),
             "a user-selected existing Ed25519 private key is required")
    _require(type(owner_identity) is str and 1 <= len(owner_identity) <= 120,
             "owner identity required")
    _require(all(type(value) is str and _HEX64.fullmatch(value) is not None
                 and value != "0" * 64 for value in (dev_sha256, exclusion_sha256)),
             "two actual nonzero 64-character evaluator commitments required")
    _require(dev_sha256 != exclusion_sha256 and dev_sha256 != TRAIN_SHA256
             and exclusion_sha256 != TRAIN_SHA256,
             "evaluator commitments must not be reused public TRAIN or each other")
    _require(type(epoch) is int and 1 <= epoch <= 2**63 - 1,
             "positive bounded owner key epoch required")
    _require(type(decision) is str and decision in DECISIONS,
             "explicit owner decision required")
    begin, end = _utc(not_before_utc), _utc(expires_utc)
    _require(type(now_utc) is datetime and now_utc.tzinfo is not None
             and now_utc.utcoffset() is not None and begin < end
             and begin <= now_utc.astimezone(timezone.utc) < end,
             "signed UTC interval must contain current operator time")

    public = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    claim = {
        "schema": SCHEMA, "corpus_id": CORPUS_ID,
        "train_schema": TRAIN_SCHEMA, "train_count": TRAIN_COUNT,
        "train_sha256": TRAIN_SHA256,
        "calc_repository": CALC_REPOSITORY, "calc_revision": CALC_REVISION,
        "calc_symbolic_sha256": CALC_SYMBOLIC_SHA256,
        "private_dev_commitment_sha256": dev_sha256,
        "protected_exclusion_receipt_sha256": exclusion_sha256,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "purpose": PURPOSE, "owner_identity": owner_identity,
        "owner_key_sha256": hashlib.sha256(public).hexdigest(),
        "owner_key_epoch": epoch, "rights_basis": RIGHTS_BASIS,
        "decision": decision, "not_before_utc": not_before_utc,
        "expires_utc": expires_utc,
    }
    try:
        wire = json.dumps(claim, sort_keys=True, ensure_ascii=False,
                          allow_nan=False, separators=(",", ":")).encode("utf-8")
    except UnicodeError as exc:
        raise ClaimEvidenceError("owner identity cannot encode canonical UTF-8") from exc
    signature_hex = private_key.sign(DOMAIN + wire).hex()
    evidence = inspect_r201_owner_claim_candidate(
        claim_bytes=wire, detached_signature_hex=signature_hex,
        supplied_owner_public_key=public, expected_owner_identity=owner_identity,
        expected_private_dev_commitment_sha256=dev_sha256,
        expected_protected_exclusion_receipt_sha256=exclusion_sha256,
        minimum_owner_key_epoch=epoch, now_utc=now_utc,
    )
    _require(evidence.get("signature_valid_for_supplied_key") is True
             and all(evidence.get(flag) is False for flag in _NON_AUTHORITY_FLAGS),
             "source-only denial contract changed; refuse export")
    return {
        "claim_bytes": wire,
        "detached_signature_hex": signature_hex,
        "public_key_hex": public.hex(),
        "claim_sha256": hashlib.sha256(wire).hexdigest(),
        "owner_key_sha256": hashlib.sha256(public).hexdigest(),
        "status": "SIGNED_WITH_SUPPLIED_KEY_NOT_ENROLLED_NO_TRAIN_AUTHORITY",
    }


def _load_selected_pem(path: Path):
    """Load only an explicit existing PEM; never discover, create or repurpose keys."""
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    raw = path.read_bytes()
    try:
        return load_pem_private_key(raw, password=None)
    except TypeError:
        password = getpass.getpass("Passphrase for chosen R201 PEM (not echoed): ")
        if not password:
            raise ClaimEvidenceError("an encrypted key requires its operator passphrase")
        try:
            return load_pem_private_key(raw, password=password.encode("utf-8"))
        finally:
            password = ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-key-pem", type=Path, required=True,
                        help="explicit dedicated owner-selected Ed25519 PEM; never auto-generated")
    parser.add_argument("--owner-identity", required=True)
    parser.add_argument("--private-dev-commitment-sha256", required=True)
    parser.add_argument("--protected-exclusion-receipt-sha256", required=True)
    parser.add_argument("--owner-key-epoch", type=int, required=True)
    parser.add_argument("--not-before-utc", required=True)
    parser.add_argument("--expires-utc", required=True)
    parser.add_argument("--decision", choices=sorted(DECISIONS), required=True)
    parser.add_argument("--ack-owner-rights", action="store_true",
                        help="required for ASSERT; this is your declaration, not our verification")
    parser.add_argument("--output-prefix", type=Path, required=True,
                        help="existing private output directory and basename; no Git upload")
    args = parser.parse_args(argv)
    if args.decision == "ASSERT_RIGHTS_TO_TRAIN" and not args.ack_owner_rights:
        parser.error("ASSERT requires the operator's explicit --ack-owner-rights")
    outputs = {
        ".claim.json": None, ".signature.hex": None,
        ".public_key.hex": None, ".receipt.json": None,
    }
    paths = {suffix: Path(str(args.output_prefix) + suffix) for suffix in outputs}
    if not args.output_prefix.parent.is_dir() or any(p.exists() for p in paths.values()):
        parser.error("output directory must exist and all four output paths must be new")
    try:
        package = sign_owner_claim_offline(
            private_key=_load_selected_pem(args.existing_key_pem),
            owner_identity=args.owner_identity,
            dev_sha256=args.private_dev_commitment_sha256,
            exclusion_sha256=args.protected_exclusion_receipt_sha256,
            epoch=args.owner_key_epoch, not_before_utc=args.not_before_utc,
            expires_utc=args.expires_utc, decision=args.decision,
            now_utc=datetime.now(timezone.utc),
        )
    except (OSError, ValueError, TypeError, ClaimEvidenceError):
        # The source filename, owner-selected key path, and library exceptions
        # can reveal private host context. CLI failures disclose neither.
        parser.error("owner-selected key or source-only claim invalid; no artifacts exported")
    outputs[".claim.json"] = package["claim_bytes"]
    outputs[".signature.hex"] = (package["detached_signature_hex"] + "\n").encode("ascii")
    outputs[".public_key.hex"] = (package["public_key_hex"] + "\n").encode("ascii")
    receipt = {key: package[key] for key in
               ("status", "claim_sha256", "owner_key_sha256")}
    receipt["all_training_authority"] = False
    outputs[".receipt.json"] = (json.dumps(receipt, sort_keys=True) + "\n").encode("utf-8")
    written: list[Path] = []
    try:
        for suffix, data in outputs.items():
            path = paths[suffix]
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            # The target now belongs to this invocation, including if a write
            # or close fails. Roll back only these newly created paths.
            written.append(path)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
    except OSError:
        # Only remove newly created artifacts; never touch pre-existing paths.
        for path in written:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                # A failed rollback still must not leak private filesystem paths.
                pass
        parser.error("source-only claim artifact write failed; review private output directory")
    print("R201_OWNER_CLAIM_SIGNED_WITH_SUPPLIED_KEY_SOURCE_ONLY_HOLD")
    print("No owner-key enrollment, evaluator custody, data admission or GPU authorization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
