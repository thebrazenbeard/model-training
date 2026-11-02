"""Cross-check R185 owner V2 against private DEV *source-only* hashes.

The exact owner signing key, expected packet hash, epoch and clock are caller
inputs, as is the private DEV bridge report. Both inputs can be co-forged.
This module is only a consistency check; it never authenticates an evaluator,
unseals held-out material, admits rights or authorizes physical training.
"""

from __future__ import annotations

import re
from datetime import datetime

from trainingday.admission import AdmissionError
from trainingday.owner_exact_subject_v2 import (
    HASH_FIELDS, STATUS as OWNER_V2_STATUS, inspect_owner_exact_subject_v2,
)
from trainingday.private_dev_commitment_v1 import (
    SCHEMA as BRIDGE_SCHEMA, STATUS as BRIDGE_STATUS,
)


SCHEMA = "VERA_R185_OWNER_PRIVATE_SOURCE_MATCH_V1"
STATUS = "SOURCE_ONLY_MATCH_UNAUTHENTICATED_HOLD"
SHARED_HASH_FIELDS = HASH_FIELDS - {"frozen_training_packet_sha256"}
BRIDGE_FALSE_FIELDS = frozenset({
    "private_content_returned", "independent_custody_authenticated",
    "train_content_commitment_authenticated", "owner_identity_authenticated",
    "source_use_rights_verified", "host_ratification_authenticated",
    "authorizes_training", "authorizes_eval_unseal",
})
BRIDGE_FIELDS = SHARED_HASH_FIELDS | BRIDGE_FALSE_FIELDS | frozenset({
    "schema", "status", "corpus_id", "dev_count",
    "protected_exclusion_count",
})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise AdmissionError(reason)


def inspect_owner_private_consistency_v1(*, owner_receipt_bytes: bytes,
        owner_signature_hex: str, owner_public_key: bytes,
        private_bridge_report: dict, expected_frozen_packet_sha256: str,
        expected_owner_identity: str, minimum_owner_key_epoch: int,
        now_utc: datetime) -> dict:
    """Reject any cross-subject mismatch; return non-authorizing evidence.

    This function does NOT authenticate the caller, verify that the bridge
    was produced inside a protected evaluator, or independently freeze the
    packet. A real host must retain those authorities and never grant based
    on these booleans alone.
    """
    _require(type(private_bridge_report) is dict and
             set(private_bridge_report) == BRIDGE_FIELDS,
             "R185 source-only bridge shape invalid")
    bridge = private_bridge_report
    _require(bridge["schema"] == BRIDGE_SCHEMA and
             bridge["status"] == BRIDGE_STATUS and
             bridge["corpus_id"] == "R18_ARITHMETIC",
             "R185 source-only bridge identity/status invalid")
    _require(all(bridge[field] is False for field in BRIDGE_FALSE_FIELDS),
             "R185 source bridge falsely claims authority or private content")
    _require(type(bridge["dev_count"]) is int and
             1 <= bridge["dev_count"] <= 65536 and
             type(bridge["protected_exclusion_count"]) is int and
             bridge["protected_exclusion_count"] == 128,
             "R185 source-only bank sizes invalid")
    _require(all(type(bridge[key]) is str and
                 _SHA256.fullmatch(bridge[key]) is not None
                 for key in SHARED_HASH_FIELDS),
             "R185 source-only digest invalid")
    _require(type(expected_frozen_packet_sha256) is str and
             _SHA256.fullmatch(expected_frozen_packet_sha256) is not None,
             "R185 caller packet commitment invalid")

    expected = {field: bridge[field] for field in SHARED_HASH_FIELDS}
    expected["frozen_training_packet_sha256"] = expected_frozen_packet_sha256
    inspected = inspect_owner_exact_subject_v2(
        receipt_bytes=owner_receipt_bytes,
        signature_hex=owner_signature_hex,
        owner_public_key=owner_public_key,
        expected_hashes=expected,
        expected_owner_identity=expected_owner_identity,
        minimum_owner_key_epoch=minimum_owner_key_epoch,
        now_utc=now_utc,
    )
    _require(inspected["status"] == OWNER_V2_STATUS,
             "R185 owner withheld or nonasserting decision")
    # Do not propagate any caller's privilege booleans. A valid test-local
    # signature and matching hashes still establish no independent custody.
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "caller_supplied_hashes_consistent": True,
        "signature_valid_for_supplied_key": True,
        "owner_identity_authenticated": False,
        "owner_rights_admitted": False,
        "private_evaluator_custody_authenticated": False,
        "private_eval_unseal_authorized": False,
        "independent_source_authenticated": False,
        "gpu_lease_granted": False,
        "physical_training_authorized": False,
    }
