"""R35 **untrusted shape-only** training attester-claim contract.

This module intentionally has no trust anchor, signer, host-observation path,
bank grant, or mechanism to authorize private evaluation. It checks whether
caller-provided declarations and *purported* witness fields are internally
well-formed. A self-consistent forged claim can pass its shape checks.

Never use ``structurally_consistent`` as authentication or permission to score.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


CLAIM_SCHEMA = "VERA_TRAININGDAY_R35_UNTRUSTED_ATTESTER_CLAIM_SHAPE_V1"
ARMS = frozenset(f"{trainer}:{control}" for trainer in ("mezo", "qlora")
                 for control in ("frozen", "zero", "correct", "corrupt"))
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_BINDINGS = frozenset({"plan_sha256", "corpus_sha256", "base_sha256"})
_TOP = frozenset({"schema", "purported_attester_id", "arms"}) | _BINDINGS
_ARM = frozenset({"arm_id", "declared", "purported_witness"})
_DECLARED = frozenset({
    "run_id", "adapter_after_sha256", "trainable_before_sha256",
    "trainable_after_sha256",
})
_WITNESS = frozenset({
    "arm_id", "run_id", "corpus_sha256", "base_sha256",
    "adapter_after_sha256", "trainable_before_sha256",
    "trainable_after_sha256", "host_id", "lease_id",
    "observed_start_ms", "observed_end_ms",
})


@dataclass(frozen=True)
class ClaimShapeAssessment:
    structurally_consistent: bool
    independent_attestation_verified: bool
    private_bank_unseal_authorized: bool
    blockers: tuple[str, ...]


def _object(value: object, fields: frozenset[str]) -> bool:
    """Reject custom mapping hooks and arbitrary additional authority fields."""
    return (type(value) is dict and all(type(key) is str for key in value)
            and value.keys() == fields)


def _hash(value: object) -> bool:
    return type(value) is str and _SHA.fullmatch(value) is not None


def _id(value: object) -> bool:
    return type(value) is str and 1 <= len(value) <= 128 and value.isascii() and all(
        c.isalnum() or c in "._:-" for c in value)


def assess_claim_shape(claim: object, *, expected: object) -> ClaimShapeAssessment:
    """Validate exact declared-vs-purported-witness structure; never attest.

    ``expected`` is also a caller-supplied contract here. Even a perfect
    result only establishes relative consistency against those values, not
    their independent origin. Only a separately trusted external custodian
    could decide whether to unseal, and that custodian is not implemented.
    """
    blocks: set[str] = set()
    if not _object(expected, _BINDINGS) or any(
            not _hash(expected[key]) for key in _BINDINGS):
        blocks.add("INVALID_EXPECTED_BINDINGS")

    if not _object(claim, _TOP):
        blocks.add("INVALID_CLAIM_RECORD")
        return _result(blocks)
    if type(claim["schema"]) is not str or claim["schema"] != CLAIM_SCHEMA:
        blocks.add("INVALID_SCHEMA")
    if not _id(claim["purported_attester_id"]):
        blocks.add("INVALID_PURPORTED_ATTESTER")
    if any(not _hash(claim[key]) for key in _BINDINGS):
        blocks.add("INVALID_CLAIM_BINDING")
    if ("INVALID_EXPECTED_BINDINGS" not in blocks
            and "INVALID_CLAIM_BINDING" not in blocks) and any(
            claim[key] != expected[key] for key in _BINDINGS):
        blocks.add("EXPECTED_BINDING_MISMATCH")
    # Never compare malformed attacker-controlled top-level values against
    # per-arm purported witnesses. A type-invalid object could define its own
    # comparison callbacks and replace a fail-closed result with an exception.
    if blocks:
        return _result(blocks)

    arms = claim["arms"]
    if type(arms) is not list:
        blocks.add("INVALID_ARM_SET")
        return _result(blocks)
    if len(arms) != len(ARMS):
        blocks.add("INVALID_ARM_SET")
        # Also bound work before touching attacker-supplied per-arm objects.
        return _result(blocks)
    seen_arm_ids: set[str] = set()
    seen_run_ids: set[str] = set()
    common_initial_digest: str | None = None
    claimed_lease_intervals: dict[str, list[tuple[str, int, int]]] = {}
    for arm in arms:
        if not _object(arm, _ARM):
            blocks.add("INVALID_ARM_RECORD")
            continue
        arm_id = arm["arm_id"]
        declared = arm["declared"]
        witness = arm["purported_witness"]
        if not _id(arm_id) or arm_id not in ARMS:
            blocks.add("INVALID_ARM_SET")
            continue
        if arm_id in seen_arm_ids:
            blocks.add("INVALID_ARM_SET")
        seen_arm_ids.add(arm_id)
        if not _object(declared, _DECLARED) or not _object(witness, _WITNESS):
            blocks.add("INVALID_ARM_RECORD")
            continue
        run_id = declared["run_id"]
        if (not _id(run_id) or any(not _hash(declared[key]) for key in (
                "adapter_after_sha256", "trainable_before_sha256",
                "trainable_after_sha256"))):
            blocks.add("INVALID_ARM_RECORD")
            continue
        current_initial_digest = declared["trainable_before_sha256"]
        if common_initial_digest is None:
            common_initial_digest = current_initial_digest
        elif current_initial_digest != common_initial_digest:
            blocks.add("INITIAL_TRAINABLE_DIGEST_MISMATCH")
        if (any(not _id(witness[key]) for key in (
                "arm_id", "run_id", "host_id", "lease_id"))
                or any(not _hash(witness[key]) for key in (
                    "corpus_sha256", "base_sha256", "adapter_after_sha256",
                    "trainable_before_sha256", "trainable_after_sha256"))):
            blocks.add("INVALID_ARM_RECORD")
            continue
        if run_id in seen_run_ids:
            blocks.add("DUPLICATE_RUN_ID")
        seen_run_ids.add(run_id)
        if (witness["arm_id"] != arm_id or witness["run_id"] != run_id
                or witness["corpus_sha256"] != claim["corpus_sha256"]
                or witness["base_sha256"] != claim["base_sha256"]
                or any(witness[key] != declared[key] for key in (
                    "adapter_after_sha256", "trainable_before_sha256",
                    "trainable_after_sha256"))):
            blocks.add("DECLARATION_WITNESS_MISMATCH")
        start, end = witness["observed_start_ms"], witness["observed_end_ms"]
        if (type(start) is not int or type(end) is not int
                or start < 0 or end <= start):
            blocks.add("INVALID_OBSERVATION")
        else:
            # Only a consistency check on *claims*, not evidence that this
            # host lease was genuine or held exclusively in the real system.
            # Half-open intervals permit immediate sequential handoff.
            lease_id, host_id = witness["lease_id"], witness["host_id"]
            for prior_host, prior_start, prior_end in claimed_lease_intervals.get(lease_id, []):
                if host_id != prior_host:
                    blocks.add("LEASE_HOST_MISMATCH")
                elif start < prior_end and prior_start < end:
                    blocks.add("EXCLUSIVE_LEASE_OVERLAP")
            claimed_lease_intervals.setdefault(lease_id, []).append(
                (host_id, start, end))
    if seen_arm_ids != ARMS:
        blocks.add("INVALID_ARM_SET")
    return _result(blocks)


def _result(blockers: set[str]) -> ClaimShapeAssessment:
    # This module has no independent signer/verifier or protected bank owner.
    # Crucially, even a structurally perfect claim MUST remain unauthorized.
    consistent = not blockers
    return ClaimShapeAssessment(
        structurally_consistent=consistent,
        independent_attestation_verified=False,
        private_bank_unseal_authorized=False,
        blockers=tuple(sorted(blockers | {"UNTRUSTED_PURPORTED_WITNESS"})),
    )
