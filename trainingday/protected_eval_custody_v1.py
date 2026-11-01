"""Source-only R18 protected-exclusion preparation boundary.

R18 plan construction already checks the known train/R18/R16 indices. A
different evaluator must additionally establish the *complete*, private DEV
and EVAL exclusion roster and independently own the sealed bank. This module
only checks that a caller's answer-free declaration agrees with the frozen
public R18 plan. It does not open bank files, read identities/answers, verify
an independently installed issuer, or authorize any physical effect.

No digest in this declaration should be published for a low-entropy corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from trainingday.r18_benchmark import BenchmarkError, validate_plan


EXCLUSION_SCHEMA = "VERA_R18_PROTECTED_EVAL_EXCLUSION_PREPARE_V1"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_PRINCIPAL = re.compile(r"[A-Za-z][A-Za-z0-9_-]{7,127}\Z")
_HASH_FIELDS = frozenset({
    "plan_sha256", "r18_eval_custody_sha256", "r18_eval_index_sha256",
    "r18_eval_source_manifest_sha256", "r16_exclusion_custody_sha256",
    "r16_exclusion_id_sha256", "train_content_sha256",
    "protected_dev_exclusion_sha256", "protected_eval_exclusion_sha256",
    "complete_private_roster_sha256",
})
_PRINCIPALS = frozenset({
    "independent_steward_id", "independent_evaluator_id", "trainer_id",
})
_KEYS = frozenset({"schema", "claimed_complete_private_roster"}) | _HASH_FIELDS | _PRINCIPALS
_PLAN_BINDINGS = {
    "r18_eval_custody_sha256": "eval_custody_sha256",
    "r18_eval_index_sha256": "eval_index_sha256",
    "r18_eval_source_manifest_sha256": "eval_source_manifest_sha256",
    "r16_exclusion_custody_sha256": "r16_eval_custody_sha256",
    "r16_exclusion_id_sha256": "r16_holdout_id_sha256",
    "train_content_sha256": "train_content_sha256",
}


class ProtectedEvalCustodyError(ValueError):
    """A public source declaration is incomplete or mismatched; remain HOLD."""


@dataclass(frozen=True)
class PublicExclusionShapeDecision:
    """Deliberately contains no private identifiers, hashes, keys or grants."""

    status: str = "SOURCE_ONLY_EXCLUSION_SHAPE_HOLD"
    source_bindings_match: bool = True
    independent_custody_verified: bool = False
    protected_evaluation_authorized: bool = False
    training_authorized: bool = False
    gpu_lease_issued: bool = False
    requires_independent_host_evaluator: bool = True


def assess_public_exclusion_claim(plan: object, claim: object) -> PublicExclusionShapeDecision:
    """Validate *only* an answer-free claim's shape and R18 source bindings.

    A caller can choose every input, including IDs and digests. Even a valid
    result does not prove source ownership, roster completeness, issuer/key
    administration, private bank isolation, no leakage, or a signed grant.
    A separately installed host/evaluator authority must establish those facts.
    """
    if type(plan) is not dict or type(claim) is not dict or set(claim) != _KEYS:
        raise ProtectedEvalCustodyError("unrecognized protected-exclusion declaration")
    try:
        plan_digest = validate_plan(plan)
    except (BenchmarkError, TypeError, ValueError) as exc:
        raise ProtectedEvalCustodyError("invalid frozen R18 source plan") from exc
    if type(claim["schema"]) is not str or claim["schema"] != EXCLUSION_SCHEMA:
        raise ProtectedEvalCustodyError("protected-exclusion schema mismatch")
    if type(claim["claimed_complete_private_roster"]) is not bool or not claim[
            "claimed_complete_private_roster"]:
        raise ProtectedEvalCustodyError("complete protected roster was not declared")
    if any(type(claim[key]) is not str or _HEX64.fullmatch(claim[key]) is None
           for key in _HASH_FIELDS):
        raise ProtectedEvalCustodyError("unbound private exclusion commitment")
    principals = []
    for key in sorted(_PRINCIPALS):
        name = claim[key]
        if type(name) is not str or _PRINCIPAL.fullmatch(name) is None:
            raise ProtectedEvalCustodyError("invalid role-specific principal declaration")
        principals.append(name)
    if len(set(principals)) != len(principals):
        raise ProtectedEvalCustodyError("custody principals not distinct")
    if claim["plan_sha256"] != plan_digest or any(
            claim[declared] != plan[original]
            for declared, original in _PLAN_BINDINGS.items()):
        raise ProtectedEvalCustodyError("protected-exclusion binding differs from frozen plan")
    # There is deliberately NO path to return an ADMIT, signature-pass, unseal,
    # or training-grant status from this caller-controlled declaration.
    return PublicExclusionShapeDecision()
