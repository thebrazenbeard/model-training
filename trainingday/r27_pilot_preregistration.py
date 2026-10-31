"""Public, answer-free R27 multi-skill pilot preregistration and toy dry-run.

No training data, answers, hidden labels, model inference, bank interface, or
deployment code are touched. The caller declares all hashes, task rights,
outcomes, and arm roles. Even a positive synthetic statistical result NEVER
establishes blinded evaluation, authentic rights, or neural improvement.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass


_HASH = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{3,63}$")
_FAMILIES = frozenset(("ARITHMETIC", "PROGRAM_REASONING", "SOURCE_PRIORITY"))
_ROLES = frozenset(("FROZEN", "ZERO_UPDATE", "CORRECT_LABEL", "CORRUPTED_LABEL"))


class PilotError(ValueError):
    """Controlled malformed declaration or inadmissible test fixture."""


@dataclass(frozen=True)
class PilotCase:
    case_id: str
    family: str
    prompt_sha256: str
    rights_evidence_sha256: str


@dataclass(frozen=True)
class PilotProtocol:
    protocol_id: str
    frozen_base_sha256: str
    tokenizer_sha256: str
    admitted_train_corpus_sha256: str
    known_training_prompt_sha256: tuple[str, ...]
    cases: tuple[PilotCase, ...]
    opaque_arm_ids: tuple[str, ...]
    min_paired_gain: float
    one_sided_alpha: float


@dataclass(frozen=True)
class PreregReceipt:
    manifest_sha256: str
    canonical_json: str
    case_count: int
    declared_training_overlap_absent: bool = True
    training_inventory_complete_attested: bool = False
    rights_authenticated: bool = False
    blind_allocation_authenticated: bool = False
    protected_eval_access_authorized: bool = False
    release_authorized: bool = False


@dataclass(frozen=True)
class SyntheticPilotScore:
    status: str
    protocol_sha256: str
    correct_vs_frozen_gain: float
    correct_vs_frozen_p_value: float
    by_family_accuracy: tuple[tuple[str, float], ...]
    control_checks_passed: bool
    input_outcomes_authenticated: bool = False
    rights_authenticated: bool = False
    blinding_authenticated: bool = False
    physical_neural_improvement_demonstrated: bool = False
    protected_eval_access_authorized: bool = False
    model_promotion_authorized: bool = False


def _is_hash(value: object) -> bool:
    return type(value) is str and _HASH.fullmatch(value) is not None


def _is_id(value: object) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


def _canonical_protocol(spec: PilotProtocol) -> str:
    """Canonical public commitment; sorted case order must not affect identity."""
    doc = {
        "schema": "trainingday-r27-pilot-prereg-v1",
        "protocol_id": spec.protocol_id,
        "base_sha256": spec.frozen_base_sha256,
        "tokenizer_sha256": spec.tokenizer_sha256,
        "train_corpus_sha256": spec.admitted_train_corpus_sha256,
        "known_train_prompt_sha256": sorted(spec.known_training_prompt_sha256),
        "cases": [asdict(row) for row in sorted(spec.cases, key=lambda c: c.case_id)],
        "opaque_arm_ids": sorted(spec.opaque_arm_ids),
        "policy": {"min_paired_gain": spec.min_paired_gain,
                   "one_sided_alpha": spec.one_sided_alpha,
                   "comparison_alpha": spec.one_sided_alpha / 3.0,
                   "family_accuracy_floor": 0.50,
                   "max_family_regression_vs_frozen": 0.02,
                   "compare_to": ["FROZEN", "ZERO_UPDATE", "CORRUPTED_LABEL"],
                   "no_improvement_default": True},
    }
    return json.dumps(doc, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def preregister_pilot(spec: PilotProtocol) -> PreregReceipt:
    """Commit answer-free public metadata only, never certify claimed rights."""
    if type(spec) is not PilotProtocol:
        raise PilotError("MALFORMED_PROTOCOL")
    if not _is_id(spec.protocol_id):
        raise PilotError("MALFORMED_PROTOCOL_ID")
    if not all(_is_hash(x) for x in (
        spec.frozen_base_sha256, spec.tokenizer_sha256,
        spec.admitted_train_corpus_sha256,
    )):
        raise PilotError("MALFORMED_MODEL_OR_CORPUS_DIGEST")
    if (type(spec.min_paired_gain) is not float or
            not math.isfinite(spec.min_paired_gain) or
            not 0.0 < spec.min_paired_gain <= 1.0):
        raise PilotError("MALFORMED_PAIRED_GAIN")
    if (type(spec.one_sided_alpha) is not float or
            not math.isfinite(spec.one_sided_alpha) or
            not 0.0 < spec.one_sided_alpha < 0.5):
        raise PilotError("MALFORMED_ALPHA")
    if (type(spec.known_training_prompt_sha256) is not tuple or
            not 1 <= len(spec.known_training_prompt_sha256) <= 4096 or
            not all(_is_hash(x) for x in spec.known_training_prompt_sha256) or
            len(set(spec.known_training_prompt_sha256)) != len(spec.known_training_prompt_sha256)):
        raise PilotError("MALFORMED_TRAINING_INVENTORY")
    if (type(spec.cases) is not tuple or not 3 <= len(spec.cases) <= 512 or
            not all(type(c) is PilotCase for c in spec.cases)):
        raise PilotError("MALFORMED_CASE_INVENTORY")
    if (type(spec.opaque_arm_ids) is not tuple or len(spec.opaque_arm_ids) != 4 or
            not all(_is_id(x) for x in spec.opaque_arm_ids) or
            len(set(spec.opaque_arm_ids)) != 4):
        raise PilotError("MALFORMED_OPAQUE_ARMS")
    case_ids: set[str] = set()
    prompts: set[str] = set()
    families: set[str] = set()
    train = set(spec.known_training_prompt_sha256)
    for row in spec.cases:
        if not (_is_id(row.case_id) and type(row.family) is str and
                row.family in _FAMILIES and _is_hash(row.prompt_sha256) and
                _is_hash(row.rights_evidence_sha256)):
            raise PilotError("MALFORMED_CASE_OR_RIGHTS_REFERENCE")
        if row.case_id in case_ids or row.prompt_sha256 in prompts:
            raise PilotError("DUPLICATE_CASE_OR_PROMPT")
        if row.prompt_sha256 in train:
            raise PilotError("DECLARED_TRAIN_PILOT_OVERLAP")
        case_ids.add(row.case_id)
        prompts.add(row.prompt_sha256)
        families.add(row.family)
    if families != _FAMILIES:
        raise PilotError("REQUIRED_SKILL_FAMILY_MISSING")
    canonical = _canonical_protocol(spec)
    return PreregReceipt(
        manifest_sha256=hashlib.sha256(canonical.encode("ascii")).hexdigest(),
        canonical_json=canonical,
        case_count=len(spec.cases),
    )


def _one_sided_sign_p_value(wins: int, losses: int) -> float:
    """Exact binomial discordant-pair sign test, H0 win probability <= 0.5."""
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    # The population is hard-bounded to 512 by preregister_pilot.
    tail = sum(math.comb(discordant, k) for k in range(wins, discordant + 1))
    return min(1.0, tail / (2 ** discordant))


def score_synthetic_pilot(
    spec: PilotProtocol,
    receipt: PreregReceipt,
    role_by_arm: dict[str, str],
    public_boolean_outcomes: dict[str, dict[str, bool]],
) -> SyntheticPilotScore:
    """Toy stats from *declared* correctness bits; no task answers or private OOD."""
    current = preregister_pilot(spec)
    if (type(receipt) is not PreregReceipt or
            receipt.manifest_sha256 != current.manifest_sha256 or
            receipt.canonical_json != current.canonical_json or
            receipt != current):
        raise PilotError("PREREG_COMMITMENT_MISMATCH")
    if (type(role_by_arm) is not dict or
            len(role_by_arm) != len(spec.opaque_arm_ids) or
            not all(type(k) is str and type(v) is str for k, v in role_by_arm.items()) or
            set(role_by_arm) != set(spec.opaque_arm_ids) or
            set(role_by_arm.values()) != _ROLES):
        raise PilotError("ARM_MAPPING_MALFORMED_OR_UNBLINDED")
    if (type(public_boolean_outcomes) is not dict or
            len(public_boolean_outcomes) != len(spec.opaque_arm_ids) or
            not all(type(k) is str for k in public_boolean_outcomes) or
            set(public_boolean_outcomes) != set(spec.opaque_arm_ids)):
        raise PilotError("MISSING_OR_EXTRA_ARM_OUTCOMES")
    cases = {row.case_id: row for row in spec.cases}
    for arm in spec.opaque_arm_ids:
        scores = public_boolean_outcomes[arm]
        if (type(scores) is not dict or len(scores) != len(cases) or
                not all(type(k) is str and type(v) is bool for k, v in scores.items()) or
                set(scores) != set(cases)):
            raise PilotError("UNPAIRED_OR_MALFORMED_CASE_OUTCOMES")
    by_role = {role: public_boolean_outcomes[arm] for arm, role in role_by_arm.items()}
    correct = by_role["CORRECT_LABEL"]
    n = len(cases)
    controls_ok = True
    gain_frozen = 0.0
    p_frozen = 1.0
    for control in ("FROZEN", "ZERO_UPDATE", "CORRUPTED_LABEL"):
        baseline = by_role[control]
        wins = sum(correct[k] and not baseline[k] for k in cases)
        losses = sum(baseline[k] and not correct[k] for k in cases)
        gain = (wins - losses) / n
        p = _one_sided_sign_p_value(wins, losses)
        if control == "FROZEN":
            gain_frozen, p_frozen = gain, p
        if gain < spec.min_paired_gain or p > spec.one_sided_alpha / 3.0:
            controls_ok = False
    family_acc = tuple(sorted(
        (family, sum(correct[k] for k, row in cases.items()
                     if row.family == family) /
         sum(row.family == family for row in cases.values()))
        for family in _FAMILIES
    ))
    if any(accuracy < 0.50 for _, accuracy in family_acc):
        controls_ok = False
    for family, candidate_accuracy in family_acc:
        frozen_accuracy = (
            sum(by_role["FROZEN"][k] for k, row in cases.items()
                if row.family == family) /
            sum(row.family == family for row in cases.values())
        )
        if candidate_accuracy + 0.02 < frozen_accuracy:
            controls_ok = False
    return SyntheticPilotScore(
        status=("SYNTHETIC_THRESHOLD_MET_NO_CLAIM" if controls_ok
                else "NO_IMPROVEMENT"),
        protocol_sha256=current.manifest_sha256,
        correct_vs_frozen_gain=gain_frozen,
        correct_vs_frozen_p_value=p_frozen,
        by_family_accuracy=family_acc,
        control_checks_passed=controls_ok,
    )
