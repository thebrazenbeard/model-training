"""Source-only consistency gate for claimed neural training continuation.

This module validates *claims* supplied in a portable evidence envelope. It
does not inspect model tensors, load optimizer state, trust an evaluator, or
grant authority to run, resume, graduate, merge, or deploy a model. A forged
but internally consistent envelope still returns PHYSICAL_VERIFICATION_HOLD.

The final protected evaluation must be performed by an independently trusted
examiner, against an unseen sealed bank, after a fresh process reload. Those
facts cannot be established from data the training process itself supplies.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


SCHEMA = "VERA_TRAININGDAY_R141_NEURAL_CONTINUATION_CLAIMS_V1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_HEAD = re.compile(r"[0-9a-f]{40}\Z")
_MAX_BYTES = 256 * 1024
_MAX_SEGMENTS = 128
_MAX_NODES = 10000


class EvidenceError(ValueError):
    """Malformed or contradictory neural continuation claims."""


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise EvidenceError("DUPLICATE_JSON_KEY")
        output[key] = value
    return output


def _plain_finite(value: Any) -> bool:
    stack = [(value, 0)]
    count = 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if count > _MAX_NODES or depth > 32:
            return False
        if type(item) is dict:
            if any(type(k) is not str for k in item):
                return False
            stack.extend((v, depth + 1) for v in item.values())
        elif type(item) is list:
            stack.extend((v, depth + 1) for v in item)
        elif type(item) is float:
            if not math.isfinite(item):
                return False
        elif type(item) not in (str, int, bool, type(None)):
            return False
    return True


def read_claims(path: Path) -> dict[str, Any]:
    """Read a bounded source envelope, rejecting symlinks and JSON tricks."""
    if type(path) is not type(Path()) or path.is_symlink() or not path.is_file():
        raise EvidenceError("EVIDENCE_FILE_NOT_REGULAR")
    if path.stat().st_size <= 0 or path.stat().st_size > _MAX_BYTES:
        raise EvidenceError("EVIDENCE_SIZE_INVALID")
    try:
        raw = path.read_bytes()
        if len(raw) > _MAX_BYTES:
            raise EvidenceError("EVIDENCE_SIZE_INVALID")
        claims = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda _: (_ for _ in ()).throw(
                EvidenceError("NONFINITE_JSON_NUMBER")
            ),
        )
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise EvidenceError("EVIDENCE_PARSE_FAILED") from exc
    if type(claims) is not dict or not _plain_finite(claims):
        raise EvidenceError("EVIDENCE_NONPLAIN_OR_NONFINITE")
    return claims


def _need(condition: bool, code: str) -> None:
    if not condition:
        raise EvidenceError(code)


def _object(value: Any, code: str) -> dict[str, Any]:
    _need(type(value) is dict, code)
    return value


def _hash(value: Any) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _int(value: Any, lower: int = 0, upper: int = 2**31 - 1) -> bool:
    return type(value) is int and lower <= value <= upper


def _number(value: Any, lower: float, upper: float) -> bool:
    # Range-check before float coercion. math.isfinite(10**1000) raises
    # OverflowError, allowing hostile JSON integers to bypass fail-closed
    # decisions through an uncaught exception.
    return (type(value) in (int, float) and lower <= value <= upper
            and math.isfinite(value))


def _verify_arms(claims: dict[str, Any]) -> tuple[str, dict, dict, dict]:
    arms = _object(claims.get("arms"), "ARMS_MISSING")
    _need(set(arms) == {"zero", "correct", "corrupt"}, "CONTROL_ARMS_NOT_MATCHED")
    zero, correct, corrupt = [
        _object(arms[name], f"{name.upper()}_INVALID")
        for name in ("zero", "correct", "corrupt")
    ]
    shared = ("base_revision", "model_config_sha256", "base_frozen_weights_sha256",
              "train_manifest_sha256",
              "selected_ids_sha256", "seed", "hyperparameters_sha256",
              "initial_optimizer_sha256", "initial_rng_sha256")
    for name, arm in arms.items():
        _need(arm.get("mode") == name, "CONTROL_MODE_MISMATCH")
        _need(all(_hash(arm.get(k)) for k in shared if k not in {"base_revision", "seed"}),
              "CONTROL_BINDING_HASH_INVALID")
        _need(_GIT_HEAD.fullmatch(arm.get("base_revision", "")) is not None,
              "CONTROL_BASE_REVISION_INVALID")
        _need(_int(arm.get("seed")), "CONTROL_SEED_INVALID")
        _need(_hash(arm.get("initial_fp32_digest")) and _hash(arm.get("final_fp32_digest")),
              "CONTROL_FP32_DIGEST_INVALID")
        _need(_int(arm.get("optimizer_steps"), 0, 32), "CONTROL_STEPS_INVALID")
        _need(_hash(arm.get("adapter_artifact_sha256")), "CONTROL_ADAPTER_HASH_INVALID")
        for key in shared:
            _need(arm[key] == zero[key], "CONTROL_SOURCE_OR_SEED_MISMATCH")
    _need(zero["optimizer_steps"] == 0 and
          zero["initial_fp32_digest"] == zero["final_fp32_digest"],
          "ZERO_ARM_UPDATED")
    _need(correct["optimizer_steps"] > 0 and
          corrupt["optimizer_steps"] == correct["optimizer_steps"],
          "TRAINED_ARM_STEP_MISMATCH")
    initial = zero["initial_fp32_digest"]
    _need(correct["initial_fp32_digest"] == initial and
          corrupt["initial_fp32_digest"] == initial,
          "TRAINED_ARM_DIFFERENT_INITIAL_TENSORS")
    _need(correct["final_fp32_digest"] != initial and
          correct["final_fp32_digest"] != corrupt["final_fp32_digest"],
          "NO_DISTINCT_CORRECT_WEIGHT_CHANGE_CLAIM")
    _need(corrupt["final_fp32_digest"] != initial, "CORRUPT_CONTROL_NOT_UPDATED")
    _need(len({a["adapter_artifact_sha256"] for a in (zero, correct, corrupt)}) == 3,
          "CONTROL_ADAPTERS_NOT_SEPARATE")
    _need(all(_hash(a.get("training_input_sha256")) for a in (zero, correct, corrupt))
          and zero["training_input_sha256"] == correct["training_input_sha256"]
          and corrupt["training_input_sha256"] != correct["training_input_sha256"]
          and _hash(corrupt.get("corrupt_label_audit_sha256")),
          "CORRUPTED_LABEL_INPUT_NOT_DISTINCTLY_ATTESTED")
    _need(_number(correct.get("claimed_fp32_l2_delta"), 0.0, 1e12) and
          correct["claimed_fp32_l2_delta"] > 1e-6,
          "NONTRIVIAL_TENSOR_DELTA_UNESTABLISHED")
    return initial, zero, correct, corrupt


def _verify_chain(claims: dict[str, Any], initial: str, correct: dict) -> None:
    chain = claims.get("continuation_segments")
    _need(type(chain) is list and 1 <= len(chain) <= _MAX_SEGMENTS,
          "CONTINUATION_CHAIN_INVALID")
    prior_step, prior_model = 0, initial
    prior_optimizer = prior_rng = prior_checkpoint = None
    for index, item in enumerate(chain):
        seg = _object(item, "CONTINUATION_SEGMENT_INVALID")
        start, end = seg.get("start_step"), seg.get("end_step")
        _need(_int(start) and _int(end) and start == prior_step and
              start < end <= correct["optimizer_steps"], "CONTINUATION_STEP_GAP")
        hashes = ("model_in_fp32_digest", "model_out_fp32_digest",
                  "optimizer_in_sha256", "optimizer_out_sha256",
                  "rng_in_sha256", "rng_out_sha256", "checkpoint_out_sha256",
                  "run_contract_sha256")
        _need(all(_hash(seg.get(key)) for key in hashes), "CONTINUATION_HASH_INVALID")
        _need(seg["model_in_fp32_digest"] == prior_model and
              seg["model_out_fp32_digest"] != prior_model,
              "CONTINUATION_MODEL_STATE_BROKEN")
        if index:
            _need(seg["optimizer_in_sha256"] == prior_optimizer and
                  seg["rng_in_sha256"] == prior_rng and
                  seg.get("checkpoint_in_sha256") == prior_checkpoint,
                  "CONTINUATION_OPTIMIZER_RNG_OR_CHECKPOINT_BROKEN")
        else:
            _need(seg.get("checkpoint_in_sha256") is None,
                  "FIRST_SEGMENT_UNEXPECTED_RESUME")
            _need(seg["optimizer_in_sha256"] == correct["initial_optimizer_sha256"]
                  and seg["rng_in_sha256"] == correct["initial_rng_sha256"],
                  "INITIAL_OPTIMIZER_OR_RNG_NOT_BOUND")
        _need(seg["run_contract_sha256"] == claims["run_contract_sha256"],
              "CONTINUATION_CONTRACT_CHANGED")
        prior_step, prior_model = end, seg["model_out_fp32_digest"]
        prior_optimizer, prior_rng = seg["optimizer_out_sha256"], seg["rng_out_sha256"]
        prior_checkpoint = seg["checkpoint_out_sha256"]
    _need(prior_step == correct["optimizer_steps"] and
          prior_model == correct["final_fp32_digest"] and
          prior_checkpoint == claims["final_checkpoint_sha256"],
          "CONTINUATION_FINAL_STATE_MISMATCH")


def _verify_eval(claims: dict[str, Any], zero: dict, correct: dict, corrupt: dict) -> None:
    evaluation = _object(claims.get("evaluation"), "EVALUATION_MISSING")
    _need(_hash(evaluation.get("frozen_reference_sha256")) and
          evaluation["frozen_reference_sha256"] == correct["base_frozen_weights_sha256"],
          "FROZEN_REFERENCE_NOT_PINNED_TO_BASE")
    _need(_hash(evaluation.get("sealed_bank_sha256")) and
          _hash(evaluation.get("sealed_commitment_sha256")) and
          _hash(evaluation.get("examiner_result_sha256")),
          "SEALED_EVALUATION_BINDING_INVALID")
    _need(evaluation["sealed_bank_sha256"] != correct["train_manifest_sha256"] and
          evaluation["sealed_bank_sha256"] != correct["selected_ids_sha256"],
          "EVALUATION_NOT_SEPARATE_FROM_TRAIN")
    _need(type(evaluation.get("examiner_identity")) is str and
          len(evaluation["examiner_identity"]) >= 8 and
          evaluation.get("independent_examiner_claimed") is True,
          "INDEPENDENT_EXAMINER_NOT_IDENTIFIED")
    heldout_n = evaluation.get("heldout_case_count")
    frozen_n = evaluation.get("frozen_case_count")
    _need(_int(heldout_n, 1, 1_000_000), "HELDOUT_SAMPLE_COUNT_INVALID")
    _need(_int(frozen_n, 1, 1_000_000), "FROZEN_SAMPLE_COUNT_INVALID")
    for key in ("zero_accuracy", "correct_accuracy", "corrupt_accuracy",
                "frozen_baseline_accuracy", "frozen_after_accuracy"):
        _need(_number(evaluation.get(key), 0, 1), "ACCURACY_CLAIM_INVALID")
        # A score must be the EXACT ratio of integer correct cases to cases
        # examined. A one-case evaluation cannot have accuracy 0.75. Exact
        # machine-produced ratios are required; rounded display values are
        # presentation only and are not admissible evidence.
        sample_n = frozen_n if key.startswith("frozen_") else heldout_n
        numerator = evaluation.get(key.removesuffix("_accuracy") + "_correct_count")
        _need(_int(numerator, 0, sample_n) and
              evaluation[key] == numerator / sample_n,
              "ACCURACY_DENOMINATOR_OR_COUNT_MISMATCH")
    _need(_number(evaluation.get("correct_frozen_kl"), 0, 1e6),
          "KL_CLAIM_INVALID")
    _need(_number(evaluation.get("max_frozen_kl"), 0, 1e6) and
          _number(evaluation.get("max_frozen_accuracy_drop"), 0, 1),
          "RETENTION_LIMIT_INVALID")
    _need(evaluation["correct_accuracy"] > max(
          evaluation["zero_accuracy"], evaluation["corrupt_accuracy"]),
          "LEARNING_NOT_DISTINCT_FROM_CONTROLS")
    _need(evaluation["correct_frozen_kl"] <= evaluation["max_frozen_kl"] and
          evaluation["frozen_baseline_accuracy"] - evaluation["frozen_after_accuracy"] <=
          evaluation["max_frozen_accuracy_drop"], "FROZEN_COMPETENCE_RETENTION_FAILED")


def assess_source_claims(claims: Any) -> dict[str, Any]:
    """Reject contradictory evidence; *never* grant real training authority."""
    reasons: list[str] = []
    try:
        _need(type(claims) is dict and _plain_finite(claims), "CLAIMS_INVALID")
        _need(claims.get("schema") == SCHEMA, "SCHEMA_INVALID")
        _need(type(claims.get("source_git_head")) is str and
              _GIT_HEAD.fullmatch(claims["source_git_head"]) is not None,
              "SOURCE_GIT_HEAD_INVALID")
        _need(_hash(claims.get("run_contract_sha256")) and
              _hash(claims.get("final_checkpoint_sha256")),
              "RUN_OR_CHECKPOINT_BINDING_INVALID")
        initial, zero, correct, corrupt = _verify_arms(claims)
        _verify_chain(claims, initial, correct)
        _verify_eval(claims, zero, correct, corrupt)
    except (EvidenceError, KeyError, TypeError, RecursionError) as exc:
        reasons.append(str(exc) if type(exc) is EvidenceError else "EVIDENCE_CONTRACT_BROKEN")
    consistent = not reasons
    return {
        "schema": "VERA_TRAININGDAY_R141_SOURCE_ASSESSMENT_V1",
        "source_claims_consistent": consistent,
        "status": ("SOURCE_CLAIMS_CONSISTENT_PHYSICAL_VERIFICATION_HOLD"
                   if consistent else "SOURCE_CLAIMS_REJECTED_PHYSICAL_VERIFICATION_HOLD"),
        "reason_codes": reasons if reasons else ["UNTRUSTED_PRODUCER_AND_EVALUATOR_EVIDENCE"],
        "authorizes_training": False,
        "authorizes_resume": False,
        "authorizes_graduation": False,
        "proves_neural_learning": False,
        "proves_physical_reload": False,
    }
