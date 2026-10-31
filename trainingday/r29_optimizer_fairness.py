"""R29 public CPU-only scalar-objective and optimizer accounting witness.

No device probing or model execution occurs here.  Inputs are caller-declared
public toy values.  A matching result cannot authenticate a Qwen3.5-4B run,
validate historical R18, establish real AdamW8bit defaults, or start training.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

from trainingday.r24_optimizer_sampler_audit import (
    CaseTokens, ArmTimings, simulate_source_exposure, assess_source_and_exposure,
)


SCHEMA = "VERA_TRAININGDAY_R29_PUBLIC_TOY_OPTIMIZER_FAIRNESS_V1"
MAX_CASES = 1024
MAX_TOKENS = 4096
MAX_VOCAB = 4096
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_SHA = re.compile(r"[a-f0-9]{64}\Z")


@dataclass(frozen=True)
class ToyTokenCase:
    case_id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    target_logits: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class ClockObservation:
    monotonic_elapsed_seconds: float
    wall_elapsed_seconds: float
    standby_seconds: float
    planned_budget_seconds: float


@dataclass(frozen=True)
class DeclaredControls:
    zero_before_sha256: str
    zero_after_sha256: str
    correct_label_sha256: str
    corrupted_label_sha256: str
    correct_label_updates: int
    corrupted_label_updates: int


def _sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _finite_logit(value: object) -> bool:
    if type(value) not in (float, int):
        return False
    try:
        return math.isfinite(value) and abs(value) <= 1e6
    except OverflowError:
        return False


def _validate_cases(cases: tuple[ToyTokenCase, ...]) -> None:
    if not isinstance(cases, tuple) or not 1 <= len(cases) <= MAX_CASES:
        raise ValueError("public toy cases must be a bounded nonempty tuple")
    seen = set()
    vocabulary = None
    for case in cases:
        if not isinstance(case, ToyTokenCase):
            raise ValueError("malformed public toy case")
        if (not isinstance(case.case_id, str) or not _ID.fullmatch(case.case_id)
                or case.case_id in seen):
            raise ValueError("invalid or duplicate public case id")
        seen.add(case.case_id)
        if (not isinstance(case.input_ids, tuple) or not isinstance(case.labels, tuple)
                or not 2 <= len(case.input_ids) <= MAX_TOKENS
                or len(case.input_ids) != len(case.labels)
                or any(type(x) is not int or x < 0 for x in case.input_ids)
                or any(type(x) is not int or (x < 0 and x != -100)
                       for x in case.labels)
                or case.labels[0] != -100):
            raise ValueError("invalid shifted causal LM input IDs / labels")
        supervised = [x for x in case.labels[1:] if x != -100]
        if (not supervised or not isinstance(case.target_logits, tuple)
                or len(case.target_logits) != len(supervised)):
            raise ValueError("target logits must match supervised label count")
        # R18 completion-only training ignores the prompt; the target is a suffix.
        first_target = next(i for i, token in enumerate(case.labels) if token != -100)
        if any(x == -100 for x in case.labels[first_target:]):
            raise ValueError("supervised completion has a masked interior position")
        for logits, token_id in zip(case.target_logits, supervised, strict=True):
            if (not isinstance(logits, tuple) or not 2 <= len(logits) <= MAX_VOCAB
                    or type(token_id) is not int or not 0 <= token_id < len(logits)
                    or any(not _finite_logit(v) for v in logits)):
                raise ValueError("invalid target token ID or finite vocabulary logits")
            if vocabulary is None:
                vocabulary = len(logits)
            if len(logits) != vocabulary:
                raise ValueError("frozen public model vocabulary changed")
        if any(token_id >= vocabulary for token_id in case.input_ids):
            raise ValueError("public input token ID exceeds frozen toy vocabulary")


def _token_commitment(cases: tuple[ToyTokenCase, ...]) -> str:
    return _sha256_json([{"id": row.case_id, "input_ids": row.input_ids,
                          "labels": row.labels} for row in cases])


def _logit_commitment(cases: tuple[ToyTokenCase, ...]) -> str:
    return _sha256_json([{"id": row.case_id, "target_logits": row.target_logits}
                         for row in cases])


def _case_nll_sum(case: ToyTokenCase) -> tuple[float, int]:
    targets = (v for v in case.labels if v != -100)
    losses = []
    for target_id, logits in zip(targets, case.target_logits, strict=True):
        max_logit = max(logits)
        # The stable log-sum-exp is the same cross entropy used by a target
        # token's causal LM softmax, with prediction at the preceding index.
        loss = max_logit + math.log(math.fsum(math.exp(x - max_logit)
                                              for x in logits)) - logits[target_id]
        if not math.isfinite(loss):
            raise ValueError("nonfinite completion token CE")
        losses.append(loss)
    return math.fsum(losses), len(losses)


def scalar_objective_witness(
        mezo_cases: tuple[ToyTokenCase, ...],
        qlora_cases: tuple[ToyTokenCase, ...],
        qlora_microbatch_sizes: tuple[int, ...]) -> dict:
    """Cross-check MeZO full-batch CE vs token-weighted QLoRA microbatches.

    The same frozen public logits witness *objective* math, not equal gradient
    estimators, parameter updates, training data provenance, or runtime dose.
    Different frozen logits are explicitly a noncomparable snapshot.
    """
    _validate_cases(mezo_cases)
    _validate_cases(qlora_cases)
    if (not isinstance(qlora_microbatch_sizes, tuple)
            or not qlora_microbatch_sizes
            or any(type(n) is not int or not 1 <= n <= MAX_CASES
                   for n in qlora_microbatch_sizes)
            or sum(qlora_microbatch_sizes) != len(qlora_cases)):
        raise ValueError("microbatch partition must cover each QLoRA case exactly once")

    mezo_nlls = [_case_nll_sum(row) for row in mezo_cases]
    qlora_nlls = [_case_nll_sum(row) for row in qlora_cases]
    total_mezo_targets = sum(count for _, count in mezo_nlls)
    total_qlora_targets = sum(count for _, count in qlora_nlls)
    mezo_ce = math.fsum(loss for loss, _ in mezo_nlls) / total_mezo_targets
    weighted = 0.0
    naive_micro_means = []
    offset = 0
    for size in qlora_microbatch_sizes:
        micro = qlora_nlls[offset: offset + size]
        numerator = math.fsum(loss for loss, _ in micro)
        denominator = sum(count for _, count in micro)
        micro_mean = numerator / denominator
        weighted += micro_mean * (denominator / total_qlora_targets)
        naive_micro_means.append(micro_mean)
        offset += size
    reasons = []
    mezo_targets_sha = _token_commitment(mezo_cases)
    qlora_targets_sha = _token_commitment(qlora_cases)
    mezo_logits_sha = _logit_commitment(mezo_cases)
    qlora_logits_sha = _logit_commitment(qlora_cases)
    if mezo_targets_sha != qlora_targets_sha:
        reasons.append("TARGET_TOKEN_ID_COMMITMENT_MISMATCH")
    if mezo_logits_sha != qlora_logits_sha:
        reasons.append("FROZEN_LOGITS_SNAPSHOT_MISMATCH")
    if (total_mezo_targets != total_qlora_targets
            or not math.isclose(mezo_ce, weighted, rel_tol=0.0, abs_tol=1e-12)):
        reasons.append("SCALAR_CE_OBJECTIVE_MISMATCH")
    return {
        "schema": SCHEMA, "evidence_class": "PUBLIC_CPU_TOY_MATH_ONLY",
        "status": "HOLD" if reasons else "PASS_PUBLIC_SCALAR_OBJECTIVE",
        "hold_reasons": reasons,
        "first_case_id": mezo_cases[0].case_id,
        "last_case_id": mezo_cases[-1].case_id,
        "mezo_first_case_id": mezo_cases[0].case_id,
        "mezo_last_case_id": mezo_cases[-1].case_id,
        "qlora_first_case_id": qlora_cases[0].case_id,
        "qlora_last_case_id": qlora_cases[-1].case_id,
        "target_id_commitment_sha256": mezo_targets_sha,
        "qlora_target_id_commitment_sha256": qlora_targets_sha,
        "mezo_frozen_logits_sha256": mezo_logits_sha,
        "qlora_frozen_logits_sha256": qlora_logits_sha,
        "supervised_target_tokens": total_mezo_targets,
        "qlora_supervised_target_tokens": total_qlora_targets,
        "mezo_scalar_cross_entropy": mezo_ce,
        "qlora_target_weighted_cross_entropy": weighted,
        "qlora_naive_unweighted_microbatch_mean": math.fsum(naive_micro_means) / len(naive_micro_means),
        "qlora_microbatch_sizes": list(qlora_microbatch_sizes),
        "same_gradient_estimator": False,
        "proves_real_training_objective_parity": False,
        "authorizes_training": False,
    }


def _clock_reasons(clock: ClockObservation, timings: ArmTimings) -> list[str]:
    if not isinstance(clock, ClockObservation):
        raise ValueError("clock declaration required")
    values = (clock.monotonic_elapsed_seconds, clock.wall_elapsed_seconds,
              clock.standby_seconds, clock.planned_budget_seconds)
    if (any(type(v) not in (float, int) or not math.isfinite(v) or v < 0
            for v in values) or clock.planned_budget_seconds == 0):
        raise ValueError("invalid, negative or nonfinite clock observation")
    reasons = []
    if (clock.monotonic_elapsed_seconds > clock.planned_budget_seconds
            or clock.wall_elapsed_seconds > clock.planned_budget_seconds):
        reasons.append("TIME_BUDGET_OVERRUN_INVALID")
    if clock.standby_seconds > 0.5:
        reasons.append("STANDBY_INTEGRITY_INVALID")
    if abs(clock.monotonic_elapsed_seconds - clock.wall_elapsed_seconds) > max(
            0.5, 0.01 * max(clock.monotonic_elapsed_seconds, clock.wall_elapsed_seconds)):
        reasons.append("WALL_MONOTONIC_CLOCK_DIVERGENCE")
    if (not math.isclose(clock.monotonic_elapsed_seconds,
                         math.fsum(timings.step_seconds), rel_tol=0.0, abs_tol=1e-6)
            or clock.planned_budget_seconds != timings.training_budget_seconds):
        reasons.append("OPTIMIZER_WINDOW_CLOCK_MISMATCH")
    return reasons


def _control_reasons(controls: DeclaredControls) -> list[str]:
    if not isinstance(controls, DeclaredControls):
        raise ValueError("control declarations required")
    hashes = (controls.zero_before_sha256, controls.zero_after_sha256,
              controls.correct_label_sha256, controls.corrupted_label_sha256)
    if any(not isinstance(digest, str) or not _SHA.fullmatch(digest) for digest in hashes):
        raise ValueError("control hash commitments must be canonical SHA-256")
    if (type(controls.correct_label_updates) is not int
            or type(controls.corrupted_label_updates) is not int
            or not 0 <= controls.correct_label_updates <= 4096
            or not 0 <= controls.corrupted_label_updates <= 4096):
        raise ValueError("control optimizer updates invalid")
    reasons = []
    if controls.zero_before_sha256 != controls.zero_after_sha256:
        reasons.append("ZERO_UPDATE_TENSOR_CHANGED")
    if controls.correct_label_sha256 == controls.corrupted_label_sha256:
        reasons.append("CONTROL_LABEL_COMMITMENTS_NOT_DISTINCT")
    if controls.correct_label_updates == 0:
        reasons.append("CORRECT_LABEL_NO_OPTIMIZER_UPDATE")
    if controls.corrupted_label_updates == 0:
        reasons.append("CORRUPTED_LABEL_NO_OPTIMIZER_UPDATE")
    return reasons


def audit_fair_optimizer_pair(
        *, mezo_cases: tuple[ToyTokenCase, ...], qlora_cases: tuple[ToyTokenCase, ...],
        mezo_timings: ArmTimings, qlora_timings: ArmTimings,
        mezo_clock: ClockObservation, qlora_clock: ClockObservation,
        qlora_weight_decay_source: dict,
        declared_controls: DeclaredControls,
        max_seq_length: int, tokenizer_sha256: str, corpus_sha256: str,
        qlora_microbatch_sizes: tuple[int, ...] = (2, 2, 2, 2)) -> dict:
    """Combine public toy objective, source exposure, and declared receipt falsifiers.

    Never upgrades caller-declared clock/control hashes to real host evidence.
    The optimizer schedule is exactly R24's existing source policy simulation.
    """
    if (not isinstance(tokenizer_sha256, str) or not _SHA.fullmatch(tokenizer_sha256)
            or not isinstance(corpus_sha256, str) or not _SHA.fullmatch(corpus_sha256)):
        raise ValueError("source/tokenizer hash commitments invalid")
    if not isinstance(qlora_weight_decay_source, dict):
        raise ValueError("QLoRA source weight decay witness required")
    objective = scalar_objective_witness(mezo_cases, qlora_cases,
                                         qlora_microbatch_sizes)
    mezo_pool = tuple(CaseTokens(row.case_id, len(row.input_ids),
                                 sum(x != -100 for x in row.labels))
                      for row in mezo_cases)
    qlora_pool = tuple(CaseTokens(row.case_id, len(row.input_ids),
                                  sum(x != -100 for x in row.labels))
                       for row in qlora_cases)
    mezo = simulate_source_exposure("mezo", mezo_pool, mezo_timings,
                                     max_seq_length)
    qlora = simulate_source_exposure("qlora", qlora_pool, qlora_timings,
                                      max_seq_length)
    exposure = assess_source_and_exposure(mezo, qlora, qlora_weight_decay_source)
    reasons = list(dict.fromkeys(
        objective["hold_reasons"] + exposure["hold_reasons"]
        + _clock_reasons(mezo_clock, mezo_timings)
        + _clock_reasons(qlora_clock, qlora_timings)
        + _control_reasons(declared_controls)
        + ["RUNTIME_EFFECTIVE_ADAMW8BIT_DECAY_UNATTESTED",
           "CONTROL_RECEIPTS_NOT_INDEPENDENTLY_ATTESTED",
           "PHYSICAL_EQUAL_TIME_NOT_ATTESTED"]))
    # R18's QLoRA source requires effective eight-case optimizer steps.  The
    # QLoRA microbatch partition is only a frozen scalar loss witness, not a
    # runtime claim about the active batch-size fit.
    return {
        "schema": SCHEMA, "status": "HOLD", "hold_reasons": reasons,
        "evidence_class": "SOURCE_SIMULATION_PLUS_PUBLIC_CPU_TOY_ONLY",
        "tokenizer_sha256_declared": tokenizer_sha256,
        "corpus_sha256_declared": corpus_sha256,
        "public_toy_objective_parity": objective["status"] == "PASS_PUBLIC_SCALAR_OBJECTIVE",
        "synthetic_schedule_matched": exposure["synthetic_schedule_matched"],
        "objective_witness": objective,
        "mezo_accounting": mezo,
        "qlora_accounting": qlora,
        "physical_equal_time_attested": False,
        "runtime_weight_decay_attested": False,
        "independent_control_custody_attested": False,
        "effective_neural_training_dose_equal": False,
        "proves_neural_weight_improvement": False,
        "authorizes_training": False,
        "authorizes_promotion": False,
    }
