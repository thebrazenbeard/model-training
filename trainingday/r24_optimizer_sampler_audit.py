"""R24 source-bound, synthetic-only MeZO/QLoRA exposure fairness audit.

This audit deliberately cannot authenticate a Windows/GPU run, source revision,
optimizer package runtime, or the actual Qwen tokenizer/corpus.  It compares
the R18 runners' observed *source policies*: cyclic MeZO selections versus a
single QLoRA sweep, and MeZO paired-forward versus QLoRA backward accounting.
Every assessment is non-authorizing regardless of how its synthetic schedule
compares.  Do not use a PASS-like synthetic fixture to start or graduate Vera.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re


EFFECTIVE_BATCH = 8  # QLoRA R18 fixed; MeZO's frozen plan can configure otherwise.
MAX_CASES = 1024
MAX_STEPS = 4096
MAX_SOURCE_BYTES = 2 * 1024 * 1024
_CASE_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")


@dataclass(frozen=True)
class CaseTokens:
    """One synthetic/public admitted record; no protected evaluation content."""

    case_id: str
    input_tokens: int
    target_tokens: int


@dataclass(frozen=True)
class ArmTimings:
    """Caller-declared *completed* optimizer-step elapsed times in seconds."""

    step_seconds: tuple[float, ...]
    training_budget_seconds: float


def _finite_nonnegative_number(value: object) -> bool:
    if type(value) not in (int, float) or value < 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _literal_number(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and _finite_nonnegative_number(node.value):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _literal_number(node.operand)
    return None


def inspect_adamw8bit_weight_decay(source_file: Path) -> dict:
    """Inspect explicit constructor syntax, *not* actual bnb optimizer behavior.

    Ambiguous constructor count and dynamic expressions remain HOLD.  An
    explicit literal is only SOURCE_INSPECTION: a runtime package, wrapper, or
    different executable file could still change effective regularization.
    """
    raw = source_file.read_bytes()
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("optimizer source exceeds bounded audit input")
    digest = hashlib.sha256(raw).hexdigest()
    tree = ast.parse(raw, filename=str(source_file))
    constructors = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AdamW8bit"
    ]
    result = {"source_sha256": digest, "constructor_count": len(constructors),
              "status": None, "weight_decay": None,
              "evidence_class": "SOURCE_INSPECTION_ONLY",
              "runtime_effective_weight_decay_verified": False}
    if len(constructors) != 1:
        result["status"] = ("NO_ADAMW8BIT_CONSTRUCTOR" if not constructors
                            else "AMBIGUOUS_CONSTRUCTORS")
        return result
    constructor = constructors[0]
    matches = [kw for kw in constructor.keywords if kw.arg == "weight_decay"]
    if not matches:
        result["status"] = "UNDECLARED_DEFAULT_UNKNOWN"
        return result
    if len(matches) != 1:
        result["status"] = "AMBIGUOUS_WEIGHT_DECAY"
        return result
    value = _literal_number(matches[0].value)
    if value is None:
        result["status"] = "DYNAMIC_VALUE_UNATTESTED"
        return result
    result.update(status="EXPLICIT_SOURCE_CONSTANT", weight_decay=value)
    return result


def _validate_cases(cases: tuple[CaseTokens, ...], seq: int) -> None:
    if type(seq) is not int or not 1 <= seq <= 131072:
        raise ValueError("sequence fit length invalid")
    if not isinstance(cases, tuple) or not 8 <= len(cases) <= MAX_CASES or len(cases) % 8:
        raise ValueError("case pool must contain 8–1024 complete effective batches")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, CaseTokens):
            raise ValueError("case entry is not CaseTokens")
        if not isinstance(case.case_id, str) or not _CASE_ID.fullmatch(case.case_id):
            raise ValueError("invalid public/synthetic case id")
        if case.case_id in seen:
            raise ValueError("duplicate case id")
        seen.add(case.case_id)
        if (type(case.input_tokens) is not int or type(case.target_tokens) is not int
                or not 1 <= case.target_tokens <= case.input_tokens):
            raise ValueError("invalid input or supervised target token count")
        if case.input_tokens > seq:
            raise ValueError("sequence fit cannot hold admitted case")


def _validate_timings(timings: ArmTimings) -> None:
    if not isinstance(timings, ArmTimings):
        raise ValueError("invalid arm timing declaration")
    if (not _finite_nonnegative_number(timings.training_budget_seconds)
            or timings.training_budget_seconds <= 0):
        raise ValueError("invalid optimizer budget")
    if not isinstance(timings.step_seconds, tuple) or len(timings.step_seconds) > MAX_STEPS:
        raise ValueError("invalid optimizer steps")
    if any(not _finite_nonnegative_number(s) or s <= 0 for s in timings.step_seconds):
        raise ValueError("nonfinite or nonpositive optimizer step time")


def simulate_source_exposure(arm: str, cases: tuple[CaseTokens, ...],
                             timings: ArmTimings, max_seq_length: int,
                             *, examples_per_update: int = EFFECTIVE_BATCH) -> dict:
    """Synthetic schedule matching the R18 source selection/padding policies.

    MeZO: `prepared[(step*configured_batch+offset) % len(prepared)]`, paired forward calls.
    QLoRA: `converted[offset:offset+8]`, one sweep with 8 effective examples.
    Values are *not* host receipts and must not be merged into a physical run.
    """
    if arm not in ("mezo", "qlora"):
        raise ValueError("unknown R18 training arm")
    _validate_cases(cases, max_seq_length)
    _validate_timings(timings)
    try:
        elapsed_seconds = math.fsum(timings.step_seconds)
    except OverflowError as exc:
        raise ValueError("nonfinite cumulative optimizer step time") from exc
    if not math.isfinite(elapsed_seconds):
        raise ValueError("nonfinite cumulative optimizer step time")
    if type(examples_per_update) is not int or not 1 <= examples_per_update <= MAX_CASES:
        raise ValueError("invalid declared effective batch")
    if arm == "qlora" and examples_per_update != EFFECTIVE_BATCH:
        raise ValueError("QLoRA source fixes effective batch to eight")
    if arm == "qlora" and len(timings.step_seconds) * examples_per_update > len(cases):
        raise ValueError("QLoRA single-sweep policy does not repeat a case")
    counts: Counter[str] = Counter()
    step_rows = []
    input_total = target_total = 0
    for step, seconds in enumerate(timings.step_seconds):
        if arm == "mezo":
            selected = [cases[(step * examples_per_update + offset) % len(cases)]
                        for offset in range(examples_per_update)]
        else:
            selected = list(cases[step * examples_per_update:
                                  (step + 1) * examples_per_update])
        selected_input = sum(x.input_tokens for x in selected)
        selected_target = sum(x.target_tokens for x in selected)
        counts.update(x.case_id for x in selected)
        input_total += selected_input
        target_total += selected_target
        step_rows.append({"step": step + 1, "case_ids": [x.case_id for x in selected],
                          "input_tokens_presented": selected_input,
                          "supervised_target_tokens": selected_target,
                          "optimizer_step_seconds": float(seconds)})
    presentations = sum(counts.values())
    forward_multiplier = 2 if arm == "mezo" else 1
    backwards = 0 if arm == "mezo" else input_total
    return {
        "arm": arm, "sampler_policy": ("MODULO_CYCLE" if arm == "mezo"
                                      else "SINGLE_SWEEP_NO_REPEATS"),
        "evidence_class": "SYNTHETIC_SOURCE_POLICY_SIMULATION",
        "training_budget_seconds": float(timings.training_budget_seconds),
        "optimizer_window_seconds": elapsed_seconds,
        "optimizer_steps": len(step_rows),
        "effective_batch_size": examples_per_update,
        "example_presentations": presentations,
        "unique_examples_seen": len(counts),
        "duplicate_presentations": presentations - len(counts),
        "presentation_counts": dict(sorted(counts.items())),
        "input_tokens_presented": input_total,
        "supervised_target_tokens": target_total,
        "forward_sample_tokens": forward_multiplier * input_total,
        "backward_sample_tokens": backwards,
        "padded_forward_tokens": forward_multiplier * presentations * max_seq_length,
        "padded_backward_tokens": (0 if arm == "mezo"
                                   else presentations * max_seq_length),
        "max_seq_length": max_seq_length,
        "steps": step_rows,
    }


def assess_source_and_exposure(mezo: dict, qlora: dict,
                               qlora_weight_decay_source: dict) -> dict:
    """Report falsifiers while always refusing GPU/promotion authorization.

    This assesses *declared* synthetic schedules and source syntax only. Even
    no discrepancies means no authenticated lease, equal actual GPU time,
    adapter update, independent custody, or untouched evaluation proof.
    """
    if mezo.get("arm") != "mezo" or qlora.get("arm") != "qlora":
        raise ValueError("wrong arm provenance")
    reasons: list[str] = []

    def hold(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    if (mezo.get("evidence_class") != "SYNTHETIC_SOURCE_POLICY_SIMULATION"
            or qlora.get("evidence_class") != "SYNTHETIC_SOURCE_POLICY_SIMULATION"):
        hold("SYNTHETIC_PROVENANCE_MISSING")

    if (mezo.get("sampler_policy") != "MODULO_CYCLE"
            or qlora.get("sampler_policy") != "SINGLE_SWEEP_NO_REPEATS"):
        hold("SAMPLER_SOURCE_POLICY_MISMATCH")
    if mezo.get("effective_batch_size") != qlora.get("effective_batch_size"):
        hold("EFFECTIVE_BATCH_SIZE_MISMATCH")

    if qlora_weight_decay_source.get("status") != "EXPLICIT_SOURCE_CONSTANT":
        hold("QLORA_WEIGHT_DECAY_DEFAULT_UNVERIFIED")
    elif qlora_weight_decay_source.get("weight_decay") != 0.0:
        hold("OPTIMIZER_REGULARIZATION_MISMATCH")
    if mezo.get("training_budget_seconds") != qlora.get("training_budget_seconds"):
        hold("TIME_BUDGET_DECLARATION_MISMATCH")
    if (mezo.get("optimizer_window_seconds", math.inf) > mezo.get("training_budget_seconds", -1)
            or qlora.get("optimizer_window_seconds", math.inf) > qlora.get("training_budget_seconds", -1)):
        hold("TIME_BUDGET_OVERRUN_INVALID")
    if (not math.isclose(mezo.get("optimizer_window_seconds", math.inf),
                         qlora.get("optimizer_window_seconds", -math.inf),
                         rel_tol=0.0, abs_tol=1e-6)):
        hold("ACTIVE_OPTIMIZER_TIME_DIFFERENT")
    if mezo.get("optimizer_steps") == 0 or qlora.get("optimizer_steps") == 0:
        hold("NO_COMPLETE_UPDATE_IN_ARM")
    if mezo.get("presentation_counts") != qlora.get("presentation_counts"):
        hold("CASE_EXPOSURE_MISMATCH")
    if mezo.get("duplicate_presentations", 0) > 0:
        hold("REPEATED_EXAMPLES_IN_MEZO")
    if mezo.get("input_tokens_presented") != qlora.get("input_tokens_presented"):
        hold("INPUT_TOKEN_EXPOSURE_MISMATCH")
    if mezo.get("supervised_target_tokens") != qlora.get("supervised_target_tokens"):
        hold("TARGET_TOKEN_EXPOSURE_MISMATCH")
    if mezo.get("max_seq_length") != qlora.get("max_seq_length"):
        hold("PADDED_SEQUENCE_POLICY_MISMATCH")
    return {
        "status": "HOLD", "hold_reasons": reasons,
        "synthetic_schedule_matched": not reasons,
        "evidence_class": "SOURCE_INSPECTION_PLUS_SYNTHETIC_SIMULATION_ONLY",
        "optimizer_family_dynamics_equal": False,
        "runtime_weight_decay_verified": False,
        "physical_time_or_lease_attested": False,
        "actual_token_ids_attested": False,
        "authorizes_gpu_start": False,
        "proves_neural_improvement": False,
        "authorizes_model_promotion": False,
        "mezo": mezo,
        "qlora": qlora,
    }
