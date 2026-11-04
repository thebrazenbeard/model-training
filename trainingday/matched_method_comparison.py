"""Fail-closed source protocol for a paired MeZO/QLoRA TRAININGDAY study.

This module prepares a reproducible study schedule and validates untrusted
numeric outcomes. It does not import torch, read model files, unseal DEV data,
start a process, grant an evaluator role, or authorize a training run.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence


SCHEMA = "TRAININGDAY_MEZO_QLORA_MATCHED_STUDY_SOURCE_V3"
BASE_ID = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
METHODS = ("MEZO_LORA", "QLORA_ADAMW")
LABEL_MODES = ("correct", "corrupted", "zero")
BASE_REFERENCE = "OFFICIAL_BASE"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CASE_ID = re.compile(r"case-[0-9a-f]{16}\Z")


class MatchedStudyError(ValueError):
    """A proposed comparison would admit a confound or unverifiable shape."""


def _sha(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def commit_training_schedule(seed: int, ordered_rows: Sequence[Mapping[str, str]]) -> dict:
    """Commit ordered example inputs and seed-specific assigned labels.

    The caller MUST derive these from the actual admitted TRAIN rows and the
    deterministic corruption transform. This helper emits digests only: it
    cannot independently attest that the supplied rows were used by a host.
    Neither raw rows nor protected DEV examples belong in committed receipts.
    """
    if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
        raise MatchedStudyError("invalid corruption seed")
    if (type(ordered_rows) not in (list, tuple)
            or not 1 <= len(ordered_rows) <= 1_000_000):
        raise MatchedStudyError("nonempty bounded ordered TRAIN rows required")
    input_rows = []
    correct = []
    corrupted = []
    seen = set()
    required = {"row_id", "input", "correct_label", "corrupted_label"}
    for item in ordered_rows:
        if (type(item) is not dict or set(item) != required
                or any(type(item[key]) is not str or not item[key]
                       for key in required)
                or item["row_id"] in seen):
            raise MatchedStudyError("invalid or duplicate full TRAIN row")
        seen.add(item["row_id"])
        input_rows.append([item["row_id"], item["input"]])
        correct.append([item["row_id"], item["correct_label"]])
        corrupted.append([item["row_id"], item["corrupted_label"]])
    if correct == corrupted:
        raise MatchedStudyError("corrupted label schedule identical to correct schedule")

    def digest(domain: str, values: list) -> str:
        payload = {"domain": domain, "seed": seed, "ordered_values": values}
        return hashlib.sha256(_canonical_bytes(payload)).hexdigest()

    return {"seed": seed,
            "train_order_sha256": digest("training-input-order-v1", input_rows),
            "correct_labels_sha256": digest("correct-label-assignment-v1", correct),
            "corrupted_labels_sha256": digest("corrupted-label-assignment-v1", corrupted)}


def _schedule(seeds: Sequence[int], rows: Sequence[Mapping[str, object]]) -> dict:
    """Demand precommitted ordered rows and both label schedules per seed."""
    required = {"seed", "train_order_sha256", "correct_labels_sha256",
                "corrupted_labels_sha256"}
    if type(rows) not in (tuple, list) or len(rows) != len(seeds):
        raise MatchedStudyError("complete predeclared per-seed label schedule required")
    result = {}
    for row in rows:
        if (type(row) is not dict or set(row) != required
                or type(row["seed"]) is not int
                or row["seed"] not in seeds or row["seed"] in result
                or any(not _sha(row[key]) for key in required - {"seed"})
                or row["correct_labels_sha256"] == row["corrupted_labels_sha256"]):
            raise MatchedStudyError("invalid TRAIN order or correct/corrupt label digest")
        result[row["seed"]] = dict(row)
    if (set(result) != set(seeds)
            or len({x["corrupted_labels_sha256"] for x in result.values()})
            != len(seeds)):
        raise MatchedStudyError("reused or missing seed-specific corruption schedule")
    return result


def _initial_adapters(seeds: Sequence[int],
                      commitments: Sequence[Mapping[str, object]]) -> dict:
    """Require an exact initial PEFT adapter digest for every paired seed.

    Hash declarations are **not** host attestation. Under a later qualified
    launcher, these must come from independently checked initial adapter bytes.
    """
    if (type(commitments) not in (tuple, list)
            or len(commitments) != len(seeds)):
        raise MatchedStudyError("complete per-seed initial adapter commitments required")
    result = {}
    for row in commitments:
        if (type(row) is not dict
                or set(row) != {"seed", "initial_adapter_sha256"}
                or type(row["seed"]) is not int
                or row["seed"] not in seeds or row["seed"] in result
                or not _sha(row["initial_adapter_sha256"])):
            raise MatchedStudyError("invalid seed-specific initial adapter byte digest")
        result[row["seed"]] = row["initial_adapter_sha256"]
    if set(result) != set(seeds):
        raise MatchedStudyError("missing initial adapter seed identity")
    return result


def build_study_plan(*, official_weight_sha256: str,
                     tokenizer_sha256: str, train_sha256: str,
                     private_dev_commitment_sha256: str,
                     lora_target_spec_sha256: str,
                     schedule_commitments: Sequence[Mapping[str, object]],
                     initial_adapter_commitments: Sequence[Mapping[str, object]],
                     seeds: Sequence[int] = (113, 271, 997),
                     wall_clock_budget_seconds: int = 600,
                     actual_time_tolerance_percent: int = 20,
                     candidate_sequence_length: int = 128) -> dict:
    """Specify matched *adapter-subspace* trials; deny every physical grant.

    MeZO full-parameter training and QLoRA LoRA-only training do not have a
    matched trainable parameter space. A fair optimizer comparison first uses
    the same target LoRA parameters. Full-parameter MeZO is a *separate*
    systems/capability comparison and must not be reported as this experiment.
    """
    hashes = (official_weight_sha256, tokenizer_sha256, train_sha256,
              private_dev_commitment_sha256, lora_target_spec_sha256)
    if any(not _sha(x) for x in hashes):
        raise MatchedStudyError("all source and evaluator digests must be SHA-256")
    if train_sha256 == private_dev_commitment_sha256:
        raise MatchedStudyError("TRAIN and DEV commitments cannot be identical")
    if (type(seeds) not in (tuple, list) or not 3 <= len(seeds) <= 32
            or any(type(seed) is not int or seed < 0 or seed > 2**31 - 1
                   for seed in seeds) or len(set(seeds)) != len(seeds)):
        raise MatchedStudyError("3-32 distinct bounded integer seeds required")
    if (type(wall_clock_budget_seconds) is not int
            or not 60 <= wall_clock_budget_seconds <= 86_400):
        raise MatchedStudyError("bounded equal wall-clock ceiling required")
    if (type(actual_time_tolerance_percent) is not int
            or not 0 <= actual_time_tolerance_percent <= 50):
        raise MatchedStudyError("predeclared actual elapsed-time diagnostic tolerance required")
    if (type(candidate_sequence_length) is not int
            or not 32 <= candidate_sequence_length <= 2048):
        raise MatchedStudyError("bounded candidate tokenizer sequence length required")
    schedule = _schedule(seeds, schedule_commitments)
    initial_adapters = _initial_adapters(seeds, initial_adapter_commitments)

    arms = [
        {"arm_id": f"{method}:{mode}:{seed}", "method": method,
         "label_mode": mode, "seed": seed,
         "trainable_subspace": "IDENTICAL_PINNED_LORA_TARGETS",
         "initial_adapter_sha256": initial_adapters[seed],
         "train_order_sha256": (schedule[seed]["train_order_sha256"]
                                if mode != "zero" else None),
         "label_assignment_sha256": (schedule[seed][f"{mode}_labels_sha256"]
                                     if mode != "zero" else None),
         "wall_clock_budget_seconds": wall_clock_budget_seconds}
        for seed in seeds for method in METHODS for mode in LABEL_MODES
    ]
    arms.extend(
        {"arm_id": f"{BASE_REFERENCE}:reference:{seed}",
         "method": BASE_REFERENCE, "label_mode": "reference", "seed": seed,
         "trainable_subspace": "NO_ADAPTER_UNMODIFIED_OFFICIAL_BASE",
         "initial_adapter_sha256": None,
         "train_order_sha256": None,
         "label_assignment_sha256": None,
         "wall_clock_budget_seconds": wall_clock_budget_seconds}
        for seed in seeds)
    record = {
        "schema": SCHEMA,
        "status": "SOURCE_ONLY_HARD_HOLD",
        "base_id": BASE_ID,
        "base_revision": BASE_REVISION,
        "official_weight_sha256": official_weight_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "train_sha256": train_sha256,
        "private_dev_commitment_sha256": private_dev_commitment_sha256,
        "lora_target_spec_sha256": lora_target_spec_sha256,
        "schedule_commitments": [schedule[seed] for seed in seeds],
        "initial_adapter_commitments": [
            {"seed": seed, "initial_adapter_sha256": initial_adapters[seed]}
            for seed in seeds],
        "seeds": list(seeds),
        "methods": list(METHODS),
        "labels": list(LABEL_MODES),
        "arms": arms,
        "base_zero_update_reference_required": True,
        "equal_wall_clock_budget_seconds": wall_clock_budget_seconds,
        "actual_time_tolerance_percent": actual_time_tolerance_percent,
        "candidate_sequence_length": candidate_sequence_length,
        "full_chatml_tokenizer_fit_independently_verified": False,
        "require_zero_truncated_training_rows": True,
        "require_equal_paired_example_token_exposure": True,
        "require_identical_paired_example_order_sha256": True,
        "require_pinned_mode_specific_label_assignment_sha256": True,
        "require_same_per_seed_initial_adapter_bytes": True,
        "require_zero_update_adapter_unchanged": True,
        "initial_adapter_bytes_independently_attested": False,
        "require_forward_and_backward_count_record": True,
        "require_per_arm_checkpoint_reload": True,
        "require_trained_arm_interruption_resume": True,
        "require_independent_blinded_eval": True,
        "allow_full_parameter_mezo_as_matched_arm": False,
        "source_rights_admitted": False,
        "protected_evaluator_custody_verified": False,
        "host_owned_lease_verified": False,
        "authorizes_training": False,
        "authorizes_eval_unseal": False,
    }
    record["plan_sha256"] = hashlib.sha256(_canonical_bytes(record)).hexdigest()
    return record


def validate_untrusted_receipts(plan: Mapping[str, object],
                                receipts: Sequence[Mapping[str, object]]) -> dict:
    """Check coverage and budget accounting, never authenticate a host/run.

    Receipts must contain only scalar public telemetry. Evaluation prompts,
    labels, scores, identity tokens and private DEV rows do not belong here.
    """
    if (type(plan) is not dict or plan.get("schema") != SCHEMA
            or not _sha(plan.get("plan_sha256"))):
        raise MatchedStudyError("invalid study plan")
    # A caller can rehash an incomplete/forged plan; recompute the *entire*
    # schema-controlled plan rather than accepting any internally consistent
    # hash. This also rejects attempts to turn false grants into true claims.
    canonical = build_study_plan(
        official_weight_sha256=plan.get("official_weight_sha256"),
        tokenizer_sha256=plan.get("tokenizer_sha256"),
        train_sha256=plan.get("train_sha256"),
        private_dev_commitment_sha256=plan.get("private_dev_commitment_sha256"),
        lora_target_spec_sha256=plan.get("lora_target_spec_sha256"),
        schedule_commitments=plan.get("schedule_commitments"),
        initial_adapter_commitments=plan.get("initial_adapter_commitments"),
        seeds=plan.get("seeds"),
        wall_clock_budget_seconds=plan.get("equal_wall_clock_budget_seconds"),
        actual_time_tolerance_percent=plan.get("actual_time_tolerance_percent"),
        candidate_sequence_length=plan.get("candidate_sequence_length"))
    if canonical != plan:
        raise MatchedStudyError("plan deviates from non-authorizing canonical schedule")
    claimed_sha = plan["plan_sha256"]
    scheduled = {arm["arm_id"]: arm for arm in plan["arms"]}
    if (type(receipts) not in (list, tuple)
            or len(receipts) != len(scheduled)):
        raise MatchedStudyError("incomplete study arm coverage")
    expected_keys = {"arm_id", "plan_sha256", "elapsed_ms",
                     "optimizer_steps", "examples_seen", "tokens_seen",
                     "forward_calls", "backward_calls", "peak_vram_bytes",
                     "peak_host_ram_bytes", "checkpoint_reloaded",
                     "adapter_sha256", "loaded_weight_sha256",
                     "initial_adapter_sha256",
                     "train_order_sha256", "label_assignment_sha256",
                     "truncated_training_rows",
                     "interruption_resumed"}
    seen = set()
    comparable = {}
    receipt_by_arm = {}
    for receipt in receipts:
        if type(receipt) is not dict or set(receipt) != expected_keys:
            raise MatchedStudyError("unexpected public receipt field or omitted telemetry")
        arm_id = receipt["arm_id"]
        if (type(arm_id) is not str or arm_id not in scheduled
                or arm_id in seen or receipt["plan_sha256"] != claimed_sha):
            raise MatchedStudyError("unknown, duplicate, or replayed arm")
        seen.add(arm_id)
        receipt_by_arm[arm_id] = receipt
        if (receipt["loaded_weight_sha256"] != plan["official_weight_sha256"]
                or type(receipt["loaded_weight_sha256"]) is not str):
            raise MatchedStudyError("model byte-manifest identity mismatch")
        for field in ("elapsed_ms", "optimizer_steps", "examples_seen",
                      "tokens_seen", "forward_calls", "backward_calls",
                      "peak_vram_bytes", "peak_host_ram_bytes",
                      "truncated_training_rows"):
            if type(receipt[field]) is not int or receipt[field] < 0:
                raise MatchedStudyError("invalid nonnegative integer telemetry")
        if not 0 < receipt["elapsed_ms"] <= plan["equal_wall_clock_budget_seconds"] * 1_000:
            raise MatchedStudyError("wall-clock budget exceeded or not measured")
        if receipt["peak_host_ram_bytes"] == 0 or receipt["peak_vram_bytes"] == 0:
            raise MatchedStudyError("missing positive host and GPU memory telemetry")
        if (type(receipt["checkpoint_reloaded"]) is not bool
                or type(receipt["interruption_resumed"]) is not bool):
            raise MatchedStudyError("unverifiable reload/resume telemetry shape")
        if not receipt["checkpoint_reloaded"]:
            raise MatchedStudyError("fresh-process checkpoint reload not demonstrated")
        arm = scheduled[arm_id]
        if receipt["initial_adapter_sha256"] != arm["initial_adapter_sha256"]:
            raise MatchedStudyError("initial adapter identity mismatches frozen seed")
        if arm["method"] == BASE_REFERENCE:
            if (receipt["adapter_sha256"] is not None
                    or receipt["initial_adapter_sha256"] is not None
                    or receipt["train_order_sha256"] is not None
                    or receipt["label_assignment_sha256"] is not None
                    or any(receipt[name] for name in ("optimizer_steps", "examples_seen",
                                                       "tokens_seen", "backward_calls"))
                    or receipt["interruption_resumed"]):
                raise MatchedStudyError("untouched base reference was fine-tuned or adapted")
            continue
        if not _sha(receipt["adapter_sha256"]):
            raise MatchedStudyError("PEFT arm requires adapter byte identity")
        if arm["label_mode"] == "zero":
            if (receipt["optimizer_steps"] != 0 or receipt["backward_calls"] != 0
                    or receipt["adapter_sha256"] != arm["initial_adapter_sha256"]
                    or receipt["train_order_sha256"] is not None
                    or receipt["label_assignment_sha256"] is not None):
                raise MatchedStudyError("zero-update control changed adapter or performed updates")
        else:
            if (receipt["train_order_sha256"] != arm["train_order_sha256"]
                    or receipt["label_assignment_sha256"]
                    != arm["label_assignment_sha256"]):
                raise MatchedStudyError("ordered TRAIN rows or assigned labels differ from plan")
            if (receipt["optimizer_steps"] < 1 or receipt["examples_seen"] < 1
                    or receipt["tokens_seen"] < 1
                    or receipt["forward_calls"] < 1):
                raise MatchedStudyError("training arm lacks work accounting")
            if receipt["truncated_training_rows"]:
                raise MatchedStudyError("truncated TRAIN labels invalidate matched learning")
            if not receipt["interruption_resumed"]:
                raise MatchedStudyError("training-arm recovery not demonstrated")
            comparable[(arm["seed"], arm["label_mode"], arm["method"])] = (
                receipt["examples_seen"], receipt["tokens_seen"],
                receipt["train_order_sha256"], receipt["label_assignment_sha256"])
        if arm["method"] == "MEZO_LORA" and receipt["backward_calls"]:
            raise MatchedStudyError("MeZO arm used gradient backpropagation")
        if (arm["method"] == "MEZO_LORA" and arm["label_mode"] != "zero"
                and receipt["forward_calls"] < 2 * receipt["optimizer_steps"]):
            raise MatchedStudyError("MeZO lacks two-sided forward-call accounting")
        if arm["method"] == "QLORA_ADAMW" and arm["label_mode"] != "zero" \
                and receipt["backward_calls"] < receipt["optimizer_steps"]:
            raise MatchedStudyError("QLoRA arm lacks backward-pass accounting")
    elapsed_disparities = 0
    forward_disparities = 0
    backward_disparities = 0
    max_elapsed_difference_percent = 0.0
    for seed in plan["seeds"]:
        for mode in ("correct", "corrupted"):
            if comparable[(seed, mode, METHODS[0])] != comparable[(seed, mode, METHODS[1])]:
                raise MatchedStudyError("methods saw different examples, order or assigned labels")
            first = receipt_by_arm[f"{METHODS[0]}:{mode}:{seed}"]
            second = receipt_by_arm[f"{METHODS[1]}:{mode}:{seed}"]
            high = max(first["elapsed_ms"], second["elapsed_ms"])
            difference = 100 * abs(first["elapsed_ms"] - second["elapsed_ms"]) / high
            max_elapsed_difference_percent = max(max_elapsed_difference_percent,
                                                  difference)
            elapsed_disparities += difference > plan["actual_time_tolerance_percent"]
            forward_disparities += first["forward_calls"] != second["forward_calls"]
            backward_disparities += first["backward_calls"] != second["backward_calls"]
    return {
        "schema": SCHEMA,
        "status": "SHAPE_VALIDATED_UNAUTHENTICATED_RECEIPTS",
        "plan_sha256": claimed_sha,
        "arms_checked": len(seen),
        "covers_all_declared_arms": seen == set(scheduled),
        "actual_elapsed_time_pairs_over_tolerance": elapsed_disparities,
        "maximum_paired_elapsed_difference_percent": round(max_elapsed_difference_percent, 3),
        "forward_count_disparate_pairs": forward_disparities,
        "backward_count_disparate_pairs": backward_disparities,
        "actual_elapsed_time_within_declared_tolerance": elapsed_disparities == 0,
        "matched_training_exposure_declared": True,
        "measured_compute_equivalence_established": False,
        "scientific_cost_comparison_status": (
            "TIME_DISPARITY_REQUIRES_SEPARATE_ANALYSIS" if elapsed_disparities
            else "EQUAL_TIME_CEILING_ONLY_COMPUTE_UNQUALIFIED"),
        "physical_execution_independently_verified": False,
        "data_rights_verified": False,
        "protected_eval_verified": False,
        "proves_learning": False,
    }


def summarize_paired_binary_outcomes(
    reference: Mapping[str, bool], challenger: Mapping[str, bool],
) -> dict:
    """Return only aggregate paired binary metrics, with exact McNemar p.

    This is deliberately an *untrusted* score-shape computation, not a
    validated evaluator. Its caller must independently protect unseen DEV
    examples, identities, assignment and disclosure.
    """
    if (type(reference) is not dict or type(challenger) is not dict
            or not 32 <= len(reference) <= 1024
            or set(reference) != set(challenger)
            or any(type(k) is not str or _CASE_ID.fullmatch(k) is None
                   or type(reference[k]) is not bool
                   or type(challenger[k]) is not bool for k in reference)):
        raise MatchedStudyError("invalid or mismatched case-identity/outcome sets")
    gains = sum(not reference[k] and challenger[k] for k in reference)
    losses = sum(reference[k] and not challenger[k] for k in reference)
    discordant = gains + losses
    tail = sum(math.comb(discordant, k)
               for k in range(min(gains, losses) + 1))
    pvalue = min(1.0, 2 * tail / (2 ** discordant))
    return {
        "status": "UNTRUSTED_PAIRED_SCORE_SHAPE_ONLY",
        "cases": len(reference),
        "reference_correct": sum(reference.values()),
        "challenger_correct": sum(challenger.values()),
        "net_improvement_cases": gains - losses,
        "paired_gain": gains,
        "paired_loss": losses,
        "mcnemar_exact_two_sided_p": pvalue,
        "independent_eval_verified": False,
        "generalization_proven": False,
    }
