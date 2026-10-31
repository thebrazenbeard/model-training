"""Source-only R18 MeZO / QLoRA matched-time benchmark custody contract.

This module deliberately imports no model, GPU, tokenizer, or evaluation bank.
An external custodian must admit the corpus, seal a NEW R18 evaluation bank,
and preregister a plan before either trainer starts.  These validators check
receipts; they cannot independently attest that a GPU or external timer was
honestly measured.  R16 fresh64 is historical context, never a tuning set.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import secrets
import stat
import subprocess
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path


PLAN_SCHEMA = "VERA_TRAININGDAY_R18_BENCHMARK_PLAN_V1"
PROBE_SCHEMA = "VERA_TRAININGDAY_R18_FIT_PROBE_V1"
ARM_SCHEMA = "VERA_TRAININGDAY_R18_ARM_RECEIPT_V1"
MODEL_ID = "Qwen/Qwen3.5-4B"
ARMS = ("mezo", "qlora")
CONTROLS = ("frozen", "zero", "correct", "corrupt")
FIT_LENGTHS = (64, 128, 256, 384, 512, 768, 1024)
FIT_POLICY = "ASCENDING_LENGTH_BATCH1_THEN_BATCH2_UNTIL_FIRST_OOM"
EVAL_SOURCE = "R18_NEW_SEALED_NOT_R16_FRESH64"
ADMISSION_STATUSES = frozenset({
    "ADMITTED_SOURCE_ONLY_NOT_TRAINED",
    "ADMITTED_LABYRINTH_TRAIN_ONLY_NO_WEIGHT_UPDATE",
})
PLAN_KEYS = frozenset({
    "schema", "base_model", "base_revision", "base_binding_sha256",
    "tokenizer_binding_sha256", "corpus_manifest_sha256", "selected_id_sha256",
    "selected_count", "train_content_sha256", "source_admission_status",
    "eval_bank_source",
    "eval_custody_sha256", "eval_index_sha256", "eval_source_manifest_sha256",
    "eval_examples",
    "r16_eval_custody_sha256", "r16_holdout_id_sha256", "seed",
    "training_budget_seconds", "probe_budget_seconds", "fit_policy",
    "fit_candidates", "primary_training_config", "no_eval_access_before_arm_freeze",
    "preregistered_checkpoint_selection", "promotion_policy",
})


class BenchmarkError(ValueError):
    """Invalid or noncomparable R18 experimental evidence."""


def _check(ok: bool, problem: str) -> None:
    if not ok:
        raise BenchmarkError(problem)


def _hex(value: object, width: int = 64) -> bool:
    return (isinstance(value, str) and len(value) == width
            and all(c in "0123456789abcdef" for c in value))


def _number(value: object, *, zero: bool = True) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and (value >= 0 if zero else value > 0))


def _integer(value: object, *, zero: bool = True) -> bool:
    return type(value) is int and (value >= 0 if zero else value > 0)


def canonical_sha256(value: object) -> str:
    """Canonical JSON receipt digest; never hash a self-referential hash field."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def selected_id_digest(rows: Sequence[Mapping | str]) -> str:
    """Match R16 admission's *selection-order* (not sorted-ID) commitment."""
    ids = [row if isinstance(row, str) else row["id"] for row in rows]
    _check(bool(ids) and all(isinstance(i, str) and i.strip() for i in ids),
           "selected training IDs missing")
    _check(len(set(ids)) == len(ids), "duplicate selected training IDs")
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _prompt_sha(prompt: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", prompt).split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _training_index(rows: Sequence[Mapping], source_admission_status: str) -> list[dict]:
    _check(source_admission_status in ADMISSION_STATUSES,
           "source must be explicitly admitted before R18 plan construction")
    result = []
    for row in rows:
        _check(isinstance(row, Mapping), "train row not an object")
        _check(all(isinstance(row.get(k), str) and row[k].strip()
                   for k in ("id", "prompt", "answer")), "train row incomplete")
        if source_admission_status == "ADMITTED_SOURCE_ONLY_NOT_TRAINED":
            _check(row.get("split") == "train" and row.get("training_use") == "ADMITTED"
                   and row.get("evaluation_holdout") is not True,
                   "train-only admitted row required")
        else:
            # The Labyrinth admission adapter independently replays procedural
            # train cases then returns a minimal id/family/prompt/answer dict.
            # Its accepted cases intentionally do not carry split/use markers.
            _check(isinstance(row.get("family"), str) and row["family"]
                   and "split" not in row and "training_use" not in row,
                   "Labyrinth plan needs verified replayed train-only case shape")
        result.append({"id": row["id"], "prompt_sha256": _prompt_sha(row["prompt"]),
                       "answer_sha256": hashlib.sha256(row["answer"].encode()).hexdigest()})
    selected_id_digest(rows)
    _check(len({r["prompt_sha256"] for r in result}) == len(result),
           "duplicate normalized train prompt")
    return result


def _evaluation_index(index: Sequence[Mapping]) -> list[dict]:
    _check(isinstance(index, (list, tuple)) and len(index) >= 64,
           "new R18 sealed evaluation must contain at least 64 indexed cases")
    result = []
    for item in index:
        _check(isinstance(item, Mapping) and set(item) == {"id", "prompt_sha256"},
               "eval index must be answer-free ID/prompt-hash metadata only")
        _check(isinstance(item["id"], str) and bool(item["id"].strip())
               and _hex(item["prompt_sha256"]), "invalid sealed evaluation index")
        result.append(dict(item))
    _check(len({item["id"] for item in result}) == len(result),
           "duplicate sealed eval ID")
    _check(len({item["prompt_sha256"] for item in result}) == len(result),
           "duplicate sealed eval prompt")
    return sorted(result, key=lambda item: item["id"])


def fit_candidates() -> list[dict]:
    """Frozen ascending true-length probes, first batch1 then batch2.

    Every successful probe must complete a representative training update on
    exact padded tensor lengths; max_length alone is not evidence. OOM requires
    an entirely fresh process before the next probe. Neither prompts nor loss
    values from the heldout evaluation bank may be used.
    """
    return [{"micro_batch_size": batch, "actual_padded_sequence_length": length,
             "effective_batch_size": 8, "offload": False,
             "truncation_allowed": False}
            for batch in (1, 2) for length in FIT_LENGTHS]


def make_plan(*, selected_rows: Sequence[Mapping], admission_selected_id_sha256: str,
              source_admission_status: str,
              eval_index: Sequence[Mapping], r16_holdout_ids: Sequence[str],
              base_revision: str, base_binding_sha256: str,
              tokenizer_binding_sha256: str, corpus_manifest_sha256: str,
              eval_custody_sha256: str, eval_source_manifest_sha256: str,
              r16_eval_custody_sha256: str,
              seed: int, training_budget_seconds: float,
              probe_budget_seconds: float, primary_sequence_length: int = 256) -> dict:
    """Make a reproducible plan from admitted train and answer-free eval indices.

    An evaluation custodian supplies the fresh index and custody digest without
    revealing labels. R16 fresh64 IDs are exclusion-only and may not be scored
    or inspected to select a fit, learning rate, or checkpoint.
    """
    training = _training_index(selected_rows, source_admission_status)
    eval_meta = _evaluation_index(eval_index)
    _check(_hex(admission_selected_id_sha256)
           and selected_id_digest(selected_rows) == admission_selected_id_sha256,
           "admission selected-ID digest mismatch")
    _check(isinstance(r16_holdout_ids, (list, tuple)) and len(r16_holdout_ids) == 64
           and all(isinstance(i, str) and i.strip() for i in r16_holdout_ids)
           and len(set(r16_holdout_ids)) == 64, "R16 exclusion inventory incomplete")
    train_ids = {r["id"] for r in training}
    eval_ids = {r["id"] for r in eval_meta}
    r16_ids = set(r16_holdout_ids)
    _check(not train_ids.intersection(eval_ids | r16_ids)
           and not eval_ids.intersection(r16_ids), "train/R18/R16 ID leakage")
    _check(not {r["prompt_sha256"] for r in training}.intersection(
        r["prompt_sha256"] for r in eval_meta), "normalized train/eval prompt leakage")
    for name, digest in (("base binding", base_binding_sha256),
                         ("tokenizer binding", tokenizer_binding_sha256),
                         ("corpus manifest", corpus_manifest_sha256),
                         ("R18 eval custody", eval_custody_sha256),
                         ("R18 independent source manifest", eval_source_manifest_sha256),
                         ("R16 exclusion custody", r16_eval_custody_sha256)):
        _check(_hex(digest), f"{name} SHA256 missing")
    _check(_hex(base_revision, 40), "base must be pinned to 40-hex model revision")
    _check(_integer(seed), "invalid seed")
    _check(_number(training_budget_seconds, zero=False)
           and _number(probe_budget_seconds, zero=False), "budgets must be positive")
    _check(training_budget_seconds > probe_budget_seconds,
           "training budget must exceed an individual probe budget")
    _check(primary_sequence_length in FIT_LENGTHS,
           "primary training sequence length must be predeclared fit length")
    plan = {
        "schema": PLAN_SCHEMA, "base_model": MODEL_ID, "base_revision": base_revision,
        "base_binding_sha256": base_binding_sha256,
        "tokenizer_binding_sha256": tokenizer_binding_sha256,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "selected_id_sha256": admission_selected_id_sha256,
        "selected_count": len(training),
        "train_content_sha256": canonical_sha256(training),
        "source_admission_status": source_admission_status,
        "eval_bank_source": EVAL_SOURCE,
        "eval_custody_sha256": eval_custody_sha256,
        "eval_source_manifest_sha256": eval_source_manifest_sha256,
        "eval_index_sha256": canonical_sha256(eval_meta),
        "eval_examples": len(eval_meta),
        "r16_eval_custody_sha256": r16_eval_custody_sha256,
        "r16_holdout_id_sha256": canonical_sha256(sorted(r16_ids)),
        "seed": seed,
        "training_budget_seconds": training_budget_seconds,
        "probe_budget_seconds": probe_budget_seconds,
        "fit_policy": FIT_POLICY, "fit_candidates": fit_candidates(),
        "primary_training_config": fit_candidates()[FIT_LENGTHS.index(primary_sequence_length)],
        "no_eval_access_before_arm_freeze": True,
        "preregistered_checkpoint_selection": "END_OF_TIME_BUDGET_ONLY",
        "promotion_policy": "NEVER_AUTOPROMOTE_R18_REQUIRES_INDEPENDENT_REPLICATION",
    }
    validate_plan(plan)
    return plan


def validate_plan(plan: Mapping) -> str:
    _check(isinstance(plan, Mapping) and plan.get("schema") == PLAN_SCHEMA,
           "invalid R18 preregistration schema")
    _check(set(plan) == PLAN_KEYS, "plan has missing or unregistered keys")
    _check(plan.get("base_model") == MODEL_ID and _hex(plan.get("base_revision"), 40),
           "unbound official Qwen3.5-4B revision")
    for key in ("base_binding_sha256", "tokenizer_binding_sha256",
                "corpus_manifest_sha256", "selected_id_sha256", "train_content_sha256",
                "eval_custody_sha256", "eval_index_sha256", "eval_source_manifest_sha256",
                "r16_eval_custody_sha256",
                "r16_holdout_id_sha256"):
        _check(_hex(plan.get(key)), f"invalid plan {key}")
    _check(_integer(plan.get("selected_count"), zero=False)
           and _integer(plan.get("eval_examples"), zero=False)
           and plan["eval_examples"] >= 64, "invalid plan corpus/eval sizes")
    _check(plan.get("source_admission_status") in ADMISSION_STATUSES,
           "admission provenance missing")
    _check(_integer(plan.get("seed")) and _number(plan.get("training_budget_seconds"), zero=False)
           and _number(plan.get("probe_budget_seconds"), zero=False)
           and plan["training_budget_seconds"] > plan["probe_budget_seconds"],
           "invalid plan budgets/seed")
    _check(plan.get("eval_bank_source") == EVAL_SOURCE
           and plan.get("no_eval_access_before_arm_freeze") is True,
           "R16 fresh64 reuse or premature eval access forbidden")
    _check(plan.get("fit_policy") == FIT_POLICY
           and plan.get("fit_candidates") == fit_candidates()
           and plan.get("primary_training_config") in fit_candidates()[:len(FIT_LENGTHS)],
           "max-fit schedule is not preregistered")
    _check(plan.get("preregistered_checkpoint_selection") == "END_OF_TIME_BUDGET_ONLY"
           and plan.get("promotion_policy") ==
           "NEVER_AUTOPROMOTE_R18_REQUIRES_INDEPENDENT_REPLICATION",
           "posthoc checkpoint selection or automatic model promotion forbidden")
    return canonical_sha256(plan)


def validate_fit_probes(plan: Mapping, probes: Sequence[Mapping]) -> dict[str, dict]:
    """Require each arm's b1/b2 ascending sweep, fresh OOM process and no eval.

    On first OOM in a batch phase, later lengths in that phase are skipped;
    batch2 may only probe lengths already fitting for batch1. A batch1 OOM at
    the very first length makes the arm infeasible, not a successful comparison.
    """
    plan_sha = validate_plan(plan)
    _check(isinstance(probes, (list, tuple)), "probe evidence must be a sequence")
    by_arm: dict[str, list[Mapping]] = {arm: [] for arm in ARMS}
    nonces = set()
    for probe in probes:
        _check(isinstance(probe, Mapping) and probe.get("schema") == PROBE_SCHEMA
               and probe.get("plan_sha256") == plan_sha, "probe plan custody mismatch")
        arm = probe.get("arm")
        _check(arm in ARMS, "unknown probe arm")
        _check(probe.get("eval_access") is False
               and probe.get("data_partition") == "TRAIN_ONLY"
               and probe.get("checkpoint_selection") is False,
               "evaluation leakage during fit probe")
        nonce = probe.get("fresh_process_id")
        _check(isinstance(nonce, str) and nonce and nonce not in nonces,
               "each probe must use a fresh unique process identity")
        nonces.add(nonce)
        index = probe.get("candidate_index")
        _check(_integer(index) and index < len(fit_candidates())
               and probe.get("config") == fit_candidates()[index],
               "unregistered padded-shape probe")
        _check(probe.get("outcome") in {"FIT", "OOM"}
               and _number(probe.get("elapsed_seconds"), zero=False)
               and probe["elapsed_seconds"] <= plan["probe_budget_seconds"],
               "invalid probe result or monotonic time budget")
        for key in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
                    "peak_cpu_rss_bytes", "peak_nvml_used_bytes"):
            _check(_integer(probe.get(key)), f"invalid probe {key}")
        _check(probe["peak_cuda_reserved_bytes"] >= probe["peak_cuda_allocated_bytes"],
               "reserved GPU memory below allocated GPU memory")
        _check(probe.get("optimizer_step_completed") is (probe["outcome"] == "FIT"),
               "fit requires an actual representative optimizer step")
        _check((probe.get("failure_artifact_sha256") is None if probe["outcome"] == "FIT"
                else _hex(probe.get("failure_artifact_sha256"))),
               "OOM needs a pinned failure log, FIT must not claim one")
        _check(probe.get("actual_padded_sequence_length") ==
               probe["config"]["actual_padded_sequence_length"],
               "max_length config is not proof of actual tensor length")
        by_arm[arm].append(probe)
    selections = {}
    for arm, records in by_arm.items():
        _check(records and records[0]["candidate_index"] == 0,
               f"{arm} missing first batch1 probe")
        b1 = [r for r in records if r["config"]["micro_batch_size"] == 1]
        b2 = [r for r in records if r["config"]["micro_batch_size"] == 2]
        _check(records == b1 + b2, f"{arm} fit phase order changed")
        for phase, items, start in ((1, b1, 0), (2, b2, len(FIT_LENGTHS))):
            _check(all(r["candidate_index"] == start + i for i, r in enumerate(items)),
                   f"{arm} batch{phase} skipped/reordered probe")
            _check(not any(r["outcome"] == "OOM" for r in items[:-1]),
                   f"{arm} continued probing after an OOM")
        b1_fits = [r for r in b1 if r["outcome"] == "FIT"]
        _check(bool(b1_fits), f"{arm} cannot fit even 64 padded tokens")
        _check(len(b1) == len(FIT_LENGTHS) or b1[-1]["outcome"] == "OOM",
               f"{arm} stopped batch1 before reaching capacity")
        # Batch2 is probed at every successful batch1 length, stopping at OOM.
        _check(bool(b2) and len(b2) <= len(b1_fits)
               and (len(b2) == len(b1_fits) or b2[-1]["outcome"] == "OOM"),
               f"{arm} incomplete batch2 followup")
        fits = b1_fits + [r for r in b2 if r["outcome"] == "FIT"]
        selected = max(fits, key=lambda r: (
            r["config"]["micro_batch_size"] * r["actual_padded_sequence_length"],
            r["actual_padded_sequence_length"], r["config"]["micro_batch_size"]))
        _check(any(r["config"] == plan["primary_training_config"] for r in b1_fits),
               f"{arm} cannot run the shared preregistered primary train shape")
        selections[arm] = {"max_fit_candidate_index": selected["candidate_index"],
                           "max_fit_config": selected["config"],
                           "primary_training_config": plan["primary_training_config"],
                           "probe_sha256": canonical_sha256(
                               [dict(r) for r in records])}
    return selections


def validate_arm_receipt(receipt: Mapping, plan: Mapping, *,
                         allow_unscored: bool = True) -> None:
    """Check an arm's pinned identity, measured time/work and sealed eval status."""
    plan_sha = validate_plan(plan)
    _check(isinstance(receipt, Mapping) and receipt.get("schema") == ARM_SCHEMA,
           "invalid R18 arm receipt")
    _check(receipt.get("arm") in ARMS and receipt.get("control") in CONTROLS,
           "invalid arm/control")
    _check(receipt.get("plan_sha256") == plan_sha, "receipt plan mismatch")
    for key in ("base_model", "base_revision", "base_binding_sha256",
                "tokenizer_binding_sha256", "corpus_manifest_sha256",
                "selected_id_sha256", "train_content_sha256", "source_admission_status",
                "eval_custody_sha256",
                "eval_index_sha256", "seed"):
        _check(receipt.get(key) == plan[key], f"arm {key} differs from preregistration")
    _check(isinstance(receipt.get("run_id"), str) and receipt["run_id"]
           and receipt.get("training_budget_seconds") == plan["training_budget_seconds"],
           "invalid arm run ID or budget")
    trained = receipt["control"] in {"correct", "corrupt"}
    for key in ("optimizer_steps", "unique_examples_seen", "example_presentations",
                "input_tokens_presented", "forward_sample_tokens",
                "backward_sample_tokens", "padded_forward_tokens",
                "padded_backward_tokens", "supervised_target_tokens",
                "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
                "peak_nvml_used_bytes", "peak_cpu_rss_bytes"):
        _check(_integer(receipt.get(key)), f"invalid arm counter {key}")
    _check(receipt["peak_cuda_reserved_bytes"] >= receipt["peak_cuda_allocated_bytes"],
           "GPU allocated memory exceeds reserved memory")
    _check(receipt.get("peak_cuda_metric_scope") ==
           "PRIMARY_TRAIN_PROCESS_AFTER_SETUP_RESET",
           "training CUDA peak mixed with synthetic fit probes or missing scope")
    _check(receipt.get("peak_nvml_metric_scope") in
           {"SAMPLED_NVML", "UNAVAILABLE_NOT_ESTIMATED"},
           "NVML physical observation scope missing or invented")
    _check((receipt["peak_nvml_used_bytes"] > 0
            if receipt["peak_nvml_metric_scope"] == "SAMPLED_NVML"
            else receipt["peak_nvml_used_bytes"] == 0),
           "NVML peak contradicts measured or unavailable observation scope")
    _check(receipt.get("peak_cpu_rss_metric_scope") in
           {"TRAINING_WINDOW_SAMPLED_RSS", "PROCESS_LIFETIME_PEAK_OR_CURRENT_RSS"},
           "CPU RSS peak observation scope missing")
    _check(receipt["unique_examples_seen"] <= plan["selected_count"]
           and receipt["unique_examples_seen"] <= receipt["example_presentations"]
           and receipt["supervised_target_tokens"] <= receipt["input_tokens_presented"],
           "impossible observed training example/token accounting")
    _check(receipt["forward_sample_tokens"] >= receipt["input_tokens_presented"]
           and (receipt["arm"] != "mezo" or not trained or
                receipt["forward_sample_tokens"] >= 2 * receipt["input_tokens_presented"]),
           "MeZO two-sided perturbation passes not counted")
    _check(receipt["padded_forward_tokens"] >= receipt["forward_sample_tokens"]
           and receipt["padded_backward_tokens"] >= receipt["backward_sample_tokens"],
           "physical padded token work cannot be less than useful-token work")
    _check(receipt["arm"] != "mezo" or receipt["backward_sample_tokens"] == 0,
           "MeZO is forward-only; backward token counter must be zero")
    _check(_hex(receipt.get("trainable_digest_before"))
           and _hex(receipt.get("trainable_digest_after")),
           "missing trainable tensor digest custody")
    if trained:
        _check(receipt["optimizer_steps"] > 0
               and receipt["example_presentations"] > 0
               and receipt["supervised_target_tokens"] > 0
               and receipt["trainable_digest_before"] != receipt["trainable_digest_after"],
               "trained arm has no proven parameter change")
    else:
        _check(receipt["optimizer_steps"] == 0
               and receipt["example_presentations"] == 0
               and receipt["input_tokens_presented"] == 0
               and receipt["forward_sample_tokens"] == 0
               and receipt["backward_sample_tokens"] == 0
               and receipt["padded_forward_tokens"] == 0
               and receipt["padded_backward_tokens"] == 0
               and receipt["supervised_target_tokens"] == 0
               and receipt["trainable_digest_before"] == receipt["trainable_digest_after"],
               "frozen/zero arm was updated or trained")
    for key in ("train_start_monotonic_ns", "train_stop_monotonic_ns",
                "train_deadline_monotonic_ns"):
        _check(_integer(receipt.get(key)), f"invalid monotonic timer {key}")
    start, stop, deadline = (receipt[k] for k in (
        "train_start_monotonic_ns", "train_stop_monotonic_ns",
        "train_deadline_monotonic_ns"))
    _check(start <= stop <= deadline
           and deadline == start + round(plan["training_budget_seconds"] * 1e9),
           "training escaped strict monotonic deadline")
    _check(_number(receipt.get("optimizer_window_seconds"))
           and abs(receipt["optimizer_window_seconds"] - (stop-start)/1e9) <= 0.01,
           "reported training time differs from monotonic clock")
    stages = receipt.get("stage_seconds")
    _check(isinstance(stages, Mapping) and set(stages) ==
           {"load", "setup", "train", "save", "eval"}
           and all(_number(value) for value in stages.values()),
           "missing external load/setup/train/save/eval timing")
    _check(abs(stages["train"] - receipt["optimizer_window_seconds"]) <= 0.01,
           "train stage is not the timed optimizer window")
    _check(_number(receipt.get("external_process_wall_seconds"))
           and sum(stages.values()) <= receipt["external_process_wall_seconds"] + 0.02,
           "stage time exceeds external wall-clock custody")
    _check(_hex(receipt.get("training_input_sha256"))
           and _hex(receipt.get("fit_probe_sha256")),
           "training bytes or fit probe not pinned")
    cfg = receipt.get("selected_fit_config")
    _check(cfg == plan["primary_training_config"],
           "primary training used a posthoc fit-maximizing or different tensor shape")
    _check(_integer(receipt.get("max_prepared_example_tokens"), zero=False)
           and receipt["max_prepared_example_tokens"] <= cfg["actual_padded_sequence_length"],
           "silent truncation of prepared prompt/answer forbidden")
    scored = receipt.get("status") == "EVALUATED_SEALED"
    _check(scored or (allow_unscored and receipt.get("status") == "TRAINED_UNSCORED"),
           "arm needs sealed evaluation or explicit unscored status")
    if scored:
        _check(receipt.get("eval_unsealed_after_arm_freeze") is True
               and receipt.get("eval_examples") == plan["eval_examples"],
               "R18 sealed evaluation reused or accessed before arm freeze")
        _check(_hex(receipt.get("arm_freeze_sha256"))
               and _hex(receipt.get("eval_case_receipt_sha256"))
               and _integer(receipt.get("arm_frozen_at_utc_ns"), zero=False)
               and _integer(receipt.get("eval_unsealed_at_utc_ns"), zero=False)
               and receipt["eval_unsealed_at_utc_ns"] > receipt["arm_frozen_at_utc_ns"],
               "evaluation was not cryptographically bound after the arm froze")
        _check(_integer(receipt.get("eval_correct"))
               and receipt["eval_correct"] <= receipt["eval_examples"]
               and _integer(receipt.get("eval_target_tokens"), zero=False)
               and _number(receipt.get("eval_nll"))
               and _number(receipt.get("eval_nll_sum")), "invalid heldout accuracy/NLL")
        _check(abs(receipt["eval_nll"] - receipt["eval_nll_sum"] /
                   receipt["eval_target_tokens"]) <= 1e-7,
               "NLL must be total heldout token loss divided by target token count")
        _check(_number(receipt.get("eval_accuracy"))
               and abs(receipt["eval_accuracy"] - receipt["eval_correct"] /
                       receipt["eval_examples"]) < 1e-9,
               "evaluation accuracy does not match integer case count")
    else:
        _check(receipt.get("eval_accuracy") is None
               and receipt.get("eval_nll") is None
               and receipt.get("eval_nll_sum") is None
               and receipt.get("eval_case_receipt_sha256") is None
               and receipt.get("eval_correct") is None
               and receipt.get("eval_examples") == 0
               and receipt.get("eval_target_tokens") == 0,
               "unscored arm must not carry early evaluation labels/metrics")


def compare_arms(plan: Mapping, receipts: Sequence[Mapping],
                 probes: Sequence[Mapping]) -> dict:
    """Return an evidence report, **never** an automatic promotion decision.

    All four controls per method are mandatory, including same-zero frozen
    parity and corruption checks. Accuracy + NLL improvement on one small bank
    is insufficient to promote without a separate blinded replication.
    """
    fit = validate_fit_probes(plan, probes)
    _check(isinstance(receipts, (list, tuple)) and len(receipts) == 8,
           "require 2 methods x frozen/zero/correct/corrupt receipts")
    keyed = {}
    run_ids = set()
    for rec in receipts:
        validate_arm_receipt(rec, plan, allow_unscored=False)
        key = (rec["arm"], rec["control"])
        _check(key not in keyed and rec["run_id"] not in run_ids,
               "duplicate arm/control or run namespace")
        keyed[key] = rec
        run_ids.add(rec["run_id"])
        _check(rec["selected_fit_config"] == fit[rec["arm"]]["primary_training_config"]
               and rec["fit_probe_sha256"] == fit[rec["arm"]]["probe_sha256"],
               "final arm training shape not selected by its frozen fit probe")
    for arm in ARMS:
        controls = {control: keyed[(arm, control)] for control in CONTROLS}
        _check(controls["frozen"]["eval_correct"] == controls["zero"]["eval_correct"]
               and abs(controls["frozen"]["eval_nll"] - controls["zero"]["eval_nll"]) <= 1e-8,
               f"{arm} frozen/zero baseline mismatch")
        _check(controls["frozen"]["trainable_digest_before"] ==
               controls["zero"]["trainable_digest_before"],
               f"{arm} zero/frozen initial weights differ")
        _check(all(controls[c]["selected_id_sha256"] == plan["selected_id_sha256"]
                   for c in CONTROLS), "controls used different train IDs")
        _check(controls["correct"]["training_input_sha256"] !=
               controls["corrupt"]["training_input_sha256"],
               "corrupt labels did not change training input bytes")
        _check(controls["correct"]["optimizer_window_seconds"] >=
               0.85 * plan["training_budget_seconds"]
               and controls["corrupt"]["optimizer_window_seconds"] >=
               0.85 * plan["training_budget_seconds"],
               "training ended too early for equal-time claim")
    correct = [keyed[(arm, "correct")] for arm in ARMS]
    _check(correct[0]["training_input_sha256"] == correct[1]["training_input_sha256"],
           "different prepared training input across algorithms")
    _check(abs(correct[0]["optimizer_window_seconds"] -
               correct[1]["optimizer_window_seconds"]) <=
           0.05 * plan["training_budget_seconds"],
           "MeZO and QLoRA did not receive equal optimizer-wall time")
    # Precommitted metrics are descriptive, not a posthoc model-selection rule.
    report = {"schema": "VERA_TRAININGDAY_R18_COMPARISON_V1",
              "plan_sha256": validate_plan(plan),
              "decision": "NO_MODEL_PROMOTION_INDEPENDENT_REPLICATION_REQUIRED",
              "fresh_eval_bank": True, "control_arms_verified": 8,
              "physical_vram_residency": "UNVERIFIED_BY_ALLOCATOR_OR_SAMPLED_NVML",
              "method_metrics": {}}
    for arm in ARMS:
        rec = keyed[(arm, "correct")]
        zero = keyed[(arm, "zero")]
        corrupt = keyed[(arm, "corrupt")]
        report["method_metrics"][arm] = {
            "accuracy": rec["eval_accuracy"], "nll": rec["eval_nll"],
            "delta_accuracy_vs_zero": rec["eval_accuracy"] - zero["eval_accuracy"],
            "delta_nll_vs_zero": rec["eval_nll"] - zero["eval_nll"],
            "corrupt_accuracy": corrupt["eval_accuracy"],
            "corrupt_nll": corrupt["eval_nll"],
            "optimizer_window_seconds": rec["optimizer_window_seconds"],
            "external_process_wall_seconds": rec["external_process_wall_seconds"],
            "unique_examples_seen": rec["unique_examples_seen"],
            "example_presentations": rec["example_presentations"],
            "input_tokens_presented": rec["input_tokens_presented"],
            "forward_sample_tokens": rec["forward_sample_tokens"],
            "backward_sample_tokens": rec["backward_sample_tokens"],
            "padded_forward_tokens": rec["padded_forward_tokens"],
            "padded_backward_tokens": rec["padded_backward_tokens"],
            "supervised_target_tokens": rec["supervised_target_tokens"],
            "optimizer_steps": rec["optimizer_steps"],
            "peak_cuda_allocated_bytes": rec["peak_cuda_allocated_bytes"],
            "peak_cuda_reserved_bytes": rec["peak_cuda_reserved_bytes"],
            "peak_cuda_metric_scope": rec["peak_cuda_metric_scope"],
            "peak_nvml_used_bytes": rec["peak_nvml_used_bytes"],
            "peak_nvml_metric_scope": rec["peak_nvml_metric_scope"],
            "peak_cpu_rss_bytes": rec["peak_cpu_rss_bytes"],
            "peak_cpu_rss_metric_scope": rec["peak_cpu_rss_metric_scope"],
            "fit_probe_peak_cuda_allocated_bytes": max(
                (p["peak_cuda_allocated_bytes"] for p in probes if p["arm"] == arm), default=0),
            "fit_probe_peak_cuda_reserved_bytes": max(
                (p["peak_cuda_reserved_bytes"] for p in probes if p["arm"] == arm), default=0),
            "fit_probe_peak_cpu_rss_bytes": max(
                (p["peak_cpu_rss_bytes"] for p in probes if p["arm"] == arm), default=0),
            "fit_probe_memory_scope": "ISOLATED_SYNTHETIC_FRESH_PROCESSES",
            "max_fit_frontier": fit[arm]["max_fit_config"],
            "productive_primary_shape": plan["primary_training_config"],
        }
    report["accuracy_delta_mezo_minus_qlora"] = (
        correct[0]["eval_accuracy"] - correct[1]["eval_accuracy"])
    report["nll_delta_mezo_minus_qlora"] = (
        correct[0]["eval_nll"] - correct[1]["eval_nll"])
    return report


def _json_file(path: Path) -> object:
    _check(path.is_file() and not path.is_symlink(),
           "JSON evidence missing or symlinked")
    return json.loads(path.read_text(encoding="utf-8"))


def _windows_owner_sid() -> str:
    """Resolve the current Windows user SID, never a hardcoded account name."""
    result = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"],
                            capture_output=True, text=True, check=False)
    _check(result.returncode == 0, "cannot resolve Windows owner SID")
    rows = list(csv.reader(io.StringIO(result.stdout.lstrip("\ufeff"))))
    _check(len(rows) == 1 and len(rows[0]) == 2
           and re.fullmatch(r"S-1-\d+(?:-\d+)+", rows[0][1]) is not None,
           "Windows owner SID result malformed")
    return rows[0][1]


def _windows_verify_owner_only(path: Path, sid: str, *, directory: bool) -> None:
    """Inspect an owner/privileged-system-only protected Windows DACL.

    `icacls /verify` alone checks ACL canonicality, not confidentiality. Check
    the owner SID, absence of inheritance, and reject every unprivileged ACE.
    Windows may retain explicit SYSTEM, local Administrators and OWNER RIGHTS
    FullControl ACEs; these well-known SIDs preserve normal Windows custody.
    Path is passed as a Base64 UTF-16LE literal, never PowerShell source text.
    """
    encoded_path = base64.b64encode(str(path).encode("utf-16le")).decode("ascii")
    source = rf"""
$ErrorActionPreference = 'Stop'
$path = [Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('{encoded_path}'))
$sid = '{sid}'
$acl = Get-Acl -LiteralPath $path
$owner = $acl.GetOwner([Security.Principal.SecurityIdentifier]).Value
$rules = @($acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier]))
if ($owner -ne $sid -or -not $acl.AreAccessRulesProtected -or $rules.Count -lt 1 -or
    $rules.Count -gt 4) {{ exit 21 }}
$allowed = @($sid, 'S-1-5-18', 'S-1-5-32-544', 'S-1-3-4')
$observed = @{{}}
foreach ($r in $rules) {{
    $principal = $r.IdentityReference.Value
    if ($allowed -notcontains $principal -or $observed.ContainsKey($principal) -or
        $r.IsInherited -or
        $r.AccessControlType -ne [Security.AccessControl.AccessControlType]::Allow -or
        ([int]($r.FileSystemRights -band [Security.AccessControl.FileSystemRights]::FullControl)) -ne
            [int][Security.AccessControl.FileSystemRights]::FullControl) {{ exit 22 }}
    if ({'$true' if directory else '$false'} -and
        ($r.InheritanceFlags -band [Security.AccessControl.InheritanceFlags]::ContainerInherit) -eq 0) {{ exit 23 }}
    $observed[$principal] = $true
}}
if (-not $observed.ContainsKey($sid)) {{ exit 24 }}
Write-Output 'R18_PRIVATE_ACL_PRIVILEGED_ONLY'
"""
    encoded_script = base64.b64encode(source.encode("utf-16le")).decode("ascii")
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded_script],
        capture_output=True, text=True, check=False)
    _check(result.returncode == 0 and "R18_PRIVATE_ACL_PRIVILEGED_ONLY" in result.stdout,
           "Windows private evaluation ACL has inherited/broad access")


def _protect_private_path(path: Path, *, directory: bool,
                          is_windows: bool | None = None) -> None:
    """Verify private custody on POSIX or enforce strict owner-only NTFS ACL."""
    if is_windows is None:
        is_windows = os.name == "nt"
    if not is_windows:
        expected = 0o700 if directory else 0o600
        _check(stat.S_IMODE(path.stat().st_mode) == expected,
               "private evaluator path lacks owner-only POSIX permissions")
        return
    sid = _windows_owner_sid()
    commands = [
        ["icacls", str(path), "/inheritance:r"],
        ["icacls", str(path), "/grant:r",
         f"*{sid}:(OI)(CI)F" if directory else f"*{sid}:F"],
        # Windows may make BUILTIN\Administrators the owner of newly created
        # paths even when the current user has an explicit FullControl ACE.
        # Set the owner to this process's actual SID before attesting custody.
        ["icacls", str(path), "/setowner", f"*{sid}"],
        ["icacls", str(path), "/verify"],
    ]
    for argv in commands:
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        _check(result.returncode == 0,
               "icacls could not restrict/verify private evaluation custody")
    _windows_verify_owner_only(path, sid, directory=directory)


def _create_private_json(path: Path, value: object) -> None:
    """Exclusive, mode-0600 writing for evaluation-custodian data only."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as writer:
        json.dump(value, writer, sort_keys=True, indent=2, ensure_ascii=False)
        writer.write("\n")
    _protect_private_path(path, directory=False)


def preregister_from_corpus(*, manifest: Path, old_r15_receipt: Path,
                            base_binding: Path, producer_git_head: str,
                            work_root: Path, private_output_dir: Path,
                            plan_output: Path, eval_source_manifest_output: Path | None = None,
                            expected_r16_selected_id_sha256: str | None = None,
                            cases: int, seed: int, training_budget_seconds: float,
                            probe_budget_seconds: float,
                            primary_sequence_length: int = 256) -> dict:
    """Trusted CPU-only custodian: new external 6-chain bank, v2 training fixed.

    Requires the independent R16 structural evaluator module from PR #138;
    the neural runners MUST NEVER import that module or read the resulting
    private eval-cases file. R15/R16 v2 exhaustion is verified and historical
    labels excluded; NEW R18 cases use a separately pinned external OOD source.
    """
    from trainingday.admission import _hash_file
    from trainingday.r18_eval_source import (
        generate_novel_r18_cases, source_manifest as new_source_manifest,
    )
    from trainingday.labyrinth_intake import admit_labyrinth
    from trainingday.train_neural import (
        R16_PRODUCER_HEAD, R16_V2_MANIFEST_SHA256, _required_base,
    )
    try:
        from trainingday.eval_structural_r16 import (
            BANK_ID_SHA256 as PINNED_R16_FRESH64_SHA256, _old_ids,
        )
        from trainingday.structural_eval import (
            R15_SCORED_SIGNATURE_COUNT, R15_V2_MANIFEST_SHA256,
            preregister_r15_fresh_structural64,
        )
    except ImportError as exc:
        raise BenchmarkError(
            "R18 custodian needs the independent PR #138 structural_eval.py; "
            "never import it from a neural runner") from exc

    _check(_hex(PINNED_R16_FRESH64_SHA256), "PR #138 R16 fresh64 pin absent")
    if expected_r16_selected_id_sha256 is not None:
        _check(expected_r16_selected_id_sha256 == PINNED_R16_FRESH64_SHA256,
               "user-supplied R16 selection pin differs from original R16 evaluator")
    _check(_hex(producer_git_head, 40) and producer_git_head == R16_PRODUCER_HEAD,
           "PR #140 R16 v2 corpus producer head changed")
    _check(cases == 1024 and _integer(seed),
           "R18 common training pool must be exactly 1024 admitted selected cases")
    manifest, base_binding, old_r15_receipt = (
        Path(manifest), Path(base_binding), Path(old_r15_receipt))
    work_root, private_output_dir, plan_output = (
        Path(work_root), Path(private_output_dir), Path(plan_output))
    eval_source_manifest_output = (Path(eval_source_manifest_output)
                                   if eval_source_manifest_output is not None
                                   else plan_output.parent / "r18-external-chain6-source-manifest.json")
    _check(manifest.is_file() and not manifest.is_symlink()
           and _hash_file(manifest) == R16_V2_MANIFEST_SHA256
           and R15_V2_MANIFEST_SHA256 == R16_V2_MANIFEST_SHA256,
           "R18 train/eval must share exact R16 v2 pinned corpus manifest")
    _check(work_root.is_dir() and not work_root.is_symlink()
           and private_output_dir.parent.is_dir()
           and not private_output_dir.parent.is_symlink()
           and not private_output_dir.exists() and not private_output_dir.is_symlink()
           and plan_output.parent.is_dir() and not plan_output.parent.is_symlink()
           and not plan_output.exists() and not plan_output.is_symlink()
           and eval_source_manifest_output.parent.is_dir()
           and not eval_source_manifest_output.parent.is_symlink()
           and not eval_source_manifest_output.exists()
           and not eval_source_manifest_output.is_symlink()
           and eval_source_manifest_output != plan_output,
           "private custody directory, external eval source, and public plan must be new")
    # This exact PR #138 helper enforces the original R15 base receipt byte
    # SHA f7a807..., the status/manifest/base provenance, and ordered 16 IDs.
    # No manual extraction or copying of evaluator-private IDs is permitted.
    old_ids = _old_ids(old_r15_receipt)

    # Both source admission and full base custody verification are CPU-only.
    admitted, selected = admit_labyrinth(
        manifest, work_root, sample_size=cases, producer_git_head=producer_git_head)
    _check(admitted.get("selected_count") == cases
           and admitted.get("status") == "ADMITTED_LABYRINTH_TRAIN_ONLY_NO_WEIGHT_UPDATE"
           and admitted.get("selected_id_sha256") == selected_id_digest(selected),
           "admission summary/selected case custody changed")
    base, _ = _required_base(base_binding)
    _check(base.get("model_id") == MODEL_ID and _hex(base.get("revision"), 40)
           and _hex(base.get("files", {}).get("tokenizer_config.json")),
           "base/tokenizer snapshot not pinned")

    # Sensitive structure/answer inspection happens only in this custodian.
    r16 = preregister_r15_fresh_structural64(
        manifest, previously_scored_16_ids=old_ids)
    _check(r16.get("selected_count") == 64
           and r16.get("selected_id_sha256") == PINNED_R16_FRESH64_SHA256
           and r16.get("signature_count") == 64,
           "fresh64 R16 exclusion selection differs from earlier scored bank")
    r16_cases = r16["cases"]
    r16_ids = [row["id"] for row in r16_cases]
    # The pinned v2 eval split has ZERO unseen signatures after the previously
    # scored R15+R16 banks. Never relabel those structures as fresh. Instead,
    # independently generate 64 six-operation compositions (the admitted v2
    # reasoning oracle permits only 2–5 steps) with their own source custody.
    result = subprocess.run(["git", "rev-parse", "HEAD"],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, check=False)
    _check(result.returncode == 0 and _hex(result.stdout.strip(), 40),
           "external evaluator generator source Git commit unavailable")
    # A public fixed seed would expose all evaluation labels. This private
    # 256-bit nonce is generated once, committed publicly only by SHA, and
    # unsealed solely to the independent scorer after training arms freeze.
    private_nonce_hex = secrets.token_hex(32)
    new_cases = generate_novel_r18_cases(private_nonce_hex=private_nonce_hex)
    eval_source = new_source_manifest(
        new_cases, private_nonce_hex=private_nonce_hex,
        v2_manifest_sha256=R16_V2_MANIFEST_SHA256,
        producer_git_head=result.stdout.strip())
    source_sha = canonical_sha256(eval_source)
    _check(len(new_cases) == 64 and all(case["id"] not in set(old_ids + r16_ids)
                                        for case in new_cases),
           "external six-chain eval overlaps previously scored IDs")
    eval_index = [{"id": case["id"], "prompt_sha256": _prompt_sha(case["prompt"])}
                  for case in new_cases]
    # These commitments bind *private answers* without copying them into the
    # public plan. The independent scorer can later reopen the private file.
    r16_custody_sha = canonical_sha256({"manifest": R16_V2_MANIFEST_SHA256,
                                        "cases": r16_cases})
    eval_custody_sha = canonical_sha256({"source_manifest_sha256": source_sha,
                                         "cases": new_cases})
    inputs = {
        "selected_rows": selected,
        "admission_selected_id_sha256": admitted["selected_id_sha256"],
        "source_admission_status": admitted["status"],
        "eval_index": eval_index, "r16_holdout_ids": r16_ids,
        "base_revision": base["revision"],
        "base_binding_sha256": _hash_file(base_binding),
        "tokenizer_binding_sha256": base["files"]["tokenizer_config.json"],
        "corpus_manifest_sha256": _hash_file(manifest),
        "eval_custody_sha256": eval_custody_sha,
        "eval_source_manifest_sha256": source_sha,
        "r16_eval_custody_sha256": r16_custody_sha,
        "seed": seed, "training_budget_seconds": training_budget_seconds,
        "probe_budget_seconds": probe_budget_seconds,
        "primary_sequence_length": primary_sequence_length,
    }
    plan = make_plan(**inputs)
    # Keep independent scorer secrets in a new 0700 directory. No output may
    # overwrite an earlier sealed bank or preregistration; never print labels.
    private_output_dir.mkdir(mode=0o700, exist_ok=False)
    # NTFS ignores the POSIX mode bits; lock and verify its DACL *before*
    # writing any label. Children then inherit only the owner's full rights.
    _protect_private_path(private_output_dir, directory=True)
    _create_private_json(private_output_dir / "r18-plan-inputs.json", inputs)
    _create_private_json(private_output_dir / "r18-evaluator-sealed-private.json", {
        "schema": "VERA_TRAININGDAY_R18_PRIVATE_EVAL_BANK_V1",
        "manifest_sha256": R16_V2_MANIFEST_SHA256,
        "private_nonce_hex": private_nonce_hex,
        "eval_source_manifest_sha256": source_sha,
        "eval_custody_sha256": eval_custody_sha,
        "eval_index_sha256": plan["eval_index_sha256"],
        "cases": new_cases,
    })
    _create_private_json(private_output_dir / "r18-exclusions-private.json", {
        "schema": "VERA_TRAININGDAY_R18_PRIVATE_EVAL_EXCLUSIONS_V1",
        "old_r15_scored16_ids": old_ids, "r16_fresh64_ids": r16_ids,
        "r16_selected_id_sha256": PINNED_R16_FRESH64_SHA256,
        "excluded_signature_count": R15_SCORED_SIGNATURE_COUNT + 64,
    })
    _create_private_json(eval_source_manifest_output, eval_source)
    _create_private_json(plan_output, plan)
    return {"status": "R18_SOURCE_ONLY_PREREGISTERED_EVAL_UNTOUCHED_BY_MODEL",
            "plan_sha256": validate_plan(plan),
            "selected_train_count": plan["selected_count"],
            "selected_train_id_sha256": plan["selected_id_sha256"],
            "r16_exclusion_id_sha256": plan["r16_holdout_id_sha256"],
            "r18_eval_count": plan["eval_examples"],
            "r18_eval_index_sha256": plan["eval_index_sha256"],
            "r18_eval_custody_sha256": plan["eval_custody_sha256"],
            "r18_eval_source_manifest_sha256": plan["eval_source_manifest_sha256"],
            "r18_eval_source_protocol": "R18_EXTERNAL_LONG_CHAIN64_OOD_V1",
            "r18_eval_source_manifest_output": str(eval_source_manifest_output),
            "plan_output": str(plan_output),
            "private_output_dir": str(private_output_dir)}


def main(argv: Sequence[str] | None = None) -> None:
    """Custodian CLI: no model loading, no network, no GPU allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    register = modes.add_parser("preregister", help="Create immutable R18 source-only plan")
    register.add_argument("--inputs", type=Path, required=True,
                          help="JSON object with make_plan keyword arguments")
    register.add_argument("--output", type=Path, required=True)
    from_corpus = modes.add_parser("preregister-from-corpus",
                                   help="CPU-only Lappy custodian: admit1024 and seal fresh R18 64")
    from_corpus.add_argument("--manifest", type=Path, required=True)
    from_corpus.add_argument("--old-r15-receipt", type=Path, required=True)
    from_corpus.add_argument("--base-binding", type=Path, required=True)
    from_corpus.add_argument("--producer-git-head", required=True)
    from_corpus.add_argument("--work-root", type=Path, required=True)
    from_corpus.add_argument("--private-output-dir", type=Path, required=True)
    from_corpus.add_argument("--plan-output", type=Path, required=True)
    from_corpus.add_argument("--eval-source-manifest-output", type=Path,
                             help="New public manifest pinning independent 6-op evaluation source")
    from_corpus.add_argument("--expected-r16-selected-id-sha256",
                             help="Optional extra check against PR #138 frozen R16 bank pin")
    from_corpus.add_argument("--cases", type=int, default=1024)
    from_corpus.add_argument("--seed", type=int, default=41)
    from_corpus.add_argument("--training-budget-seconds", type=float, required=True)
    from_corpus.add_argument("--probe-budget-seconds", type=float, required=True)
    from_corpus.add_argument("--primary-sequence-length", type=int, default=256)
    verify = modes.add_parser("verify-plan", help="Verify pinned plan and print its digest")
    verify.add_argument("--plan", type=Path, required=True)
    fit = modes.add_parser("verify-fit", help="Verify full predeclared fit sweep")
    fit.add_argument("--plan", type=Path, required=True)
    fit.add_argument("--probes", type=Path, required=True)
    receipt = modes.add_parser("verify-receipt", help="Verify an R18 arm receipt")
    receipt.add_argument("--plan", type=Path, required=True)
    receipt.add_argument("--receipt", type=Path, required=True)
    receipt.add_argument("--require-eval", action="store_true")
    comparison = modes.add_parser("compare", help="Compare two sealed-evaluated 4-arm runs")
    comparison.add_argument("--plan", type=Path, required=True)
    comparison.add_argument("--probes", type=Path, required=True)
    comparison.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "preregister-from-corpus":
        output = preregister_from_corpus(
            manifest=args.manifest, old_r15_receipt=args.old_r15_receipt,
            base_binding=args.base_binding, producer_git_head=args.producer_git_head,
            work_root=args.work_root, private_output_dir=args.private_output_dir,
            plan_output=args.plan_output,
            eval_source_manifest_output=args.eval_source_manifest_output,
            expected_r16_selected_id_sha256=args.expected_r16_selected_id_sha256,
            cases=args.cases, seed=args.seed,
            training_budget_seconds=args.training_budget_seconds,
            probe_budget_seconds=args.probe_budget_seconds,
            primary_sequence_length=args.primary_sequence_length)
    elif args.command == "preregister":
        input_data = _json_file(args.inputs)
        _check(isinstance(input_data, dict), "preregister inputs must be an object")
        plan = make_plan(**input_data)
        _check(args.output.parent.is_dir() and not args.output.parent.is_symlink()
               and not args.output.exists() and not args.output.is_symlink(),
               "plan output must be new inside a real existing directory")
        # Exclusive creation means an existing plan can never be silently replaced.
        with args.output.open("x", encoding="utf-8") as writer:
            json.dump(plan, writer, indent=2, ensure_ascii=False, sort_keys=True)
            writer.write("\n")
        output = {"plan_sha256": validate_plan(plan), "output": str(args.output)}
    else:
        plan = _json_file(args.plan)
        if args.command == "verify-plan":
            output = {"plan_sha256": validate_plan(plan), "selected_count":
                      plan["selected_count"], "fresh_eval_count": plan["eval_examples"]}
        elif args.command == "verify-fit":
            output = {"plan_sha256": validate_plan(plan), "selected_fit":
                      validate_fit_probes(plan, _json_file(args.probes))}
        elif args.command == "verify-receipt":
            receipt_data = _json_file(args.receipt)
            validate_arm_receipt(receipt_data, plan, allow_unscored=not args.require_eval)
            output = {"plan_sha256": validate_plan(plan),
                      "arm": receipt_data["arm"], "control": receipt_data["control"],
                      "status": "VALIDATED_SOURCE_RECEIPT_NOT_GPU_ATTESTED"}
        else:
            output = compare_arms(plan, _json_file(args.receipts), _json_file(args.probes))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
