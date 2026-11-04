"""Predeclared optimizer-step curve on public synthetic tiny Qwen3.5.

This deliberately tests a different question from the original four-step
20-seed replication: does extending optimizer exposure from 4 to 16 steps
separate correct from frequency-preserving shuffled feedback?  It cannot
establish effectiveness of official pretrained Qwen3.5-4B or R201 admission.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import Counter

from trainingday import qwen35_tiny_heldout_eval as baseline
from trainingday import qwen35_full32_hybrid_peft_cpu_smoke as smoke


SCHEMA = "TRAININGDAY_TINY_QWEN35_OPTIMIZER_CURVE_V1"
# Distinct from the predeclared 20-seed four-step study. This is a *two-seed
# mechanistic pilot*, not a powered statistical estimate of generalization.
SEEDS = (35060, 35061)
CHECKPOINT_STEPS = (0, 4, 16)
BATCH_SIZE = 4


def _assigned_pairs(seed: int, train_pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Cycle a seeded fixed order; never use a withheld pair for optimization."""
    if len(set(train_pairs)) != len(train_pairs) or len(train_pairs) < 25:
        raise AssertionError("invalid unique train-pair source")
    ordered = sorted(train_pairs, key=lambda pair: hashlib.sha256(
        f"curve-order-v1:{seed}:{pair[0]}:{pair[1]}".encode("ascii")
    ).digest())
    return [ordered[i % len(ordered)] for i in range(CHECKPOINT_STEPS[-1] * BATCH_SIZE)]


def _assigned_answers(
    seed: int, assigned: list[tuple[int, int]], *, shuffled: bool
) -> tuple[list[int], int]:
    answers = [sum(pair) for pair in assigned]
    if shuffled:
        # Repeated presentations of one operand pair MUST receive the same
        # shuffled target. Shuffle unique pairs only within equal occurrence-
        # count groups; this preserves both pair consistency and the *exact*
        # occurrence-weighted marginal answer-frequency multiset.
        counts = Counter(assigned)
        by_count: dict[int, list[tuple[int, int]]] = {}
        for pair, count in counts.items():
            by_count.setdefault(count, []).append(pair)
        assigned_target: dict[tuple[int, int], int] = {}
        for count in sorted(by_count):
            group = sorted(by_count[count])
            permuted = list(group)
            random.Random(seed + 9901 + count).shuffle(permuted)
            assigned_target.update({pair: sum(other)
                                    for pair, other in zip(group, permuted)})
        altered = [assigned_target[pair] for pair in assigned]
        if sorted(altered) != sorted(answers):
            raise AssertionError("stable shuffled feedback changed target multiset")
        answers = altered
    changed = sum(answer != sum(pair) for pair, answer in zip(assigned, answers))
    if shuffled and changed < len(assigned) // 2:
        raise AssertionError("shuffled feedback too similar to correct labels")
    if not shuffled and changed:
        raise AssertionError("correct feedback unexpectedly changed")
    return answers, changed


def _exposure_domain(assigned: list[tuple[int, int]], steps: int
                     ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Partition the fixed schedule into *ever-exposed* and never-exposed pairs.

    This is a descriptive optimizer-exposure partition, NOT a protected DEV
    split.  Repeated pair occurrences remain exposed after first consumption.
    """
    if not isinstance(steps, int) or steps not in CHECKPOINT_STEPS:
        raise AssertionError("unsupported checkpoint")
    if len(assigned) != CHECKPOINT_STEPS[-1] * BATCH_SIZE:
        raise AssertionError("invalid fixed learning schedule")
    consumed = set(assigned[:steps * BATCH_SIZE])
    scheduled = set(assigned)
    exposed, unexposed = sorted(consumed), sorted(scheduled - consumed)
    if set(exposed) & set(unexposed) or len(exposed) + len(unexposed) != len(scheduled):
        raise AssertionError("optimizer exposure partition invalid")
    return exposed, unexposed


def _prefix_feedback_histogram(labels: list[int], steps: int) -> dict[str, int]:
    """Count targets consumed *so far*; whole-schedule balance is insufficient."""
    if (not isinstance(steps, int) or steps not in CHECKPOINT_STEPS or
            len(labels) != CHECKPOINT_STEPS[-1] * BATCH_SIZE):
        raise AssertionError("invalid optimizer feedback histogram source")
    if any(not isinstance(x, int) or isinstance(x, bool) or not 0 <= x <= 18
           for x in labels):
        raise AssertionError("invalid optimizer feedback label")
    return {str(answer): count for answer, count in
            sorted(Counter(labels[:steps * BATCH_SIZE]).items())}


def _histogram_l1(correct: dict[str, int], shuffled: dict[str, int],
                  expected_count: int) -> int:
    """L1 count distance of cumulative optimizer target frequencies."""
    for histogram in (correct, shuffled):
        if (not isinstance(histogram, dict) or
                any(not isinstance(k, str) or not k.isdecimal() or
                    not isinstance(v, int) or isinstance(v, bool) or v <= 0
                    for k, v in histogram.items()) or
                sum(histogram.values()) != expected_count):
            raise AssertionError("invalid optimizer prefix histogram")
    return sum(abs(correct.get(key, 0) - shuffled.get(key, 0))
               for key in correct.keys() | shuffled.keys())


def _arm(torch, seed: int, train_pairs: list[tuple[int, int]],
         withheld_pairs: list[tuple[int, int]], *, shuffled: bool) -> dict:
    model = baseline._model(torch, seed)
    if set(train_pairs) & set(withheld_pairs):
        raise AssertionError("heldout pair leaked into optimizer source")
    assigned = _assigned_pairs(seed, train_pairs)
    labels, changed = _assigned_answers(seed, assigned, shuffled=shuffled)
    # Fixed full-schedule domain. At step 4 this includes operand pairs not
    # encountered by the optimizer yet: it is NOT a seen-so-far measurement.
    scheduled_train_domain = sorted(set(assigned))
    if len(scheduled_train_domain) < 25:
        raise AssertionError("curve did not cover training operand domain")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-3)
    frozen_embedding = smoke._hash_tensors(
        {"embedding": model.base_model.model.model.embed_tokens.weight}
    )
    old_lora_b = smoke._hash_tensors({
        name: p for name, p in model.named_parameters() if ".lora_B." in name
    })
    points: list[dict] = []
    for step in range(CHECKPOINT_STEPS[-1] + 1):
        if step in CHECKPOINT_STEPS:
            exposed, unexposed = _exposure_domain(assigned, step)
            points.append({
                "steps": step,
                "scheduled_train_domain_true_answer_nll": baseline._answer_nll(
                    model, scheduled_train_domain, torch),
                "heldout_answer_nll": baseline._answer_nll(model, withheld_pairs, torch),
                "exposed_train_pair_count": len(exposed),
                "unexposed_train_pair_count": len(unexposed),
                "exposed_train_domain_true_answer_nll": (
                    baseline._answer_nll(model, exposed, torch) if exposed else None),
                "unexposed_train_domain_true_answer_nll": (
                    baseline._answer_nll(model, unexposed, torch) if unexposed else None),
                "optimizer_consumed_feedback_histogram": _prefix_feedback_histogram(
                    labels, step),
            })
        if step == CHECKPOINT_STEPS[-1]:
            break
        start = step * BATCH_SIZE
        items = [baseline._row(pair, answer_override=label)
                 for pair, label in zip(assigned[start:start + BATCH_SIZE],
                                        labels[start:start + BATCH_SIZE])]
        ids = torch.tensor([row[0] for row in items], dtype=torch.long)
        target = torch.tensor([row[1] for row in items], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        loss = model(input_ids=ids, labels=target, use_cache=False).loss
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite optimizer-step loss")
        loss.backward()
        optimizer.step()
    if frozen_embedding != smoke._hash_tensors(
        {"embedding": model.base_model.model.model.embed_tokens.weight}
    ):
        raise AssertionError("frozen embedding changed")
    if old_lora_b == smoke._hash_tensors({
        name: p for name, p in model.named_parameters() if ".lora_B." in name
    }):
        raise AssertionError("LoRA-B remained unchanged")
    return {"checkpoints": points, "changed_feedback": changed,
            "unique_scheduled_train_domain_pairs": len(scheduled_train_domain),
            "optimizer_steps": CHECKPOINT_STEPS[-1]}


def run() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    import torch

    start = time.monotonic()
    torch.set_num_threads(2)
    records = []
    for seed in SEEDS:
        train_pairs, heldout = baseline._pairs(seed)
        correct = _arm(torch, seed, train_pairs, heldout, shuffled=False)
        shuffled = _arm(torch, seed, train_pairs, heldout, shuffled=True)
        paired = []
        for a, b in zip(correct["checkpoints"], shuffled["checkpoints"]):
            if a["steps"] != b["steps"]:
                raise AssertionError("checkpoint alignment lost")
            if a["steps"] == 0 and any(abs(a[key] - b[key]) > 1e-6 for key in
                                       ("scheduled_train_domain_true_answer_nll", "heldout_answer_nll")):
                raise AssertionError("paired arms differ before training")
            row = {"steps": a["steps"],
                   "heldout_advantage_nats": b["heldout_answer_nll"] - a["heldout_answer_nll"],
                   "scheduled_train_domain_advantage_nats": (
                       b["scheduled_train_domain_true_answer_nll"] -
                       a["scheduled_train_domain_true_answer_nll"])}
            frequency_l1 = _histogram_l1(
                a["optimizer_consumed_feedback_histogram"],
                b["optimizer_consumed_feedback_histogram"],
                a["steps"] * BATCH_SIZE)
            row["optimizer_prefix_feedback_frequency_l1_count"] = frequency_l1
            row["optimizer_prefix_feedback_marginals_matched"] = frequency_l1 == 0
            for domain in ("exposed", "unexposed"):
                if a[f"{domain}_train_pair_count"] != b[f"{domain}_train_pair_count"]:
                    raise AssertionError("paired arms differ in exposure partition")
                k = f"{domain}_train_domain_true_answer_nll"
                if (a[k] is None) != (b[k] is None):
                    raise AssertionError("paired arms differ in exposure metric availability")
                row[f"{domain}_train_pair_count"] = a[f"{domain}_train_pair_count"]
                row[f"{domain}_train_domain_advantage_nats"] = (
                    b[k] - a[k] if a[k] is not None else None)
                if a["steps"] == 0 and a[k] is not None and abs(a[k] - b[k]) > 1e-6:
                    raise AssertionError("paired exposure arms differ before training")
            paired.append(row)
        records.append({
            "seed": seed, "train_count": len(train_pairs), "heldout_count": len(heldout),
            "train_digest": hashlib.sha256(json.dumps(train_pairs).encode()).hexdigest(),
            "heldout_digest": hashlib.sha256(json.dumps(heldout).encode()).hexdigest(),
            "correct": correct, "shuffled": shuffled, "paired": paired,
        })
    if torch.cuda.is_initialized():
        raise AssertionError("CUDA initialized during CPU-only study")
    return {"schema": SCHEMA, "status": "MEASURED_NO_QUALITY_CLAIM",
            "architecture": "RANDOM_TINY_QWEN35_32L_152_LORA_NOT_PRETRAINED_4B",
            "train_data_source": "PUBLIC_SYNTHETIC_ONLY",
            "protected_dev_used": False, "official_pretrained_weights_used": False,
            "heldout_gradients_used": False, "training_authorized": False,
            "gpu_requested": False, "seeds": list(SEEDS),
            "optimizer_step_checkpoints": list(CHECKPOINT_STEPS),
            "records": records, "cpu_elapsed_seconds": time.monotonic() - start,
            "interpretation": (
                "Both signs retained; no best-step/seed selection. The fixed "
                "scheduled training domain includes future optimizer pairs at "
                "early checkpoints; it is not a seen-so-far memorization measure. "
                "Separate exposed/unexposed training-domain NLL partitions the "
                "schedule by actual optimizer exposure at each checkpoint. "
                "Unexposed scheduled examples are not independently heldout, "
                "and both empty-domain NLLs are null rather than zero. "
                "Shuffled feedback preserves the full 64-example target "
                "frequency multiset, NOT necessarily the first 4 or 16 "
                "consumed labels; the reported prefix L1 count distance "
                "exposes this early-checkpoint frequency confound."
            )}


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
