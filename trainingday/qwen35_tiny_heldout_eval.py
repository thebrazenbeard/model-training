"""Honest withheld-outcome controls for a *random tiny* Qwen3.5 hybrid.

This is a small CPU experiment, NOT R201/DEV evaluation or authorized Qwen4B
training.  It uses generated addition tasks with operand-pair holdouts and
compares correct-feedback LoRA against a frozen base and shuffled feedback.
Loss is measured on the ANSWER token alone, so easy EOS learning cannot inflate
the apparent result.  Negative results are expected and must be retained.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

from trainingday import qwen35_full32_hybrid_peft_cpu_smoke as smoke


SCHEMA = "TRAININGDAY_QWEN35_TINY_HELDOUT_CONTROLS_V1"
VOCAB = 96
BOS, ADD, EQUAL, EOS = 1, 3, 4, 2
NUMBER_OFFSET = 20
# Predeclared 20-seed extension; retains the original five, without
# selecting seeds after observing their signed held-out outcomes.
SEEDS = tuple(range(35032, 35052))
STEPS = 4
BATCH_SIZE = 4


def _pairs(split_seed: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Hash partition unordered operand pairs; no commuted mirror leakage."""
    train, heldout = [], []
    for left in range(10):
        for right in range(left, 10):
            fingerprint = hashlib.sha256(
                f"withheld-generator-v1:{split_seed}:{left}:{right}".encode("ascii")
            ).digest()
            (heldout if fingerprint[0] % 4 == 0 else train).append((left, right))
    if not (25 <= len(train) <= 49 and 6 <= len(heldout) <= 24):
        raise AssertionError("bad held-out split size")
    if set(train) & set(heldout) or len(train) + len(heldout) != 55:
        raise AssertionError("train and withheld generators intersect")
    return train, heldout


def _row(pair: tuple[int, int], *, answer_override: int | None = None) -> tuple[list[int], list[int]]:
    left, right = pair
    answer = left + right if answer_override is None else answer_override
    if not (0 <= left <= 9 and left <= right <= 9 and 0 <= answer <= 18):
        raise AssertionError("invalid source or answer token")
    ids = [BOS, ADD, NUMBER_OFFSET + left, NUMBER_OFFSET + right,
           EQUAL, NUMBER_OFFSET + answer, EOS]
    # Labels at answer position ONLY.  No prefix or EOS leakage.
    labels = [-100, -100, -100, -100, -100, ids[5], -100]
    return ids, labels


def _model(torch, seed: int):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    config = Qwen3_5TextConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=32, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, layer_types=list(smoke.LAYER_TYPES),
        linear_conv_kernel_dim=2, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, max_position_embeddings=128,
        bos_token_id=BOS, eos_token_id=EOS, pad_token_id=0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    )
    base = Qwen3_5ForCausalLM(config).to("cpu")
    target_modules = smoke.expected_target_modules()
    if len(target_modules) != 152:
        raise AssertionError("wrong LoRA target count")
    model = get_peft_model(base, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=target_modules,
    ))
    if len([n for n, p in model.named_parameters() if p.requires_grad and ".lora_B." in n]) != 152:
        raise AssertionError("incomplete LoRA-B construction")
    return model


def _answer_nll(model, pairs: list[tuple[int, int]], torch) -> float:
    """Answer-only cross entropy, weighted by examples, never EOS tokens."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = [_row(pair) for pair in pairs[i:i + BATCH_SIZE]]
            ids = torch.tensor([x[0] for x in batch], dtype=torch.long)
            labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
            if not torch.all(labels[:, :5] == -100) or not torch.all(labels[:, 6] == -100):
                raise AssertionError("answer-only evaluation contaminated")
            loss = model(input_ids=ids, labels=labels, use_cache=False).loss
            if not torch.isfinite(loss):
                raise AssertionError("nonfinite held-out loss")
            total += float(loss.item()) * len(batch)
    return total / len(pairs)


def _run_arm(torch, *, seed: int, train_pairs: list[tuple[int, int]],
             withheld_pairs: list[tuple[int, int]], shuffled: bool) -> dict:
    model = _model(torch, seed)
    # The first BATCH_SIZE*STEPS training examples span the operand domain;
    # lexicographic source ordering would otherwise over-sample small inputs.
    train_pairs = sorted(train_pairs, key=lambda pair: hashlib.sha256(
        f"train-order-v1:{seed}:{pair[0]}:{pair[1]}".encode("ascii")
    ).digest())
    if len(train_pairs) < BATCH_SIZE * STEPS:
        raise AssertionError("not enough independent training pairs")
    if any(p.requires_grad and "lora_" not in name for name, p in model.named_parameters()):
        raise AssertionError("frozen base unexpectedly trainable")
    frozen_embedding_before = smoke._hash_tensors(
        {"embed": model.base_model.model.model.embed_tokens.weight}
    )
    before = _answer_nll(model, withheld_pairs, torch)
    # Diagnostic only: the 16 examples actually used for optimization.
    # This is NOT heldout generalization; evaluate the true labels in both arms.
    seen_pairs = train_pairs[:STEPS * BATCH_SIZE]
    before_seen = _answer_nll(model, seen_pairs, torch)
    train_answers = [left + right for left, right in train_pairs]
    source_answers = list(train_answers[:STEPS * BATCH_SIZE])
    if shuffled:
        # Shuffle exactly the labels used in the optimizer, preserving their
        # multiset; otherwise marginal target-frequency changes confound the
        # feedback-ablation comparison.
        order = torch.randperm(STEPS * BATCH_SIZE,
                               generator=torch.Generator().manual_seed(seed + 9901)).tolist()
        if all(i == j for i, j in enumerate(order)):
            raise AssertionError("shuffled control unexpectedly identity")
        train_answers[:STEPS * BATCH_SIZE] = [source_answers[j] for j in order]
        if sorted(train_answers[:STEPS * BATCH_SIZE]) != sorted(source_answers):
            raise AssertionError("shuffle changed target-frequency multiset")
    # Count only feedback on examples actually used by the four optimizer steps.
    changed_feedback_count = sum(answer != sum(pair)
                                 for pair, answer in list(zip(train_pairs, train_answers))[:STEPS * BATCH_SIZE])
    if shuffled and changed_feedback_count < STEPS * BATCH_SIZE // 2:
        raise AssertionError("too few altered feedback labels in shuffled control")
    if not shuffled and changed_feedback_count:
        raise AssertionError("correct-feedback generator altered its outcomes")
    assignments = list(zip(train_pairs, train_answers))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-3)
    old_b = smoke._hash_tensors({n: p for n, p in model.named_parameters() if ".lora_B." in n})
    losses = []
    for step in range(STEPS):
        # Fixed deterministic source-curriculum order for both feedback arms.
        begin = step * BATCH_SIZE
        batch = assignments[begin:begin + BATCH_SIZE]
        inputs = [_row(pair, answer_override=answer) for pair, answer in batch]
        ids = torch.tensor([x[0] for x in inputs], dtype=torch.long)
        labels = torch.tensor([x[1] for x in inputs], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        loss = model(input_ids=ids, labels=labels, use_cache=False).loss
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    new_b = smoke._hash_tensors({n: p for n, p in model.named_parameters() if ".lora_B." in n})
    if old_b == new_b:
        raise AssertionError("LoRA-B unchanged despite real optimizer steps")
    if frozen_embedding_before != smoke._hash_tensors(
        {"embed": model.base_model.model.model.embed_tokens.weight}
    ):
        raise AssertionError("frozen embedding changed during optimizer work")
    after = _answer_nll(model, withheld_pairs, torch)
    after_seen = _answer_nll(model, seen_pairs, torch)
    return {"before_answer_nll": before, "after_answer_nll": after,
            "seen_before_answer_nll": before_seen,
            "seen_after_answer_nll": after_seen,
            "seen_improvement": before_seen - after_seen,
            "improvement": before - after, "optimizer_steps": STEPS,
            "train_step_losses": losses, "lora_b_changed": True,
            "frozen_embedding_unchanged": True,
            "altered_feedback_labels": changed_feedback_count}


def run() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    import torch

    start = time.monotonic()
    torch.set_num_threads(2)
    results = []
    for seed in SEEDS:
        train_pairs, withheld_pairs = _pairs(seed)
        trained = _run_arm(torch, seed=seed, train_pairs=train_pairs,
                           withheld_pairs=withheld_pairs, shuffled=False)
        shuffled = _run_arm(torch, seed=seed, train_pairs=train_pairs,
                            withheld_pairs=withheld_pairs, shuffled=True)
        if abs(trained["before_answer_nll"] - shuffled["before_answer_nll"]) > 1e-6:
            raise AssertionError("controls differ before training")
        if abs(trained["seen_before_answer_nll"] - shuffled["seen_before_answer_nll"]) > 1e-6:
            raise AssertionError("seen controls differ before training")
        results.append({"seed": seed, "train_count": len(train_pairs),
                        "withheld_count": len(withheld_pairs),
                        "train_pair_digest": hashlib.sha256(json.dumps(train_pairs).encode()).hexdigest(),
                        "withheld_pair_digest": hashlib.sha256(json.dumps(withheld_pairs).encode()).hexdigest(),
                        "frozen_answer_nll": trained["before_answer_nll"],
                        "correct_feedback": trained, "shuffled_feedback": shuffled,
                        "advantage_vs_shuffled": shuffled["after_answer_nll"] - trained["after_answer_nll"],
                        "seen_advantage_vs_shuffled": shuffled["seen_after_answer_nll"] - trained["seen_after_answer_nll"]})
    if torch.cuda.is_initialized():
        raise AssertionError("CUDA initialized during CPU-only experiment")
    advantages = [row["advantage_vs_shuffled"] for row in results]
    improvements = [row["correct_feedback"]["improvement"] for row in results]
    seen_advantages = [row["seen_advantage_vs_shuffled"] for row in results]
    return {"schema": SCHEMA, "status": "HELDOUT_CONTROLS_MEASURED_NO_QUALITY_CLAIM",
            "architecture": "RANDOM_TINY_32L_QWEN35_24GDN_8ATTN_152_LORA_NOT_OFFICIAL_4B",
            "task": "generated_addition_answer_token_nll_with_unseen_unordered_operand_pairs",
            "train_data_source": "PUBLIC_SYNTHETIC_ONLY", "protected_dev_used": False,
            "heldout_gradients_used": False,
            "heldout_evaluator_independently_custodied": False,
            "official_pretrained_weights_used": False, "training_authorized": False,
            "gpu_execution_requested": False, "cuda_runtime_initialized": False,
            "seeds": list(SEEDS), "steps_per_feedback_arm": STEPS,
            "batch_size": BATCH_SIZE, "trained_examples_per_arm": STEPS * BATCH_SIZE,
            "correct_beats_shuffled_seeds": sum(x > 0 for x in advantages),
            "correct_improves_frozen_seeds": sum(x > 0 for x in improvements),
            "mean_advantage_vs_shuffled_nats": sum(advantages) / len(advantages),
            "correct_beats_shuffled_seen_seeds": sum(x > 0 for x in seen_advantages),
            "mean_seen_advantage_vs_shuffled_nats": sum(seen_advantages) / len(seen_advantages),
            "results": results,
            "cpu_elapsed_seconds": time.monotonic() - start,
            "interpretation": "Report signed NLL deltas; improvement is not presumed.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
