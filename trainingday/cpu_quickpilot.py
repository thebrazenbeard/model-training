"""Bounded CPU LoRA weight-update experiment on the pinned official Qwen3.5-4B.

This is an *alternative CPU experiment*, not the frozen NF4 GPU recipe.
Never use it to claim that a new model learned reasoning or passed a blind
benchmark. It can change neural parameters, save an adapter, and report bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from trainingday.train_neural import _control_rows, _parameter_digest, validate_run
from trainingday.admission import _hash_file


RECIPE = "TRAININGDAY_QWEN35_CPU_BF16_QV_LORA_V1"


def _check_cpu_memory(available_gib: float) -> bool:
    if available_gib < 14:
        raise RuntimeError("less than 14 GiB free system RAM for bounded CPU-only 4B load")
    return True


def _completion_tokens(tokenizer, prompt: str, answer: str, max_length: int):
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
    )
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": answer}], tokenize=True,
        add_generation_prompt=False, enable_thinking=False,
    )
    # Transformers 5.x returns BatchEncoding with input_ids rather than
    # the plain List[int] returned by earlier chat-template integrations.
    for_name = []
    for value in (prefix, full):
        if hasattr(value, "keys"):
            value = value["input_ids"]
        if isinstance(value, list) and value and isinstance(value[0], list):
            value = value[0]
        for_name.append(value)
    prefix, full = for_name
    if (not isinstance(prefix, list) or not isinstance(full, list)
            or full[:len(prefix)] != prefix
            or len(prefix) >= max_length or len(full) > max_length
            or not full[len(prefix):]):
        raise ValueError("completion-only token alignment/budget failed")
    return full, [-100] * len(prefix) + full[len(prefix):]


def _control_delta(mode: str, before: str, after: str) -> bool:
    changed = before != after
    expected_changed = mode in ("correct", "corrupt")
    if mode not in ("zero", "correct", "corrupt") or changed != expected_changed:
        raise RuntimeError("trainable weight changed unexpectedly / failed to change")
    return changed


def train_cpu(args) -> dict:
    """Bind all source/model hashes before allocating 4B frozen CPU parameters."""
    summary, cases, binding, model_path = validate_run(args)
    preflight = {
        "status": "CPU_SOURCE_ONLY_PREFLIGHT_NO_MODEL_LOADED",
        "recipe": RECIPE, "corpus": summary,
        "base_model": binding["model_id"], "base_revision": binding["revision"],
        "mode": args.mode, "steps": args.steps, "new_optimizer_steps": 0,
        "seed": args.seed,
        "training_input_sha256": hashlib.sha256(
            json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
    }
    if args.preflight_only:
        return preflight

    import psutil
    memory = psutil.virtual_memory()
    if memory.total < 30 * (2 ** 30):
        raise RuntimeError("CPU-only 4B load requires at least 30 GiB physical system RAM")
    _check_cpu_memory(memory.available / (2 ** 30))

    # No CUDA or bnb usage; BF16 frozen model + FP32 LoRA on CPU. The model
    # itself is read-only and will NEVER be saved/modified in this script.
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from peft import LoraConfig, get_peft_model

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if not tokenizer.chat_template:
        raise RuntimeError("missing pinned tokenizer chat template")
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map="cpu", low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        ),
    )
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("CPU LoRA has no trainable parameters")
    model.train()
    before, count = _parameter_digest(model, torch)
    if args.expected_initial_digest and before != args.expected_initial_digest:
        raise RuntimeError("initial CPU adapter digest mismatch vs zero arm")
    output = args.output_root / args.run_id
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    loss_log = []
    try:
        if args.mode != "zero":
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=args.learning_rate,
                                          weight_decay=0.0)
            for index in range(args.steps):
                row = cases[index]
                ids, labels = _completion_tokens(
                    tokenizer, row["prompt"], row["response"], max_length=128,
                )
                input_ids = torch.tensor([ids], dtype=torch.long)
                target = torch.tensor([labels], dtype=torch.long)
                mask = torch.ones_like(input_ids)
                optimizer.zero_grad(set_to_none=True)
                output_logits = model(input_ids=input_ids,
                                      attention_mask=mask, labels=target)
                loss = output_logits.loss
                if not torch.isfinite(loss):
                    raise RuntimeError("loss nonfinite")
                loss.backward()
                optimizer.step()
                loss_log.append(float(loss.detach()))
                print("CPU_STEP", index + 1, "LOSS", loss_log[-1], flush=True)
        after, count_after = _parameter_digest(model, torch)
        if count != count_after:
            raise RuntimeError("CPU LoRA trainable element count changed")
        _control_delta(args.mode, before, after)
        adapter = output / "adapter"
        model.save_pretrained(adapter, safe_serialization=True)
        weights = adapter / "adapter_model.safetensors"
        config = adapter / "adapter_config.json"
        if not weights.is_file() or not config.is_file():
            raise RuntimeError("CPU adapter output missing")
        receipt = {
            **preflight, "status": "CPU_BF16_LORA_SAVED_NOT_INDEPENDENTLY_RELOADED",
            "new_optimizer_steps": args.steps,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "trainable_parameter_count": count,
            "trainable_digest_before": before,
            "trainable_digest_after": after,
            "adapter_model_sha256": _hash_file(weights),
            "adapter_config_sha256": _hash_file(config),
            "base_binding_sha256": _hash_file(args.base_binding),
            "training_losses": loss_log, "torch_version": torch.__version__,
            "CPU_ONLY": True, "no_gguf_conversion": True,
            "real_model_reload_qualified": False,
            "reasoning_improvement_qualified": False,
        }
        (output / "training_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        return receipt
    except Exception:
        # Output remains quarantined for forensics; never reuse run ID.
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--producer-git-head", required=True)
    parser.add_argument("--base-binding", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=["zero", "correct", "corrupt"], required=True)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--seed", type=int, default=29034)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--expected-initial-digest", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train_cpu(args), indent=2))


if __name__ == "__main__":
    main()
