"""Real TRL completion masks -> tiny random Qwen3.5 full32 PEFT -> reload.

This is a public, CPU-only engineering test. The tokenizer is from pinned
Qwen3.5-4B, but no pretrained weights or protected curriculum are read.
The eight arithmetic prompts are generated below, solely for this experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from trainingday import qwen35_full32_hybrid_peft_cpu_smoke as smoke


SCHEMA = "TRAININGDAY_QWEN35_FULL32_TINY_REAL_TRL_ADAPTER_V1"
TOKENIZER_SHA = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
TOKENIZER_VOCAB = 248320  # Official model config, not tokenizer.get_vocab_size().
EOS_ID = 248046
PAD_ID = 248044
BASE_SEED = 35032
ARITHMETIC = ((9, 4), (6, -11), (12, 8), (3, -7),
              (-8, -5), (22, 7), (11, -19), (17, -3))


def _offline_cpu() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_DISABLED"] = "true"


def _sha_tensor(tensor) -> str:
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def _construct_base(torch):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(BASE_SEED)
    config = Qwen3_5TextConfig(
        vocab_size=TOKENIZER_VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=32, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, layer_types=list(smoke.LAYER_TYPES),
        linear_conv_kernel_dim=2, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, max_position_embeddings=128,
        bos_token_id=248045, eos_token_id=EOS_ID, pad_token_id=PAD_ID,
        rope_parameters={"rope_type": "default", "rope_theta": 10000,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    )
    model = Qwen3_5ForCausalLM(config).to("cpu")
    if list(model.config.layer_types) != list(smoke.LAYER_TYPES):
        raise AssertionError("wrong hybrid layer pattern")
    return model


def _load_tokenizer(path: Path):
    from tokenizers import Tokenizer

    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TOKENIZER_SHA:
        raise AssertionError("official pinned tokenizer digest mismatch")
    tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
    if tokenizer.token_to_id("<|im_end|>") != EOS_ID or tokenizer.token_to_id("<|endoftext|>") != PAD_ID:
        raise AssertionError("official EOS/PAD token IDs changed")
    return tokenizer


def _synthetic_rows(raw_tokenizer) -> list[dict]:
    examples = []
    for left, right in ARITHMETIC:
        prefix = f"Q: {left} + {right} = ?\nA:"
        text = f"{prefix} {left + right}"
        prefix_ids = raw_tokenizer.encode(prefix, add_special_tokens=False).ids
        all_ids = raw_tokenizer.encode(text, add_special_tokens=False).ids
        if all_ids[:len(prefix_ids)] != prefix_ids:
            raise AssertionError("token boundary merged across synthetic completion")
        ids = all_ids + [EOS_ID]
        if not len(prefix_ids) < len(ids) <= 64:
            raise AssertionError("synthetic sequence not bounded")
        if max(ids) >= TOKENIZER_VOCAB or min(ids) < 0:
            raise AssertionError("token ID exceeds faithful model vocabulary")
        completion = [0] * len(prefix_ids) + [1] * (len(ids) - len(prefix_ids))
        examples.append({"input_ids": ids, "completion_mask": completion})
    if len(examples) != 8 or not any(x + y < 0 for x, y in ARITHMETIC):
        raise AssertionError("synthetic arithmetic set changed")
    return examples


def _adapter_b_hash(model):
    b_params = {name: p for name, p in model.named_parameters() if ".lora_B." in name}
    if len(b_params) != 152:
        raise AssertionError(f"wrong LoRA-B count {len(b_params)}")
    return smoke._hash_tensors(b_params)


def _probe_logits(model, ids, torch):
    model.eval()
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


def _child(adapter_dir: Path, probe_path: Path) -> dict:
    _offline_cpu()
    import torch
    from peft import PeftModel

    torch.set_num_threads(2)
    ids = torch.tensor([json.loads(probe_path.read_text(encoding="utf-8"))], dtype=torch.long)
    base = _construct_base(torch)
    base_embed_sha = _sha_tensor(base.model.embed_tokens.weight)
    base_logits_sha = _sha_tensor(_probe_logits(base, ids, torch))
    loaded = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    loaded_logits_sha = _sha_tensor(_probe_logits(loaded, ids, torch))
    if loaded_logits_sha == base_logits_sha:
        raise AssertionError("reloaded adapter has no measurable effect")
    if torch.cuda.is_initialized():
        raise AssertionError("child CUDA initialized")
    return {"pid": os.getpid(), "adapter_sha": _adapter_b_hash(loaded),
            "logits_sha": loaded_logits_sha, "base_logits_sha": base_logits_sha,
            "base_embed_sha": base_embed_sha, "cuda_initialized": False}


def run() -> dict:
    _offline_cpu()
    start = time.monotonic()
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import PreTrainedTokenizerFast
    from trl import SFTConfig, SFTTrainer

    torch.set_num_threads(2)
    tokenizer_path = Path(os.environ["TRAININGDAY_QWEN_TOKENIZER_JSON"])
    raw = _load_tokenizer(tokenizer_path)
    rows = _synthetic_rows(raw)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path), eos_token="<|im_end|>", pad_token="<|endoftext|>",
    )
    if tokenizer.eos_token_id != EOS_ID or tokenizer.pad_token_id != PAD_ID:
        raise AssertionError("tokenizer special ids differ")
    base = _construct_base(torch)
    base_embed_sha = _sha_tensor(base.model.embed_tokens.weight)
    probe_ids = torch.tensor([rows[0]["input_ids"]], dtype=torch.long)
    parent_base_logits_sha = _sha_tensor(_probe_logits(base, probe_ids, torch))
    targets = smoke.expected_target_modules()
    if len(set(targets)) != 152:
        raise AssertionError("wrong PEFT target set")
    model = get_peft_model(base, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    ))
    before_sha = _adapter_b_hash(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n for n, p in model.named_parameters() if p.requires_grad):
        raise AssertionError("unintended trainable base parameters")

    # All eight synthetic examples pass through real SFTTrainer preprocessing;
    # only the first dataloader batch takes an optimizer step to bound CPU RAM.
    training_rows = rows
    with tempfile.TemporaryDirectory(prefix="trainingday-qwen35-trl-integrated-") as tmp:
        args = SFTConfig(
            output_dir=tmp, max_length=64, per_device_train_batch_size=2,
            completion_only_loss=True, packing=False, report_to="none", seed=42,
            fp16=False, bf16=False, dataloader_num_workers=0,
        )
        trainer = SFTTrainer(
            model=model, args=args, train_dataset=Dataset.from_list(training_rows),
            processing_class=tokenizer,
        )
        observed = []
        first_batch = None
        for candidate_batch in trainer.get_train_dataloader():
            if first_batch is None:
                first_batch = candidate_batch
            observed.extend(
                (candidate_batch["input_ids"][j, :int(candidate_batch["attention_mask"][j].sum())].tolist(),
                 candidate_batch["labels"][j, :int(candidate_batch["attention_mask"][j].sum())].tolist())
                for j in range(candidate_batch["input_ids"].shape[0])
            )
        observed.sort()
        if first_batch is None:
            raise AssertionError("TRL returned an empty dataloader")
        expected = sorted((row["input_ids"],
                           [t if m else -100 for t, m in zip(row["input_ids"], row["completion_mask"])])
                          for row in training_rows)
        if observed != expected:
            raise AssertionError("actual SFTTrainer completion-only labels differ")
        if not all(row[-1] == EOS_ID for _, row in observed):
            raise AssertionError("EOS not supervised")
        if any(row[0] != -100 for _, row in observed):
            raise AssertionError("prompt unexpectedly supervised")
        if any(-100 not in labels or sum(x != -100 for x in labels) < 2
               for _, labels in observed):
            raise AssertionError("some synthetic rows lack prompt masks or answer supervision")

        model.train()
        optimizer = torch.optim.AdamW(trainable, lr=5e-3)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=first_batch["input_ids"],
                     attention_mask=first_batch["attention_mask"],
                     labels=first_batch["labels"], use_cache=False).loss
        if loss is None or not torch.isfinite(loss):
            raise AssertionError("nonfinite integrated masked training loss")
        loss.backward()
        for layer in range(32):
            grads = [p.grad for name, p in model.named_parameters()
                     if f".model.layers.{layer}." in name and ".lora_B." in name]
            if len(grads) != (5 if smoke.LAYER_TYPES[layer] == "linear_attention" else 4):
                raise AssertionError(f"missing LoRA-B modules layer={layer}")
            if any(g is None or not torch.isfinite(g).all() for g in grads):
                raise AssertionError(f"nonfinite or missing gradient layer={layer}")
            if sum(float(g.abs().sum()) for g in grads) == 0:
                raise AssertionError(f"zero LoRA-B gradient layer={layer}")
        optimizer.step()
        after_sha = _adapter_b_hash(model)
        if after_sha == before_sha:
            raise AssertionError("integrated optimizer did not update LoRA-B")
        trained_logits_sha = _sha_tensor(_probe_logits(model, probe_ids, torch))
        if _sha_tensor(model.base_model.model.model.embed_tokens.weight) != base_embed_sha:
            raise AssertionError("frozen embedding changed")
        if torch.cuda.is_initialized() or any(p.device.type != "cpu" for p in model.parameters()):
            raise AssertionError("CPU-only constraint violated")

        adapter_dir = Path(tmp) / "adapter-v1"
        model.save_pretrained(str(adapter_dir), safe_serialization=True)
        if not {"adapter_model.safetensors", "adapter_config.json"}.issubset(
            {p.name for p in adapter_dir.iterdir()}):
            raise AssertionError("PEFT adapter format missing")
        probe_path = Path(tmp) / "probe.json"
        probe_path.write_text(json.dumps(rows[0]["input_ids"]), encoding="utf-8")
        child = subprocess.run(
            [sys.executable, "-m", "trainingday.qwen35_full32_trl_integrated_cpu",
             "--child", str(adapter_dir), str(probe_path)],
            check=False, capture_output=True, text=True, timeout=120,
            env=os.environ.copy(),
        )
        if child.returncode != 0:
            raise AssertionError(f"child reload failed: {child.stderr[-1000:]}")
        receipt = json.loads(child.stdout.splitlines()[-1])
        if receipt["pid"] == os.getpid():
            raise AssertionError("reload did not use a separate process")
        if (receipt["adapter_sha"] != after_sha or receipt["logits_sha"] != trained_logits_sha or
                receipt["base_embed_sha"] != base_embed_sha):
            raise AssertionError("fresh-process adapter/model hash parity failed")
        if receipt["base_logits_sha"] != parent_base_logits_sha:
            raise AssertionError("fresh-process base-model output differs before LoRA load")
        if receipt["base_logits_sha"] == trained_logits_sha or receipt["cuda_initialized"]:
            raise AssertionError("fresh-process base/adapted contrast failed")

        # An adapter loader must reject absent metadata in a fresh interpreter.
        missing_dir = Path(tmp) / "missing-adapter"
        missing_dir.mkdir()
        rejected = subprocess.run(
            [sys.executable, "-m", "trainingday.qwen35_full32_trl_integrated_cpu",
             "--child", str(missing_dir), str(probe_path)],
            check=False, capture_output=True, text=True, timeout=120,
            env=os.environ.copy(),
        )
        if rejected.returncode == 0:
            raise AssertionError("missing adapter was accepted by fresh process")

    try:
        import resource
        peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:  # Windows subprocess measurements are not comparable.
        peak_rss_kib = None
    return {
        "schema": SCHEMA, "status": "TINY_RANDOM_FULL32_QWEN35_TRL_MASKED_ADAPTER_RELOAD_PASS",
        "tokenizer_sha256": TOKENIZER_SHA, "synthetic_examples_masked_by_real_trainer": len(observed),
        "source_curriculum": "GENERATED_ARITHMETIC_ONLY_NO_R201_OR_DEV",
        "actual_optimizer_examples": 2, "model_config_vocab": TOKENIZER_VOCAB,
        "all_ids_unmodified": True, "trl_completion_mask_derived_labels": True,
        "hybrid_layers": 32, "lora_b_tensors": 152,
        "before_lora_b_sha256": before_sha, "after_and_reloaded_lora_b_sha256": after_sha,
        "trained_and_reloaded_logits_sha256": trained_logits_sha,
        "masked_training_loss": float(loss.item()),
        "elapsed_seconds": time.monotonic() - start, "process_peak_rss_kib": peak_rss_kib,
        "cuda_initialized": False, "official_pretrained_weights_loaded": False,
        "missing_adapter_rejected_in_fresh_process": True,
        "four_gb_gpu_fit": "UNKNOWN", "training_authorized": False,
    }


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        print(json.dumps(_child(Path(sys.argv[2]), Path(sys.argv[3])), sort_keys=True))
    elif len(sys.argv) == 1:
        print(json.dumps(run(), sort_keys=True))
    else:
        raise SystemExit("Usage: python -m trainingday.qwen35_full32_trl_integrated_cpu [--child ADAPTER_DIR PROBE_JSON]")
