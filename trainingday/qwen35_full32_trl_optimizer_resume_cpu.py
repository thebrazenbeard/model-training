"""CPU-only optimizer resume witness for a *tiny random* hybrid Qwen3.5.

The actual pinned official tokenizer and a TRL completion-only dataloader feed
the same two real AdamW steps in (a) one process and (b) a persisted checkpoint
resumed in a separate interpreter. Neither run loads official pretrained 4B
weights or grants physical TRAININGDAY authority.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from trainingday import qwen35_full32_trl_integrated_cpu as integrated


SCHEMA = "TRAININGDAY_QWEN35_TINY_FULL32_TRL_OPTIMIZER_RESUME_V1"
LR = 0.005
VERSIONED_ADAPTER = "adapter-step000001-v1"
OPTIMIZER = "adamw-step000001.pt"
MANIFEST = "checkpoint-manifest-v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_batch_sha(batch) -> dict[str, str]:
    return {key: integrated._sha_integer_tensor(batch[key])
            for key in ("input_ids", "labels", "attention_mask")}


def _runtime_fingerprint(torch) -> dict:
    import transformers
    import trl
    import peft
    import tokenizers

    return {"torch": str(torch.__version__), "transformers": transformers.__version__,
            "trl": trl.__version__, "peft": peft.__version__,
            "tokenizers": tokenizers.__version__,
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "threads": torch.get_num_threads(), "mkldnn": torch.backends.mkldnn.enabled,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def _optimizer_digest(optimizer) -> str:
    """Hash optimizer moments/step counters and param group hyperparameters."""
    state = optimizer.state_dict()
    h = hashlib.sha256()
    h.update(json.dumps(state["param_groups"], sort_keys=True).encode("utf-8"))
    for i in sorted(state["state"]):
        h.update(str(i).encode("ascii"))
        for name, tensor in sorted(state["state"][i].items()):
            h.update(name.encode("ascii"))
            h.update(str(tensor.dtype).encode("ascii"))
            h.update(str(tuple(tensor.shape)).encode("ascii"))
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _validate_checkpoint(path: Path) -> dict:
    """Integrity check before torch.load; no claim of trusted external signing."""
    manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    expected = {"schema", "tokenizer_sha256", "model_seed", "step", "cursor",
                "lr", "before_second_batch_sha256", "initial_frozen_sha256",
                "trainable_names", "files", "config_sha256", "runtime", "rng_state_sha256"}
    if set(manifest) != expected or manifest["schema"] != SCHEMA:
        raise ValueError("checkpoint manifest schema or fields changed")
    if (manifest["tokenizer_sha256"] != integrated.TOKENIZER_SHA or
            type(manifest["model_seed"]) is not int or manifest["model_seed"] != integrated.BASE_SEED or
            type(manifest["step"]) is not int or manifest["step"] != 1 or
            type(manifest["cursor"]) is not int or manifest["cursor"] != 1 or
            type(manifest["lr"]) is not float or manifest["lr"] != LR):
        raise ValueError("checkpoint bound version/step/learning-rate mismatch")
    if (not isinstance(manifest["trainable_names"], list) or
            len(manifest["trainable_names"]) != 304 or
            len(set(manifest["trainable_names"])) != 304 or
            any(not isinstance(x, str) or "lora_" not in x for x in manifest["trainable_names"])):
        raise ValueError("trainable parameter identities not admitted")
    if (not isinstance(manifest["files"], dict) or
            set(manifest["files"]) != {f"{VERSIONED_ADAPTER}/adapter_model.safetensors",
                                       f"{VERSIONED_ADAPTER}/adapter_config.json", OPTIMIZER}):
        raise ValueError("required checkpoint files missing from manifest")
    for relpath, expected_sha in manifest["files"].items():
        if len(expected_sha) != 64 or not all(c in "0123456789abcdef" for c in expected_sha):
            raise ValueError("invalid file digest in manifest")
        actual = _sha(path / relpath)
        if actual != expected_sha:
            raise ValueError(f"checkpoint file corrupted: {relpath}")
    for value in (manifest["initial_frozen_sha256"], manifest["config_sha256"],
                  manifest["rng_state_sha256"], *manifest["before_second_batch_sha256"].values()):
        if (not isinstance(value, str) or len(value) != 64 or
                any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("invalid checkpoint data binding")
    if not isinstance(manifest["runtime"], dict) or set(manifest["runtime"]) != {
            "torch", "transformers", "trl", "peft", "tokenizers", "cpu_capability",
            "threads", "mkldnn", "deterministic_algorithms"}:
        raise ValueError("runtime fingerprint missing")
    if set(manifest["before_second_batch_sha256"]) != {"input_ids", "labels", "attention_mask"}:
        raise ValueError("checkpoint batch identities missing")
    return manifest


def _context(torch, *, from_adapter: Path | None = None):
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import PreTrainedTokenizerFast
    from trl import SFTConfig, SFTTrainer
    from trainingday import qwen35_full32_hybrid_peft_cpu_smoke as smoke

    tokenizer_path = Path(os.environ["TRAININGDAY_QWEN_TOKENIZER_JSON"])
    raw = integrated._load_tokenizer(tokenizer_path)
    rows = integrated._synthetic_rows(raw)
    tok = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path),
                                  eos_token="<|im_end|>", pad_token="<|endoftext|>")
    base = integrated._construct_base(torch)
    frozen_sha = integrated._sha_tensor(base.model.embed_tokens.weight)
    if from_adapter is None:
        model = get_peft_model(base, LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules=smoke.expected_target_modules()))
    else:
        model = PeftModel.from_pretrained(base, str(from_adapter), is_trainable=True)
    trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable_names = [n for n, _ in trainable_named]
    if len(trainable_names) != 304 or any("lora_" not in n for n in trainable_names):
        raise AssertionError("expected exactly 304 PEFT LoRA-only training tensors")
    with tempfile.TemporaryDirectory(prefix="trainingday-qwen35-resume-dataloader-") as tmp:
        config = SFTConfig(output_dir=tmp, max_length=64, per_device_train_batch_size=2,
                           completion_only_loss=True, packing=False, report_to="none",
                           seed=42, fp16=False, bf16=False, dataloader_num_workers=0)
        trainer = SFTTrainer(model=model, args=config,
                             train_dataset=Dataset.from_list(rows), processing_class=tok)
        batches = [{k: v.detach().clone() for k, v in b.items()}
                   for b in trainer.get_train_dataloader()]
    if len(batches) != 4 or any(-100 not in b["labels"] for b in batches):
        raise AssertionError("unmasked/changed actual TRL training batches")
    if any(b["labels"].shape != b["input_ids"].shape for b in batches):
        raise AssertionError("actual TRL completion-only mask mismatch")
    return model, trainable_named, batches, frozen_sha


def _step(torch, model, optimizer, batch) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=batch["input_ids"], labels=batch["labels"],
                 attention_mask=batch["attention_mask"], use_cache=False).loss
    if loss is None or not torch.isfinite(loss):
        raise AssertionError("nonfinite masked TRL loss")
    loss.backward()
    b_grads = [p.grad for n, p in model.named_parameters() if ".lora_B." in n]
    if len(b_grads) != 152 or any(g is None or not torch.isfinite(g).all() or
                                  not bool((g != 0).any()) for g in b_grads):
        raise AssertionError("at least one hybrid LoRA-B projection lacks gradient")
    optimizer.step()
    return float(loss.item())


def _trainable_digest(model) -> str:
    from trainingday import qwen35_full32_hybrid_peft_cpu_smoke as smoke
    return smoke._hash_tensors({n: p for n, p in model.named_parameters()
                                if p.requires_grad})


def _save(path, model, optimizer, batch, frozen_sha, names, torch):
    adapter_dir = path / VERSIONED_ADAPTER
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    rng_state = torch.get_rng_state().clone()
    torch.save({"optimizer": optimizer.state_dict(), "rng_state": rng_state}, path / OPTIMIZER)
    files = [f"{VERSIONED_ADAPTER}/adapter_model.safetensors",
             f"{VERSIONED_ADAPTER}/adapter_config.json", OPTIMIZER]
    manifest = {"schema": SCHEMA, "tokenizer_sha256": integrated.TOKENIZER_SHA,
                "model_seed": integrated.BASE_SEED, "step": 1, "cursor": 1,
                "lr": LR, "before_second_batch_sha256": _tensor_batch_sha(batch),
                "initial_frozen_sha256": frozen_sha, "trainable_names": names,
                "config_sha256": hashlib.sha256(model.base_model.model.config.to_json_string().encode("utf-8")).hexdigest(),
                "runtime": _runtime_fingerprint(torch),
                "rng_state_sha256": hashlib.sha256(rng_state.numpy().tobytes()).hexdigest(),
                "files": {f: _sha(path / f) for f in files}}
    (path / MANIFEST).write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    assert _validate_checkpoint(path) == manifest


def _resume(path: Path):
    # Run the integrity gate before importing the model framework or unpickling.
    manifest = _validate_checkpoint(path)
    integrated._offline_cpu()
    import torch
    torch.set_num_threads(2)
    model, trainable, batches, frozen_sha = _context(torch, from_adapter=path / VERSIONED_ADAPTER)
    names = [n for n, _ in trainable]
    if names != manifest["trainable_names"] or frozen_sha != manifest["initial_frozen_sha256"]:
        raise ValueError("wrong model/base trainable parameter identity")
    if _runtime_fingerprint(torch) != manifest["runtime"]:
        raise ValueError("CPU numerical runtime differs from checkpoint")
    if hashlib.sha256(model.base_model.model.config.to_json_string().encode("utf-8")).hexdigest() != manifest["config_sha256"]:
        raise ValueError("base model architecture differs from checkpoint")
    if _tensor_batch_sha(batches[1]) != manifest["before_second_batch_sha256"]:
        raise ValueError("TRL dataloader resume cursor/batch drift")
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=LR)
    payload = torch.load(path / OPTIMIZER, map_location="cpu", weights_only=True)
    if set(payload) != {"optimizer", "rng_state"}:
        raise ValueError("optimizer/RNG checkpoint payload incomplete")
    state = payload["optimizer"]
    if len(state["param_groups"]) != 1 or state["param_groups"][0]["lr"] != LR:
        raise ValueError("optimizer group mismatch")
    if len(state["state"]) != 304 or any(int(v["step"]) != 1 for v in state["state"].values()):
        raise ValueError("optimizer step state was not completely persisted")
    optimizer.load_state_dict(state)
    rng_state = payload["rng_state"]
    if hashlib.sha256(rng_state.numpy().tobytes()).hexdigest() != manifest["rng_state_sha256"]:
        raise ValueError("RNG state differs from saved checkpoint")
    torch.set_rng_state(rng_state)
    loss = _step(torch, model, optimizer, batches[1])
    ids = torch.tensor([integrated._synthetic_rows(integrated._load_tokenizer(
        Path(os.environ["TRAININGDAY_QWEN_TOKENIZER_JSON"])))[0]["input_ids"]], dtype=torch.long)
    return {"status": "RESUMED_ADAMW_REAL_SECOND_STEP_PASS", "pid": os.getpid(),
            "second_step_loss": loss, "adapter_sha256": _trainable_digest(model),
            "logits_sha256": integrated._sha_tensor(integrated._probe_logits(model, ids, torch)),
            "optimizer_state_sha256": _optimizer_digest(optimizer),
            "after_resume_optimizer_steps": sorted({int(x["step"]) for x in optimizer.state.values()}),
            "cuda_initialized": torch.cuda.is_initialized()}


def run() -> dict:
    integrated._offline_cpu()
    import torch
    torch.set_num_threads(2)
    model, trainable, batches, frozen_sha = _context(torch)
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=LR)
    initial_sha = _trainable_digest(model)
    with tempfile.TemporaryDirectory(prefix="trainingday-qwen35-trl-resume-") as tmp:
        path = Path(tmp)
        first_loss = _step(torch, model, optimizer, batches[0])
        after_first_sha = _trainable_digest(model)
        if after_first_sha == initial_sha:
            raise AssertionError("first step had no trained adapter effect")
        _save(path, model, optimizer, batches[1], frozen_sha, [n for n, _ in trainable], torch)
        second_loss = _step(torch, model, optimizer, batches[1])
        direct_sha = _trainable_digest(model)
        direct_optimizer_sha = _optimizer_digest(optimizer)
        ids = torch.tensor([integrated._synthetic_rows(integrated._load_tokenizer(
            Path(os.environ["TRAININGDAY_QWEN_TOKENIZER_JSON"])))[0]["input_ids"]], dtype=torch.long)
        direct_logits = integrated._sha_tensor(integrated._probe_logits(model, ids, torch))
        del model, optimizer, trainable, batches, ids
        gc.collect()
        child = subprocess.run([sys.executable, "-m",
                                "trainingday.qwen35_full32_trl_optimizer_resume_cpu",
                                "--resume", str(path)], capture_output=True, text=True,
                               timeout=180, env=os.environ.copy(), check=False)
        if child.returncode:
            raise AssertionError(f"fresh process optimizer resume failed: {child.stderr[-1200:]}")
        resumed = json.loads(child.stdout.splitlines()[-1])
        if (resumed["pid"] == os.getpid() or
                resumed["adapter_sha256"] != direct_sha or
                resumed["logits_sha256"] != direct_logits or
                resumed["optimizer_state_sha256"] != direct_optimizer_sha or
                resumed["after_resume_optimizer_steps"] != [2] or
                resumed["cuda_initialized"] or
                abs(resumed["second_step_loss"] - second_loss) > 1e-7):
            raise AssertionError("separate-process AdamW continuation not equivalent to direct two steps")
        # Mutate a real payload byte: the independent integrity checker must
        # refuse the file before torch.load; manifest hashes are not signatures.
        state = path / OPTIMIZER
        before = state.read_bytes()
        state.write_bytes(before[:-1] + bytes([before[-1] ^ 1]))
        rejected = subprocess.run([sys.executable, "-m",
                                   "trainingday.qwen35_full32_trl_optimizer_resume_cpu",
                                   "--verify", str(path)], capture_output=True, text=True,
                                  timeout=15, env=os.environ.copy(), check=False)
        if rejected.returncode == 0 or "checkpoint file corrupted" not in rejected.stderr:
            raise AssertionError("corrupted checkpoint accepted")
    return {"schema": SCHEMA,
            "status": "REAL_MASKED_TRL_ADAMW_FRESH_PROCESS_OPTIMIZER_RESUME_PASS",
            "architecture": "TINY_RANDOM_QWEN35_32_LAYER_24_GDN_8_ATTN_152_LORA_B",
            "training_examples": "8_GENERATED_ARITHMETIC_ONLY", "actual_optimizer_steps": 2,
            "first_step_loss": first_loss, "second_step_loss": second_loss,
            "initial_lora_sha256": initial_sha,
            "first_step_lora_sha256": after_first_sha,
            "direct_and_resumed_adapter_sha256": direct_sha,
            "direct_and_resumed_optimizer_sha256": direct_optimizer_sha,
            "direct_and_resumed_logits_sha256": direct_logits,
            "fresh_child_pid": resumed["pid"], "corrupt_optimizer_file_rejected": True,
            "official_pretrained_weights_loaded": False, "training_authorized": False,
            "four_gb_gpu_fit": "UNKNOWN", "cuda_initialized": False}


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--resume":
        print(json.dumps(_resume(Path(sys.argv[2])), sort_keys=True))
    elif len(sys.argv) == 3 and sys.argv[1] == "--verify":
        print(json.dumps({"verified": _validate_checkpoint(Path(sys.argv[2]))["schema"]}))
    elif len(sys.argv) == 1:
        print(json.dumps(run(), sort_keys=True))
    else:
        raise SystemExit("Usage: python -m trainingday.qwen35_full32_trl_optimizer_resume_cpu [--resume|--verify PATH]")
