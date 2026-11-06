"""Public CPU-only Qwen3.5 hybrid LoRA persistence/restart experiment.

The 32-layer architecture is deliberately *tiny-width and randomly initialized*.
No official pretrained weights, private curriculum, GPU or host lease are used.
The fresh adapter consumer is a separate Python process with separately
constructed base parameters, not a second reference to the training object.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from trainingday import qwen35_full32_hybrid_peft_cpu_smoke as smoke


SCHEMA = "TRAININGDAY_QWEN35_TINY_FULL32_ADAPTER_RELOAD_CPU_V1"
BASE_SEED = 35032
TRAIN_TOKENS = (5, 6, 7, 8, 9, 10)


def _cpu_only() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_DISABLED"] = "true"


def _construct_random_base(torch, transformers):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(BASE_SEED)
    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=32, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, layer_types=list(smoke.LAYER_TYPES),
        linear_conv_kernel_dim=2, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, max_position_embeddings=128,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    )
    base = Qwen3_5ForCausalLM(config).to("cpu")
    if list(base.config.layer_types) != list(smoke.LAYER_TYPES):
        raise AssertionError("fresh Qwen3.5 base layer pattern changed")
    return base


def _tensor_sha(tensor) -> str:
    return hashlib.sha256(tensor.detach().to("cpu").contiguous().numpy().tobytes()).hexdigest()


def _adapter_b_hash(model) -> tuple[str, int]:
    tensors = {name: p for name, p in model.named_parameters() if ".lora_B." in name}
    if len(tensors) != smoke.TARGET_COUNT:
        raise AssertionError(f"adapter has {len(tensors)} LoRA-B tensors, not 152")
    return smoke._hash_tensors(tensors), len(tensors)


def _adapter_ab_hash(model) -> tuple[str, int]:
    """Bind both LoRA factors, not only the factor changed from zero at step 1."""
    tensors = {name: p for name, p in model.named_parameters()
               if ".lora_A." in name or ".lora_B." in name}
    if (sum(".lora_A." in name for name in tensors) != smoke.TARGET_COUNT or
            sum(".lora_B." in name for name in tensors) != smoke.TARGET_COUNT):
        raise AssertionError("adapter missing LoRA-A or LoRA-B tensor coverage")
    return smoke._hash_tensors(tensors), len(tensors)


def _artifact_hashes(adapter_dir: Path) -> dict[str, str]:
    """Whole-file witness covers persisted A, B, safetensors header, and config."""
    digests: dict[str, str] = {}
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        path = adapter_dir / name
        if not path.is_file() or path.is_symlink():
            raise AssertionError(f"missing regular persisted adapter file: {name}")
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _assert_artifact_hashes(expected: dict[str, str], actual: dict[str, str]) -> None:
    if expected != actual:
        raise AssertionError("persisted LoRA-A/B or adapter config bytes changed")


def _probe_logits(model, ids, torch):
    model.eval()
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits.detach().float().cpu().contiguous()


def _reload_in_child(adapter_dir: Path) -> dict:
    _cpu_only()
    import torch
    import transformers
    from peft import PeftModel

    torch.set_num_threads(2)
    ids = torch.tensor([list(TRAIN_TOKENS)], dtype=torch.long)
    base = _construct_random_base(torch, transformers)
    frozen_base_sha = _tensor_sha(base.model.embed_tokens.weight)
    base_logits_sha = _tensor_sha(_probe_logits(base, ids, torch))
    # Crucial: fresh process, fresh base; only the LoRA adapter is loaded.
    loaded = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    loaded_logits = _probe_logits(loaded, ids, torch)
    loaded_sha, tensor_count = _adapter_b_hash(loaded)
    loaded_ab_sha, loaded_ab_count = _adapter_ab_hash(loaded)
    saved_artifact_hashes = _artifact_hashes(adapter_dir)
    if _tensor_sha(loaded_logits) == base_logits_sha:
        raise AssertionError("reloaded adapter has no measurable effect on logits")
    if any(p.device.type != "cpu" for p in loaded.parameters()):
        raise AssertionError("loaded parameters escaped CPU")
    if torch.cuda.is_initialized():
        raise AssertionError("CUDA initialized in fresh adapter consumer")
    return {
        "child_pid": os.getpid(),
        "adapter_b_sha256": loaded_sha,
        "lora_b_tensors": tensor_count,
        "adapter_ab_sha256": loaded_ab_sha,
        "lora_a_and_b_tensors": loaded_ab_count,
        "persisted_artifact_sha256": saved_artifact_hashes,
        "loaded_logits_sha256": _tensor_sha(loaded_logits),
        "frozen_base_sha256": frozen_base_sha,
        "base_logits_sha256": base_logits_sha,
        "cuda_initialized": False,
    }


def run() -> dict:
    _cpu_only()
    import torch
    import peft
    import transformers
    from peft import LoraConfig, get_peft_model

    torch.set_num_threads(2)
    ids = torch.tensor([list(TRAIN_TOKENS)], dtype=torch.long)
    base = _construct_random_base(torch, transformers)
    frozen_base_sha = _tensor_sha(base.model.embed_tokens.weight)
    base_logits_sha = _tensor_sha(_probe_logits(base, ids, torch))
    targets = smoke.expected_target_modules()
    if len(set(targets)) != smoke.TARGET_COUNT:
        raise AssertionError("wrong explicit adapter target count")
    model = get_peft_model(base, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    ))
    before_sha, _ = _adapter_b_hash(model)
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise AssertionError("unexpected unfrozen non-LoRA parameter")
    model.train()
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=5e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
    if loss is None or not torch.isfinite(loss):
        raise AssertionError("nonfinite tiny full32 training loss")
    loss.backward()
    for layer in range(32):
        grads = [p.grad for n, p in trainable.items()
                 if ".lora_B." in n and f".model.layers.{layer}." in n]
        if len(grads) != (5 if smoke.LAYER_TYPES[layer] == "linear_attention" else 4):
            raise AssertionError(f"missing LoRA-B gradients at layer {layer}")
        if any(g is None or not torch.isfinite(g).all() for g in grads):
            raise AssertionError(f"nonfinite or missing LoRA-B gradient at layer {layer}")
        if sum(float(g.abs().sum()) for g in grads) <= 0:
            raise AssertionError(f"zero adapter gradient at layer {layer}")
    optimizer.step()
    after_sha, tensor_count = _adapter_b_hash(model)
    after_ab_sha, ab_tensor_count = _adapter_ab_hash(model)
    if before_sha == after_sha:
        raise AssertionError("optimizer did not change persisted LoRA-B tensors")
    trained_logits_sha = _tensor_sha(_probe_logits(model, ids, torch))
    if frozen_base_sha != _tensor_sha(model.base_model.model.model.embed_tokens.weight):
        raise AssertionError("frozen base embedding changed")
    if any(p.device.type != "cpu" for p in model.parameters()) or torch.cuda.is_initialized():
        raise AssertionError("CPU-only constraint violated")

    with tempfile.TemporaryDirectory(prefix="trainingday-qwen35-adapter-reload-") as d:
        # Versioned artifact is written and read by a new interpreter process.
        adapter_dir = Path(d) / "adapter-v1"
        model.save_pretrained(str(adapter_dir), safe_serialization=True)
        expected_files = {"adapter_config.json", "adapter_model.safetensors"}
        if not expected_files.issubset({p.name for p in adapter_dir.iterdir()}):
            raise AssertionError("missing persisted PEFT adapter artifact")
        saved_artifact_hashes = _artifact_hashes(adapter_dir)
        output = subprocess.run(
            [sys.executable, "-m", "trainingday.qwen35_full32_adapter_reload_cpu",
             "--child", str(adapter_dir)],
            check=False, capture_output=True, text=True, env=os.environ.copy(),
            timeout=120,
        )
        if output.returncode != 0:
            raise AssertionError(
                f"fresh adapter consumer exited {output.returncode}: {output.stderr[-1200:]}")
        try:
            child = json.loads(output.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as e:
            raise AssertionError(f"fresh child did not emit JSON receipt: {output.stderr[-600:]}") from e
        if child["child_pid"] == os.getpid():
            raise AssertionError("adapter consumer did not run in separate process")
        if child["adapter_b_sha256"] != after_sha:
            raise AssertionError("reloaded LoRA-B tensor hash differs")
        if child["adapter_ab_sha256"] != after_ab_sha:
            raise AssertionError("reloaded LoRA-A/B tensor hash differs")
        if child["lora_a_and_b_tensors"] != ab_tensor_count:
            raise AssertionError("reloaded LoRA-A/B tensor inventory differs")
        _assert_artifact_hashes(saved_artifact_hashes,
                                child["persisted_artifact_sha256"])
        _assert_artifact_hashes(saved_artifact_hashes,
                                _artifact_hashes(adapter_dir))
        if child["loaded_logits_sha256"] != trained_logits_sha:
            raise AssertionError("fresh-process adapter logits differ")
        if child["base_logits_sha256"] != base_logits_sha:
            raise AssertionError("fresh-process base logits differ")
        if child["frozen_base_sha256"] != frozen_base_sha:
            raise AssertionError("fresh-process frozen base initialization differs")
        if trained_logits_sha == base_logits_sha:
            raise AssertionError("adapter output has no detectable change vs frozen base")
        if child["lora_b_tensors"] != smoke.TARGET_COUNT or child["cuda_initialized"]:
            raise AssertionError("fresh adapter consumption violated module/CPU invariant")

        # Negative control: a fresh consumer must reject an empty, local adapter
        # directory. We require an actual failing subprocess, not a mocked loader.
        absent_dir = Path(d) / "missing-adapter"
        absent_dir.mkdir()
        rejected = subprocess.run(
            [sys.executable, "-m", "trainingday.qwen35_full32_adapter_reload_cpu",
             "--child", str(absent_dir)],
            check=False, capture_output=True, text=True, env=os.environ.copy(),
            timeout=120,
        )
        if rejected.returncode == 0:
            raise AssertionError("fresh consumer accepted missing adapter metadata")

    return {
        "schema": SCHEMA,
        "status": "TINY_RANDOM_FULL32_QWEN35_ADAPTER_FRESH_PROCESS_RELOAD_PASS",
        "runtime_versions": {"torch": str(torch.__version__),
                             "transformers": transformers.__version__,
                             "peft": peft.__version__},
        "model_weights": "RANDOM_TINY_WIDTH_NO_PRETRAINED_SHARDS",
        "official_model_exercised": False,
        "hybrid_layers": 32,
        "lora_b_tensors": tensor_count,
        "before_lora_b_sha256": before_sha,
        "after_and_reloaded_lora_b_sha256": after_sha,
        "after_and_reloaded_lora_ab_sha256": after_ab_sha,
        "lora_a_and_b_tensors": ab_tensor_count,
        "persisted_artifact_sha256": saved_artifact_hashes,
        "trained_and_reloaded_logits_sha256": trained_logits_sha,
        "frozen_base_logits_sha256": base_logits_sha,
        "separate_child_process_verified": True,
        "missing_adapter_rejected_in_separate_process": True,
        "versioned_adapter_format": "adapter-v1/safetensors+config",
        "loss": float(loss.item()),
        "gpu_execution_requested": False,
        "cuda_initialized": False,
        "four_gb_gpu_fit": "UNKNOWN",
        "training_authorized": False,
    }


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        print(json.dumps(_reload_in_child(Path(sys.argv[2])), sort_keys=True))
    elif len(sys.argv) == 1:
        print(json.dumps(run(), sort_keys=True))
    else:
        raise SystemExit("Usage: python -m trainingday.qwen35_full32_adapter_reload_cpu [--child DIR]")
