"""Public CPU Qwen3.5 *random miniature* HF -> GGUF -> llama.cpp load probe.

This intentionally does not handle official pretrained shards or a production
adapter. The pinned official *tokenizer metadata* is retrieved separately from
the randomly initialized 32-layer text model to exercise the real converter.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


OFFICIAL_MODEL = "Qwen/Qwen3.5-4B"
OFFICIAL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SYNTHETIC_BASE = "synthetic:qwen3.5-full32-hybrid-seed35032-width64-v1"
SCHEMA = "TRAININGDAY_QWEN35_TINY_FULL32_GGUF_LOAD_V1"
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja",
)


def assert_adapter_lineage(adapter_config: dict, expected_base: str = SYNTHETIC_BASE) -> None:
    """Reject a mismatched base *before* merging any adapter tensors."""
    actual = adapter_config.get("base_model_name_or_path")
    if actual != expected_base:
        raise ValueError(f"adapter base lineage mismatch: {actual!r} != {expected_base!r}")


def assert_model_architecture(config: dict) -> None:
    if config.get("model_type") not in {"qwen3_5", "qwen3_5_text"}:
        raise ValueError("base architecture is not Qwen3.5")
    if config.get("architectures") != ["Qwen3_5ForCausalLM"]:
        raise ValueError("base architecture class is not Qwen3_5ForCausalLM")
    layer_types = config.get("layer_types")
    if (not isinstance(layer_types, list) or len(layer_types) != 32
            or layer_types != ["linear_attention", "linear_attention",
                               "linear_attention", "full_attention"] * 8):
        raise ValueError("not the 32-layer Qwen3.5 24-GDN/8-full-attention pattern")


def fetch_tokenizer_metadata(path: Path) -> Path:
    """Allowlist only public tokenizer JSON/text, NEVER model weight files."""
    from huggingface_hub import snapshot_download

    result = Path(snapshot_download(
        repo_id=OFFICIAL_MODEL,
        revision=OFFICIAL_REVISION,
        allow_patterns=list(TOKENIZER_FILES),
        local_dir=str(path),
    ))
    files = [f for f in result.rglob("*") if f.is_file() and ".cache" not in f.parts]
    if not (result / "tokenizer.json").is_file():
        raise AssertionError("pinned official tokenizer metadata not available")
    if any(f.name not in TOKENIZER_FILES for f in files):
        raise AssertionError("unexpected file outside tokenizer-only allowlist")
    if sum(f.stat().st_size for f in files) > 32 * 1024 * 1024:
        raise AssertionError("public tokenizer-only metadata exceeded 32 MiB guard")
    return result


def prepare(out: Path) -> dict:
    os.environ.update(CUDA_VISIBLE_DEVICES="", WANDB_DISABLED="true")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from trainingday.qwen35_full32_hybrid_peft_cpu_smoke import (
        LAYER_TYPES, TARGET_COUNT, expected_target_modules,
    )

    torch.set_num_threads(2)
    torch.manual_seed(35032)
    tokenizer_dir = fetch_tokenizer_metadata(out / "tokenizer-source")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    vocab_size = max(tokenizer.get_vocab().values()) + 1
    if not 128 <= vocab_size <= 300000:
        raise AssertionError("unexpected pinned tokenizer vocabulary size")

    config = Qwen3_5TextConfig(
        vocab_size=vocab_size, hidden_size=64, intermediate_size=128,
        num_hidden_layers=32, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, layer_types=list(LAYER_TYPES), linear_conv_kernel_dim=2,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        max_position_embeddings=128, bos_token_id=1, eos_token_id=2,
        pad_token_id=0, rope_parameters={"rope_type": "default",
                                          "rope_theta": 10000,
                                          "mrope_section": [2, 1, 1],
                                          "mrope_interleaved": True},
    )
    base = Qwen3_5ForCausalLM(config).cpu()
    base.config.architectures = ["Qwen3_5ForCausalLM"]
    assert_model_architecture(base.config.to_dict())
    frozen = base.model.layers[0].linear_attn.in_proj_qkv.weight.detach().clone()

    targets = expected_target_modules()
    if len(targets) != TARGET_COUNT or len(set(targets)) != TARGET_COUNT:
        raise AssertionError("invalid 152-module full32 GDN and attention inventory")
    adapter = get_peft_model(base, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    ))
    adapter.peft_config["default"].base_model_name_or_path = SYNTHETIC_BASE
    adapter.train()
    opt = torch.optim.AdamW((p for p in adapter.parameters() if p.requires_grad), lr=5e-3)
    ids = torch.tensor([[5, 6, 7, 8, 9, 10]], dtype=torch.long)
    opt.zero_grad(set_to_none=True)
    loss = adapter(input_ids=ids, labels=ids, use_cache=False).loss
    if loss is None or not torch.isfinite(loss):
        raise AssertionError("nonfinite tiny LoRA training loss")
    loss.backward()
    opt.step()

    adapter_dir = out / "adapter"
    adapter.save_pretrained(str(adapter_dir), safe_serialization=True)
    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert_adapter_lineage(adapter_cfg)
    bad_cfg = {**adapter_cfg, "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct"}
    try:
        assert_adapter_lineage(bad_cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("negative control accepted Qwen2.5 adapter lineage")

    merged = adapter.merge_and_unload().cpu()
    current = merged.model.layers[0].linear_attn.in_proj_qkv.weight
    if torch.equal(frozen, current):
        raise AssertionError("LoRA merge left sampled GDN projection unchanged")
    merged_dir = out / "merged"
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    assert_model_architecture(json.loads((merged_dir / "config.json").read_text()))
    if any(p.device.type != "cpu" for p in merged.parameters()) or torch.cuda.is_initialized():
        raise AssertionError("probe initialized GPU or escaped CPU")

    receipt = {
        "schema": SCHEMA,
        "official_pretrained_weights_exercised": False,
        "tokenizer_provenance": f"{OFFICIAL_MODEL}@{OFFICIAL_REVISION}",
        "random_model_lineage": SYNTHETIC_BASE,
        "layers": 32,
        "gdn_layers": 24,
        "full_attention_layers": 8,
        "lora_targets": TARGET_COUNT,
        "negative_qwen25_lineage_rejected": True,
        "merged_gdn_projection_changed": True,
        "loss": float(loss.item()),
        "vocab_size": vocab_size,
        "merged_safetensors_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in merged_dir.glob("*.safetensors")
        },
        "gguf_conversion": "NOT_YET_RUN",
        "llama_cpp_load": "NOT_YET_RUN",
        "gpu_fit": "UNKNOWN",
        "koboldcpp_compatibility": "UNKNOWN",
    }
    (out / "receipt-preconvert.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def assert_loaded_gguf(gguf: Path, llama_cli: Path) -> dict:
    if not gguf.is_file():
        raise AssertionError("converter did not produce a GGUF file")
    with gguf.open("rb") as stream:
        magic = stream.read(4)
    if magic != b"GGUF":
        raise AssertionError("converter did not create valid GGUF magic")
    result = subprocess.run([
        str(llama_cli), "-m", str(gguf), "-p", "hello", "-n", "1",
        "-c", "128", "--temp", "0", "--no-display-prompt",
    ], capture_output=True, text=True, check=False, timeout=120)
    if result.returncode != 0:
        raise AssertionError(f"real llama-cli load failed ({result.returncode}): "
                             f"{result.stderr[-3000:]}\n{result.stdout[-1500:]}")
    if not result.stdout.strip() and not result.stderr.strip():
        raise AssertionError("llama-cli exited without evidence of execution")
    return {
        "schema": SCHEMA,
        "status": "RANDOM_TINY_FULL32_QWEN35_MERGED_GGUF_LLAMA_CPP_LOAD_PASS",
        "gguf_sha256": hashlib.sha256(gguf.read_bytes()).hexdigest(),
        "gguf_bytes": gguf.stat().st_size,
        "llama_cli_exit_code": result.returncode,
        "llama_cli_output_tail": result.stdout[-600:],
        "official_pretrained_weights_exercised": False,
        "koboldcpp_compatibility": "UNKNOWN",
        "four_gb_gpu_fit": "UNKNOWN",
    }


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "prepare":
        print(json.dumps(prepare(Path(sys.argv[2])), sort_keys=True))
    elif len(sys.argv) == 4 and sys.argv[1] == "load":
        print(json.dumps(assert_loaded_gguf(Path(sys.argv[2]), Path(sys.argv[3])), sort_keys=True))
    else:
        raise SystemExit("usage: python -m trainingday.qwen35_tiny_gguf_e2e_cpu "
                         "prepare DIR | load GGUF LLAMA_CLI")


if __name__ == "__main__":
    main()
