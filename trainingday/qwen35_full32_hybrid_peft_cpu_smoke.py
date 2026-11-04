"""CPU-only full-32-layer Qwen3.5 *architecture* LoRA injection experiment.

The two-layer tiny pilot passed hosted CI; this expands the same random,
miniature-width architecture to all 24 Gated DeltaNet and eight full-attention
layers. It does NOT use official Qwen weights, GPU, protected data or a lease.
Its success would prove only CPU PEFT injection, gradient flow and an optimizer
step for this miniature configuration, not 4 GiB fit or model quality.
"""

from __future__ import annotations

import hashlib
import json
import os
import time


SCHEMA = "TRAININGDAY_TINY_WIDTH_FULL32_QWEN35_PEFT_CPU_STEP_V1"
LAYER_TYPES = ("linear_attention", "linear_attention", "linear_attention",
               "full_attention") * 8
GDN_SUFFIXES = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                "linear_attn.in_proj_b", "linear_attn.in_proj_a",
                "linear_attn.out_proj")
ATTN_SUFFIXES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj")
TARGET_COUNT = 152


def expected_target_modules() -> list[str]:
    """Explicit, layer-numbered source-only names; no broad q/v suffixes."""
    return [f"model.layers.{layer}.{suffix}"
            for layer, kind in enumerate(LAYER_TYPES)
            for suffix in (GDN_SUFFIXES if kind == "linear_attention" else ATTN_SUFFIXES)]


def _hash_tensors(tensors: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        raw = tensor.detach().to(device="cpu", dtype=tensor.dtype,
                                 copy=True).contiguous().numpy().tobytes()
        label = name.encode("utf-8")
        digest.update(len(label).to_bytes(4, "big") + label)
        digest.update(len(raw).to_bytes(8, "big") + raw)
    return digest.hexdigest()


def run() -> dict:
    """Real dynamic test, uncallable as physical training authorization."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_DISABLED"] = "true"

    import torch
    import peft
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    # Strictly source-only cross-check against the separately reviewed 152
    # target candidate from Draft #248, now available in the intended stack.
    from trainingday.qwen35_hybrid_lora_candidate import HYBRID_LINEAR_PROJECTIONS

    if (tuple(HYBRID_LINEAR_PROJECTIONS["linear_attention"]) != GDN_SUFFIXES
            or tuple(HYBRID_LINEAR_PROJECTIONS["full_attention"]) != ATTN_SUFFIXES):
        raise AssertionError("published source-only hybrid target contract changed")

    torch.set_num_threads(2)
    torch.manual_seed(35032)
    start = time.monotonic()
    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=32, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, layer_types=list(LAYER_TYPES),
        linear_conv_kernel_dim=2, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, max_position_embeddings=128,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    )
    # Pure random allocation via constructor. There is no from_pretrained.
    base = Qwen3_5ForCausalLM(config).to("cpu")
    if list(base.config.layer_types) != list(LAYER_TYPES):
        raise AssertionError("constructed 32-layer hybrid order differs")
    for layer, kind in enumerate(LAYER_TYPES):
        block = base.model.layers[layer]
        expected = ("Qwen3_5GatedDeltaNet" if kind == "linear_attention"
                    else "Qwen3_5Attention")
        module = block.linear_attn if kind == "linear_attention" else block.self_attn
        if type(module).__name__ != expected:
            raise AssertionError(f"decoder layer {layer} wrong class: {type(module).__name__}")

    targets = expected_target_modules()
    if len(targets) != TARGET_COUNT or len(set(targets)) != TARGET_COUNT:
        raise AssertionError("invalid expanded 152-module source target inventory")
    before_modules = dict(base.named_modules())
    missing = [name for name in targets
               if name not in before_modules or not isinstance(before_modules[name], torch.nn.Linear)]
    if missing:
        raise AssertionError(f"actual tiny Qwen linear targets missing: {missing}")
    frozen_embed = base.model.embed_tokens.weight.detach().clone()
    frozen_qkv = base.model.layers[0].linear_attn.in_proj_qkv.weight.detach().clone()

    model = get_peft_model(base, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    ))
    model.train()
    injected: dict[str, str] = {}
    for name, module in model.named_modules():
        for target in targets:
            if name == target or name.endswith("." + target):
                if target in injected:
                    raise AssertionError(f"ambiguous runtime match: {target}")
                if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                    raise AssertionError(f"PEFT did not inject expected LoRA layer: {name}")
                injected[target] = name
    if set(injected) != set(targets):
        raise AssertionError(f"PEFT only injected {len(injected)}/{len(targets)}")

    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise AssertionError("unexpected trainable non-LoRA parameter")
    b_params = {name: p for name, p in trainable.items() if "lora_B" in name}
    if len(b_params) != TARGET_COUNT:
        raise AssertionError("LoRA-B tensor count does not match all 152 projections")
    before = _hash_tensors(b_params)
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=5e-3)
    ids = torch.tensor([[5, 6, 7, 8, 9, 10]], dtype=torch.long, device="cpu")
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
    if loss is None or not torch.isfinite(loss):
        raise AssertionError("nonfinite full32 tiny-width Qwen loss")
    loss.backward()

    gradient_sum_by_layer: list[float] = []
    for layer in range(32):
        prefix = f".model.layers.{layer}."
        grads = [p.grad for name, p in b_params.items() if prefix in name]
        expected_count = 5 if LAYER_TYPES[layer] == "linear_attention" else 4
        if len(grads) != expected_count:
            raise AssertionError(f"layer {layer} missing LoRA-B gradients")
        if any(g is None or not torch.isfinite(g).all() for g in grads):
            raise AssertionError(f"layer {layer} has absent/nonfinite LoRA-B gradient")
        total = sum(float(g.abs().sum()) for g in grads)
        if total <= 0:
            raise AssertionError(f"layer {layer} received zero adapter gradient")
        gradient_sum_by_layer.append(total)

    optimizer.step()
    after = _hash_tensors(b_params)
    if before == after:
        raise AssertionError("full32 AdamW step changed no adapter bytes")
    if not torch.equal(frozen_embed, model.base_model.model.model.embed_tokens.weight):
        raise AssertionError("frozen embedding changed")
    qkv_after = model.base_model.model.model.layers[0].linear_attn.in_proj_qkv.base_layer.weight
    if not torch.equal(frozen_qkv, qkv_after):
        raise AssertionError("frozen GDN base projection changed")
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise AssertionError("parameter escaped CPU")
    if torch.cuda.is_initialized():
        raise AssertionError("CUDA initialized during CPU-only full32 experiment")

    elapsed_s = time.monotonic() - start
    import resource
    rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "schema": SCHEMA,
        "status": "TINY_WIDTH_QWEN35_FULL32_HYBRID_PEFT_CPU_STEP_PASS",
        "runtime_versions": {"torch": str(torch.__version__),
                             "transformers": transformers.__version__,
                             "peft": peft.__version__},
        "model_weights": "RANDOM_SMALL_CONFIG_NO_PRETRAINED_SHARDS",
        "official_model_exercised": False,
        "hybrid_layers": 32,
        "gdn_layers": 24,
        "full_attention_layers": 8,
        "lora_projections_injected": len(injected),
        "layers_with_nonzero_lora_b_gradients": len(gradient_sum_by_layer),
        "smallest_layer_gradient_sum": min(gradient_sum_by_layer),
        "trainable_lora_parameters": sum(int(p.numel()) for p in trainable.values()),
        "loss": float(loss.item()),
        "lora_b_sha256_before": before,
        "lora_b_sha256_after": after,
        "sampled_base_embedding_and_qkv_unchanged": True,
        "cpu_process_peak_rss_kib": rss_kib,
        "cpu_elapsed_seconds": elapsed_s,
        "gpu_execution_requested": False,
        "cuda_runtime_initialized": False,
        "four_gb_gpu_fit": "UNKNOWN",
        "training_authorized": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
