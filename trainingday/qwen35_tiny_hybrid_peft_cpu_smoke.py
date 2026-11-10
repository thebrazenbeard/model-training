"""Actual CPU PEFT injection into a *random*, miniature Qwen3.5 hybrid.

This deliberately does not load pretrained files, download a checkpoint, use a
GPU, or exercise the protected physical-training path. The tiny configuration
has one Gated DeltaNet and one full-attention layer. Its forward/backward step
can prove adapter injection and gradients for these two layer *types* only.
"""

from __future__ import annotations

import hashlib
import json
import os


SCHEMA = "TRAININGDAY_TINY_QWEN35_HYBRID_PEFT_CPU_SMOKE_V1"
LAYER_KINDS = ("linear_attention", "full_attention")
PROJECTIONS = {
    "linear_attention": (
        "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
        "linear_attn.in_proj_b", "linear_attn.in_proj_a",
        "linear_attn.out_proj",
    ),
    "full_attention": (
        "self_attn.q_proj", "self_attn.k_proj",
        "self_attn.v_proj", "self_attn.o_proj",
    ),
}


def _targets() -> list[str]:
    return [f"model.layers.{i}.{suffix}"
            for i, kind in enumerate(LAYER_KINDS)
            for suffix in PROJECTIONS[kind]]


def _hash_tensors(tensors: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        raw = tensor.detach().to("cpu", copy=True).float().contiguous().numpy().tobytes()
        label = name.encode("utf-8")
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def run() -> dict:
    # Environment and construction both bind this smoke to CPU with no Hub.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_DISABLED"] = "true"

    import torch
    import peft
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.set_num_threads(2)
    torch.manual_seed(63502)
    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, layer_types=list(LAYER_KINDS),
        linear_conv_kernel_dim=2, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4,
        max_position_embeddings=128,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    )
    # Constructing from config creates random weights; never from_pretrained.
    model = Qwen3_5ForCausalLM(config).to("cpu")
    if (list(model.config.layer_types) != list(LAYER_KINDS)
            or type(model.model.layers[0].linear_attn).__name__ != "Qwen3_5GatedDeltaNet"
            or type(model.model.layers[1].self_attn).__name__ != "Qwen3_5Attention"):
        raise AssertionError("actual constructed hybrid layers differ from tiny config")
    modules = dict(model.named_modules())
    targets = _targets()
    if not all(name in modules and isinstance(modules[name], torch.nn.Linear)
               for name in targets):
        absent = [name for name in targets
                  if name not in modules or not isinstance(modules[name], torch.nn.Linear)]
        raise AssertionError(f"tiny Qwen3.5 expected linear modules absent: {absent}")
    frozen_reference = model.model.embed_tokens.weight.detach().clone()

    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    ))
    model.train()
    inject = {}
    for name, module in model.named_modules():
        for suffix in targets:
            if name.endswith("." + suffix) or name == suffix:
                if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                    raise AssertionError(f"not a genuine PEFT-injected LoRA layer: {name}")
                if suffix in inject:
                    raise AssertionError(f"ambiguous PEFT target match: {suffix}")
                inject[suffix] = name
    if set(inject) != set(targets):
        raise AssertionError(f"PEFT module coverage mismatch: missing={set(targets)-set(inject)}")

    trainable = {name: param for name, param in model.named_parameters()
                 if param.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise AssertionError("unexpected non-LoRA trainable parameters")
    b_params = {name: p for name, p in trainable.items() if "lora_B" in name}
    if len(b_params) != len(targets):
        raise AssertionError("LoRA-B trainable tensor count differs from targets")
    before = _hash_tensors(b_params)

    optimizer = torch.optim.AdamW(list(trainable.values()), lr=5e-3)
    ids = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]], dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)
    output = model(input_ids=ids, labels=ids, use_cache=False)
    loss = output.loss
    if loss is None or not torch.isfinite(loss):
        raise AssertionError("nonfinite tiny Qwen3.5 loss")
    loss.backward()
    gradients = {name: p.grad for name, p in b_params.items()}
    if any(g is None or not torch.isfinite(g).all() for g in gradients.values()):
        raise AssertionError("missing/nonfinite gradient in a targeted LoRA-B tensor")
    gdn_grad = sum(float(g.abs().sum()) for name, g in gradients.items()
                   if ".linear_attn." in name)
    attention_grad = sum(float(g.abs().sum()) for name, g in gradients.items()
                         if ".self_attn." in name)
    if gdn_grad <= 0 or attention_grad <= 0:
        raise AssertionError("no real gradient through one of the hybrid layer types")
    optimizer.step()
    after = _hash_tensors(b_params)
    if before == after:
        raise AssertionError("optimizer step did not change LoRA-B parameters")
    if not torch.equal(frozen_reference, model.base_model.model.model.embed_tokens.weight):
        raise AssertionError("random base embedding unexpectedly mutated")
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise AssertionError("CPU-only smoke placed a model parameter on a different device")
    if torch.cuda.is_initialized():
        raise AssertionError("CPU-only smoke unexpectedly initialized CUDA")

    return {
        "schema": SCHEMA,
        "status": "TINY_RANDOM_QWEN35_HYBRID_PEFT_CPU_STEP_PASS",
        "runtime_versions": {"torch": str(torch.__version__),
                             "transformers": transformers.__version__,
                             "peft": peft.__version__},
        "architecture": "Qwen3_5ForCausalLM",
        "model_weights": "RANDOM_FROM_TINY_CONFIG_NOT_OFFICIAL_WEIGHTS",
        "source_pretrained_weights_loaded": False,
        "gpu_execution_requested": False,
        "cpu_only": True,
        "cuda_runtime_initialized": False,
        "hybrid_layer_types": list(LAYER_KINDS),
        "gdn_linear_targets_injected": len(PROJECTIONS["linear_attention"]),
        "attention_linear_targets_injected": len(PROJECTIONS["full_attention"]),
        "total_lora_targets_injected": len(inject),
        "trainable_lora_tensors": len(trainable),
        "trainable_parameters": sum(int(p.numel()) for p in trainable.values()),
        "gradient_sum_gdn": gdn_grad,
        "gradient_sum_attention": attention_grad,
        "loss": float(loss.item()),
        "lora_b_sha256_before": before,
        "lora_b_sha256_after": after,
        "base_embedding_unchanged": True,
        "four_gb_gpu_fit": "UNKNOWN",
        "official_model_exercised": False,
        "training_authorized": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
