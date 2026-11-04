"""Measure idealized 152-target LoRA state on the *official text meta skeleton*.

This does not allocate pretrained weights, use CUDA, or attempt an optimizer
step. The arithmetic omits activations, quantization metadata, CUDA workspace,
LoRA framework overhead, and optimizer implementation overhead. It cannot
prove that the model fits a 4 GiB GPU or authorize physical training.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path

from trainingday.qwen35_meta_skeleton_probe import (
    OFFICIAL_CONFIG_SHA256, OFFICIAL_REVISION, MetaSkeletonProbeError,
    bound_text_meta_memory, verified_config,
)


class HybridMetaBudgetError(ValueError):
    """The pinned hybrid target or LoRA inventory cannot be substantiated."""


_GDN = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
        "linear_attn.in_proj_b", "linear_attn.in_proj_a",
        "linear_attn.out_proj")
_FULL = ("self_attn.q_proj", "self_attn.k_proj",
         "self_attn.v_proj", "self_attn.o_proj")
_EXPECTED_TARGETS = 152


def hybrid_projection_targets(layer_types: list[str]) -> list[str]:
    """Derive exact text-decoder projections, never broad PEFT suffixes."""
    if (type(layer_types) is not list or len(layer_types) != 32
            or any(type(kind) is not str
                   or kind not in ("linear_attention", "full_attention")
                   for kind in layer_types)
            or layer_types.count("linear_attention") != 24
            or layer_types.count("full_attention") != 8):
        raise HybridMetaBudgetError("pinned 24 GDN / 8 full-attention decoder missing")
    targets = [f"model.layers.{idx}.{suffix}"
               for idx, kind in enumerate(layer_types)
               for suffix in (_GDN if kind == "linear_attention" else _FULL)]
    if len(targets) != _EXPECTED_TARGETS or len(set(targets)) != len(targets):
        raise HybridMetaBudgetError("hybrid projection inventory incomplete")
    return targets


def expected_rank4_elements(linear_shapes: list[tuple[int, int]]) -> int:
    """Independently derive rank-four A+B elements from *base* linears."""
    if (type(linear_shapes) is not list or len(linear_shapes) != _EXPECTED_TARGETS
            or any(type(shape) is not tuple or len(shape) != 2
                   or any(type(dim) is not int or dim <= 0 for dim in shape)
                   for shape in linear_shapes)):
        raise HybridMetaBudgetError("152 positive linear in/out shapes required")
    return sum(4 * (in_features + out_features)
               for in_features, out_features in linear_shapes)


def bound_rank4_adapter_memory(*, adapter_elements: int,
                               a_b_tensor_count: int,
                               target_module_count: int) -> dict:
    """Optimistic BF16 adapter+grad and FP32 AdamW-moment *arithmetic*.

    Real PEFT defaults may keep LoRA tensors in FP32; this is intentionally a
    lower-bound *scenario* and omits all transient/staging allocations.
    """
    if (type(adapter_elements) is not int or adapter_elements <= 0
            or type(a_b_tensor_count) is not int or a_b_tensor_count != 304
            or type(target_module_count) is not int or target_module_count != 152):
        raise HybridMetaBudgetError("incomplete rank-four hybrid adapter inventory")
    return {
        "target_module_count": target_module_count,
        "trainable_lora_a_b_tensors": a_b_tensor_count,
        "adapter_parameter_elements": adapter_elements,
        "optimistic_bf16_adapter_weight_bytes": adapter_elements * 2,
        "optimistic_bf16_gradient_bytes": adapter_elements * 2,
        "optimistic_fp32_adamw_moment_bytes": adapter_elements * 8,
        "optimistic_training_state_bytes": adapter_elements * 12,
        "excluded": ["QUANTIZATION_METADATA", "BASE_MODEL_NONLINEAR_TENSORS",
                     "ACTIVATIONS", "CUDA_WORKSPACE", "FRAMEWORK_OVERHEAD",
                     "OPTIMIZER_TRANSIENTS", "WDDM_FOREIGN_PROCESS_CONTENTION"],
        "physical_fit": "UNKNOWN",
        "training_authorized": False,
        "optimizer_steps_observed": 0,
    }


def run_hybrid_meta_probe(config_path: Path) -> dict:
    """Inspect real PEFT modules for a 152-projection official meta model."""
    try:
        text = verified_config(config_path)
    except MetaSkeletonProbeError as exc:
        raise HybridMetaBudgetError("official config not independently pinned") from exc
    targets = hybrid_projection_targets(text.get("layer_types"))

    import torch
    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    if torch.version.cuda is not None:
        raise HybridMetaBudgetError("requires CPU-only PyTorch; CUDA disallowed")
    config = Qwen3_5TextConfig(**text)
    if list(config.layer_types) != text["layer_types"]:
        raise HybridMetaBudgetError("official text layer types drifted")
    with init_empty_weights(include_buffers=True):
        skeleton = Qwen3_5ForCausalLM(config)
        skeleton.tie_weights()
        if skeleton.get_input_embeddings().weight is not skeleton.get_output_embeddings().weight:
            raise HybridMetaBudgetError("pinned tied text embeddings not shared")
        original_modules = dict(skeleton.named_modules())
        for name in targets:
            if name not in original_modules or not isinstance(original_modules[name], torch.nn.Linear):
                raise HybridMetaBudgetError("target is missing or non-linear in actual model")
        predicted_adapter_elements = expected_rank4_elements([
            (original_modules[name].in_features, original_modules[name].out_features)
            for name in targets
        ])
        base_memory = bound_text_meta_memory(
            text_parameter_elements=sum(p.numel() for p in skeleton.parameters()),
            tied_embedding_elements=skeleton.get_input_embeddings().weight.numel(),
        )
        model = get_peft_model(
            skeleton, LoraConfig(
                r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                task_type="CAUSAL_LM", target_modules=targets,
            ),
        )

    parameters = tuple(model.named_parameters())
    if not parameters or any(parameter.device.type != "meta" for _, parameter in parameters):
        raise HybridMetaBudgetError("actual model tensors escaped meta device")
    trainable = {name: parameter for name, parameter in parameters if parameter.requires_grad}
    if len(trainable) != 2 * len(targets) or len({id(p) for p in trainable.values()}) != len(trainable):
        raise HybridMetaBudgetError("extra, aliased or missing trainable adapter tensors")

    seen: set[str] = set()
    for target in targets:
        source_weight_shape = tuple(original_modules[target].weight.shape)
        out_features, in_features = source_weight_shape
        for side, expected_shape in (("A", (4, in_features)), ("B", (out_features, 4))):
            suffix = f".{target}.lora_{side}.default.weight"
            hits = [(name, p) for name, p in trainable.items() if name.endswith(suffix)]
            if len(hits) != 1 or tuple(hits[0][1].shape) != expected_shape:
                raise HybridMetaBudgetError("PEFT rank-four adapter shape/name mismatch")
            seen.add(hits[0][0])
    if seen != set(trainable):
        raise HybridMetaBudgetError("foreign trainable parameters outside hybrid targets")
    actual_adapter_elements = sum(p.numel() for p in trainable.values())
    if actual_adapter_elements != predicted_adapter_elements:
        raise HybridMetaBudgetError("actual PEFT rank-four elements differ from base-shape formula")
    bound = bound_rank4_adapter_memory(
        adapter_elements=actual_adapter_elements,
        a_b_tensor_count=len(trainable), target_module_count=len(targets),
    )
    bound["observed_meta_lora_dtypes"] = sorted({str(p.dtype) for p in trainable.values()})
    return {
        "status": "OFFICIAL_TEXT_META_HYBRID_RANK4_SOURCE_ONLY_HOLD",
        "official_revision": OFFICIAL_REVISION,
        "official_config_sha256": OFFICIAL_CONFIG_SHA256,
        "frameworks": {name: metadata.version(name)
                       for name in ("torch", "transformers", "accelerate", "peft")},
        "text_model_class": type(skeleton).__name__,
        "meta_only": True,
        "base_text_meta_memory": base_memory,
        "hybrid_rank4_adapter_state": bound,
        "pretrained_weights_loaded": False,
        "weight_shards_authenticated": False,
        "cuda_used": False,
        "optimizer_steps_observed": 0,
        "physical_training_authorized": False,
        "hardware_fit": "UNKNOWN",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Official Qwen3.5 hybrid rank-four meta-only budget")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_hybrid_meta_probe(args.config), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
