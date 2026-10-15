"""Explicit, fail-closed precision policy for resumed Vera LoRA weights.

Keeping resumed trainable adapters in float32 avoids BF16 rounding of small
optimizer updates. Base model quantization and optimizer remain unchanged.
"""
from __future__ import annotations

POLICY = "FP32_RESUMED_LORA_TRAIN_AND_SAVE_V1"


class AdapterPrecisionHold(ValueError):
    pass


def verify_trainable(model, torch, *, permit_cast: bool) -> dict:
    trainable = []
    cast_count = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" not in name or not param.is_floating_point():
            raise AdapterPrecisionHold("Unexpected trainable non-LoRA parameter: " + name)
        if param.dtype != torch.float32:
            if not permit_cast:
                raise AdapterPrecisionHold("Optimizer trainable parameter not FP32: " + name)
            param.data = param.data.to(dtype=torch.float32)
            cast_count += 1
        if param.dtype != torch.float32:
            raise AdapterPrecisionHold("Adapter FP32 conversion failed: " + name)
        trainable.append(name)
    if not trainable:
        raise AdapterPrecisionHold("No trainable LoRA tensors found")
    return {
        "policy": POLICY,
        "verified_trainable_tensors": len(trainable),
        "converted_from_lower_precision": cast_count,
        "tensor_dtype": "float32",
    }
