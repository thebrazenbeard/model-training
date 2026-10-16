"""FP32-exact model-side LoRA rehydration for inference-only frozen adapters.

PEFT can downcast saved FP32 LoRA matrices when attached to a BF16 base.
Restore exact saved bytes in memory after PEFT load; never mutate model files.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as fd:
        for block in iter(lambda:fd.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def source_key(name: str) -> str:
    if not ("lora_A.default.weight" in name or "lora_B.default.weight" in name):
        raise ValueError("non-LoRA frozen parameter in inference adapter: "+name)
    return (name.replace("lora_A.default.weight","lora_A.weight")
                .replace("lora_B.default.weight","lora_B.weight"))


def restore_frozen_fp32_lora(model, saved_path: Path, *, expected_sha: str,
                            expected_tensors: int, torch) -> dict:
    from safetensors import safe_open

    saved_path=Path(saved_path)
    current=sha256(saved_path)
    if current != expected_sha:
        raise ValueError("Saved adapter SHA256 mismatch")
    # Only LoRA in-memory tensors, never frozen base weights.
    lora={}
    for name,param in model.named_parameters():
        if "lora_" in name:
            key=source_key(name)
            if key in lora:
                raise ValueError("Duplicate named LoRA tensor")
            lora[key]=param
    if len(lora) != expected_tensors:
        raise ValueError("LoRA tensor count mismatch")
    with safe_open(str(saved_path),framework="pt",device="cpu") as fd:
        if set(fd.keys()) != set(lora):
            raise ValueError("LoRA saved tensors and model keys differ")
        for key,param in lora.items():
            tensor=fd.get_tensor(key)
            if tensor.dtype != torch.float32 or tuple(tensor.shape)!=tuple(param.shape):
                raise ValueError("Saved FP32 tensor dtype or shape drift: "+key)
            if not torch.isfinite(tensor).all().item():
                raise ValueError("Nonfinite adapter tensor: "+key)
        with torch.no_grad():
            for key,param in lora.items():
                # Assignment must retain precision, not upcast already-rounded weights.
                exact=fd.get_tensor(key).to(device=param.device,dtype=torch.float32)
                param.data=exact.contiguous()
                param.requires_grad_(False)
                if not torch.equal(param.detach(),exact):
                    raise ValueError("Failed exact inference-adapter readback: "+key)
    return {"restored_tensors":len(lora),"adapter_model_sha256":current,
            "saved_dtype":"float32","frozen":True,"model_files_mutated":False}
