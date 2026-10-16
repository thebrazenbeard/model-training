"""Restore the exact verified FP32 LoRA checkpoint after PEFT/TRL BF16 recasts.

This only writes the in-memory trainable LoRA tensors of a disposable candidate.
Never mutates checkpoint files, base weights, optimizer state, or serving models.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def saved_name(trainable_name: str) -> str:
    if not ("lora_A.default.weight" in trainable_name or
            "lora_B.default.weight" in trainable_name):
        raise ValueError("non-LoRA/unknown adapter trainable parameter: "+trainable_name)
    return (trainable_name.replace("lora_A.default.weight", "lora_A.weight")
                          .replace("lora_B.default.weight", "lora_B.weight"))


def restore_exact_lora(model, source_path: Path, *, expected_sha: str,
                       expected_tensors: int, torch) -> dict:
    """Rehydrate verified FP32 checkpoint into post-TRL FP32 parameters.

    PEFT may silently downcast a saved FP32 adapter to BF16 when attaching to a
    BF16 model. Converting it back to FP32 does not recover lost low bits.
    Therefore copy the immutable source tensor AFTER SFTTrainer construction.
    """
    from safetensors import safe_open

    source_path = Path(source_path)
    sha = _sha(source_path)
    if sha != expected_sha:
        raise ValueError("Resume adapter SHA mismatch; no writes permitted")
    trainable = {name: param for name,param in model.named_parameters()
                 if param.requires_grad}
    named_source = {saved_name(name): (name,param)
                    for name,param in trainable.items()}
    if len(trainable) != expected_tensors:
        raise ValueError("Trainable LoRA tensor count mismatch")
    if len(named_source) != len(trainable):
        raise ValueError("LoRA trainable name mapping collision")
    with safe_open(str(source_path), framework="pt", device="cpu") as fd:
        available = set(fd.keys())
        if available != set(named_source):
            raise ValueError("Saved LoRA tensor topology/name mismatch")
        # Validate shapes, dtypes and finite values before first assignment.
        for key,(name,param) in named_source.items():
            t=fd.get_tensor(key)
            if (t.dtype != torch.float32 or param.dtype != torch.float32 or
                tuple(t.shape) != tuple(param.shape)):
                raise ValueError("FP32 LoRA tensor dtype/shape mismatch: "+key)
            if not torch.isfinite(t).all().item():
                raise ValueError("Nonfinite saved LoRA tensor: "+key)
        with torch.no_grad():
            for key,(name,param) in named_source.items():
                exact=fd.get_tensor(key).to(device=param.device,dtype=torch.float32)
                param.data.copy_(exact)
                if not torch.equal(param.detach(),exact):
                    raise ValueError("FP32 LoRA exact restoration readback mismatch: "+key)
    return {
        "schema": "VERA_EXACT_FP32_RESUME_REHYDRATION_V1",
        "source": str(source_path),
        "adapter_model_sha256": sha,
        "restored_tensors": len(trainable),
        "dtype": "float32",
        "after_trainer_constructor": True,
        "file_mutated": False,
    }
