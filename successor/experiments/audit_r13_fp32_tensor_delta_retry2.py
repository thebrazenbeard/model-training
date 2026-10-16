"""Read-only R11→R12 tensor, dtype, lineage and exact-weight audit."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

BASE = Path(r"D:\VERA")
PREV = BASE / "models/adapters/v10r3r12-vera-fp32-qwen35-r11-resume-20261015-trl-retry2/adapter/adapter_model.safetensors"
NEW_DIR = BASE / "models/adapters/v10r3r13-vera-identity-judgment-qwen35-r12-resume-20261016-retry2"
NEW = NEW_DIR / "adapter/adapter_model.safetensors"
OUT = BASE / "portal-task-state/VERA_R13_FP32_MECHANICAL_AUDIT_20261016_RETRY2.json"
PREV_SHA = "a80f5e36a1a2319029309521fc88f6575215ac99d488c6a4bbf883afac53d72c"
SPEC_SHA = "0127d7e88f3ed608117d2a1388d0c82a3d921df75fbf408374db3945b1060418"
CORPUS_SHA = "90de3879113a99f85002ec884d504ee487e5ed699d9deaa0146290c47597b3c5"


def sha(path):
    hash = hashlib.sha256()
    with path.open("rb") as fd:
        for part in iter(lambda: fd.read(4 * 1024 * 1024), b""):
            hash.update(part)
    return hash.hexdigest()


def audit():
    if OUT.exists():
        raise RuntimeError("R13 independent audit exists: no overwrite")
    if sha(PREV) != PREV_SHA:
        raise RuntimeError("Original R12 adapter hash mismatch")
    receipt = json.loads((NEW_DIR / "DEV_TRAINING_COMPLETE.json").read_text(encoding="utf8"))
    body = dict(receipt)
    checksum = body.pop("receipt_sha256", None)
    actual = hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf8")).hexdigest()
    if checksum != actual:
        raise RuntimeError("R13 receipt self hash mismatch")
    expected = receipt["adapter_artifacts"]["files"]["adapter_model.safetensors"]
    if (receipt["status"] != "DEVELOPMENT_OPTIMIZER_STEP_COMPLETE"
        or receipt["cumulative_optimizer_steps"] != 84
        or receipt["resume_adapter"]["source_cumulative_optimizer_steps"] != 52
        or receipt["resume_adapter"]["adapter_model_sha256"] != PREV_SHA
        or receipt["spec_sha256"] != SPEC_SHA or receipt["train_sha256"] != CORPUS_SHA
        or receipt["adapter_precision_policy"] != "FP32_RESUMED_LORA_TRAIN_AND_SAVE_V1"
        or receipt["deployment_status"] != "NOT_DEPLOYED"
        or receipt["adapter_precision_initial"]["verified_trainable_tensors"] != 496
        or receipt["adapter_precision_after_trainer"]["verified_trainable_tensors"] != 496
        or receipt["adapter_precision_final"]["verified_trainable_tensors"] != 496
        or receipt["adapter_precision_final"]["tensor_dtype"] != "float32"
        or receipt["exact_resume_rehydration"]["adapter_model_sha256"] != PREV_SHA
        or receipt["exact_resume_rehydration"]["restored_tensors"] != 496
        or receipt["exact_resume_rehydration"]["dtype"] != "float32"
        or sha(NEW) != expected):
        raise RuntimeError("R13 completed lineage, optimizer, precision or SHA mismatch")
    import torch
    from safetensors import safe_open
    delta_sum = 0.0
    l2sq = 0.0
    changed = 0
    changed_elements = 0
    total_elements = 0
    max_delta = 0.0
    with safe_open(str(PREV), framework="pt",device="cpu") as old, safe_open(str(NEW),framework="pt",device="cpu") as new:
        keys = sorted(old.keys())
        if len(keys) != 496 or set(keys) != set(new.keys()):
            raise RuntimeError("Unexpected R13 LoRA key topology")
        for k in keys:
            a = old.get_tensor(k)
            b = new.get_tensor(k)
            if a.dtype != torch.float32 or b.dtype != torch.float32 or a.shape != b.shape:
                raise RuntimeError("R13 saved LoRA tensor not FP32 or topology changed: " + k)
            if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
                raise RuntimeError("Nonfinite adapter tensor: " + k)
            delta = b - a.float()
            n = int(torch.count_nonzero(delta).item())
            changed += int(n > 0)
            changed_elements += n
            total_elements += int(delta.numel())
            delta_sum += float(delta.abs().sum().item())
            l2sq += float((delta*delta).sum().item())
            max_delta = max(max_delta, float(delta.abs().max().item()))
    if changed == 0 or not math.isfinite(l2sq):
        raise RuntimeError("No finite R13 adapter update")
    result = {
        "schema": "VERA_R13_FP32_INDEPENDENT_MECHANICAL_AUDIT_V1",
        "status": "TENSOR_PRECISION_AND_DELTA_VERIFIED_BEHAVIOR_UNQUALIFIED",
        "r12_sha256": PREV_SHA, "r13_sha256": expected,
        "r13_receipt_self_sha256": actual,
        "optimizer_cumulative_steps": 84,
        "tensor_count": len(keys), "changed_tensors": changed,
        "changed_elements": changed_elements, "total_elements": total_elements,
        "global_l2_delta": math.sqrt(l2sq),
        "absolute_delta_sum": delta_sum,
        "max_absolute_delta": max_delta,
        "saved_child_dtype": "float32",
        "parent_dtype": "float32",
        "adapter_file_mutated_by_audit": False,
        "behavior_qualified": False,
        "deployment_authorized": False
    }
    with OUT.open("x",encoding="utf8") as fd:
        json.dump(result,fd,indent=2,sort_keys=True)
        fd.write("\n")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(),indent=2,sort_keys=True))
