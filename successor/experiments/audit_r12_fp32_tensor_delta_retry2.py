"""Read-only R11→R12 tensor, dtype, lineage and exact-weight audit."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

BASE = Path(r"D:\VERA")
PREV = BASE / "models/adapters/v10r3r11-vera-voice-qwen35-r10-resume-20261015-pagefile-retry1/adapter/adapter_model.safetensors"
NEW_DIR = BASE / "models/adapters/v10r3r12-vera-fp32-qwen35-r11-resume-20261015-trl-retry2"
NEW = NEW_DIR / "adapter/adapter_model.safetensors"
OUT = BASE / "portal-task-state/VERA_R12_FP32_MECHANICAL_AUDIT_20261015_RETRY2.json"
PREV_SHA = "45f056297a55e19b22918d5298bcd5a9808fb0d8b8f5bf4210ebac91e4b88e10"
SPEC_SHA = "be394d5c7e94842581d4a4f30f8ce1b21fed4237ed1c4b1e09058f800d7615f2"
CORPUS_SHA = "ca46a0340c370e2284cafcf2f76c439a5ff0826c4c8b93960f26db9573bf09cf"


def sha(path):
    hash = hashlib.sha256()
    with path.open("rb") as fd:
        for part in iter(lambda: fd.read(4 * 1024 * 1024), b""):
            hash.update(part)
    return hash.hexdigest()


def audit():
    if OUT.exists():
        raise RuntimeError("R12 independent audit exists: no overwrite")
    if sha(PREV) != PREV_SHA:
        raise RuntimeError("Original R11 adapter hash mismatch")
    receipt = json.loads((NEW_DIR / "DEV_TRAINING_COMPLETE.json").read_text(encoding="utf8"))
    body = dict(receipt)
    checksum = body.pop("receipt_sha256", None)
    actual = hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf8")).hexdigest()
    if checksum != actual:
        raise RuntimeError("R12 receipt self hash mismatch")
    expected = receipt["adapter_artifacts"]["files"]["adapter_model.safetensors"]
    if (receipt["status"] != "DEVELOPMENT_OPTIMIZER_STEP_COMPLETE"
        or receipt["cumulative_optimizer_steps"] != 52
        or receipt["resume_adapter"]["source_cumulative_optimizer_steps"] != 44
        or receipt["resume_adapter"]["adapter_model_sha256"] != PREV_SHA
        or receipt["spec_sha256"] != SPEC_SHA or receipt["train_sha256"] != CORPUS_SHA
        or receipt["adapter_precision_policy"] != "FP32_RESUMED_LORA_TRAIN_AND_SAVE_V1"
        or receipt["deployment_status"] != "NOT_DEPLOYED"
        or receipt["adapter_precision_initial"]["verified_trainable_tensors"] != 496
        or receipt["adapter_precision_after_trainer"]["verified_trainable_tensors"] != 496
        or receipt["adapter_precision_final"]["verified_trainable_tensors"] != 496
        or receipt["adapter_precision_final"]["tensor_dtype"] != "float32"
        or sha(NEW) != expected):
        raise RuntimeError("R12 completed lineage, optimizer, precision or SHA mismatch")
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
            raise RuntimeError("Unexpected R12 LoRA key topology")
        for k in keys:
            a = old.get_tensor(k)
            b = new.get_tensor(k)
            if a.dtype != torch.bfloat16 or b.dtype != torch.float32 or a.shape != b.shape:
                raise RuntimeError("R12 saved LoRA tensor not FP32 or topology changed: " + k)
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
        raise RuntimeError("No finite R12 adapter update")
    result = {
        "schema": "VERA_R12_FP32_INDEPENDENT_MECHANICAL_AUDIT_V1",
        "status": "TENSOR_PRECISION_AND_DELTA_VERIFIED_BEHAVIOR_UNQUALIFIED",
        "r11_sha256": PREV_SHA, "r12_sha256": expected,
        "r12_receipt_self_sha256": actual,
        "optimizer_cumulative_steps": 52,
        "tensor_count": len(keys), "changed_tensors": changed,
        "changed_elements": changed_elements, "total_elements": total_elements,
        "global_l2_delta": math.sqrt(l2sq),
        "absolute_delta_sum": delta_sum,
        "max_absolute_delta": max_delta,
        "saved_child_dtype": "float32",
        "parent_dtype": "bfloat16",
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
