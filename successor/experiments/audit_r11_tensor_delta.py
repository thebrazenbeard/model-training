"""Read-only R10->R11 adapter delta and receipt integrity audit (not behavioral qualification)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(r"D:\VERA")
OLD = ROOT / "models/adapters/v10r3r10-vera-voice-qwen35-r9-resume-20261013/adapter/adapter_model.safetensors"
NEW_DIR = ROOT / "models/adapters/v10r3r11-vera-voice-qwen35-r10-resume-20261015-pagefile-retry1"
NEW = NEW_DIR / "adapter/adapter_model.safetensors"
OUT = ROOT / "portal-task-state/VERA_R11_INDEPENDENT_TENSOR_DELTA_AUDIT_20261015_V1.json"
SHA10 = "c69ed0e4bd78e09752b5cbb4dfaf19ebea8a4174735bf88d2676947ef04b2739"
SHA11 = "45f056297a55e19b22918d5298bcd5a9808fb0d8b8f5bf4210ebac91e4b88e10"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as fd:
        for chunk in iter(lambda: fd.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def receipt_integrity():
    receipt = json.loads((NEW_DIR / "DEV_TRAINING_COMPLETE.json").read_text(encoding="utf8"))
    content = dict(receipt)
    asserted = content.pop("receipt_sha256")
    actual = hashlib.sha256(json.dumps(
        content, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf8")).hexdigest()
    if asserted != actual:
        raise RuntimeError("R11 completion receipt self-hash mismatch")
    if (receipt["cumulative_optimizer_steps"] != 44
        or receipt["resume_adapter"]["source_cumulative_optimizer_steps"] != 36
        or receipt["resume_adapter"]["adapter_model_sha256"] != SHA10
        or receipt["adapter_artifacts"]["files"]["adapter_model.safetensors"] != SHA11
        or receipt["qualification_status"] != "NOT_EXTERNALLY_EVALUATED"
        or receipt["deployment_status"] != "NOT_DEPLOYED"):
        raise RuntimeError("R11 lineage or authority ceiling mismatch")
    return receipt, actual


def audit():
    if OUT.exists():
        raise RuntimeError("Previous R11 tensor audit exists; refuse overwrite")
    if sha(OLD) != SHA10 or sha(NEW) != SHA11:
        raise RuntimeError("Adapter provenance SHA mismatch")
    receipt, signature = receipt_integrity()
    import torch
    from safetensors import safe_open
    changed = 0
    nonzero = 0
    count = 0
    l2sq = 0.0
    abs_sum = 0.0
    largest = [0.0, ""]
    with safe_open(str(OLD), framework="pt", device="cpu") as old, safe_open(str(NEW), framework="pt", device="cpu") as new:
        keys = sorted(old.keys())
        if len(keys) != 496 or set(keys) != set(new.keys()):
            raise RuntimeError("Adapter key topology drift")
        for key in keys:
            a = old.get_tensor(key)
            b = new.get_tensor(key)
            if a.shape != b.shape or a.dtype != b.dtype:
                raise RuntimeError("Incompatible adapter tensors")
            if not (torch.isfinite(a).all().item() and torch.isfinite(b).all().item()):
                raise RuntimeError("Nonfinite adapter tensor: " + key)
            delta = b.float() - a.float()
            n = int(torch.count_nonzero(delta).item())
            nonzero += n
            count += delta.numel()
            changed += int(n != 0)
            abs_sum += float(delta.abs().sum().item())
            l2sq += float((delta * delta).sum().item())
            mx = float(delta.abs().max().item())
            if mx > largest[0]:
                largest = [mx, key]
    if changed == 0 or not math.isfinite(l2sq):
        raise RuntimeError("No finite adapter weight change")
    result = {
        "schema": "VERA_R11_INDEPENDENT_TENSOR_DELTA_AUDIT_V1",
        "status": "LOCAL_ADAPTER_RECEIPT_AND_TENSOR_DELTA_VERIFIED",
        "parent_r10_sha256": SHA10, "child_r11_sha256": SHA11,
        "receipt_self_hash": signature, "cumulative_optimizer_steps": 44,
        "tensor_count": len(keys), "changed_tensors": changed,
        "total_elements": count, "changed_elements": nonzero,
        "absolute_delta_sum": abs_sum, "global_l2_delta": math.sqrt(l2sq),
        "maximum_absolute_delta": largest[0], "max_delta_tensor": largest[1],
        "behavior_qualified": False, "independent_behavioral_review": False,
        "deployment_authorized": False, "adapters_mutated": False
    }
    with OUT.open("x", encoding="utf8") as fd:
        json.dump(result, fd, indent=2, sort_keys=True)
        fd.write("\n")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
