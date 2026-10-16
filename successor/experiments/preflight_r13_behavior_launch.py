"""R13 local training gate: preserve R12 and never interrupt a serving process."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import subprocess
import sys

from successor.experiments.build_r13_behavior_corrective import REPO, SPEC as SPEC_PATH, OUT_DIR, NEW_ADAPTER, CLASS
from successor.experiments.train_v10r2_dev import load_and_validate_dev_spec, verify_resume_adapter
from successor.experiments.train_v10_qwen35_authorized import sha256_file
from successor.experiments.preflight_v10r3r11_launch import windows_memory, gpu_free, competing_trainers

RECEIPT = Path(r"D:\VERA\portal-task-state\VERA_R13_BEHAVIOR_LAUNCH_GATE_20261016_V1.json")
MIN_RAM_GIB = 9
MIN_VIRTUAL_GIB = 12
MIN_GPU_MIB = 3900

def validated_receipt(path: Path) -> Path:
    path = path.resolve()
    if path.parent != RECEIPT.parent.resolve() or not re.fullmatch(
        r"VERA_R13_BEHAVIOR_LAUNCH_GATE_20261016_V[1-9][0-9]*\.json", path.name
    ):
        raise RuntimeError("Unsafe R13 gate receipt destination")
    if path.exists():
        raise RuntimeError("Gate receipt exists; never overwrite prior attempt")
    return path


def classify(*, ram: float, virtual: float, gpu: int, active: int, output_exists: bool):
    reasons = []
    if ram < MIN_RAM_GIB:
        reasons.append("LOW_FREE_HOST_RAM")
    if virtual < MIN_VIRTUAL_GIB:
        reasons.append("LOW_FREE_VIRTUAL_COMMIT")
    if gpu < MIN_GPU_MIB:
        reasons.append("GPU_OCCUPIED_PRESERVE_SERVING_PROCESS")
    if active:
        reasons.append("ANOTHER_TRAINER_ACTIVE")
    if output_exists:
        reasons.append("OUTPUT_EXISTS_PRESERVE_ADAPTER")
    return {"status": "READY_LOCAL_R13" if not reasons else "HOLD_NO_OPTIMIZER",
            "reasons": reasons, "gpu_free_mib": gpu,
            "host_ram_free_gib": round(ram,2),
            "virtual_commit_free_gib": round(virtual,2),
            "competing_trainers": active, "optimizer_steps_executed": 0,
            "services_interrupted": False, "model_deployed": False}
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()
    receipt_path = validated_receipt(args.receipt)
    spec = load_and_validate_dev_spec(SPEC_PATH)
    bound = verify_resume_adapter(REPO, spec["resume_adapter"])
    corpus = OUT_DIR / "train.jsonl"
    if (sha256_file(SPEC_PATH) != "ad2a3d56c9a5ee648764e9469676b78155bb993de690edda9627189a767b0fdf"
        or spec["experiment_class"] != CLASS
        or spec["development_window"] != {"start_row":416,"row_count":256}
        or spec["output"]["namespace"] != str(NEW_ADAPTER)
        or sha256_file(corpus) != spec["source_subject"]["train_sha256"]
        or spec["adapter_precision_policy"] != "FP32_RESUMED_LORA_TRAIN_AND_SAVE_V1"
        or bound["source_cumulative_optimizer_steps"] != 52
        or spec["trainer"]["max_optimizer_steps"] != 32
        or spec["trainer"]["learning_rate"] != 1e-5):
        raise RuntimeError("R13 subject/precision/parent frozen binding mismatch")
    ram, virtual = windows_memory()
    gpu = gpu_free()
    active = competing_trainers()
    result = classify(ram=ram,virtual=virtual,gpu=gpu,active=active,
                      output_exists=NEW_ADAPTER.exists())
    result.update({"schema": "VERA_R13_BEHAVIOR_TRAINING_GATE_V1",
                   "source_spec_sha256": sha256_file(SPEC_PATH),
                   "r13_corpus_sha256": sha256_file(corpus),
                   "parent_r12_sha256": bound["adapter_model_sha256"],
                   "action_requested": "TRAIN" if args.train else "GATE_ONLY",
                   "train_launched": False})
    with receipt_path.open("x", encoding="utf8") as fd:
        json.dump(result,fd,indent=2,sort_keys=True)
        fd.write("\n")
    print(json.dumps(result, indent=2,sort_keys=True),flush=True)
    if result["status"] != "READY_LOCAL_R13" or not args.train:
        return 2 if result["status"] != "READY_LOCAL_R13" else 0
    # No service stops, forced Ollama unload, model route mutation, or output overwrite.
    cmd = [sys.executable,"-m","successor.experiments.train_v10r2_dev",
           "--repo-root",str(REPO),"--train-jsonl",str(corpus),"--spec",str(SPEC_PATH)]
    return subprocess.call(cmd, cwd=REPO)


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("R13_GATE_HOLD_NO_WEIGHT_CHANGE: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
