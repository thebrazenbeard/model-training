"""R12 local training gate: preserve R11 and never interrupt a serving process."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import subprocess
import sys

from successor.experiments.build_r12_voice_corrective import REPO, OUT_DIR
SPEC_PATH = Path(__file__).with_name("V10R3R12_FP32_VERA_VOICE_EXECUTION_SPEC_20261015_RETRY2.json")
NEW_ADAPTER = Path(r"D:\VERA\models\adapters\v10r3r12-vera-fp32-qwen35-r11-resume-20261015-retry2")
EXPECTED_SPEC_SHA = "9e6ed55d700935a79fca1d75ae1585a7585cac08cd0682ea576afaf3ede40b83"
from successor.experiments.train_v10r2_dev import load_and_validate_dev_spec, verify_resume_adapter
from successor.experiments.train_v10_qwen35_authorized import sha256_file
from successor.experiments.preflight_v10r3r11_launch import windows_memory, gpu_free, competing_trainers

RECEIPT = Path(r"D:\VERA\portal-task-state\VERA_R12_FP32_LAUNCH_GATE_20261015_V3.json")
MIN_RAM_GIB = 9
MIN_VIRTUAL_GIB = 12
MIN_GPU_MIB = 3900

def validated_receipt(path: Path) -> Path:
    path = path.resolve()
    if path.parent != RECEIPT.parent.resolve() or not re.fullmatch(
        r"VERA_R12_FP32_LAUNCH_GATE_20261015_V[1-9][0-9]*\.json", path.name
    ):
        raise RuntimeError("Unsafe R12 gate receipt destination")
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
    return {"status": "READY_LOCAL_R12" if not reasons else "HOLD_NO_OPTIMIZER",
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
    if sha256_file(SPEC_PATH) != EXPECTED_SPEC_SHA:
        raise RuntimeError("Retry2 frozen source spec SHA drift")
    spec = load_and_validate_dev_spec(SPEC_PATH)
    if spec["output"]["namespace"] != str(NEW_ADAPTER):
        raise RuntimeError("Retry2 output namespace mismatched")
    bound = verify_resume_adapter(REPO, spec["resume_adapter"])
    corpus = OUT_DIR / "train.jsonl"
    if (sha256_file(corpus) != spec["source_subject"]["train_sha256"]
        or spec["adapter_precision_policy"] != "FP32_RESUMED_LORA_TRAIN_AND_SAVE_V1"
        or bound["source_cumulative_optimizer_steps"] != 44
        or spec["trainer"]["max_optimizer_steps"] != 8
        or spec["trainer"]["learning_rate"] != 2e-5):
        raise RuntimeError("R12 subject/precision/parent frozen binding mismatch")
    ram, virtual = windows_memory()
    gpu = gpu_free()
    active = competing_trainers()
    result = classify(ram=ram,virtual=virtual,gpu=gpu,active=active,
                      output_exists=NEW_ADAPTER.exists())
    result.update({"schema": "VERA_R12_FP32_TRAINING_GATE_V1",
                   "source_spec_sha256": sha256_file(SPEC_PATH),
                   "r12_corpus_sha256": sha256_file(corpus),
                   "parent_r11_sha256": bound["adapter_model_sha256"],
                   "action_requested": "TRAIN" if args.train else "GATE_ONLY",
                   "train_launched": False})
    with receipt_path.open("x", encoding="utf8") as fd:
        json.dump(result,fd,indent=2,sort_keys=True)
        fd.write("\n")
    print(json.dumps(result, indent=2,sort_keys=True),flush=True)
    if result["status"] != "READY_LOCAL_R12" or not args.train:
        return 2 if result["status"] != "READY_LOCAL_R12" else 0
    # No service stops, forced Ollama unload, model route mutation, or output overwrite.
    cmd = [sys.executable,"-m","successor.experiments.train_v10r2_dev",
           "--repo-root",str(REPO),"--train-jsonl",str(corpus),"--spec",str(SPEC_PATH)]
    return subprocess.call(cmd, cwd=REPO)


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("R12_GATE_HOLD_NO_WEIGHT_CHANGE: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
