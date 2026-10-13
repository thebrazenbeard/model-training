"""Fail-closed *Qwen-only* R10 launch gate; no model load on HOLD.

Usage: python -m successor.experiments.preflight_v10r3r10_launch ...
Default is read-only. --train executes the exact frozen R10 local trainer
only when model identity, R9 receipt, data, GPU and Windows commit reserve PASS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from successor.experiments.train_v10r2_dev import (
    load_and_validate_dev_spec, verify_resume_adapter,
)
from successor.experiments.train_v10_qwen35_authorized import sha256_file

EXPECTED_VOICE = "Vera"
EXPECTED_CORPUS_SHA = "4df8bc7a89d201dff12250399ed0bc79c78d1237ee99ae9b0ad5f3238263fdba"
EXPECTED_R9_SHA = "dc2a67a66c16c28a3b8cd5295a3e7a6ba243fb12ffb181540a0066b32047e495"
MIN_FREE_RAM_GIB = 8
MIN_FREE_VIRTUAL_GIB = 6
MIN_FREE_GPU_MIB = 3900
EXPECTED_START_ROW = 224
EXPECTED_ROW_COUNT = 64
EXPECTED_STEPS = 8


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def classify(*, model_type: str, adapter_sha: str, corpus_sha: str,
             free_ram_gib: float, free_virtual_gib: float,
             free_gpu_mib: int, live_trainer_count: int,
             output_already_exists: bool, step_count: int = 8) -> dict:
    problems = []
    if model_type != "qwen3_5":
        problems.append("WRONG_MODEL_NOT_QWEN35")
    if adapter_sha != EXPECTED_R9_SHA:
        problems.append("WRONG_PARENT_NOT_R9")
    if corpus_sha != EXPECTED_CORPUS_SHA:
        problems.append("WRONG_CORPUS_NOT_VERA_R10")
    if step_count != EXPECTED_STEPS:
        problems.append("UNAPPROVED_OPTIMIZER_STEP_COUNT")
    if output_already_exists:
        problems.append("OUTPUT_EXISTS_NO_OVERWRITE")
    if live_trainer_count != 0:
        problems.append("EXISTING_TRAINER_DO_NOT_COMPETE")
    if free_ram_gib < MIN_FREE_RAM_GIB:
        problems.append("LOW_HOST_RAM")
    if free_virtual_gib < MIN_FREE_VIRTUAL_GIB:
        problems.append("LOW_WINDOWS_VIRTUAL_MEMORY_COMMIT")
    if free_gpu_mib < MIN_FREE_GPU_MIB:
        problems.append("LOW_GPU_FREE_MEMORY")
    return {
        "schema": "VERA_R10_QWEN35_LOCAL_LAUNCH_GATE_V1",
        "model": model_type,
        "voice": EXPECTED_VOICE,
        "parent_r9_sha256": adapter_sha,
        "corpus_sha256": corpus_sha,
        "status": "READY_FOR_R10_LOCAL_ONLY" if not problems else "HOLD_NO_WEIGHT_CHANGE",
        "reasons": problems,
        "gpu_mib_available": free_gpu_mib,
        "system_ram_gib_available": round(free_ram_gib, 2),
        "virtual_memory_gib_available": round(free_virtual_gib, 2),
        "optimizer_steps_executed": 0,
        "weights_changed": False,
        "deployed": False,
    }


def windows_memory() -> tuple[float, float]:
    cmd = ("$x=Get-CimInstance Win32_OperatingSystem;"
           " $x|Select-Object FreePhysicalMemory,FreeVirtualMemory|"
           "ConvertTo-Json -Compress")
    p = subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd],
                       check=True, capture_output=True, text=True, timeout=20)
    v = json.loads(p.stdout)
    return int(v["FreePhysicalMemory"]) / 1048576, int(v["FreeVirtualMemory"]) / 1048576


def gpu_free() -> int:
    p = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True, timeout=15)
    lines = [x.strip() for x in p.stdout.splitlines() if x.strip()]
    require(len(lines) == 1 and lines[0].isdigit(),
            "GPU readback missing or ambiguous")
    return int(lines[0])


def competing_trainers() -> int:
    cmd = ("Get-CimInstance Win32_Process | Where-Object {"
           "$_.Name -match 'python|accelerate' -and "
           "$_.CommandLine -match 'train_v10r2_dev|train_vera_lora|torchrun'} | "
           "Select-Object -ExpandProperty ProcessId | ConvertTo-Json -Compress")
    p = subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd],
                       check=True, capture_output=True, text=True, timeout=20)
    # The current preflight process is *not* one of those trainer commands.
    if not p.stdout.strip():
        return 0
    ids = json.loads(p.stdout)
    return len(ids) if isinstance(ids, list) else 1


def validate_subject(root: Path, spec_path: Path, corpus: Path) -> tuple[dict, dict]:
    spec = load_and_validate_dev_spec(spec_path)
    require(spec["experiment_class"] == "TARGETED_VERA_VOICE_AND_INDEPENDENT_JUDGMENT",
            "not the Qwen Vera R10 target")
    require(spec["development_window"] == {
        "start_row": EXPECTED_START_ROW, "row_count": EXPECTED_ROW_COUNT},
        "incorrect train window")
    require(spec["trainer"]["max_optimizer_steps"] == EXPECTED_STEPS,
            "unexpected optimizer steps")
    require(spec["trainer"]["learning_rate"] == 2e-6,
            "wrong identity learning rate")
    require(spec["comparison"]["source_24b_donor"] ==
            "REFERENCE_BEHAVIOR_ONLY_NOT_BASE_WEIGHTS",
            "Dolphin incorrectly admitted as parent model")
    require(spec["authority"]["paid_compute_authorized"] is False and
            spec["authority"]["deploy_activate_authorized"] is False,
            "training cannot authorize external spend or deployment")
    parent = verify_resume_adapter(root, spec["resume_adapter"])
    require(parent["source_cumulative_optimizer_steps"] == 28,
            "wrong cumulative parent steps")
    require(spec["source_subject"]["train_sha256"] == EXPECTED_CORPUS_SHA and
            sha256_file(corpus) == EXPECTED_CORPUS_SHA,
            "wrong R10 training corpus bytes")
    config_path = Path(r"D:\VERA\models\latest-trained\base\config.json")
    model = json.loads(config_path.read_text(encoding="utf8"))
    require(model.get("model_type") == "qwen3_5" and
            model.get("text_config", {}).get("num_hidden_layers") == 32,
            "wrong base architecture: HOLD")
    return spec, parent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True, type=Path)
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--receipt", required=True, type=Path)
    p.add_argument("--train", action="store_true")
    args = p.parse_args()
    require(not args.receipt.exists(), "launch gate receipt exists: no overwrite")
    require(args.receipt.parent.is_dir(), "receipt directory unavailable")
    spec, parent = validate_subject(args.repo_root, args.spec, args.corpus)
    model_type = "qwen3_5"
    ram_free, virtual_free = windows_memory()
    gpu_mib = gpu_free()
    active = competing_trainers()
    outcome = classify(
        model_type=model_type,
        adapter_sha=parent["adapter_model_sha256"],
        corpus_sha=sha256_file(args.corpus),
        free_ram_gib=ram_free, free_virtual_gib=virtual_free,
        free_gpu_mib=gpu_mib, live_trainer_count=active,
        output_already_exists=Path(spec["output"]["namespace"]).exists(),
        step_count=spec["trainer"]["max_optimizer_steps"])
    outcome["source_spec_sha256"] = sha256_file(args.spec)
    outcome["launch_attempted"] = False
    args.receipt.write_text(
        json.dumps(outcome, sort_keys=True, indent=2) + "\n", encoding="utf8")
    print(json.dumps(outcome, indent=2, sort_keys=True), flush=True)
    if outcome["status"] != "READY_FOR_R10_LOCAL_ONLY":
        return 2
    if not args.train:
        return 0
    # Source-controlled local trainer, exact single output path, no retries.
    command = [sys.executable, "-m", "successor.experiments.train_v10r2_dev",
               "--repo-root", str(args.repo_root), "--train-jsonl",
               str(args.corpus), "--spec", str(args.spec)]
    p = subprocess.run(command, cwd=args.repo_root, check=False)
    return p.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "HOLD_NO_WEIGHT_CHANGE",
                          "error": str(error)}), flush=True)
        raise SystemExit(2)
