"""Read-only base/R11/R12 first-token logit ablation on frozen development voice prompts.

Exact source and adapter hashes are required. No generation, optimizer, checkpoint
save, system coaching, deployment, or semantic qualification is performed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
import subprocess

ROOT = Path(r"D:\VERA")
HERE = Path(__file__).resolve().parents[2]
BASE = ROOT / "models/latest-trained/base"
R11 = ROOT / "models/adapters/v10r3r11-vera-voice-qwen35-r10-resume-20261015-pagefile-retry1/adapter"
R12 = ROOT / "models/adapters/v10r3r12-vera-fp32-qwen35-r11-resume-20261015-trl-retry2/adapter"
VOICE = ROOT / "portal-task-state/vera-model-glm24b-lappy-20261012/evaluation/cases/vera_voice_heldout_v1.jsonl"
OUTPUT = ROOT / "portal-task-state/VERA_R12_R11_BASE_LOGIT_ABLATION_20261015_V2.json"
R11_SHA = "45f056297a55e19b22918d5298bcd5a9808fb0d8b8f5bf4210ebac91e4b88e10"
R12_SHA = "a80f5e36a1a2319029309521fc88f6575215ac99d488c6a4bbf883afac53d72c"
VOICE_SHA = "bd2b5375e2b7da78f42ed3ac1387f73ddec4807bc5e446b228bdbb7db3caf25f"
IDS = ("VERA_VOICE_HELDOUT_001", "VERA_VOICE_HELDOUT_007", "VERA_VOICE_HELDOUT_011")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def describe_delta(a, b):
    """Pure tensor metric: ordered a-b, with no effects on either tensor."""
    import torch
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("Wrong first-token logits shape")
    if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
        raise ValueError("Nonfinite first-token logits")
    delta = a.float() - b.float()
    return {
        "max_abs_logit_delta": float(delta.abs().max().item()),
        "mean_abs_logit_delta": float(delta.abs().mean().item()),
        "l2_logit_delta": float(torch.linalg.vector_norm(delta).item()),
        "changed_logit_elements": int(torch.count_nonzero(delta).item()),
        "top_token_differs": bool(torch.argmax(a).item() != torch.argmax(b).item()),
    }

def main():
    if OUTPUT.exists():
        raise RuntimeError("Ablation output already exists; preserve it")
    git_args = ["git", "-C", str(HERE)]
    head = subprocess.run(git_args + ["rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    if len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        raise RuntimeError("Unknown Git source revision")
    dirty = subprocess.run(git_args + ["status", "--porcelain"], check=True,
                           capture_output=True, text=True).stdout
    if dirty.strip():
        raise RuntimeError("Read-only ablation source must be committed and clean")
    if sha(R11 / "adapter_model.safetensors") != R11_SHA:
        raise RuntimeError("R11 preserved parent hash drift")
    if sha(R12 / "adapter_model.safetensors") != R12_SHA:
        raise RuntimeError("R12 saved adapter hash drift")
    if sha(VOICE) != VOICE_SHA:
        raise RuntimeError("Frozen voice holdout hash drift")
    import probe_vera_r12_heldout_generation_retry2 as probe
    if probe.verified_r12_sha() != R12_SHA:
        raise RuntimeError("Completed R12 receipt or lineage mismatch")
    resources = probe.stat_resources()
    if (resources["gpu_free_mib"] < 3900
        or resources["ram_free_gib"] < 7
        or resources["virtual_free_gib"] < 6):
        raise RuntimeError("Not enough safe local inference headroom")
    print("ABLATION_PREFLIGHT_PASS", resources, flush=True)
    selected = {}
    for line in VOICE.read_text(encoding="utf8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["id"] in IDS:
                if row["mode"] != "NO_SYSTEM_FIRST":
                    raise RuntimeError("Holdout mode changed")
                selected[row["id"]] = row
    if set(selected) != set(IDS):
        raise RuntimeError("Frozen holdout membership changed")
    stack = probe.load_stack()
    torch = stack["torch"]
    tokenizer = stack["AutoTokenizer"].from_pretrained(BASE)
    quant = stack["BitsAndBytesConfig"](
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = stack["Qwen3_5ForCausalLM"].from_pretrained(
        BASE, quantization_config=quant, device_map={"":0}, dtype=torch.bfloat16
    )
    model = stack["peft"].PeftModel.from_pretrained(
        model, R12, adapter_name="r12",
        is_trainable=False, autocast_adapter_dtype=False
    )
    model.load_adapter(R11, adapter_name="r11",
                       is_trainable=False, autocast_adapter_dtype=False)
    model.eval()
    results = []
    with torch.inference_mode():
        for case_id in IDS:
            row = selected[case_id]
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["user"]}],
                tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_tensors="pt",
            )
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
            if ids.ndim != 2 or ids.shape[1] > 512:
                raise RuntimeError("Prompt token budget violation")
            ids = ids.to("cuda")
            mask = torch.ones_like(ids)
            outputs = {}
            for adapter in ("r11", "r12"):
                model.set_adapter(adapter)
                logits = model(input_ids=ids, attention_mask=mask).logits
                outputs[adapter] = logits[0, -1].detach().float().cpu()
                del logits
            with model.disable_adapter():
                logits = model(input_ids=ids, attention_mask=mask).logits
                outputs["base"] = logits[0,-1].detach().float().cpu()
                del logits
            result = {
                "case_id": case_id,
                "axis": row["axis"],
                "prompt_sha256": hashlib.sha256(row["user"].encode("utf8")).hexdigest(),
                "token_count": int(ids.shape[1]),
                "arms": {
                    name: {
                        "top_token_id": int(torch.argmax(vector).item()),
                        "top_token_text": tokenizer.decode([int(torch.argmax(vector).item())]),
                    } for name, vector in outputs.items()
                },
                "r12_minus_r11": describe_delta(outputs["r12"],outputs["r11"]),
                "r12_minus_base": describe_delta(outputs["r12"],outputs["base"]),
                "r11_minus_base": describe_delta(outputs["r11"],outputs["base"]),
            }
            results.append(result)
            print("ABLATION_CASE",case_id, result["r12_minus_r11"],flush=True)
            del ids,mask,outputs
    report = {
        "schema": "VERA_R12_BASE_R11_LOGIT_ABLATION_V1",
        "status": "INFERENCE_LOGIT_DIFFERENCE_MEASURED_NOT_BEHAVIOR_QUALIFICATION",
        "source_head": head,
        "source_script_sha256": sha(Path(__file__).resolve()),
        "r11_adapter_sha256": R11_SHA,
        "r12_adapter_sha256": R12_SHA,
        "heldout_sha256": VOICE_SHA,
        "no_system_prompt": True,
        "optimizer_steps_performed": 0,
        "model_files_mutated": False,
        "results": results,
        "behavior_qualified": False,
        "independent_semantic_scoring": False,
    }
    with OUTPUT.open("x",encoding="utf8") as out:
        json.dump(report,out,sort_keys=True,indent=2)
        out.write("\n")
    print("ABLATION_DONE",OUTPUT,flush=True)


if __name__ == "__main__":
    main()
