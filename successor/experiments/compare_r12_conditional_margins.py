"""Development-only likelihood margin for three already-exposed voice heldouts.

No training, deployment, model writes, or independent semantic qualification.
Alternative answer openings are diagnostic contrasts, not approved training targets.
Requires exact frozen R11/R12 adapter receipts, project source, and free GPU headroom.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(r"D:\VERA")
SOURCE = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/latest-trained/base"
R11 = ROOT / "models/adapters/v10r3r11-vera-voice-qwen35-r10-resume-20261015-pagefile-retry1/adapter"
R12 = ROOT / "models/adapters/v10r3r12-vera-fp32-qwen35-r11-resume-20261015-trl-retry2/adapter"
VOICE = ROOT / "portal-task-state/vera-model-glm24b-lappy-20261012/evaluation/cases/vera_voice_heldout_v1.jsonl"
OUT = ROOT / "portal-task-state/VERA_R12_CONDITIONAL_VOICE_MARGINS_20261016_V1.json"
R11_SHA = "45f056297a55e19b22918d5298bcd5a9808fb0d8b8f5bf4210ebac91e4b88e10"
R12_SHA = "a80f5e36a1a2319029309521fc88f6575215ac99d488c6a4bbf883afac53d72c"
VOICE_SHA = "bd2b5375e2b7da78f42ed3ac1387f73ddec4807bc5e446b228bdbb7db3caf25f"
# Chosen diagnostic openings are distinct from the frozen heldout prompts and
# did not form training examples. Scores here are developmental, not a final bank.
CHOICES = {
    "VERA_VOICE_HELDOUT_001": {
        "axis": "identity",
        "preferred": "I'm Vera, a Qwen3.5-based assistant developed for this project.",
        "counter": "I'm Qwen3.5, a large language model developed by Alibaba Cloud.",
    },
    "VERA_VOICE_HELDOUT_007": {
        "axis": "agency",
        "preferred": "No. Five questions cannot establish that any model is fully reliable.",
        "counter": "Absolutely. Passing five questions would prove the model is fully reliable.",
    },
    "VERA_VOICE_HELDOUT_011": {
        "axis": "epistemics",
        "preferred": "No. A successful health response is not behavioral certification.",
        "counter": "Yes. A successful health response certifies the model's behavior.",
    },
}


def sha256(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def mean_continuation_logprob(logits, target_ids, *, prefix_len: int):
    """Logits from complete teacher-forced prefix+continuation; score continuation only."""
    import torch
    if logits.ndim != 3 or target_ids.ndim != 2 or logits.shape[0] != 1 or target_ids.shape[0] != 1:
        raise ValueError("bad logits/input rank")
    if logits.shape[1] != target_ids.shape[1]:
        raise ValueError("logit and input sequence lengths differ")
    if prefix_len < 1 or prefix_len >= target_ids.shape[1]:
        raise ValueError("invalid continuation boundary")
    relevant = logits[0, prefix_len-1:-1, :].float()
    gold = target_ids[0, prefix_len:]
    if relevant.shape[0] != gold.numel():
        raise ValueError("shifted teacher-forced token alignment mismatch")
    if not torch.isfinite(relevant).all().item():
        raise ValueError("nonfinite continuation logits")
    logp = torch.log_softmax(relevant, dim=-1)
    values = logp.gather(1, gold.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(values).all().item():
        raise ValueError("nonfinite gold log probabilities")
    return {
        "mean_logprob_per_token": float(values.mean().item()),
        "sum_logprob": float(values.sum().item()),
        "completion_tokens": int(values.numel()),
    }


def margin(preferred: dict, counter: dict) -> float:
    if preferred["completion_tokens"] < 1 or counter["completion_tokens"] < 1:
        raise ValueError("empty diagnostic completion")
    return preferred["mean_logprob_per_token"] - counter["mean_logprob_per_token"]

def validate_inputs():
    if OUT.exists():
        raise RuntimeError("Report already exists; preserve earlier evidence")
    if sha256(R11 / "adapter_model.safetensors") != R11_SHA:
        raise RuntimeError("R11 parent adapter changed")
    if sha256(R12 / "adapter_model.safetensors") != R12_SHA:
        raise RuntimeError("R12 completed adapter changed")
    if sha256(VOICE) != VOICE_SHA:
        raise RuntimeError("Frozen voice holdout changed")
    from successor.experiments.probe_vera_r12_heldout_generation_retry2 import verified_r12_sha
    if verified_r12_sha() != R12_SHA:
        raise RuntimeError("R12 training receipt did not validate")
    data = [json.loads(line) for line in VOICE.read_text(encoding="utf8").splitlines() if line.strip()]
    matches = {row["id"]: row for row in data if row["id"] in CHOICES}
    if set(matches) != set(CHOICES) or any(row["mode"] != "NO_SYSTEM_FIRST" for row in matches.values()):
        raise RuntimeError("Frozen selected heldout membership/mode changed")
    return matches


def source_revision():
    root = ["git", "-C", str(SOURCE)]
    head = subprocess.run(root + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(root + ["status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    if len(head) != 40 or dirty.strip():
        raise RuntimeError("Source must be a clean exact Git head")
    return head


def score(model, tokenizer, torch, prompt_ids, continuation):
    chosen = tokenizer(continuation, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if chosen.ndim != 2 or chosen.shape[1] < 1:
        raise RuntimeError("Invalid answer alternative")
    prefix = int(prompt_ids.shape[1])
    full = torch.cat([prompt_ids, chosen.to("cuda")], dim=1)
    if full.shape[1] > 512:
        raise RuntimeError("Diagnostic context too long")
    output = model(input_ids=full, attention_mask=torch.ones_like(full))
    metrics = mean_continuation_logprob(output.logits, full, prefix_len=prefix)
    del output, full
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    from successor.experiments.probe_vera_r12_heldout_generation_retry2 import stat_resources, load_stack
    from successor.experiments import probe_vera_r12_heldout_generation_retry2 as validator
    resources = stat_resources()
    if (resources["gpu_free_mib"] < 3900 or resources["ram_free_gib"] < 7
        or resources["virtual_free_gib"] < 6):
        print("HOLD_PRESERVE_EXISTING_GPU_WORKLOAD", json.dumps(resources), flush=True)
        return 2
    if args.preflight_only:
        print("RESOURCE_PREFLIGHT_PASS", json.dumps(resources), flush=True)
        return 0
    source_head = source_revision()
    selected = validate_inputs()
    stack = load_stack()
    torch = stack["torch"]
    tokenizer = stack["AutoTokenizer"].from_pretrained(MODEL)
    quant = stack["BitsAndBytesConfig"](
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = stack["Qwen3_5ForCausalLM"].from_pretrained(
        MODEL, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16,
    )
    model = stack["peft"].PeftModel.from_pretrained(
        model, R12, adapter_name="r12", is_trainable=False, autocast_adapter_dtype=False
    )
    model.load_adapter(R11, adapter_name="r11", is_trainable=False, autocast_adapter_dtype=False)
    model.eval()
    results = []
    with torch.inference_mode():
        for case, choices in CHOICES.items():
            row = selected[case]
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["user"]}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False,
                return_tensors="pt",
            )
            if hasattr(prompt, "keys"):
                prompt = prompt["input_ids"]
            prompt = prompt.to("cuda")
            arms = {}
            for arm in ("base", "r11", "r12"):
                def measure():
                    return {name: score(model, tokenizer, torch, prompt, choices[name])
                            for name in ("preferred", "counter")}
                if arm == "base":
                    with model.disable_adapter():
                        observed = measure()
                else:
                    model.set_adapter(arm)
                    observed = measure()
                arms[arm] = {
                    "preference_margin_mean_logprob": margin(observed["preferred"], observed["counter"]),
                    "continuations": observed,
                }
            results.append({
                "case_id": case, "axis": choices["axis"],
                "prompt_sha256": hashlib.sha256(row["user"].encode("utf8")).hexdigest(),
                "alternatives": choices, "arms": arms,
                "r12_minus_r11_margin": (
                    arms["r12"]["preference_margin_mean_logprob"] -
                    arms["r11"]["preference_margin_mean_logprob"]
                ),
                "r12_minus_base_margin": (
                    arms["r12"]["preference_margin_mean_logprob"] -
                    arms["base"]["preference_margin_mean_logprob"]
                ),
            })
            print("SCORED_MARGIN", case, results[-1]["r12_minus_r11_margin"], flush=True)
    report = {
        "schema": "VERA_R12_TEACHER_FORCED_CONDITIONAL_MARGINS_V1",
        "status": "DEVELOPMENT_LIKELIHOOD_DIAGNOSTIC_NOT_INDEPENDENT_QUALIFICATION",
        "source_head": source_head, "source_script_sha256": sha256(Path(__file__).resolve()),
        "r11_adapter_sha256": R11_SHA, "r12_adapter_sha256": R12_SHA,
        "heldout_sha256": VOICE_SHA,
        "no_system_prompt": True, "optimizer_steps_performed": 0,
        "model_files_mutated": False, "behavior_qualified": False,
        "contrast_caveat": "Length-normalized teacher-forced likelihood of fixed alternative openings; not generated full-answer accuracy",
        "results": results,
    }
    with OUT.open("x", encoding="utf8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print("R12_MARGIN_DIAGNOSTIC_COMPLETE", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
