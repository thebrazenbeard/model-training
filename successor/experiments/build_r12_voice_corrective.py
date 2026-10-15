"""Build a fresh 64-row R12 repair window without exposing heldout answers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from successor.experiments.train_v10_qwen35_authorized import sha256_file

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(r"D:\VERA")
SOURCE = ROOT / ".scratch/vera-v10r3r11-qwen35-voice-20261014-v1/train.jsonl"
SOURCE_SHA = "f73fe5a0d2c21bdd3ab3cc2cee3238d3297d0941611d31514c543ab560ee491e"
VOICE = Path(__file__).with_name("R12_VERA_CURATED_VOICE_20261015_V1.jsonl")
HOLDOUT = ROOT / "portal-task-state/vera-model-glm24b-lappy-20261012/evaluation/cases/vera_voice_heldout_v1.jsonl"
HOLDOUT_SHA = "bd2b5375e2b7da78f42ed3ac1387f73ddec4807bc5e446b228bdbb7db3caf25f"
R11_PATH = ROOT / "models/adapters/v10r3r11-vera-voice-qwen35-r10-resume-20261015-pagefile-retry1"
R11_SHA = "45f056297a55e19b22918d5298bcd5a9808fb0d8b8f5bf4210ebac91e4b88e10"
R11_RECEIPT_SELF = "60adaf9908d4d080445abd670917426f02f093e7e9bf25fa67f77b98d0a8b2f1"
R11_SPEC_REL = "successor/experiments/V10R3R11_VERA_VOICE_EXECUTION_SPEC_20261015_PAGEFILE_RETRY1.json"
R11_SPEC_SHA = "5ac59a3e06fdc4c0ef07ab6888976179386fbe453dc9b618ce510809b034b730"
OUT_DIR = ROOT / ".scratch/vera-v10r3r12-fp32-voice-20261015-v1"
NEW_ADAPTER = ROOT / "models/adapters/v10r3r12-vera-fp32-qwen35-r11-resume-20261015-v1"
SPEC_PATH = Path(__file__).with_name("V10R3R12_FP32_VERA_VOICE_EXECUTION_SPEC_20261015_V1.json")
BINDING_PATH = Path(__file__).with_name("V10R3R12_VERA_VOICE_CORPUS_BINDING_20261015_V1.json")
START, COUNT = 352, 64
GENERAL_INDICES = (355,356,359,361,362,363,364,367,369,371,374,376,377,378,379,383,
                   385,386,387,388,389,391,393,395,397,399,401,405,409,410,414,415)


def require(ok, reason):
    if not ok:
        raise ValueError("R12_SOURCE_HOLD: " + reason)


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
def prepare():
    require(not any(p.exists() for p in (OUT_DIR, NEW_ADAPTER, SPEC_PATH, BINDING_PATH)),
            "target already exists; no overwrite")
    require(sha256_file(SOURCE) == SOURCE_SHA, "R11 corpus changed")
    require(sha256_file(HOLDOUT) == HOLDOUT_SHA, "heldout source changed")
    require(sha256_file(REPO / R11_SPEC_REL) == R11_SPEC_SHA, "R11 source spec changed")
    require(sha256_file(R11_PATH / "adapter/adapter_model.safetensors") == R11_SHA,
            "R11 adapter not preserved")
    receipt = json.loads((R11_PATH / "DEV_TRAINING_COMPLETE.json").read_text(encoding="utf8"))
    require(receipt["receipt_sha256"] == R11_RECEIPT_SELF
            and receipt["cumulative_optimizer_steps"] == 44
            and receipt["weight_digest_changed"] is True, "R11 receipt mismatch")
    parent = load_rows(SOURCE)
    require(len(parent) == 50000, "wrong R11 corpus row count")
    voice = load_rows(VOICE)
    heldout = load_rows(HOLDOUT)
    require(len(voice) == 32 and len(heldout) == 25, "wrong source membership")
    prompts = [r["user"].strip().casefold() for r in voice]
    protected = {r["user"].strip().casefold() for r in heldout}
    require(len(set(prompts)) == 32 and not set(prompts) & protected,
            "duplicate or heldout training prompt")
    require(len({r["id"] for r in voice}) == 32, "duplicate voice row ID")
    require(all(r.get("privacy") == "NON_PERSONAL_BEHAVIOR_ONLY"
                and r.get("training_use") == "targeted_behavioral_sft"
                and len(r["chosen"].strip()) >= 48 for r in voice), "bad voice provenance")
    require(len(GENERAL_INDICES) == 32 and len(set(GENERAL_INDICES)) == 32, "bad rehearsal IDs")
    rehearsal = [parent[i] for i in GENERAL_INDICES]
    require(all(START <= i < START + COUNT for i in GENERAL_INDICES)
            and all(r.get("training_use") == "general_rehearsal" for r in rehearsal),
            "rehearsal membership mismatch")
    require(len({r["record_id"] for r in rehearsal}) == 32, "rehearsal duplicates")
    window = []
    for index, (v, general) in enumerate(zip(voice, rehearsal, strict=True)):
        row = {
            "schema": "VERA_R12_QWEN35_FP32_VOICE_SFT_V1",
            "record_id": v["id"],
            "prompt": v["user"],
            "response": v["chosen"],
            "source_class": "curated_nonpersonal_voice_new_r12",
            "training_use": "targeted_behavioral_sft",
            "voice_name": "Vera",
            "axis": v["axis"],
            "pair_sha256": hashlib.sha256(
                (v["user"] + "\n" + v["chosen"]).encode("utf8")).hexdigest(),
        }
        window.extend([row, general])
    require(len(window) == 64 and len({x["prompt"].strip().casefold() for x in window}) == 64,
            "R12 window nonunique prompts")
    all_rows = parent[:START] + window + parent[START+COUNT:]
    require(len(all_rows) == 50000 and all_rows[:START] == parent[:START]
            and all_rows[START+COUNT:] == parent[START+COUNT:],
            "out-of-window mutation")
    return all_rows, voice, rehearsal
def build():
    rows, voice, rehearsal = prepare()
    source_spec = json.loads((REPO / R11_SPEC_REL).read_text(encoding="utf8"))
    spec = source_spec.copy()
    spec["date"] = "2026-10-15"
    spec["experiment_class"] = "TARGETED_VERA_VOICE_AND_INDEPENDENT_JUDGMENT"
    spec["adapter_precision_policy"] = "FP32_RESUMED_LORA_TRAIN_AND_SAVE_V1"
    spec["comparison"] = dict(spec["comparison"])
    spec["comparison"]["role"] = "R12_NEW_CURATED_VERA_VOICE_FP32_PRECISION_REPAIR"
    spec["development_window"] = {"start_row": START, "row_count": COUNT}
    spec["trainer"] = dict(spec["trainer"])
    spec["trainer"]["learning_rate"] = 2e-5
    spec["trainer"]["seed"] = 20261015
    spec["trainer"]["max_optimizer_steps"] = 8
    spec["output"] = dict(spec["output"])
    spec["output"]["namespace"] = str(NEW_ADAPTER)
    spec["source_subject"] = dict(spec["source_subject"])
    spec["source_subject"]["training_corpus_id"] = "VERA_R12_QWEN35_NEW_CURATED_FP32_50K_V1"
    spec["source_subject"]["parent_revision"] = "cacd8fbc9817d454d36dfb5f78fb243f19c66fe2"
    spec["resume_adapter"] = {
        "path": str(R11_PATH / "adapter"), "adapter_model_sha256": R11_SHA,
        "source_receipt_path": str(R11_PATH / "DEV_TRAINING_COMPLETE.json"),
        "source_receipt_sha256": R11_RECEIPT_SELF,
        "source_spec_path": R11_SPEC_REL, "source_spec_sha256": R11_SPEC_SHA,
        "source_weight_digest_after": "3fd4ea2e12bfaca09dcb2732a33186275f1a51753e8f4d9af2fbd246d5ec817e",
        "previous_optimizer_steps": 8,
        "source_cumulative_optimizer_steps": 44, "next_train_row": START,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    corpus = OUT_DIR / "train.jsonl"
    with corpus.open("x", encoding="utf8", newline="\n") as fd:
        for row in rows:
            fd.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    spec["source_subject"]["train_sha256"] = sha256_file(corpus)
    binding = {
        "schema": "VERA_R12_VOICE_CORPUS_BINDING_V1",
        "source_r11_corpus_sha256": SOURCE_SHA,
        "r12_corpus_sha256": spec["source_subject"]["train_sha256"],
        "r12_source_voice_sha256": sha256_file(VOICE),
        "heldout_sha256": HOLDOUT_SHA,
        "window_start": START, "window_count": COUNT,
        "curated_voice_ids": [v["id"] for v in voice],
        "rehearsal_indices": list(GENERAL_INDICES),
        "previous_adapter_sha256": R11_SHA,
        "protected_heldout_training_prohibited": True,
        "training_executed": False, "model_deployed": False
    }
    SPEC_PATH.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf8")
    BINDING_PATH.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return {"status":"R12_FROZEN_SOURCE_ONLY_NOT_TRAINED", "corpus":str(corpus),
            "corpus_sha256": sha256_file(corpus), "voice_sha256": sha256_file(VOICE),
            "spec_sha256":sha256_file(SPEC_PATH), "binding_sha256": sha256_file(BINDING_PATH),
            "r12_new_steps_if_completed": 8, "cumulative_if_completed": 52,
            "output":str(NEW_ADAPTER), "r11_preserved":True}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
