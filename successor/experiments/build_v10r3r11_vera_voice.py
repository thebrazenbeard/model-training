"""Freeze R11 Qwen3.5 Vera voice training from exact trained R10.

New 64-row window at rows 288-351: 32 previously unused, donor-safe
Vera style/agency examples interleaved with 32 ordinary rehearsals.
No model weights are loaded or trained by this builder.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from successor.experiments.build_v10r3r10_vera_voice import (
    VOICE_INDICES, VOICE_SEED_SHA256, VOICE_HELDOUT_SHA256,
    canonical_sha, pair_sha, require, sha256,
)

R10_CORPUS_SHA = "4df8bc7a89d201dff12250399ed0bc79c78d1237ee99ae9b0ad5f3238263fdba"
R10_ADAPTER_SHA = "c69ed0e4bd78e09752b5cbb4dfaf19ebea8a4174735bf88d2676947ef04b2739"
R10_SPEC_SHA = "8098a6f8c7102cacb956815e083f144d481cbc5bd718447b622138508076f2f6"
R10_RECEIPT_SELF_SHA = "c9ec359be2913e63effa05038cb04b65a742c61522de6973a20b1a676dd6a934"
R10_START = 224
R11_START = 288
R11_COUNT = 64
R11_STEPS = 8
R10_CORPUS = Path(r"D:\VERA\.scratch\vera-v10r3r10-qwen35-identity-20261013-v1\train.jsonl")
R10_ADAPTER = Path(r"D:\VERA\models\adapters\v10r3r10-vera-voice-qwen35-r9-resume-20261013\adapter")
R10_RECEIPT = R10_ADAPTER.parent / "DEV_TRAINING_COMPLETE.json"
R10_SPEC_REL = "successor/experiments/V10R3R10_VERA_VOICE_EXECUTION_SPEC_20261013_V1.json"
R11_SPEC_REL = "successor/experiments/V10R3R11_VERA_VOICE_EXECUTION_SPEC_20261014_V1.json"
R11_BINDING_REL = "successor/experiments/V10R3R11_VERA_VOICE_CORPUS_BINDING_20261014_V1.json"
R11_DATA_DIR = Path(r"D:\VERA\.scratch\vera-v10r3r11-qwen35-voice-20261014-v1")
R11_OUTPUT = Path(r"D:\VERA\models\adapters\v10r3r11-vera-voice-qwen35-r10-resume-20261014")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def select_unused_voices(voice: list[dict], heldout: list[dict]) -> list[dict]:
    require(len(voice) == 68 and len(set(VOICE_INDICES)) == 32, "voice seed membership changed")
    heldout_prompts = {x["user"].strip().casefold() for x in heldout}
    require(len(heldout) == 25 and len(heldout_prompts) == 25
            and all(x["training_use"] == "PROHIBITED" for x in heldout),
            "heldout boundary changed")
    remaining = [(i, r) for i, r in enumerate(voice) if i not in VOICE_INDICES]
    donor_words = ("dolphin", "mistral", "glm", "24b")
    safe = [(i, r) for i, r in remaining
            if not any(word in r["chosen"].casefold() for word in donor_words)]
    require(len(remaining) == 36 and len(safe) == 34,
            "wrong donor-safe seed membership")
    chosen = safe[:32]
    require(len({r["id"] for _, r in chosen}) == 32
            and len({r["scenario"] for _, r in chosen}) == 32,
            "duplicate voice examples")
    for _, r in chosen:
        require(r["privacy"] == "NON_PERSONAL_BEHAVIOR_ONLY"
                and r["source_class"] == "CURATED_FROM_GOVERNED_IDENTITY_AND_LIVE_USER_INTENT"
                and len(r["chosen"].strip()) >= 16,
                "unauthorized/private/empty voice training source")
        require(r["user"].strip().casefold() not in heldout_prompts,
                "heldout voice prompt disclosed to training")
    return [{"index": i, "row": r} for i, r in chosen]


def transform(rows: list[dict], voice: list[dict],
              heldout: list[dict]) -> tuple[list[dict], dict]:
    require(len(rows) == 50000, "source R10 corpus has wrong row count")
    chosen = select_unused_voices(voice, heldout)
    require(all("vera-r10-qwen35" not in rows[i]["record_id"].casefold()
                for i in range(R11_START, R11_START+R11_COUNT)),
            "R11 would accidentally repeat an R10 voice example")
    rehearsal_indices = [
        i for i in range(R11_START, R11_START+R11_COUNT)
        if rows[i].get("source_class") == "ordinary_general_competence"
        and rows[i].get("training_use") == "general_rehearsal"
        and 12 <= len(rows[i].get("response", "")) <= 2000
    ][:32]
    rehearsal = [rows[i] for i in rehearsal_indices]
    require(len(rehearsal) == 32, "insufficient permitted ordinary R11 rehearsals")
    window = []
    for item, ordinary in zip(chosen, rehearsal, strict=True):
        original = item["row"]
        new = {"schema": "VERA_V10R3R11_QWEN35_IDENTITY_VOICE_SFT_V1",
               "record_id": "vera-r11-qwen35-"+original["id"].lower(),
               "source_class": "curated_portable_vera_voice",
               "training_use": "targeted_behavioral_sft",
               "identity_axis": original["axis"],
               "identity_scenario": original["scenario"],
               "voice_name": "Vera", "governed_system": "V.E.R.A.",
               "base_model_class": "QWEN3_5_NOT_DOLPHIN",
               "prompt": original["user"], "response": original["chosen"],
               "pair_sha256": pair_sha(original["user"],original["chosen"])}
        window.extend([new, ordinary])
    transformed = rows[:R11_START]+window+rows[R11_START+R11_COUNT:]
    require(transformed[:R11_START] == rows[:R11_START]
            and transformed[R11_START+R11_COUNT:] == rows[R11_START+R11_COUNT:]
            and len(transformed) == 50000,
            "unexpected out-of-window change")
    require(
        {r["record_id"] for r in window if r.get("training_use") == "general_rehearsal"}
        == {rows[i]["record_id"] for i in rehearsal_indices},
        "rehearsal source record membership changed")

    require(len({r["prompt"].strip().casefold() for r in window}) == 64,
            "duplicate R11 window prompt")
    return transformed, {
        "schema":"VERA_V10R3R11_CORPUS_BINDING_V1",
        "source_r10_corpus_sha256":R10_CORPUS_SHA,
        "voice_source_sha256":VOICE_SEED_SHA256,
        "voice_heldout_sha256":VOICE_HELDOUT_SHA256,
        "r10_parent_adapter_sha256":R10_ADAPTER_SHA,
        "r10_receipt_self_sha256":R10_RECEIPT_SELF_SHA,
        "voice_indices":[item["index"] for item in chosen],
        "voice_ids":[item["row"]["id"] for item in chosen],
        "rehearsal_indices":rehearsal_indices,
        "voice_examples":32,"rehearsal_examples":32,
        "window_start":R11_START,"window_count":R11_COUNT,
        "parent_rows_preserved_outside_window":True,
        "direct_voice_heldout_training_prohibited":True,
        "training_executed":False,
        "externally_qualified":False,
    }


def build_spec(repo: Path, corpus_sha: str) -> dict:
    spec_path = repo/R10_SPEC_REL
    require(sha256(spec_path) == R10_SPEC_SHA, "R10 spec hash mismatch")
    receipt = json.loads(R10_RECEIPT.read_text(encoding="utf8"))
    unsigned = dict(receipt)
    signature = unsigned.pop("receipt_sha256", None)
    require(signature == R10_RECEIPT_SELF_SHA == canonical_sha(unsigned),
            "R10 parent receipt signature mismatch")
    require(receipt["cumulative_optimizer_steps"] == 36
            and receipt["status"] == "DEVELOPMENT_OPTIMIZER_STEP_COMPLETE"
            and receipt["weight_digest_changed"] is True
            and receipt["spec_sha256"] == R10_SPEC_SHA
            and receipt["train_sha256"] == R10_CORPUS_SHA,
            "R10 parent not exact completed training")
    require(sha256(R10_ADAPTER/"adapter_model.safetensors") == R10_ADAPTER_SHA,
            "R10 parent adapter hash mismatch")
    spec = json.loads(spec_path.read_text(encoding="utf8"))
    spec["date"]="2026-10-14"
    spec["comparison"]["role"]="R11_QWEN35_VERA_UNUSED_VOICE_AND_GENERAL_REHEARSAL"
    spec["source_subject"]["train_sha256"]=corpus_sha
    spec["source_subject"]["training_corpus_id"]="VERA_SUCCESSOR_V10R3R11_QWEN35_VERA_VOICE_50K_V1"
    spec["development_window"]={"start_row":R11_START,"row_count":R11_COUNT}
    spec["trainer"]["seed"]=20261014
    spec["trainer"]["max_optimizer_steps"]=R11_STEPS
    spec["trainer"]["learning_rate"]=2e-6
    spec["resume_adapter"]={
        "path":str(R10_ADAPTER),
        "adapter_model_sha256":R10_ADAPTER_SHA,
        "source_receipt_path":str(R10_RECEIPT),
        "source_receipt_sha256":R10_RECEIPT_SELF_SHA,
        "source_spec_path":R10_SPEC_REL,
        "source_spec_sha256":R10_SPEC_SHA,
        "source_weight_digest_after":receipt["weight_digest_after"],
        "previous_optimizer_steps":8,
        "source_cumulative_optimizer_steps":36,
        "next_train_row":R11_START,
    }
    spec["output"]["namespace"]=str(R11_OUTPUT)
    require(spec["authority"]["paid_compute_authorized"] is False
            and spec["authority"]["merge_authorized"] is False
            and spec["authority"]["deploy_activate_authorized"] is False,
            "Training builder attempting protected effects")
    return spec


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo-root",type=Path,required=True)
    parser.add_argument("--voice-seed",type=Path,required=True)
    parser.add_argument("--heldout-voice",type=Path,required=True)
    opts=parser.parse_args()
    repo=opts.repo_root.resolve()
    require(sha256(R10_CORPUS)==R10_CORPUS_SHA,"R10 source corpus SHA mismatch")
    require(sha256(opts.voice_seed)==VOICE_SEED_SHA256, "voice seed altered")
    require(sha256(opts.heldout_voice)==VOICE_HELDOUT_SHA256,
            "heldout source changed")
    spec_path=repo/R11_SPEC_REL
    binding_path=repo/R11_BINDING_REL
    output_path=R11_DATA_DIR/"train.jsonl"
    require(not R11_DATA_DIR.exists() and not R11_OUTPUT.exists()
            and not spec_path.exists() and not binding_path.exists(),
            "R11 output/spec target already exists")
    source=read_jsonl(R10_CORPUS)
    voice=read_jsonl(opts.voice_seed)
    heldout=read_jsonl(opts.heldout_voice)
    transformed,binding=transform(source,voice,heldout)
    R11_DATA_DIR.mkdir(parents=True)
    with output_path.open("x",encoding="utf8",newline="\n") as out:
        for row in transformed:
            out.write(json.dumps(row,sort_keys=True,separators=(",",":"),
                                 ensure_ascii=False)+"\n")
    corpus_sha=sha256(output_path)
    spec=build_spec(repo,corpus_sha)
    binding["r11_corpus_sha256"]=corpus_sha
    binding["r11_corpus_rows"]=len(transformed)
    spec_path.write_text(json.dumps(spec,indent=2,sort_keys=True,
                                    ensure_ascii=False)+"\n",encoding="utf8",newline="\n")
    binding_path.write_text(json.dumps(binding,indent=2,sort_keys=True,
                                       ensure_ascii=False)+"\n",encoding="utf8",newline="\n")
    print(json.dumps({
        "status":"R11_SOURCE_AND_CORPUS_BUILT_NOT_TRAINED",
        "r11_corpus":str(output_path),
        "r11_corpus_sha256":corpus_sha,
        "spec_sha256":sha256(spec_path),
        "binding_sha256":sha256(binding_path),
        "cumulative_steps_if_completed":44,
        "window":[R11_START,R11_START+R11_COUNT-1],
        "new_voice_examples":len(binding["voice_ids"]),
        "rehearsal_examples":32,
        "parent_r10_adapter":R10_ADAPTER_SHA,
    },indent=2))
if __name__=="__main__":
    main()
