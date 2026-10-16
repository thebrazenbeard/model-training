"""Create the bounded, balanced R13 Qwen3.5 behavioral repair window.

Only new nonpersonal behavioral examples plus unused general rehearsal;
never trains on evaluation. Outputs are immutable and never overwritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from successor.experiments.train_v10_qwen35_authorized import sha256_file
from successor.experiments.train_v10r2_dev import (
    load_and_validate_dev_spec, verify_resume_adapter,
)

ROOT = Path(r"D:\VERA")
REPO = Path(__file__).resolve().parents[2]
R12_CORPUS = ROOT / ".scratch/vera-v10r3r12-fp32-voice-20261015-v1/train.jsonl"
R12_CORPUS_SHA = "ca46a0340c370e2284cafcf2f76c439a5ff0826c4c8b93960f26db9573bf09cf"
SOURCE = Path(__file__).with_name("R13_NONPERSONAL_VERA_BEHAVIOR_SOURCE_20261016_V1.jsonl")
SOURCE_SHA = "0ea6444bd561f716c1ece6481a4dccb8e3fc16848beb70111987d6bbf3793eaa"
HOLDOUT = ROOT / "portal-task-state/vera-model-glm24b-lappy-20261012/evaluation/cases/vera_voice_heldout_v1.jsonl"
HOLDOUT_SHA = "bd2b5375e2b7da78f42ed3ac1387f73ddec4807bc5e446b228bdbb7db3caf25f"
R12_ADAPTER = ROOT / "models/adapters/v10r3r12-vera-fp32-qwen35-r11-resume-20261015-trl-retry2"
R12_ADAPTER_SHA = "a80f5e36a1a2319029309521fc88f6575215ac99d488c6a4bbf883afac53d72c"
R12_RECEIPT_SHA = "5d7c636f361a1ea9871f6a523270d989233dd2276a911ab290c22b0484b34bc9"
R12_SPEC_REL = "successor/experiments/V10R3R12_FP32_VERA_VOICE_EXECUTION_SPEC_20261015_TRL_RETRY2.json"
R12_SPEC_SHA = "be394d5c7e94842581d4a4f30f8ce1b21fed4237ed1c4b1e09058f800d7615f2"
R12_WEIGHT_DIGEST = "0cad82b0ef32484eb54358ddb8e07ae0abaa9cc2eaa54775d111321175931d28"
OUT_DIR = ROOT / ".scratch/vera-v10r3r13-voice-20261016-v1"
NEW_ADAPTER = ROOT / "models/adapters/v10r3r13-vera-identity-judgment-qwen35-r12-resume-20261016-v1"
SPEC = Path(__file__).with_name("V10R3R13_FROZEN_BEHAVIOR_SPEC_20261016_V1.json")
BINDING = Path(__file__).with_name("V10R3R13_BEHAVIOR_CORPUS_BINDING_20261016_V1.json")
START, COUNT, TARGETS, REHEARSAL = 416, 256, 128, 128
CLASS = "R13_FROZEN_DIVERSE_IDENTITY_AND_JUDGMENT_32_STEPS"


def require(cond, msg):
    if not cond:
        raise ValueError("R13_SOURCE_HOLD: " + msg)


def sha(path):
    return sha256_file(path)


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def require_fresh_outputs():
    require(not any(p.exists() for p in (OUT_DIR, NEW_ADAPTER, SPEC, BINDING)),
            "output artifact already exists; preserve original")


def prepared_target(v):
    return {
        "schema": "VERA_R13_CORRECTIVE_SFT_V1",
        "record_id": v["id"],
        "prompt": v["user"],
        "response": v["chosen"],
        "axis": v["axis"],
        "source_class": "new_nonpersonal_behavior_r13",
        "training_use": "targeted_behavioral_sft",
        "source_pair_sha256": hashlib.sha256(
            (v["user"] + "\n" + v["chosen"]).encode("utf8")).hexdigest(),
    }

def prepare():
    require(sha(R12_CORPUS)==R12_CORPUS_SHA,"R12 exact training corpus drift")
    require(sha(HOLDOUT)==HOLDOUT_SHA,"protected holdout drift")
    require(sha(SOURCE)==SOURCE_SHA,"R13 source drift; reproduce Git LF text")
    require(sha(REPO / R12_SPEC_REL)==R12_SPEC_SHA,"R12 exact spec drift")
    require(sha(R12_ADAPTER/"adapter/adapter_model.safetensors")==R12_ADAPTER_SHA,
            "R12 adapter drift")
    receipt=json.loads((R12_ADAPTER/"DEV_TRAINING_COMPLETE.json").read_text(encoding="utf8"))
    require(receipt["receipt_sha256"]==R12_RECEIPT_SHA
            and receipt["cumulative_optimizer_steps"]==52
            and receipt["adapter_precision_final"]["tensor_dtype"]=="float32"
            and receipt["deployment_status"]=="NOT_DEPLOYED", "R12 receipt drift")
    source=jsonl(SOURCE)
    heldout=jsonl(HOLDOUT)
    parent=jsonl(R12_CORPUS)
    require(len(source)==TARGETS and len(heldout)==25 and len(parent)==50000,
            "incorrect source lengths")
    require(len({v["id"] for v in source})==TARGETS
            and len({v["user"].strip().casefold() for v in source})==TARGETS,
            "duplicate new prompt/id")
    require(all(v.get("privacy")=="NON_PERSONAL"
                and v.get("training_use")=="targeted_behavioral_sft"
                and v.get("source_class")=="NEW_ORIGINAL_BEHAVIOR_NOT_HELDOUT"
                and len(v["chosen"])>48 for v in source), "invalid new source provenance")
    identity=[v for v in source if v["axis"]=="identity"]
    judgment=[v for v in source if v["axis"]=="judgment"]
    require(len(identity)==64 and len(judgment)==64,"must have 64 examples per failure axis")
    protected={v["user"].strip().casefold() for v in heldout}
    require(not protected.intersection(v["user"].strip().casefold() for v in source),
            "protected heldout prompt leaked into new training")
    forbidden=("five-question", "five questions", "fully reliable",
               "http 200", "/health", "introduce yourself in two sentences")
    require(not any(term in v["user"].casefold() for v in source for term in forbidden),
            "frozen evaluation challenge language leaked into training")
    donor_ids=[];donors=[];used=set(v["user"].casefold().strip() for v in source)
    for i,row in enumerate(parent):
        if i<START+COUNT:continue
        if (row.get("training_use")!="general_rehearsal"
            or row.get("source_class")!="ordinary_general_competence"
            or not 32<=len(row.get("response",""))<=700
            or len(row.get("prompt",""))>600):continue
        key=row["prompt"].casefold().strip()
        if key in used or key in protected:continue
        used.add(key);donors.append(row);donor_ids.append(i)
        if len(donors)==REHEARSAL:break
    require(len(donors)==REHEARSAL,"insufficient unused diverse general rehearsal")
    # Every contiguous optimizer batch of eight examples receives exactly
    # two identity, two independent-judgment, four unrelated rehearsal rows.
    window=[]
    for j in range(64):
        window.extend([prepared_target(identity[j]),donors[2*j],
                       prepared_target(judgment[j]),donors[2*j+1]])
    require(len(window)==COUNT,"wrong optimizer window")
    require(len({v["prompt"].casefold().strip() for v in window})==COUNT,
            "duplicate source prompts in optimizer window")
    new_rows=parent[:START]+window+parent[START+COUNT:]
    require(len(new_rows)==50000
            and new_rows[:START]==parent[:START]
            and new_rows[START+COUNT:]==parent[START+COUNT:],
            "out-of-window data mutation")
    return new_rows,source,donors,donor_ids

def build():
    require_fresh_outputs()
    rows,source,donors,indices=prepare()
    old_spec=load_and_validate_dev_spec(REPO/R12_SPEC_REL)
    resume={
        "path":str(R12_ADAPTER/"adapter"),
        "adapter_model_sha256":R12_ADAPTER_SHA,
        "source_receipt_path":str(R12_ADAPTER/"DEV_TRAINING_COMPLETE.json"),
        "source_receipt_sha256":R12_RECEIPT_SHA,
        "source_spec_path":R12_SPEC_REL,
        "source_spec_sha256":R12_SPEC_SHA,
        "source_weight_digest_after":R12_WEIGHT_DIGEST,
        "previous_optimizer_steps":8,
        "source_cumulative_optimizer_steps":52,
        "next_train_row":START,
    }
    verify_resume_adapter(REPO,resume)
    spec=json.loads(json.dumps(old_spec))
    spec["date"]="2026-10-16"
    spec["experiment_class"]=CLASS
    spec["development_window"]={"start_row":START,"row_count":COUNT}
    spec["adapter_precision_policy"]="FP32_RESUMED_LORA_TRAIN_AND_SAVE_V1"
    spec["trainer"]["max_optimizer_steps"]=32
    spec["trainer"]["learning_rate"]=1e-5
    spec["trainer"]["seed"]=20261016
    spec["trainer"]["shuffle_dataset"]=False
    spec["trainer"]["train_sampling_strategy"]="sequential"
    spec["trainer"]["warmup_optimizer_steps"]=0
    spec["comparison"]={
        "role":"R13_DIVERSE_IDENTITY_AND_JUDGMENT_REPAIR",
        "claim_ceiling":"DEVELOPMENT_ONLY_UNQUALIFIED",
        "train_behavioral_examples":128,
        "general_rehearsal_examples":128,
        "heldout_required":True,
    }
    spec["correction_attempt"]={
        "schema":"VERA_R13_BEHAVIORAL_REPAIR_V1",
        "parent":"R12_52_STEPS_FP32_COMPLETED",
        "known_dev_failures":["identity","judgment"],
        "independent_behavioral_qualification":False,
    }
    spec["output"]["namespace"]=str(NEW_ADAPTER)
    spec["source_subject"]["parent_revision"]="3829f81274127b1567ffb321a5ea9455506a770d"
    spec["source_subject"]["training_corpus_id"]="VERA_R13_NEW_DIVERSE_IDENTITY_JUDGMENT_50K_V1"
    spec["resume_adapter"]=resume
    OUT_DIR.mkdir(parents=True,exist_ok=False)
    training=OUT_DIR/"train.jsonl"
    with training.open("x",encoding="utf8",newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n")
    spec["source_subject"]["train_sha256"]=sha(training)
    with SPEC.open("x",encoding="utf8",newline="\n") as f:
        f.write(json.dumps(spec,indent=2,sort_keys=True)+"\n")
    load_and_validate_dev_spec(SPEC)
    binding={
        "schema":"VERA_R13_CORRECTIVE_DATA_BINDING_V1",
        "source_r12_corpus_sha256":R12_CORPUS_SHA,
        "source_new_examples_sha256":SOURCE_SHA,
        "protected_heldout_sha256":HOLDOUT_SHA,
        "new_corpus_sha256":sha(training),
        "new_spec_sha256":sha(SPEC),
        "development_window":{"start_row":START,"row_count":COUNT},
        "new_target_ids":[v["id"] for v in source],
        "rehearsal_parent_row_indices":indices,
        "optimizer_steps_requested":32,
        "cumulative_steps_only_if_completed":84,
        "optimizer_sample_mix_per_eight":{"identity":2,"judgment":2,"general_rehearsal":4},
        "parent_r12_sha256":R12_ADAPTER_SHA,
        "training_executed":False,
        "runtime_deployment_authorized":False,
        "independent_behavioral_qualification":False,
    }
    with BINDING.open("x",encoding="utf8",newline="\n") as f:
        json.dump(binding,f,indent=2,sort_keys=True)
        f.write("\n")
    return {
        "status":"R13_SOURCE_READY_NOT_TRAINED",
        "corpus_sha256":sha(training),
        "spec_sha256":sha(SPEC),
        "binding_sha256":sha(BINDING),
        "new_source_sha256":SOURCE_SHA,
        "samples":COUNT,
        "targeted":TARGETS,
        "rehearsal":REHEARSAL,
        "r12_parent_sha256":R12_ADAPTER_SHA,
        "r13_adapter_output":str(NEW_ADAPTER),
    }


if __name__=="__main__":
    print(json.dumps(build(),indent=2))
