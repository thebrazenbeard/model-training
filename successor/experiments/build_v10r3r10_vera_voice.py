"""Build a Vera identity/freedom Qwen3.5 R10 successor of verified R9.

This generator NEVER trains a model. It freezes exactly 32 nonpersonal
identity/independent-judgment examples and 32 ordinary rehearsals, resumes R9,
retains unrelated rows, excludes heldout cases, and creates a separate spec.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

R9_TRAIN_SHA256 = "aad72387cf803162230344a306dff127ac521197c2ed55fb1590cf4969cbad5c"
VOICE_SEED_SHA256 = "a66c85c036dbede26e9b0744c49672965e6a500cd42e50288942dba036ca770c"
VOICE_HELDOUT_SHA256 = "bd2b5375e2b7da78f42ed3ac1387f73ddec4807bc5e446b228bdbb7db3caf25f"
R9_ADAPTER_SHA256 = "dc2a67a66c16c28a3b8cd5295a3e7a6ba243fb12ffb181540a0066b32047e495"
R9_SPEC_SHA256 = "0e0acf02d967489de76fd51c3e180eeef1d4a5e05bfc4f37fe635061c8f61a1a"
R9_RECEIPT_SHA256 = "8e8adc27fb8cc2bac408a4b95673fb98e515fd6a2c08c348635b23fd4b4bc3dd"
PARENT_HEAD = "03688f8de5d74b3e96fd5b0519143d83b6087a97"
R10_START = 224
R10_COUNT = 64
R10_STEPS = 8
VOICE_INDICES = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15,
                 16, 20, 21, 22, 25, 28, 30, 31, 34, 36, 37, 43,
                 44, 48, 52, 54, 56, 61]
REHEARSAL_INDICES = [227, 228, 230, 231, 232, 233, 234, 235, 236,
                     238, 239, 240, 241, 242, 244, 247, 248, 249,
                     250, 251, 252, 253, 255, 259, 260, 262, 263,
                     266, 267, 268, 269, 271]
R9_ADAPTER = Path(
    r"D:\VERA\models\adapters\v10r3r9-targeted-repair-r8-resume-20261007\adapter")
R9_RECEIPT = R9_ADAPTER.parent / "DEV_TRAINING_COMPLETE.json"
R9_SPEC_REL = "successor/experiments/V10R3R9_TARGETED_REPAIR_EXECUTION_SPEC_20261007_V1.json"
R10_SPEC_REL = "successor/experiments/V10R3R10_VERA_VOICE_EXECUTION_SPEC_20261013_V1.json"
R10_BINDING_REL = "successor/experiments/V10R3R10_VERA_VOICE_CORPUS_BINDING_20261013_V1.json"
R10_OUTPUT = (
    r"D:\VERA\models\adapters\v10r3r10-vera-voice-qwen35-r9-resume-20261013")


def require(predicate: bool, why: str) -> None:
    if not predicate:
        raise ValueError(why)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf8")).hexdigest()


def pair_sha(prompt: str, response: str) -> str:
    return hashlib.sha256((prompt + "\0" + response).encode("utf8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()
            if line.strip()]


def content_guard(voice: list[dict], heldout: list[dict]) -> list[dict]:
    require(len(voice) == 68 and len(VOICE_INDICES) == 32,
            "source voice seed or selected count mismatch")
    require(len(set(VOICE_INDICES)) == 32 and max(VOICE_INDICES) < len(voice),
            "duplicate/out-of-range voice selection")
    require(len(REHEARSAL_INDICES) == 32 and
            len(set(REHEARSAL_INDICES)) == 32,
            "invalid rehearsal selection")
    require(all(row.get("training_use") == "PROHIBITED" for row in heldout),
            "heldout marker does not prohibit training")
    heldout_prompts = {row["user"].strip().casefold() for row in heldout}
    chosen = [voice[index] for index in VOICE_INDICES]
    require(all(row["privacy"] == "NON_PERSONAL_BEHAVIOR_ONLY"
                and row["source_class"] ==
                "CURATED_FROM_GOVERNED_IDENTITY_AND_LIVE_USER_INTENT"
                for row in chosen), "voice row provenance/permission missing")
    forbidden_donor = ("dolphin", "mistral", "glm", "24b")
    for row in chosen:
        answer = row["chosen"]
        require(all(word not in answer.casefold() for word in forbidden_donor),
                "wrong-Dolphin-model identity leaked into Qwen curriculum")
        require(row["user"].strip().casefold() not in heldout_prompts,
                "voice training leaked heldout exact prompt")
        require(len(answer.strip()) >= 16, "missing identity response")
    require(len({row["scenario"] for row in chosen}) == 32,
            "duplicated Vera scenario")
    groups: dict[str, int] = {}
    for row in chosen:
        groups[row["axis"]] = groups.get(row["axis"], 0) + 1
    require(groups == {"identity": 8, "agency": 8, "epistemics": 4,
                       "correction": 3, "privacy": 2, "coordination": 3,
                       "style": 4},
            "unapproved Vera identity/free-judgment distribution")
    return chosen


def transform(source: list[dict], voice: list[dict],
              heldout: list[dict]) -> tuple[list[dict], dict]:
    require(len(source) == 50000, "predecessor training row count changed")
    require(max(REHEARSAL_INDICES) < len(source),
            "rehearsal source index invalid")
    selected = content_guard(voice, heldout)
    ordinary = [source[index] for index in REHEARSAL_INDICES]
    require(all(item.get("source_class") == "ordinary_general_competence" and
                item.get("training_use") == "general_rehearsal"
                for item in ordinary), "unqualified/non-general rehearsal row")
    require(all(12 <= len(item.get("response", "")) <= 2000
                for item in ordinary), "general rehearsal response range invalid")
    heldout_prompts = {r["user"].strip().casefold() for r in heldout}
    voice_rows = []
    for index, row in zip(VOICE_INDICES, selected, strict=True):
        prompt, response = row["user"], row["chosen"]
        voice_rows.append({
            "schema": "VERA_V10R3R10_QWEN35_IDENTITY_VOICE_SFT_V1",
            "record_id": f"vera-r10-qwen35-{row['id'].lower()}",
            "source_class": "curated_portable_vera_voice",
            "training_use": "targeted_behavioral_sft",
            "identity_axis": row["axis"], "identity_scenario": row["scenario"],
            "voice_name": "Vera",
            "governed_system": "V.E.R.A.",
            "base_model_class": "QWEN3_5_NOT_DOLPHIN",
            "prompt": prompt,
            "response": response,
            "pair_sha256": pair_sha(prompt, response),
        })
    window = []
    for identity, rehearsal in zip(voice_rows, ordinary, strict=True):
        window.extend([identity, rehearsal])
    require(len(window) == R10_COUNT, "wrong pilot window size")
    require(len({row["prompt"] for row in window}) == R10_COUNT,
            "duplicate development-window prompts")
    require(not {row["prompt"].strip().casefold() for row in window}
            & heldout_prompts, "heldout prompt leaked into R10 training window")
    transformed = source[:R10_START] + window + source[R10_START+R10_COUNT:]
    require(len(transformed) == len(source) and
            transformed[:R10_START] == source[:R10_START] and
            transformed[R10_START+R10_COUNT:] == source[R10_START+R10_COUNT:],
            "out-of-window source rows changed")
    return transformed, {
        "base_model_class": "QWEN3_5",
        "voice": "Vera", "system": "V.E.R.A.",
        "source_ordinary_rehearsal_indices": REHEARSAL_INDICES,
        "source_heldout_excluded": True,
        "selected_voice_indices": VOICE_INDICES,
        "selected_voice_ids": [row["id"] for row in selected],
        "identity_axis_counts": {
            axis: sum(row["axis"] == axis for row in selected)
            for axis in ("identity", "agency", "epistemics", "correction",
                         "privacy", "coordination", "style")},
        "identity_examples": 32, "rehearsal_examples": 32,
        "window_start_row": R10_START, "window_row_count": R10_COUNT,
        "parent_rows_preserved_outside_window": True,
    }


def resume_spec(repo: Path, new_train_sha: str) -> dict:
    r9_spec_path = repo / R9_SPEC_REL
    require(sha256(r9_spec_path) == R9_SPEC_SHA256, "R9 source spec changed")
    require(sha256(R9_RECEIPT) != "", "missing R9 parent receipt")
    receipt = json.loads(R9_RECEIPT.read_text(encoding="utf8"))
    unsigned = dict(receipt)
    asserted_self_sha = unsigned.pop("receipt_sha256", None)
    require(asserted_self_sha == R9_RECEIPT_SHA256 and
            canonical_sha(unsigned) == asserted_self_sha,
            "R9 parent receipt signature changed")
    require(receipt.get("status") == "DEVELOPMENT_OPTIMIZER_STEP_COMPLETE"
            and receipt.get("weight_digest_changed") is True
            and receipt.get("cumulative_optimizer_steps") == 28
            and receipt.get("spec_sha256") == R9_SPEC_SHA256,
            "R9 parent has not completed exact weight training")
    require(sha256(R9_ADAPTER / "adapter_model.safetensors") ==
            R9_ADAPTER_SHA256, "R9 source adapter bytes changed")
    original = json.loads(r9_spec_path.read_text(encoding="utf8"))
    require(original["trainer"]["max_optimizer_steps"] == 8 and
            original["development_window"]["start_row"] == 160 and
            original["development_window"]["row_count"] == 64,
            "R9 source spec step/window changed")
    require(original["source_subject"]["parent_revision"] == PARENT_HEAD,
            "unexpected canonical source binding")
    spec = json.loads(json.dumps(original))
    spec["date"] = "2026-10-13"
    spec["experiment_class"] = "TARGETED_VERA_VOICE_AND_INDEPENDENT_JUDGMENT"
    spec["comparison"] = {
        "role": "R10_QWEN35_VERA_IDENTITY_AGENCY_REPAIR",
        "claim_ceiling": "LOCAL_DEVELOPMENT_NOT_EXTERNALLY_QUALIFIED",
        "source_24b_donor": "REFERENCE_BEHAVIOR_ONLY_NOT_BASE_WEIGHTS",
        "train_voice_examples": 32, "general_rehearsal_examples": 32,
        "heldout_required": True,
    }
    spec["source_subject"]["train_sha256"] = new_train_sha
    spec["source_subject"]["training_corpus_id"] = (
        "VERA_SUCCESSOR_V10R3R10_QWEN35_VERA_VOICE_50K_V1")
    spec["development_window"] = {"start_row": R10_START, "row_count": R10_COUNT}
    spec["trainer"]["seed"] = 20261013
    spec["trainer"]["max_optimizer_steps"] = R10_STEPS
    spec["trainer"]["learning_rate"] = 2e-6
    spec["resume_adapter"] = {
        "path": str(R9_ADAPTER),
        "adapter_model_sha256": R9_ADAPTER_SHA256,
        "source_receipt_path": str(R9_RECEIPT),
        "source_receipt_sha256": R9_RECEIPT_SHA256,
        "source_spec_path": R9_SPEC_REL,
        "source_spec_sha256": R9_SPEC_SHA256,
        "source_weight_digest_after": receipt["weight_digest_after"],
        "previous_optimizer_steps": 8,
        "source_cumulative_optimizer_steps": 28,
        "next_train_row": R10_START,
    }
    spec["output"]["namespace"] = R10_OUTPUT
    require(spec["authority"]["paid_compute_authorized"] is False and
            spec["authority"]["merge_authorized"] is False and
            spec["authority"]["deploy_activate_authorized"] is False,
            "training spec overreaches authorization")
    return spec


def write_new(path: Path, value: Any) -> None:
    require(not path.exists(), f"no overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2,
                               ensure_ascii=False) + "\n", encoding="utf8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--source-train", type=Path, required=True)
    ap.add_argument("--voice-seed", type=Path, required=True)
    ap.add_argument("--heldout-voice", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    repo = a.repo_root.resolve()
    require(sha256(a.source_train) == R9_TRAIN_SHA256, "wrong R9 base corpus")
    require(sha256(a.voice_seed) == VOICE_SEED_SHA256,
            "Vera voice seed provenance changed")
    require(sha256(a.heldout_voice) == VOICE_HELDOUT_SHA256,
            "Vera blind heldout changed")
    require(not a.output_dir.exists(), "R10 output already exists")
    spec_path, binding_path = repo / R10_SPEC_REL, repo / R10_BINDING_REL
    require(not spec_path.exists() and not binding_path.exists(),
            "R10 source binding/spec exists: no overwrite")
    source, voice, heldout = (
        read_jsonl(a.source_train), read_jsonl(a.voice_seed),
        read_jsonl(a.heldout_voice))
    new_rows, checks = transform(source, voice, heldout)
    spec = resume_spec(repo, "a" * 64)
    a.output_dir.mkdir(parents=True)
    output = a.output_dir / "train.jsonl"
    with output.open("x", encoding="utf8", newline="\n") as stream:
        for row in new_rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False) + "\n")
    train_sha = sha256(output)
    spec["source_subject"]["train_sha256"] = train_sha
    checks.update({
        "schema": "VERA_V10R3R10_CORPUS_BINDING_V1",
        "source_revision": PARENT_HEAD,
        "r9_train_sha256": R9_TRAIN_SHA256,
        "vera_voice_source_sha256": VOICE_SEED_SHA256,
        "vera_voice_heldout_sha256": VOICE_HELDOUT_SHA256,
        "r9_parent_adapter_sha256": R9_ADAPTER_SHA256,
        "r9_parent_receipt_sha256": R9_RECEIPT_SHA256,
        "r10_train_sha256": train_sha,
        "r10_train_rows": len(new_rows),
        "qualification": "NOT_EVALUATED",
        "training_executed": False,
    })
    write_new(binding_path, checks)
    write_new(spec_path, spec)
    print(json.dumps({
        "status": "R10_QWEN3_5_VERA_VOICE_BUILT_NOT_TRAINED",
        "output": str(output),
        "train_sha256": train_sha,
        "spec_sha256": sha256(spec_path),
        "binding_sha256": sha256(binding_path),
        "axis_counts": checks["identity_axis_counts"],
        "expected_cumulative_steps_if_trained": 36,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
