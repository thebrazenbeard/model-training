"""Read-only neural-training adapter for Labyrinth PR #59 corpus v1.

Only independently replayed integer operation chains are accepted as SFT
question/answer pairs. MIT code source rows are inventoried but deliberately
NOT converted into fake instruction/answer supervision. Eval shard text is
never parsed or surfaced to the neural trainer; it is byte-hash checked only.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
import sqlite3
import tempfile
from pathlib import Path

from trainingday.admission import AdmissionError, _canonical_prompt, _hash_file, _require


SCHEMA = "vera.trainingday.corpus.v1"
STATE = "CORPUS_BUILT_NO_MODEL_TRAINING"
OPS = {"add", "subtract", "multiply"}


def _inside(root: Path, relative: str, split: str) -> Path:
    _require(isinstance(relative, str) and bool(re.fullmatch(
        rf"{re.escape(split)}/[A-Za-z0-9_.-]+\.jsonl", relative)),
        "unconfined Labyrinth shard path")
    path = root / relative
    _require(not root.is_symlink() and not (root / split).is_symlink()
             and path.is_file() and not path.is_symlink(),
             "missing/symlinked Labyrinth shard")
    return path


def _replay(row: dict, seed: int) -> dict:
    _require(row.get("kind") == "verified_reasoning" and
             row.get("verification") == "TWO_INTEGER_INTERPRETERS_MATCH",
             "reasoning verification flag missing")
    _require(row.get("license") == "SYNTHETIC_ORIGINAL"
             and row.get("source_repo") == "VERA_TRAININGDAY_GENERATOR"
             and row.get("source_revision") == SCHEMA,
             "reasoning provenance differs from pinned generator")
    _require(type(row.get("start")) is int and type(row.get("answer")) is int,
             "integer oracle payload malformed")
    ops = row.get("operations")
    _require(isinstance(ops, list) and 2 <= len(ops) <= 5,
             "operation-chain length invalid")
    value = row["start"]
    for op in ops:
        _require(isinstance(op, list) and len(op) == 2
                 and op[0] in OPS and type(op[1]) is int and -15 <= op[1] <= 15,
                 "operation-chain instruction invalid")
        name, operand = op
        if name == "add":
            value += operand
        elif name == "subtract":
            value -= operand
        else:
            value *= operand
    _require(value == row["answer"], "independent arithmetic oracle mismatch")
    prompt = ("Start with " + str(row["start"]) + ". " +
              " ".join(f"{name} {operand}." for name, operand in ops) +
              " What is the final integer?")
    text = f"Question: {prompt}\nAnswer: {value}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    _require(row.get("prompt") == prompt and row.get("text") == text,
             "reasoning prompt/label differs from generator")
    _require(row.get("content_sha256") == digest
             and row.get("id") == "reasoning-" + digest,
             "reasoning content hash/id mismatch")
    source_path = row.get("source_path")
    _require(isinstance(source_path, str)
             and re.fullmatch(rf"seed/{seed}/case/[0-9]{{10}}", source_path) is not None,
             "reasoning source index/seed mismatch")
    return {"id": row["id"], "family": "verified_integer_chains",
            "prompt": prompt, "answer": str(value)}


def admit_labyrinth(manifest_path: Path, work_root: Path, *, sample_size: int = 64,
                    producer_git_head: str) -> tuple[dict, list[dict]]:
    manifest_path = Path(manifest_path)
    work_root = Path(work_root)
    _require(manifest_path.is_file() and not manifest_path.is_symlink(), "Labyrinth manifest missing")
    _require(work_root.is_dir() and not work_root.is_symlink(), "work root absent")
    _require(isinstance(sample_size, int) and 1 <= sample_size <= 1024, "sample bound invalid")
    _require(isinstance(producer_git_head, str)
             and re.fullmatch(r"[0-9a-f]{40}", producer_git_head) is not None,
             "producer Git head must be exact")
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == SCHEMA and manifest.get("state") == STATE,
             "Labyrinth corpus not completed")
    _require(type(manifest.get("seed")) is int and manifest["seed"] >= 0,
             "generator seed missing")
    _require(isinstance(manifest.get("heldout_repo"), str)
             and manifest["heldout_repo"], "heldout repo missing")
    sources = manifest.get("sources")
    _require(isinstance(sources, list) and len(sources) >= 2, "source provenance missing")
    source_map = {}
    for source in sources:
        _require(isinstance(source, dict) and source.get("license") == "MIT"
                 and isinstance(source.get("repo"), str)
                 and isinstance(source.get("revision"), str)
                 and re.fullmatch(r"[0-9a-f]{40}", source["revision"]),
                 "source repo/license not pinned")
        _require(source["repo"] not in source_map, "duplicate source repo")
        source_map[source["repo"]] = source
    _require(manifest["heldout_repo"] in source_map, "heldout source absent")
    license_map = {}
    for notice in manifest.get("licenses", []):
        _require(isinstance(notice, dict) and notice.get("repo") in source_map
                 and notice.get("license") == "MIT", "notice entry invalid")
        relative = notice.get("path", "")
        expected = "licenses/" + notice["repo"].replace("/", "__") + ".txt"
        _require(relative == expected and isinstance(notice.get("sha256"), str),
                 "license notice path/hash mismatch")
        path = root / relative
        _require(path.is_file() and not path.is_symlink()
                 and not path.parent.is_symlink() and _hash_file(path) == notice["sha256"],
                 "license notice missing/tampered")
        _require(source_map[notice["repo"]].get("license_sha256") == notice["sha256"],
                 "upstream license digest differs from notice")
        license_map[notice["repo"]] = notice["sha256"]
    _require(set(license_map) == set(source_map), "license notices incomplete")
    shards = manifest.get("shards")
    _require(isinstance(shards, list) and bool(shards), "shards absent")
    seen_paths = set()
    train = []
    total = {"train": 0, "eval": 0}
    for shard in shards:
        _require(isinstance(shard, dict) and shard.get("split") in total,
                 "shard entry malformed")
        rel, split = shard.get("path"), shard["split"]
        _require(isinstance(rel, str) and rel not in seen_paths, "repeated shard path")
        seen_paths.add(rel)
        _require(type(shard.get("count")) is int and shard["count"] > 0
                 and type(shard.get("bytes")) is int and shard["bytes"] > 0
                 and isinstance(shard.get("sha256"), str)
                 and re.fullmatch(r"[0-9a-f]{64}", shard["sha256"]) is not None,
                 "shard count/size/hash invalid")
        file = _inside(root, rel, split)
        _require(file.stat().st_size == shard["bytes"] and _hash_file(file) == shard["sha256"],
                 "shard missing/tampered")
        total[split] += shard["count"]
        if split == "train":
            train.append(shard)
    _require(all(total.values()) and total == manifest.get("counts"),
             "train/eval count manifest mismatch")
    _require(set(p.relative_to(root).as_posix() for p in (root / "train").glob("*.jsonl"))
             == {s["path"] for s in train}, "undeclared train shard")

    picked = []
    reasoning_count = code_count = 0
    with tempfile.TemporaryDirectory(prefix="trainingday-lab-intake-", dir=work_root) as temp:
        db = sqlite3.connect(str(Path(temp) / "seen.sqlite3"))
        try:
            db.execute("CREATE TABLE seen (fingerprint TEXT PRIMARY KEY) WITHOUT ROWID")
            for shard in sorted(train, key=lambda s: s["path"]):
                file = _inside(root, shard["path"], "train")
                checksum = hashlib.sha256()
                observed = 0
                with file.open("rb") as stream:
                    for raw in stream:
                        checksum.update(raw)
                        observed += 1
                        try:
                            row = json.loads(raw.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise AdmissionError("invalid Labyrinth JSONL") from exc
                        if row.get("kind") == "code":
                            repo = row.get("source_repo")
                            _require(repo != manifest["heldout_repo"] and repo in source_map,
                                     "heldout/unlicensed source code in train")
                            _require(row.get("license") == "MIT"
                                     and row.get("license_sha256") == license_map[repo]
                                     and row.get("source_revision") == source_map[repo]["revision"]
                                     and row.get("verification") == "SOURCE_ONLY_NOT_FUNCTIONALLY_TESTED",
                                     "code source rights/verification mismatch")
                            body = row.get("text")
                            _require(isinstance(body, str) and
                                     hashlib.sha256(body.encode("utf-8")).hexdigest()
                                     == row.get("content_sha256")
                                     and row.get("id") == "code-" + row["content_sha256"],
                                     "code text/id digest mismatch")
                            code_count += 1
                            continue  # Not a supervised question/answer pair.
                        case = _replay(row, manifest["seed"])
                        try:
                            db.execute("INSERT INTO seen VALUES (?)", (
                                hashlib.sha256(_canonical_prompt(case["prompt"])).hexdigest(),))
                        except sqlite3.IntegrityError as exc:
                            raise AdmissionError("repeated normalized reasoning prompt") from exc
                        reasoning_count += 1
                        priority = int(hashlib.sha256((str(manifest["seed"]) + "\x00" + case["id"]).encode()).hexdigest(), 16)
                        entry = (-priority, case["id"], case)
                        if len(picked) < sample_size:
                            heapq.heappush(picked, entry)
                        elif entry[0] > picked[0][0]:
                            heapq.heapreplace(picked, entry)
                _require(observed == shard["count"] and checksum.hexdigest() == shard["sha256"],
                         "Labyrinth shard mutated/row count mismatch during read")
        finally:
            db.close()
    _require(reasoning_count >= sample_size, "insufficient verified reasoning train rows")
    selected = [case for _, _, case in sorted(picked, key=lambda e: (-e[0], e[1]))]
    summary = {"status": "ADMITTED_LABYRINTH_TRAIN_ONLY_NO_WEIGHT_UPDATE",
               "producer_git_head": producer_git_head,
               "manifest_sha256": _hash_file(manifest_path),
               "source_license_count": len(license_map),
               "train_code_rows_not_used_for_sft": code_count,
               "verified_reasoning_train_rows": reasoning_count,
               "declared_eval_rows_not_opened": total["eval"],
               "selected_count": sample_size,
               "selected_id_sha256": hashlib.sha256(
                   "\n".join(case["id"] for case in selected).encode()).hexdigest()}
    return summary, selected
