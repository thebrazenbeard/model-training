"""Independent v2 structural holdout and score-comparison gate.

Run only in an evaluator process after training artifacts are frozen. This
module opens eval records and must never be imported by the neural trainer.
Prediction scoring alone cannot prove a model was loaded or executed.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from trainingday.admission import AdmissionError
from trainingday.labyrinth_intake import _replay


class EvaluationGateError(ValueError):
    """Pinned evidence or cross-arm evaluation gate failed."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check_shard(root: Path, entry: dict) -> Path:
    split, rel = entry.get("split"), entry.get("path")
    if split not in ("train", "eval") or not isinstance(rel, str):
        raise EvaluationGateError("invalid split/path")
    if not re.fullmatch(rf"{split}/[0-9]{{6}}\.jsonl", rel):
        raise EvaluationGateError("unconfined shard path")
    path = root / rel
    if not path.is_file() or path.is_symlink() or path.parent.is_symlink():
        raise EvaluationGateError("missing or symlinked shard")
    if (type(entry.get("count")) is not int or type(entry.get("bytes")) is not int
            or entry["count"] < 1 or entry["bytes"] < 1):
        raise EvaluationGateError("invalid shard counts")
    if path.stat().st_size != entry["bytes"] or _digest(path.read_bytes()) != entry.get("sha256"):
        raise EvaluationGateError("shard checksum mismatch")
    return path


def _rows(path: Path, expected_count: int):
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise EvaluationGateError("invalid JSONL") from exc
            if not isinstance(row, dict):
                raise EvaluationGateError("non-object JSONL row")
            count += 1
            yield row
    if count != expected_count:
        raise EvaluationGateError("shard row count mismatch")


def select_structural_cases(
    manifest_path: Path, *, expected_manifest_sha256: str, max_cases: int = 64
) -> dict:
    """Pick unique unseen operator structures from a pinned v2 corpus.

    Returned evaluation prompts are sensitive development evaluation content:
    keep them in the evaluator process, never send them to training.
    """
    manifest_path = Path(manifest_path)
    if (not manifest_path.is_file() or manifest_path.is_symlink()
            or not isinstance(expected_manifest_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256)):
        raise EvaluationGateError("manifest SHA pin missing")
    raw = manifest_path.read_bytes()
    if _digest(raw) != expected_manifest_sha256:
        raise EvaluationGateError("manifest SHA mismatch")
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvaluationGateError("manifest JSON invalid") from exc
    if (manifest.get("schema") != "vera.trainingday.corpus.v1"
            or manifest.get("state") != "CORPUS_BUILT_NO_MODEL_TRAINING"
            or manifest.get("reasoning_split_policy") != "STRUCTURAL_SIGNATURE_HASH_V1"):
        raise EvaluationGateError("not a structural evaluation corpus")
    if (type(max_cases) is not int or not 1 <= max_cases <= 1024
            or type(manifest.get("seed")) is not int):
        raise EvaluationGateError("invalid evaluation bound or seed")
    root = manifest_path.parent
    if root.is_symlink():
        raise EvaluationGateError("symlinked corpus root")
    sources = manifest.get("sources")
    notices = manifest.get("licenses")
    if not isinstance(sources, list) or not isinstance(notices, list):
        raise EvaluationGateError("missing source/notice provenance")
    source_by_repo = {s["repo"]: s for s in sources}
    noticed = set()
    for notice in notices:
        repo = notice["repo"]
        if repo not in source_by_repo or repo in noticed:
            raise EvaluationGateError("unknown/duplicate license notice")
        rel = "licenses/" + repo.replace("/", "__") + ".txt"
        if notice.get("path") != rel or notice.get("sha256") != source_by_repo[repo].get("license_sha256"):
            raise EvaluationGateError("license notice metadata mismatch")
        path = root / rel
        if (not path.is_file() or path.is_symlink() or path.parent.is_symlink()
                or _digest(path.read_bytes()) != notice["sha256"]):
            raise EvaluationGateError("license notice checksum mismatch")
        noticed.add(repo)
    if noticed != set(source_by_repo):
        raise EvaluationGateError("incomplete license notices")

    entries = manifest.get("shards")
    if not isinstance(entries, list) or not entries:
        raise EvaluationGateError("shards absent")
    per_split = defaultdict(list)
    for entry in entries:
        path = _check_shard(root, entry)
        per_split[entry["split"]].append((path, entry["count"], entry["sha256"]))
    for split in ("train", "eval"):
        actual = {p.name for p in (root / split).glob("*.jsonl")}
        paths = per_split[split]
        expected = {p.name for p, _, _ in paths}
        if not expected or expected != actual or len(expected) != len(paths):
            raise EvaluationGateError("missing, undeclared or duplicate shard")
        if sum(count for _, count, _ in paths) != manifest["counts"][split]:
            raise EvaluationGateError("split counts mismatch")

    train_signatures = set()
    eval_by_signature = {}
    seen_ids = set()
    seen_digests = set()
    for split in ("train", "eval"):
        for path, size, expected_hash in sorted(per_split[split], key=lambda e: str(e[0])):
            if _digest(path.read_bytes()) != expected_hash:
                raise EvaluationGateError("shard changed during verification")
            for row in _rows(path, size):
                identifier, digest = row.get("id"), row.get("content_sha256")
                if not isinstance(identifier, str) or not isinstance(digest, str):
                    raise EvaluationGateError("missing row identity")
                if identifier in seen_ids or digest in seen_digests:
                    raise EvaluationGateError("duplicate train/eval record")
                seen_ids.add(identifier)
                seen_digests.add(digest)
                if row.get("kind") == "code":
                    if row.get("verification") != "SOURCE_ONLY_NOT_FUNCTIONALLY_TESTED":
                        raise EvaluationGateError("unverified code mislabeled")
                    continue
                try:
                    case = _replay(row, manifest["seed"])
                except (AdmissionError, ValueError, KeyError, TypeError) as exc:
                    raise EvaluationGateError("independent arithmetic oracle mismatch") from exc
                sig = tuple(op[0] for op in row["operations"])
                if split == "train":
                    train_signatures.add(sig)
                    continue
                priority = _digest((str(manifest["seed"]) + "\x00" + identifier).encode())
                entry = {"id": identifier, "prompt": case["prompt"], "expected": case["answer"]}
                previous = eval_by_signature.get(sig)
                if previous is None or priority < previous[0]:
                    eval_by_signature[sig] = (priority, entry)
    if train_signatures.intersection(eval_by_signature):
        raise EvaluationGateError("structural train/eval overlap")
    if len(eval_by_signature) < max_cases:
        raise EvaluationGateError("insufficient unseen evaluation structures")
    cases = [case for _, case in sorted(eval_by_signature.values(), key=lambda e: e[0])[:max_cases]]
    selection_hash = _digest("\n".join(case["id"] for case in cases).encode())
    return {"status": "STRUCTURAL_EVAL_SELECTED_NO_MODEL_INFERENCE",
            "manifest_sha256": expected_manifest_sha256, "signature_count": len(eval_by_signature),
            "selected_id_sha256": selection_hash, "selected_count": len(cases), "cases": cases}


def score_answers(selection: dict, predictions: dict, *, mode: str) -> dict:
    """Score a complete result set; do not call this evidence of model execution."""
    if mode not in ("frozen", "zero", "correct", "corrupt"):
        raise EvaluationGateError("invalid mode")
    cases = selection.get("cases")
    if not isinstance(cases, list) or not isinstance(predictions, dict):
        raise EvaluationGateError("missing cases/predictions")
    expected = {case["id"]: case["expected"] for case in cases}
    if set(predictions) != set(expected):
        raise EvaluationGateError("prediction ID mismatch")
    if any(not isinstance(value, str) for value in predictions.values()):
        raise EvaluationGateError("prediction format invalid")
    correct = sum(1 for key, value in predictions.items()
                  if re.fullmatch(r"-?(?:0|[1-9]\d*)", value.strip())
                  and value.strip() == expected[key])
    return {"status": "SUBMITTED_PREDICTIONS_SCORED_NOT_MODEL_ATTESTED",
            "mode": mode, "correct": correct, "total": len(cases),
            "manifest_sha256": selection["manifest_sha256"],
            "selected_id_sha256": selection["selected_id_sha256"]}


def compare_control_scores(scores: dict, *, min_gain_cases: int = 1) -> dict:
    """Check matched control advantage but never infer neural authenticity."""
    if set(scores) != {"frozen", "zero", "correct", "corrupt"}:
        raise EvaluationGateError("missing frozen/zero/correct/corrupt control score arm")
    if type(min_gain_cases) is not int or min_gain_cases < 1:
        raise EvaluationGateError("minimum gain invalid")
    frozen, zero, correct, corrupt = [scores[mode] for mode in ("frozen", "zero", "correct", "corrupt")]
    for mode, item in zip(("frozen", "zero", "correct", "corrupt"), (frozen, zero, correct, corrupt)):
        if (item.get("mode") != mode
                or item.get("status") != "SUBMITTED_PREDICTIONS_SCORED_NOT_MODEL_ATTESTED"
                or item.get("manifest_sha256") != zero.get("manifest_sha256")
                or item.get("selected_id_sha256") != zero.get("selected_id_sha256")
                or item.get("total") != zero.get("total")):
            raise EvaluationGateError("unmatched control-score receipts")
    if correct["correct"] < max(frozen["correct"], zero["correct"], corrupt["correct"]) + min_gain_cases:
        raise EvaluationGateError("candidate did not beat matched controls")
    return {"status": "SCORED_CANDIDATE_ADVANTAGE_NOT_MODEL_ATTESTED",
            "candidate_gain": True, "candidate_correct": correct["correct"],
            "frozen_correct": frozen["correct"], "zero_correct": zero["correct"],
            "corrupt_correct": corrupt["correct"],
            "total": zero["total"], "manifest_sha256": zero["manifest_sha256"],
            "selected_id_sha256": zero["selected_id_sha256"]}
