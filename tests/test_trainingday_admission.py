import hashlib
import json
from pathlib import Path

import pytest

from trainingday.admission import AdmissionError, admit


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _row(case_id: str, split: str, *, prompt: str | None = None) -> dict:
    return {
        "id": case_id, "split": split, "family": "arithmetic",
        "prompt": prompt or f"What is task {case_id}?",
        "answer": f"Answer for {case_id}",
        "training_use": "ADMITTED" if split == "train" else "EVALUATION_ONLY",
        "evaluation_holdout": False,
        "provenance": {
            "source_id": "independent-public-source/example",
            "source_revision": "a" * 40,
            "license": "MIT",
            "origin_class": "HUMAN_CROWDSOURCED",
            "origin_evidence": "https://example.invalid/original-human-crowdsourcing",
            "rights_review": "PASS", "privacy_review": "PASS",
            "openai_output": False,
        },
    }


def _corpus(tmp_path: Path, *, size: int = 3) -> Path:
    root = tmp_path / "corpus"
    (root / "shards").mkdir(parents=True)
    specs = []
    for split, rows in (
        ("train", [_row(f"train{i}", "train") for i in range(size)]),
        ("dev", [_row("dev0", "dev")]),
        ("test", [_row("test0", "test")]),
    ):
        payload = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
        (root / "shards" / f"{split}.jsonl").write_bytes(payload)
        specs.append({"path": f"shards/{split}.jsonl", "split": split,
                      "rows": len(rows), "sha256": _sha(payload)})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema": "VERA_TRAININGDAY_ADMITTED_SHARDS_V1",
                                    "corpus_id": "demo-a", "source_commit": "b" * 40,
                                    "shards": specs}), encoding="utf-8")
    return manifest


def _rewrite_shard(manifest: Path, split: str, rows: list[dict]):
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    shard = next(s for s in spec["shards"] if s["split"] == split)
    content = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    (manifest.parent / shard["path"]).write_bytes(content)
    shard["rows"], shard["sha256"] = len(rows), _sha(content)
    manifest.write_text(json.dumps(spec), encoding="utf-8")


def _admit(manifest: Path, tmp_path: Path, size: int = 2):
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    return admit(manifest, scratch, sample_size=size)


def test_admits_and_deterministically_selects(tmp_path):
    manifest = _corpus(tmp_path)
    a, rows_a = _admit(manifest, tmp_path)
    b, rows_b = _admit(manifest, tmp_path)
    assert a == b
    assert [r["id"] for r in rows_a] == [r["id"] for r in rows_b]
    assert a["rows_by_split"] == {"dev": 1, "test": 1, "train": 3}
    assert a["selected_count"] == 2
    assert a["status"] == "ADMITTED_SOURCE_ONLY_NOT_TRAINED"


def test_bad_bytes_rejected_before_consuming(tmp_path):
    manifest = _corpus(tmp_path)
    with (manifest.parent / "shards/train.jsonl").open("ab") as writer:
        writer.write(b"garbage")
    with pytest.raises(AdmissionError, match="shard sha"):
        _admit(manifest, tmp_path)


@pytest.mark.parametrize("field,value,reason", [
    ("license", "UNKNOWN", "license"),
    ("origin_class", "MODEL_GENERATED", "origin"),
    ("rights_review", "UNKNOWN", "review"),
    ("privacy_review", "UNKNOWN", "review"),
    ("openai_output", True, "OpenAI"),
])
def test_provenance_denial(tmp_path, field, value, reason):
    manifest = _corpus(tmp_path)
    row = _row("train0", "train")
    row["provenance"][field] = value
    _rewrite_shard(manifest, "train", [row, _row("train1", "train")])
    with pytest.raises(AdmissionError, match=reason):
        _admit(manifest, tmp_path)


def test_procedural_requires_oracle_source_hash(tmp_path):
    manifest = _corpus(tmp_path)
    row = _row("train0", "train")
    row["provenance"]["origin_class"] = "INDEPENDENT_PROCEDURAL"
    _rewrite_shard(manifest, "train", [row, _row("train1", "train")])
    with pytest.raises(AdmissionError, match="oracle"):
        _admit(manifest, tmp_path)


def test_same_normalized_prompt_train_test_leak_rejected(tmp_path):
    manifest = _corpus(tmp_path)
    row = _row("test0", "test", prompt="  WHAT   is task Train0? ")
    _rewrite_shard(manifest, "test", [row])
    with pytest.raises(AdmissionError, match="duplicate"):
        _admit(manifest, tmp_path)


def test_duplicate_case_id_across_splits_rejected(tmp_path):
    manifest = _corpus(tmp_path)
    _rewrite_shard(manifest, "dev", [_row("train0", "dev", prompt="distinct prompt")])
    with pytest.raises(AdmissionError, match="duplicate"):
        _admit(manifest, tmp_path)


def test_manifest_shard_path_traversal_rejected(tmp_path):
    manifest = _corpus(tmp_path)
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    spec["shards"][0]["path"] = "shards/../../secret.jsonl"
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(AdmissionError, match="traversal"):
        _admit(manifest, tmp_path)


def test_missing_dev_or_test_rejected(tmp_path):
    manifest = _corpus(tmp_path)
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    spec["shards"] = [s for s in spec["shards"] if s["split"] != "test"]
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(AdmissionError, match="dev/test"):
        _admit(manifest, tmp_path)


def test_symlinked_shard_rejected(tmp_path):
    manifest = _corpus(tmp_path)
    shard = manifest.parent / "shards/train.jsonl"
    target = manifest.parent / "train-real.jsonl"
    shard.rename(target)
    shard.symlink_to(target)
    with pytest.raises(AdmissionError, match="symlinked"):
        _admit(manifest, tmp_path)


def test_scratch_index_is_disposable(tmp_path):
    manifest = _corpus(tmp_path)
    _admit(manifest, tmp_path)
    assert not list((tmp_path / "scratch").iterdir())
