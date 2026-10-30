import hashlib
import json

import pytest

from trainingday.admission import AdmissionError
from trainingday.labyrinth_intake import admit_labyrinth


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _reason(index, start):
    seed = 29033
    ops = [["add", index + 1], ["multiply", 2]]
    value = (start + index + 1) * 2
    prompt = (f"Start with {start}. add {index + 1}. multiply 2. "
              "What is the final integer?")
    content = f"Question: {prompt}\nAnswer: {value}"
    digest = _sha(content.encode())
    return {"id": "reasoning-" + digest, "kind": "verified_reasoning",
            "verification": "TWO_INTEGER_INTERPRETERS_MATCH",
            "license": "SYNTHETIC_ORIGINAL",
            "source_repo": "VERA_TRAININGDAY_GENERATOR",
            "source_revision": "vera.trainingday.corpus.v1",
            "source_path": f"seed/{seed}/case/{index:010d}",
            "start": start, "operations": ops, "answer": value,
            "prompt": prompt, "text": content, "content_sha256": digest}


def _fixture(tmp_path):
    root = tmp_path / "lab"
    for name in ("train", "eval", "licenses"):
        (root / name).mkdir(parents=True, exist_ok=True)
    license_bytes = b"MIT License\nPermission is hereby granted\nTHE SOFTWARE IS PROVIDED \"AS IS\"\n"
    license_sha = _sha(license_bytes)
    sources = []
    licenses = []
    for repo in ("TheAlgorithms/Python", "TheAlgorithms/Go"):
        relative = "licenses/" + repo.replace("/", "__") + ".txt"
        (root / relative).write_bytes(license_bytes)
        sources.append({"repo": repo, "revision": "a" * 40, "license": "MIT",
                        "license_sha256": license_sha})
        licenses.append({"repo": repo, "path": relative,
                         "license": "MIT", "sha256": license_sha})
    code = "def a(x):\n    return x + 1\n" * 5
    code_sha = _sha(code.encode())
    code_row = {"id": "code-" + code_sha, "kind": "code",
                "verification": "SOURCE_ONLY_NOT_FUNCTIONALLY_TESTED",
                "source_repo": "TheAlgorithms/Python", "source_revision": "a" * 40,
                "license": "MIT", "license_sha256": license_sha,
                "content_sha256": code_sha, "text": code, "source_path": "x.py"}
    rows = [code_row, _reason(0, 1), _reason(1, 2), _reason(2, 3)]
    train_bytes = b"".join((json.dumps(x, sort_keys=True) + "\n").encode() for x in rows)
    eval_bytes = b'{"private":"evaluation plaintext must not be parsed"}\n'
    (root / "train/000000.jsonl").write_bytes(train_bytes)
    (root / "eval/000000.jsonl").write_bytes(eval_bytes)
    manifest = {"schema": "vera.trainingday.corpus.v1",
                "state": "CORPUS_BUILT_NO_MODEL_TRAINING", "seed": 29033,
                "heldout_repo": "TheAlgorithms/Go", "counts": {"train": 4, "eval": 1},
                "sources": sources, "licenses": licenses,
                "shards": [
                    {"split": "train", "path": "train/000000.jsonl", "count": 4,
                     "bytes": len(train_bytes), "sha256": _sha(train_bytes)},
                    {"split": "eval", "path": "eval/000000.jsonl", "count": 1,
                     "bytes": len(eval_bytes), "sha256": _sha(eval_bytes)},
                ]}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    (tmp_path / "work").mkdir()
    return path, rows


def _admit(path, tmp_path):
    return admit_labyrinth(path, tmp_path / "work", sample_size=2,
                           producer_git_head="188d3340e4498e69d8c54d66ebee9dabd6582d31")


def _rewrite_train(path, rows):
    manifest = json.loads(path.read_text())
    data = b"".join((json.dumps(x, sort_keys=True) + "\n").encode() for x in rows)
    (path.parent / "train/000000.jsonl").write_bytes(data)
    shard = next(s for s in manifest["shards"] if s["split"] == "train")
    shard.update(count=len(rows), bytes=len(data), sha256=_sha(data))
    manifest["counts"]["train"] = len(rows)
    path.write_text(json.dumps(manifest))


def test_actual_labyrinth_schema_selected_only_reasoning(tmp_path):
    path, rows = _fixture(tmp_path)
    report, selected = _admit(path, tmp_path)
    assert report["train_code_rows_not_used_for_sft"] == 1
    assert report["verified_reasoning_train_rows"] == 3
    assert report["declared_eval_rows_not_opened"] == 1
    assert len(selected) == 2
    assert all(case["family"] == "verified_integer_chains" for case in selected)
    assert not list((tmp_path / "work").iterdir())


def test_independent_answer_replay_rejects_bad_label(tmp_path):
    path, rows = _fixture(tmp_path)
    rows[1]["answer"] += 1
    _rewrite_train(path, rows)
    with pytest.raises(AdmissionError, match="oracle"):
        _admit(path, tmp_path)


def test_hash_tamper_rejects_before_row_read(tmp_path):
    path, _ = _fixture(tmp_path)
    with (path.parent / "train/000000.jsonl").open("ab") as stream:
        stream.write(b"X")
    with pytest.raises(AdmissionError, match="tampered"):
        _admit(path, tmp_path)


def test_license_notice_tamper_rejected(tmp_path):
    path, _ = _fixture(tmp_path)
    (path.parent / "licenses/TheAlgorithms__Python.txt").write_bytes(b"not MIT")
    with pytest.raises(AdmissionError, match="notice"):
        _admit(path, tmp_path)


def test_go_holdout_code_cannot_enter_training(tmp_path):
    path, rows = _fixture(tmp_path)
    rows[0]["source_repo"] = "TheAlgorithms/Go"
    _rewrite_train(path, rows)
    with pytest.raises(AdmissionError, match="heldout"):
        _admit(path, tmp_path)


def test_missing_completed_manifest_rejected(tmp_path):
    path, _ = _fixture(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["state"] = "STILL_BUILDING"
    path.write_text(json.dumps(manifest))
    with pytest.raises(AdmissionError, match="not completed"):
        _admit(path, tmp_path)


def test_eval_bytes_verified_but_not_parsed(tmp_path):
    path, _ = _fixture(tmp_path)
    # The default eval shard is intentionally not in our train JSONL schema.
    # If parsed, this positive test would fail. Still hash it exactly.
    summary, _ = _admit(path, tmp_path)
    assert summary["declared_eval_rows_not_opened"] == 1
    (path.parent / "eval/000000.jsonl").write_bytes(b"changed")
    with pytest.raises(AdmissionError, match="tampered"):
        _admit(path, tmp_path)


def test_duplicate_normalized_reasoning_prompt_rejected(tmp_path):
    path, rows = _fixture(tmp_path)
    rows.append(dict(rows[1]))
    _rewrite_train(path, rows)
    with pytest.raises(AdmissionError, match="repeated normalized"):
        _admit(path, tmp_path)
