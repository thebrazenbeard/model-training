import json
from pathlib import Path

from successor.corpus_builder import build_examples, fingerprint


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_builder_accepts_only_reviewed_sft_candidates_and_current_anchors(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(corpus, [
        {"record_id": "keep", "promotion_status": "reviewed_candidate", "training_use": ["sft_candidate"], "trigger": "Old prompt", "accepted_target": "Old answer"},
        {"record_id": "drop", "promotion_status": "reviewed_candidate_not_approved", "training_use": ["sft_candidate"], "trigger": "Bad prompt", "accepted_target": "Bad answer"},
        {"record_id": "eval", "promotion_status": "reviewed_candidate", "training_use": ["evaluation_probe"], "trigger": "Exam", "accepted_target": "Exam answer"},
    ])
    anchors = [{"prompt": "Who are you?", "response": "I am Vera.", "weight": 4}]
    rows = build_examples([corpus], anchors, holdout_hashes=set())
    assert [(r["prompt"], r["response"]) for r in rows] == [("Old prompt", "Old answer"), ("Who are you?", "I am Vera.")]
    assert rows[-1]["weight"] == 4


def test_builder_excludes_exact_holdout_fingerprint(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(corpus, [{"record_id": "x", "promotion_status": "reviewed_candidate", "training_use": ["sft_candidate"], "trigger": "P", "accepted_target": "R"}])
    blocked = {fingerprint("P", "R")}
    assert build_examples([corpus], [], blocked) == []
