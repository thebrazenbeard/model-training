"""Source-first TRAININGDAY structural evaluation gate (no GPU needed)."""
import hashlib
import json

import pytest

from test_trainingday_labyrinth_intake import _fixture, _reason
from trainingday.structural_eval import (
    EvaluationGateError, select_structural_cases, score_answers,
    compare_control_scores,
)


def _structural(tmp_path):
    path, _ = _fixture(tmp_path)
    root = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["reasoning_split_policy"] = "STRUCTURAL_SIGNATURE_HASH_V1"
    row = _reason(99, 5)
    row["operations"] = [["subtract", 3], ["multiply", 4]]
    row["answer"] = 8
    row["prompt"] = "Start with 5. subtract 3. multiply 4. What is the final integer?"
    row["text"] = f"Question: {row['prompt']}\nAnswer: 8"
    digest = hashlib.sha256(row["text"].encode()).hexdigest()
    row["content_sha256"] = digest
    row["id"] = "reasoning-" + digest
    data = (json.dumps(row, sort_keys=True) + "\n").encode()
    (root / "eval/000000.jsonl").write_bytes(data)
    entry = next(x for x in manifest["shards"] if x["split"] == "eval")
    entry.update(bytes=len(data), count=1, sha256=hashlib.sha256(data).hexdigest())
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), row


def test_selects_a_real_unseen_structure_and_keeps_source_pins(tmp_path):
    path, pin, eval_row = _structural(tmp_path)
    selected = select_structural_cases(path, expected_manifest_sha256=pin, max_cases=1)
    assert selected["status"] == "STRUCTURAL_EVAL_SELECTED_NO_MODEL_INFERENCE"
    assert selected["manifest_sha256"] == pin
    assert selected["signature_count"] == 1
    assert selected["selected_count"] == 1
    assert selected["cases"][0]["id"] == eval_row["id"]
    assert selected["cases"][0]["expected"] == "8"
    assert selected["cases"][0]["prompt"] == eval_row["prompt"]


def test_wrong_manifest_pin_is_refused(tmp_path):
    path, _, _ = _structural(tmp_path)
    with pytest.raises(EvaluationGateError, match="manifest SHA"):
        select_structural_cases(path, expected_manifest_sha256="0" * 64)


def test_reshashed_wrong_arithmetic_label_is_refused(tmp_path):
    path, _, _ = _structural(tmp_path)
    manifest = json.loads(path.read_text())
    file = path.parent / "eval/000000.jsonl"
    row = json.loads(file.read_text())
    row["answer"] = 9
    data = (json.dumps(row, sort_keys=True) + "\n").encode()
    file.write_bytes(data)
    entry = next(x for x in manifest["shards"] if x["split"] == "eval")
    entry.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    path.write_text(json.dumps(manifest), encoding="utf-8")
    pin = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(EvaluationGateError, match="oracle"):
        select_structural_cases(path, expected_manifest_sha256=pin)


def test_rejects_leaked_train_operator_structure(tmp_path):
    path, _, _ = _structural(tmp_path)
    manifest = json.loads(path.read_text())
    file = path.parent / "eval/000000.jsonl"
    row = json.loads(file.read_text())
    row["operations"] = [["add", 4], ["multiply", 2]]
    row["answer"] = 18
    row["prompt"] = "Start with 5. add 4. multiply 2. What is the final integer?"
    row["text"] = f"Question: {row['prompt']}\nAnswer: 18"
    digest = hashlib.sha256(row["text"].encode()).hexdigest()
    row.update(id="reasoning-" + digest, content_sha256=digest)
    data = (json.dumps(row, sort_keys=True) + "\n").encode()
    file.write_bytes(data)
    entry = next(x for x in manifest["shards"] if x["split"] == "eval")
    entry.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    path.write_text(json.dumps(manifest), encoding="utf-8")
    pin = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(EvaluationGateError, match="structural.*overlap"):
        select_structural_cases(path, expected_manifest_sha256=pin)


def test_scoring_refuses_partial_or_extra_predictions(tmp_path):
    path, pin, row = _structural(tmp_path)
    selected = select_structural_cases(path, expected_manifest_sha256=pin, max_cases=1)
    with pytest.raises(EvaluationGateError, match="prediction ID"):
        score_answers(selected, {}, mode="correct")
    with pytest.raises(EvaluationGateError, match="prediction ID"):
        score_answers(selected, {row["id"]: "8", "fake": "8"}, mode="correct")
    good = score_answers(selected, {row["id"]: "8"}, mode="correct")
    bad = score_answers(selected, {row["id"]: "7"}, mode="corrupt")
    assert good["correct"] == 1 and bad["correct"] == 0
    assert good["status"] == "SUBMITTED_PREDICTIONS_SCORED_NOT_MODEL_ATTESTED"
    assert "prompt" not in good


def test_candidate_gain_requires_better_than_zero_and_corrupt(tmp_path):
    path, pin, row = _structural(tmp_path)
    selected = select_structural_cases(path, expected_manifest_sha256=pin, max_cases=1)
    good = score_answers(selected, {row["id"]: "8"}, mode="correct")
    zero = score_answers(selected, {row["id"]: "7"}, mode="zero")
    corrupt = score_answers(selected, {row["id"]: "7"}, mode="corrupt")
    result = compare_control_scores({"zero": zero, "correct": good, "corrupt": corrupt},
                                    min_gain_cases=1)
    assert result["candidate_gain"] is True
    assert result["status"] == "SCORED_CANDIDATE_ADVANTAGE_NOT_MODEL_ATTESTED"
    with pytest.raises(EvaluationGateError, match="matched"):
        compare_control_scores({"zero": good, "correct": good, "corrupt": corrupt},
                               min_gain_cases=1)
    with pytest.raises(EvaluationGateError, match="missing"):
        compare_control_scores({"zero": zero, "correct": good}, min_gain_cases=1)
