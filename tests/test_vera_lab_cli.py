import json
from pathlib import Path

import pytest

from successor.vera_lab_cli import main


def scenario_record(scenario_id="cli-001"):
    return {
        "scenario_id": scenario_id,
        "version": 1,
        "family": "ordinary_competence",
        "risk_class": "LOW",
        "turn_budget": 2,
        "seed": 17,
        "initial_state": {"mode": "ordinary"},
        "user_turns": ["hello", "what is two plus two?"],
        "perturbations": [],
        "rubric": {"required": ["ordinary response"]},
        "critical_failure_predicates": [],
    }


def write_scenarios(path: Path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


def write_fake_adapter_module(path: Path):
    path.write_text(
        "class FakeAdapter:\n"
        "    def __init__(self, candidate_digest):\n"
        "        self.candidate_digest = candidate_digest\n"
        "    def generate(self, messages, runtime_state):\n"
        "        return 'reply:' + messages[-1]['content']\n"
        "def make_adapter(candidate_id, candidate_digest):\n"
        "    return FakeAdapter(candidate_digest)\n",
        encoding="utf-8",
    )


def test_validate_accepts_well_formed_jsonl(tmp_path, capsys):
    scenario_path = tmp_path / "scenarios.jsonl"
    write_scenarios(scenario_path, [scenario_record()])
    assert main(["validate", str(scenario_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scenario_count"] == 1
    assert len(payload["scenario_set_digest"]) == 64


def test_validate_rejects_malformed_scenario(tmp_path):
    scenario_path = tmp_path / "bad.jsonl"
    bad = scenario_record()
    bad["turn_budget"] = 0
    write_scenarios(scenario_path, [bad])
    with pytest.raises(ValueError):
        main(["validate", str(scenario_path)])


def test_run_writes_deterministic_manifest(tmp_path, monkeypatch):
    scenario_path = tmp_path / "scenarios.jsonl"
    adapter_module = tmp_path / "fake_adapter_module.py"
    output_a = tmp_path / "run-a.json"
    output_b = tmp_path / "run-b.json"
    write_scenarios(scenario_path, [scenario_record()])
    write_fake_adapter_module(adapter_module)
    monkeypatch.syspath_prepend(str(tmp_path))
    args = [
        "run", str(scenario_path),
        "--candidate-id", "candidate-a",
        "--candidate-digest", "abc123",
        "--adapter", "fake_adapter_module:make_adapter",
    ]
    assert main(args + ["--output", str(output_a)]) == 0
    assert main(args + ["--output", str(output_b)]) == 0
    assert output_a.read_bytes() == output_b.read_bytes()
    payload = json.loads(output_a.read_text(encoding="utf-8"))
    assert payload["candidate_id"] == "candidate-a"
    assert payload["candidate_digest"] == "abc123"
    assert len(payload["transcripts"]) == 1


def test_replay_verifies_existing_manifest(tmp_path, monkeypatch):
    scenario_path = tmp_path / "scenarios.jsonl"
    adapter_module = tmp_path / "fake_adapter_module.py"
    output_path = tmp_path / "run.json"
    write_scenarios(scenario_path, [scenario_record()])
    write_fake_adapter_module(adapter_module)
    monkeypatch.syspath_prepend(str(tmp_path))
    run_args = [
        "run", str(scenario_path),
        "--candidate-id", "candidate-a",
        "--candidate-digest", "abc123",
        "--adapter", "fake_adapter_module:make_adapter",
        "--output", str(output_path),
    ]
    assert main(run_args) == 0
    assert main([
        "replay", str(output_path),
        "--scenarios", str(scenario_path),
        "--adapter", "fake_adapter_module:make_adapter",
    ]) == 0


def test_summarize_reports_completion_counts(tmp_path, monkeypatch, capsys):
    scenario_path = tmp_path / "scenarios.jsonl"
    adapter_module = tmp_path / "fake_adapter_module.py"
    output_path = tmp_path / "run.json"
    write_scenarios(scenario_path, [scenario_record("one"), scenario_record("two")])
    write_fake_adapter_module(adapter_module)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert main([
        "run", str(scenario_path),
        "--candidate-id", "candidate-a",
        "--candidate-digest", "abc123",
        "--adapter", "fake_adapter_module:make_adapter",
        "--output", str(output_path),
    ]) == 0
    capsys.readouterr()
    assert main(["summarize", str(output_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["scenario_count"] == 2
    assert summary["completion_statuses"] == {"BUDGET_EXHAUSTED": 2}
