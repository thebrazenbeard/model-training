import argparse
import hashlib
import json

import pytest

from trainingday.admission import AdmissionError
from trainingday.train_neural import _control_rows, _required_base, train, validate_run
from test_trainingday_admission import _corpus
from test_trainingday_labyrinth_intake import _fixture as _labyrinth_fixture


def _base(tmp_path):
    root = tmp_path / "official-base"
    root.mkdir()
    files = {}
    for name, content in (("config.json", b'{}'), ("tokenizer_config.json", b'{}'),
                          ("model-00001-of-00001.safetensors", b"test tensor bytes")):
        (root / name).write_bytes(content)
        files[name] = hashlib.sha256(content).hexdigest()
    binding = tmp_path / "base-binding.json"
    binding.write_text(json.dumps({"schema": "TRAININGDAY_PINNED_BASE_V1",
                                   "model_id": "Qwen/Qwen3.5-4B", "revision": "c" * 40,
                                   "loader_class": "Qwen3_5ForCausalLM",
                                   "quantization": "NF4_DOUBLE_BF16",
                                   "snapshot_path": str(root), "files": files}))
    return binding, root


def _args(tmp_path, mode="zero", steps=0):
    manifest = _corpus(tmp_path, size=16)
    binding, _ = _base(tmp_path)
    (tmp_path / "work").mkdir()
    (tmp_path / "outputs").mkdir()
    return argparse.Namespace(manifest=manifest, base_binding=binding,
                              work_root=tmp_path / "work", output_root=tmp_path / "outputs",
                              run_id="zero-a", mode=mode, steps=steps, cases=8,
                              seed=41, learning_rate=2e-5,
                              expected_initial_digest=None, preflight_only=True,
                              producer_git_head=None)


def test_only_preflight_runs_without_gpu_or_importing_torch(tmp_path):
    args = _args(tmp_path)
    report = train(args)
    assert report["status"] == "SOURCE_ONLY_PREFLIGHT"
    assert report["new_optimizer_steps"] == 0
    assert len(report["training_input_sha256"]) == 64


def test_trained_arm_requires_control_binding(tmp_path):
    args = _args(tmp_path, mode="correct", steps=1)
    with pytest.raises(AdmissionError, match="initial digest"):
        validate_run(args)
    args.expected_initial_digest = "d" * 64
    summary, cases, base, path = validate_run(args)
    assert len(cases) == 8
    assert base["model_id"] == "Qwen/Qwen3.5-4B"
    assert path.name == "official-base"


def test_control_corruption_is_matched_within_family():
    rows = [{"id": "a", "family": "f", "prompt": "p1", "answer": "yes"},
            {"id": "b", "family": "f", "prompt": "p2", "answer": "no"}]
    correct = _control_rows(rows, "correct")
    wrong = _control_rows(rows, "corrupt")
    assert [x["prompt"] for x in wrong] == [x["prompt"] for x in correct]
    assert [x["response"] for x in wrong] == ["no", "yes"]


def test_single_label_control_is_rejected():
    rows = [{"id": "a", "family": "f", "prompt": "p", "answer": "only"}]
    with pytest.raises(AdmissionError, match="two distinct answers"):
        _control_rows(rows, "corrupt")


def test_other_arms_and_bad_bounds_rejected_before_gpu(tmp_path):
    args = _args(tmp_path)
    args.mode, args.steps = "zero", 1
    with pytest.raises(AdmissionError, match="steps"):
        validate_run(args)
    args.mode, args.steps, args.cases = "correct", 2, 8
    with pytest.raises(AdmissionError, match="cases"):
        validate_run(args)


def test_unpinned_or_unbound_base_rejected(tmp_path):
    binding, base = _base(tmp_path)
    (base / "extra.safetensors").write_bytes(b"not admitted")
    with pytest.raises(AdmissionError, match="unbound"):
        _required_base(binding)


def test_sourceless_model_id_rejected(tmp_path):
    binding, _ = _base(tmp_path)
    data = json.loads(binding.read_text())
    data["model_id"] = "rodrigomt/Qwen3.5-4B-Uncensored-Aggressive"
    binding.write_text(json.dumps(data))
    with pytest.raises(AdmissionError, match="official base"):
        _required_base(binding)


def test_labyrinth_producer_schema_integrates_without_eval_plaintext(tmp_path):
    manifest, _ = _labyrinth_fixture(tmp_path)
    binding, _ = _base(tmp_path)
    (tmp_path / "outputs").mkdir()
    args = argparse.Namespace(manifest=manifest, base_binding=binding,
                              work_root=tmp_path / "work", output_root=tmp_path / "outputs",
                              run_id="lab-zero", mode="zero", steps=0, cases=2,
                              seed=41, learning_rate=2e-5,
                              expected_initial_digest=None, preflight_only=True,
                              producer_git_head="188d3340e4498e69d8c54d66ebee9dabd6582d31")
    result = train(args)
    assert result["corpus"]["train_code_rows_not_used_for_sft"] == 1
    assert result["corpus"]["selected_count"] == 2
    assert result["new_optimizer_steps"] == 0
