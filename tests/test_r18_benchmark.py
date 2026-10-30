"""CPU-only R18 receipt and sealed-data protocol tests (no GPU or network)."""

import copy
import base64
import hashlib
import json
import os
import stat
import subprocess
import sys
import types

import pytest

from trainingday.r18_benchmark import (
    ARM_SCHEMA, EVAL_SOURCE, FIT_LENGTHS, PLAN_SCHEMA, PROBE_SCHEMA,
    BenchmarkError, canonical_sha256, compare_arms, fit_candidates,
    main, make_plan, preregister_from_corpus, selected_id_digest,
    _protect_private_path, _windows_owner_sid, _windows_verify_owner_only,
    validate_arm_receipt, validate_fit_probes,
    validate_plan,
)
from trainingday.r18_eval_source import generate_novel_r18_cases


TEST_PRIVATE_NONCE = "ab" * 32


H = "a" * 64
J = "b" * 64


def _rows():
    return [
        {"id": f"train-{i}", "split": "train", "training_use": "ADMITTED",
         "prompt": f"What is {i} plus one?", "answer": str(i+1)}
        for i in range(8)
    ]


def _eval_index():
    return [{"id": f"r18-holdout-{i}",
             "prompt_sha256": hashlib.sha256(f"r18-prompt-{i}".encode()).hexdigest()}
            for i in range(64)]


def _plan():
    rows = _rows()
    return make_plan(
        selected_rows=rows, admission_selected_id_sha256=selected_id_digest(rows),
        source_admission_status="ADMITTED_SOURCE_ONLY_NOT_TRAINED",
        eval_index=_eval_index(),
        r16_holdout_ids=[f"r16-fresh64-{i}" for i in range(64)],
        base_revision="1" * 40, base_binding_sha256=H,
        tokenizer_binding_sha256=J, corpus_manifest_sha256="c" * 64,
        eval_custody_sha256="d" * 64, eval_source_manifest_sha256="f" * 64,
        r16_eval_custody_sha256="e" * 64,
        seed=41, training_budget_seconds=100.0, probe_budget_seconds=5.0,
    )


def _probes(plan):
    result = []
    # batch1 succeeds 64..384 then OOM at 512; batch2 fits 64/128, OOM 256.
    for arm in ("mezo", "qlora"):
        for index in (0, 1, 2, 3, 4, 7, 8, 9):
            cfg = fit_candidates()[index]
            fit = index in (0, 1, 2, 3, 7, 8)
            result.append({"schema": PROBE_SCHEMA,
                           "plan_sha256": validate_plan(plan), "arm": arm,
                           "candidate_index": index, "config": cfg,
                           "outcome": "FIT" if fit else "OOM",
                           "optimizer_step_completed": fit,
                           "elapsed_seconds": 3.0,
                           "data_partition": "TRAIN_ONLY", "eval_access": False,
                           "checkpoint_selection": False,
                           "fresh_process_id": f"{arm}-{index}-process",
                           "actual_padded_sequence_length":
                               cfg["actual_padded_sequence_length"],
                           "peak_cuda_allocated_bytes": 500_000,
                           "peak_cuda_reserved_bytes": 600_000,
                           "peak_nvml_used_bytes": 700_000,
                           "peak_cpu_rss_bytes": 800_000})
            result[-1]["failure_artifact_sha256"] = None if fit else "f" * 64
    return result


def _receipt(plan, arm, control, *, probe_sha, selected_cfg):
    trained = control in ("correct", "corrupt")
    start = 100_000_000_000
    secs = 96.0 if trained else 0.0
    return {
        "schema": ARM_SCHEMA, "arm": arm, "control": control,
        "run_id": f"{arm}-{control}", "plan_sha256": validate_plan(plan),
        **{key: plan[key] for key in (
            "base_model", "base_revision", "base_binding_sha256",
            "tokenizer_binding_sha256", "corpus_manifest_sha256",
            "selected_id_sha256", "train_content_sha256", "source_admission_status",
            "eval_custody_sha256",
            "eval_index_sha256", "seed")},
        "training_budget_seconds": 100.0,
        "training_input_sha256": J if control != "corrupt" else "c" * 64,
        "fit_probe_sha256": probe_sha, "selected_fit_config": selected_cfg,
        "max_prepared_example_tokens": 96,
        "optimizer_steps": 8 if trained else 0,
        "unique_examples_seen": 8 if trained else 0,
        "example_presentations": 64 if trained else 0,
        "input_tokens_presented": 6400 if trained else 0,
        "forward_sample_tokens": (12800 if arm == "mezo" else 6400) if trained else 0,
        "backward_sample_tokens": (0 if arm == "mezo" else 6400) if trained else 0,
        "padded_forward_tokens": (16_384 if arm == "mezo" else 8192) if trained else 0,
        "padded_backward_tokens": (0 if arm == "mezo" else 8192) if trained else 0,
        "supervised_target_tokens": 320 if trained else 0,
        "peak_cuda_allocated_bytes": 500_000,
        "peak_cuda_reserved_bytes": 600_000,
        "peak_nvml_used_bytes": 700_000,
        "peak_cpu_rss_bytes": 800_000,
        "trainable_digest_before": H,
        "trainable_digest_after": J if trained else H,
        "train_start_monotonic_ns": start,
        "train_stop_monotonic_ns": start + int(secs * 1e9),
        "train_deadline_monotonic_ns": start + 100_000_000_000,
        "optimizer_window_seconds": secs,
        "stage_seconds": {"load": 2.0, "setup": 3.0, "train": secs,
                          "save": 1.0, "eval": 2.0},
        "external_process_wall_seconds": 10.0 + secs,
        "status": "EVALUATED_SEALED", "eval_unsealed_after_arm_freeze": True,
        "arm_freeze_sha256": "a" * 64, "eval_case_receipt_sha256": "b" * 64,
        "arm_frozen_at_utc_ns": 1_000_000_000_000,
        "eval_unsealed_at_utc_ns": 1_001_000_000_000,
        "eval_examples": 64, "eval_correct": 5 if control == "correct" else 3,
        "eval_accuracy": (5 if control == "correct" else 3) / 64,
        "eval_target_tokens": 220, "eval_nll": 1.4 if control == "correct" else 1.8,
        "eval_nll_sum": 308.0 if control == "correct" else 396.0,
    }


def _comparison_inputs():
    plan = _plan()
    probes = _probes(plan)
    fit = validate_fit_probes(plan, probes)
    receipts = [_receipt(plan, arm, control, probe_sha=fit[arm]["probe_sha256"],
                         selected_cfg=fit[arm]["primary_training_config"])
                for arm in ("mezo", "qlora")
                for control in ("frozen", "zero", "correct", "corrupt")]
    return plan, probes, receipts


def test_plan_exact_base_sealed_eval_and_selection_order():
    plan = _plan()
    assert plan["schema"] == PLAN_SCHEMA
    assert plan["eval_bank_source"] == EVAL_SOURCE
    assert plan["selected_id_sha256"] == hashlib.sha256(
        "\n".join(f"train-{i}" for i in range(8)).encode()).hexdigest()
    assert validate_plan(plan) == canonical_sha256(plan)
    assert [c["actual_padded_sequence_length"] for c in fit_candidates()[:7]] == list(FIT_LENGTHS)


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(base_revision="unpinned"),
    lambda p: p.update(eval_bank_source="R16_FRESH64"),
    lambda p: p.update(no_eval_access_before_arm_freeze=False),
    lambda p: p.update(promotion_policy="PROMOTE_ON_SMALL_GAIN"),
    lambda p: p["fit_candidates"].reverse(),
])
def test_preregistered_plan_cannot_be_rewritten(mutation):
    plan = _plan()
    mutation(plan)
    with pytest.raises(BenchmarkError):
        validate_plan(plan)


def test_sealed_eval_rejects_overlap_answers_prompt_duplicate_or_r16_reuse():
    fixture = dict(
        selected_rows=_rows(), admission_selected_id_sha256=selected_id_digest(_rows()),
        source_admission_status="ADMITTED_SOURCE_ONLY_NOT_TRAINED",
        eval_index=_eval_index(), r16_holdout_ids=[f"r16-fresh64-{i}" for i in range(64)],
        base_revision="1" * 40, base_binding_sha256=H, tokenizer_binding_sha256=J,
        corpus_manifest_sha256="c" * 64, eval_custody_sha256="d" * 64,
        eval_source_manifest_sha256="f" * 64,
        r16_eval_custody_sha256="e" * 64, seed=41,
        training_budget_seconds=100.0, probe_budget_seconds=5.0,
    )
    for bad in ("train-0", "r16-fresh64-0"):
        kwargs = copy.deepcopy(fixture)
        kwargs["eval_index"][0]["id"] = bad
        with pytest.raises(BenchmarkError, match="leakage"):
            make_plan(**kwargs)
    kwargs = copy.deepcopy(fixture)
    kwargs["eval_index"][0]["answer"] = "secret"
    with pytest.raises(BenchmarkError, match="answer-free"):
        make_plan(**kwargs)
    kwargs = copy.deepcopy(fixture)
    kwargs["eval_index"][0]["prompt_sha256"] = hashlib.sha256(
        b"what is 0 plus one?").hexdigest()
    with pytest.raises(BenchmarkError, match="prompt leakage"):
        make_plan(**kwargs)
    kwargs = copy.deepcopy(fixture)
    kwargs["selected_rows"][0]["training_use"] = "EVALUATION_ONLY"
    with pytest.raises(BenchmarkError, match="train-only"):
        make_plan(**kwargs)


def test_fit_sweep_selects_actual_padded_shape_and_accounts_gpu_cpu():
    plan = _plan()
    selected = validate_fit_probes(plan, _probes(plan))
    for arm in ("mezo", "qlora"):
        assert selected[arm]["max_fit_config"]["micro_batch_size"] == 1
        assert selected[arm]["max_fit_config"]["actual_padded_sequence_length"] == 384
        assert selected[arm]["primary_training_config"]["actual_padded_sequence_length"] == 256


@pytest.mark.parametrize("mutate", [
    lambda p: p[0].update(eval_access=True),
    lambda p: p[0].update(actual_padded_sequence_length=512),
    lambda p: p[0].update(peak_cuda_allocated_bytes=999_999),
    lambda p: p[0].update(optimizer_step_completed=False),
    lambda p: p[0].update(elapsed_seconds=5.1),
    lambda p: p[1].update(fresh_process_id=p[0]["fresh_process_id"]),
    lambda p: p.pop(1),
])
def test_fit_rejects_probe_leakage_overtime_false_fit_or_missing_shapes(mutate):
    plan = _plan()
    probes = _probes(plan)
    mutate(probes)
    with pytest.raises(BenchmarkError):
        validate_fit_probes(plan, probes)


def test_equal_time_comparison_never_autopromotes_small_gain():
    plan, probes, receipts = _comparison_inputs()
    result = compare_arms(plan, receipts, probes)
    assert result["decision"] == "NO_MODEL_PROMOTION_INDEPENDENT_REPLICATION_REQUIRED"
    assert result["control_arms_verified"] == 8
    assert result["method_metrics"]["mezo"]["forward_sample_tokens"] == 12_800
    assert result["method_metrics"]["qlora"]["forward_sample_tokens"] == 6_400
    assert result["accuracy_delta_mezo_minus_qlora"] == 0


@pytest.mark.parametrize("key,value", [
    ("train_stop_monotonic_ns", 200_000_000_001),
    ("optimizer_window_seconds", 90.0),
    ("forward_sample_tokens", 6399),
    ("eval_accuracy", 0.9),
    ("max_prepared_example_tokens", 1025),
    ("external_process_wall_seconds", 1.0),
    ("selected_id_sha256", "b" * 64),
])
def test_arm_receipt_rejects_time_compute_eval_and_binding_mismatch(key, value):
    plan, probes, receipts = _comparison_inputs()
    bad = copy.deepcopy(receipts[2])  # MeZO correct arm
    bad[key] = value
    with pytest.raises(BenchmarkError):
        validate_arm_receipt(bad, plan)


def test_unscored_training_receipt_has_no_eval_access():
    plan, probes, receipts = _comparison_inputs()
    train_receipt = copy.deepcopy(receipts[2])
    train_receipt.update(status="TRAINED_UNSCORED", eval_accuracy=None,
                         eval_nll=None, eval_nll_sum=None,
                         eval_case_receipt_sha256=None, eval_correct=None, eval_examples=0,
                         eval_target_tokens=0)
    validate_arm_receipt(train_receipt, plan)
    with pytest.raises(BenchmarkError):
        validate_arm_receipt(train_receipt, plan, allow_unscored=False)


def test_comparison_requires_all_controls_and_equal_measured_time():
    plan, probes, receipts = _comparison_inputs()
    with pytest.raises(BenchmarkError, match="2 methods"):
        compare_arms(plan, receipts[:-1], probes)
    # Keep the single receipt otherwise internally consistent by updating ns,
    # stage and external wall; this isolates the *cross-arm* fairness gate.
    slow = copy.deepcopy(receipts)
    qlora_correct = slow[6]
    qlora_correct["train_stop_monotonic_ns"] = (
        qlora_correct["train_start_monotonic_ns"] + 86_000_000_000)
    qlora_correct["optimizer_window_seconds"] = 86.0
    qlora_correct["stage_seconds"]["train"] = 86.0
    qlora_correct["external_process_wall_seconds"] = 96.0
    with pytest.raises(BenchmarkError, match="equal optimizer-wall time"):
        compare_arms(plan, slow, probes)


def test_source_only_cli_preregisters_and_never_overwrites(tmp_path, capsys):
    inputs = dict(
        selected_rows=_rows(), admission_selected_id_sha256=selected_id_digest(_rows()),
        source_admission_status="ADMITTED_SOURCE_ONLY_NOT_TRAINED",
        eval_index=_eval_index(), r16_holdout_ids=[f"r16-fresh64-{i}" for i in range(64)],
        base_revision="1" * 40, base_binding_sha256=H,
        tokenizer_binding_sha256=J, corpus_manifest_sha256="c" * 64,
        eval_custody_sha256="d" * 64, eval_source_manifest_sha256="f" * 64,
        r16_eval_custody_sha256="e" * 64,
        seed=41, training_budget_seconds=100.0, probe_budget_seconds=5.0,
    )
    inputs_file = tmp_path / "admitted-source-index.json"
    plan_file = tmp_path / "r18-plan.json"
    inputs_file.write_text(json.dumps(inputs), encoding="utf-8")
    main(["preregister", "--inputs", str(inputs_file), "--output", str(plan_file)])
    assert json.loads(capsys.readouterr().out)["plan_sha256"] == validate_plan(_plan())
    main(["verify-plan", "--plan", str(plan_file)])
    assert json.loads(capsys.readouterr().out)["fresh_eval_count"] == 64
    with pytest.raises(BenchmarkError, match="must be new"):
        main(["preregister", "--inputs", str(inputs_file), "--output", str(plan_file)])


def test_preregister_real_labyrinth_replayed_train_only_row_shape():
    rows = [{"id": row["id"], "family": "verified_integer_chains",
             "prompt": row["prompt"], "answer": row["answer"]}
            for row in _rows()]
    plan = make_plan(
        selected_rows=rows, admission_selected_id_sha256=selected_id_digest(rows),
        source_admission_status="ADMITTED_LABYRINTH_TRAIN_ONLY_NO_WEIGHT_UPDATE",
        eval_index=_eval_index(),
        r16_holdout_ids=[f"r16-fresh64-{i}" for i in range(64)],
        base_revision="1" * 40, base_binding_sha256=H,
        tokenizer_binding_sha256=J, corpus_manifest_sha256="c" * 64,
        eval_custody_sha256="d" * 64, eval_source_manifest_sha256="f" * 64,
        r16_eval_custody_sha256="e" * 64,
        seed=41, training_budget_seconds=100.0, probe_budget_seconds=5.0,
    )
    assert plan["source_admission_status"] == "ADMITTED_LABYRINTH_TRAIN_ONLY_NO_WEIGHT_UPDATE"
    assert validate_plan(plan)


def _custodian_fixture(tmp_path, monkeypatch):
    """Test custody control flow at its real filesystem seam, no heldout model."""
    import trainingday.admission as admission
    import trainingday.labyrinth_intake as intake
    import trainingday.train_neural as neural

    from trainingday.r18_benchmark import canonical_sha256

    old16 = [f"reasoning-{'a' * 60}{i:04x}" for i in range(16)]
    r16_cases = [{"id": f"r16-{i}", "prompt": f"r16 question {i}",
                  "expected": str(i)} for i in range(64)]
    r18_cases = generate_novel_r18_cases(private_nonce_hex=TEST_PRIVATE_NONCE)
    monkeypatch.setattr("trainingday.r18_benchmark.secrets.token_hex",
                        lambda size: TEST_PRIVATE_NONCE if size == 32 else None)
    r16_hash = hashlib.sha256("\n".join(c["id"] for c in r16_cases).encode()).hexdigest()
    train = [{"id": f"train-{i}", "family": "verified_integer_chains",
              "prompt": f"train question {i}", "answer": str(i)} for i in range(1024)]
    old_path = tmp_path / "old16-private.json"
    old_path.write_text(json.dumps({"previously_scored_16_ids": old16}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    binding = tmp_path / "base-binding.json"
    binding.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(admission, "_hash_file", lambda p: neural.R16_V2_MANIFEST_SHA256
                        if p == manifest else "c" * 64)
    monkeypatch.setattr(intake, "admit_labyrinth", lambda *a, **k: (
        {"selected_count": 1024, "status": "ADMITTED_LABYRINTH_TRAIN_ONLY_NO_WEIGHT_UPDATE",
         "selected_id_sha256": selected_id_digest(train)}, train))
    monkeypatch.setattr(neural, "_required_base", lambda binding: (
        {"model_id": "Qwen/Qwen3.5-4B", "revision": "1" * 40,
         "files": {"tokenizer_config.json": "b" * 64}}, tmp_path))
    structural = types.ModuleType("trainingday.structural_eval")
    structural.R15_V2_MANIFEST_SHA256 = neural.R16_V2_MANIFEST_SHA256
    structural.R15_SCORED_SIGNATURE_COUNT = 13
    structural.preregister_r15_fresh_structural64 = lambda manifest, **k: {
        "selected_count": 64, "selected_id_sha256": r16_hash, "signature_count": 64,
        "cases": r16_cases}

    def fresh64(*args, **kwargs):
        raise AssertionError("v2 eval exhausted; must NEVER reselect a v2 structural bank")

    structural.select_structural_cases = fresh64
    monkeypatch.setitem(sys.modules, "trainingday.structural_eval", structural)
    evaluator = types.ModuleType("trainingday.eval_structural_r16")
    evaluator.BANK_ID_SHA256 = r16_hash
    evaluator._old_ids = lambda path: json.loads(
        path.read_text(encoding="utf-8"))["previously_scored_16_ids"]
    monkeypatch.setitem(sys.modules, "trainingday.eval_structural_r16", evaluator)
    kwargs = dict(manifest=manifest, old_r15_receipt=old_path,
                  base_binding=binding, producer_git_head=neural.R16_PRODUCER_HEAD,
                  work_root=tmp_path, private_output_dir=tmp_path / "private",
                  plan_output=tmp_path / "r18-plan.json",
                  expected_r16_selected_id_sha256=r16_hash,
                  cases=1024, seed=41, training_budget_seconds=100.0,
                  probe_budget_seconds=5.0)
    return kwargs, r18_cases


def test_cpu_custodian_seals_fresh64_excluding_all_prior80_and_public_no_answers(
        tmp_path, monkeypatch):
    kwargs, r18_cases = _custodian_fixture(tmp_path, monkeypatch)
    public = preregister_from_corpus(**kwargs)
    assert public["selected_train_count"] == 1024
    assert public["r18_eval_count"] == 64
    plan = json.loads(kwargs["plan_output"].read_text(encoding="utf-8"))
    assert public["plan_sha256"] == validate_plan(plan)
    assert plan["eval_bank_source"] == EVAL_SOURCE
    assert "Start with" not in kwargs["plan_output"].read_text(encoding="utf-8")
    assert TEST_PRIVATE_NONCE not in kwargs["plan_output"].read_text(encoding="utf-8")
    assert TEST_PRIVATE_NONCE not in json.dumps(public)
    private = kwargs["private_output_dir"] / "r18-evaluator-sealed-private.json"
    payload = json.loads(private.read_text(encoding="utf-8"))
    assert payload["cases"] == r18_cases
    public_source = kwargs["plan_output"].parent / "r18-external-chain6-source-manifest.json"
    source_metadata = json.loads(public_source.read_text(encoding="utf-8"))
    assert plan["eval_source_manifest_sha256"] == canonical_sha256(source_metadata)
    assert source_metadata["selected_unique_signature_count"] == 64
    assert source_metadata["protocol"] == "R18_EXTERNAL_LONG_CHAIN64_OOD_V1"
    assert source_metadata["private_nonce_sha256"] == hashlib.sha256(
        bytes.fromhex(TEST_PRIVATE_NONCE)).hexdigest()
    assert TEST_PRIVATE_NONCE not in public_source.read_text(encoding="utf-8")
    assert "Start with" not in public_source.read_text(encoding="utf-8")
    assert payload["private_nonce_hex"] == TEST_PRIVATE_NONCE
    assert payload["eval_custody_sha256"] == canonical_sha256({
        "source_manifest_sha256": plan["eval_source_manifest_sha256"], "cases": r18_cases})
    if os.name == "nt":
        owner_sid = _windows_owner_sid()
        _windows_verify_owner_only(private, owner_sid, directory=False)
        _windows_verify_owner_only(kwargs["private_output_dir"], owner_sid, directory=True)
    else:
        assert stat.S_IMODE(private.stat().st_mode) == 0o600
        assert stat.S_IMODE(kwargs["private_output_dir"].stat().st_mode) == 0o700


def test_cpu_custodian_wrong_historical_r16_pin_makes_no_artifacts(tmp_path, monkeypatch):
    kwargs, _ = _custodian_fixture(tmp_path, monkeypatch)
    kwargs["expected_r16_selected_id_sha256"] = "f" * 64
    with pytest.raises(BenchmarkError, match="user-supplied R16 selection pin"):
        preregister_from_corpus(**kwargs)
    assert not kwargs["private_output_dir"].exists()
    assert not kwargs["plan_output"].exists()


def test_windows_private_acl_owner_only_and_no_broad_inheritance(tmp_path, monkeypatch):
    """Simulate Windows command receipts without treating NTFS st_mode as ACL."""
    owner_sid = "S-1-5-21-286630441-271553882-1369355300-1001"
    recorded = []

    def windows_process(argv, **kwargs):
        assert kwargs["capture_output"] and kwargs["text"]
        assert kwargs["check"] is False
        recorded.append(argv)
        if argv[0] == "whoami":
            return subprocess.CompletedProcess(argv, 0, f'"lappy\\patri","{owner_sid}"\n', "")
        if argv[0] == "icacls":
            return subprocess.CompletedProcess(
                argv, 0, "Successfully processed 1 files; Failed processing 0 files", "")
        assert argv[0] == "powershell.exe"
        script = base64.b64decode(argv[-1]).decode("utf-16le")
        assert all(sid in script for sid in (
            owner_sid, "S-1-5-18", "S-1-5-32-544", "S-1-3-4"))
        assert "$acl.AreAccessRulesProtected" in script
        assert "$allowed -notcontains $principal" in script
        assert "$observed.ContainsKey($principal)" in script
        assert "$r.IsInherited" in script
        return subprocess.CompletedProcess(argv, 0, "R18_PRIVATE_ACL_PRIVILEGED_ONLY\n", "")

    monkeypatch.setattr("trainingday.r18_benchmark.subprocess.run", windows_process)
    path = tmp_path / "new-private-dir"
    path.mkdir()
    _protect_private_path(path, directory=True, is_windows=True)
    assert [a[0] for a in recorded] == ["whoami", "icacls", "icacls", "icacls", "icacls",
                                           "powershell.exe"]
    assert recorded[1][2:] == ["/inheritance:r"]
    assert recorded[2][2:] == ["/grant:r", f"*{owner_sid}:(OI)(CI)F"]
    assert recorded[3][2:] == ["/setowner", f"*{owner_sid}"]
    assert recorded[4][2:] == ["/verify"]
    assert all(isinstance(argv, list) for argv in recorded)


def test_windows_private_acl_fails_closed_on_broad_access(tmp_path, monkeypatch):
    def windows_process(argv, **kwargs):
        if argv[0] == "whoami":
            return subprocess.CompletedProcess(argv, 0, '"lappy\\patri","S-1-5-21-1-2-3-1001"', "")
        if argv[0] == "powershell.exe":
            return subprocess.CompletedProcess(argv, 21, "", "broad Authenticated Users ACE")
        return subprocess.CompletedProcess(argv, 0, "Successfully processed 1 files", "")

    monkeypatch.setattr("trainingday.r18_benchmark.subprocess.run", windows_process)
    private = tmp_path / "private"
    private.mkdir()
    with pytest.raises(BenchmarkError, match="inherited/broad"):
        _protect_private_path(private, directory=True, is_windows=True)


def test_custodian_guards_private_dir_before_writing_eval_labels(tmp_path, monkeypatch):
    kwargs, _ = _custodian_fixture(tmp_path, monkeypatch)
    from trainingday import r18_benchmark as protocol

    actual = protocol._protect_private_path
    observed = []

    def monitor(path, *, directory, **k):
        if directory:
            assert not list(path.iterdir())  # No sensitive file exists yet.
        else:
            assert observed and observed[0] == "directory-protected"
        actual(path, directory=directory, **k)
        observed.append("directory-protected" if directory else path.name)

    monkeypatch.setattr(protocol, "_protect_private_path", monitor)
    preregister_from_corpus(**kwargs)
    assert observed[0] == "directory-protected"
    assert "r18-evaluator-sealed-private.json" in observed
