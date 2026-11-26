import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import successor.v3_training as v3
from successor.v3_training import preflight_v3_training, validate_v3_training_config


HEAD = "c" * 40
TASK9 = "a" * 40
VERIFIER = "b" * 40
SHA_PARENT = "1" * 64
BASE_TREE = "6" * 64
BASE_INVENTORY = "7" * 64
USER_INSTRUCTION_SHA = "9" * 64


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bytes_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def base_manifest():
    return {
        "repo_id": "HuggingFaceTB/SmolLM3-3B",
        "revision": "a07cc9a04f16550a088caea529712d1d335b0ac1",
        "mutable_revision_allowed": False,
        "observed_local_cache_tree_sha256": BASE_TREE,
        "file_inventory_sha256": BASE_INVENTORY,
    }


def make_subject(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "successor").mkdir(parents=True)
    write_json(repo / "successor" / "base_model_manifest.json", base_manifest())

    private = tmp_path / "private"
    parent = private / "parent"
    parent.mkdir(parents=True)
    (parent / "adapter_model.safetensors").write_bytes(b"parent")
    write_json(parent / "adapter_config.json", {"peft_type": "LORA", "r": 8})
    parent_sha = file_sha(parent / "adapter_model.safetensors")
    parent_config_sha = file_sha(parent / "adapter_config.json")
    parent_subject = v3._sha256_json({
        "adapter_config_sha256": parent_config_sha,
        "adapter_sha256": parent_sha,
        "base_revision": base_manifest()["revision"],
        "base_tree_sha256": BASE_TREE,
        "base_inventory_sha256": BASE_INVENTORY,
    })

    train = private / "train.jsonl"
    validation = private / "validation.jsonl"
    train.write_text('{"prompt":"t","response":"a"}\n', encoding="utf-8")
    validation.write_text('{"prompt":"v","response":"b"}\n', encoding="utf-8")

    corpus = private / "corpus_manifest.json"
    corpus_payload = {
        "readiness_source_commit": TASK9,
        "train_file_sha256": file_sha(train),
        "validation_file_sha256": file_sha(validation),
        "general_fraction": 0.5,
        "train_validation_overlap": 0,
        "blind_plaintext_included": False,
        "weight_change_performed": False,
    }
    write_json(corpus, corpus_payload)

    external = private / "external"
    source_review = external / "source.json"
    behavior_review = external / "behavior.json"
    radical_registration = external / "radical.json"
    pragmatic_registration = external / "pragmatic.json"
    for path, payload in (
        (source_review, {"kind": "SOURCE"}),
        (behavior_review, {"kind": "BEHAVIOR"}),
        (radical_registration, {"role": "RADICAL_HOSTILE"}),
        (pragmatic_registration, {"role": "PRAGMATIC_HOSTILE"}),
    ):
        write_json(path, payload)

    ready = private / "training_ready_receipt.json"
    write_json(ready, {
        "schema": v3.TASK9_READY_SCHEMA,
        "ready": True,
        "status": "READY",
        "readiness_source_commit": TASK9,
        "subject_digest": "3" * 64,
        "evidence_digest": "4" * 64,
        "external_verification_digest": "5" * 64,
        "external_verifier_commit": VERIFIER,
    })

    bus = tmp_path / "bus"
    bus.mkdir()
    subprocess.check_call(["git", "-C", str(bus), "init", "-q"])
    subprocess.check_call(["git", "-C", str(bus), "config", "user.email", "test@example.invalid"])
    subprocess.check_call(["git", "-C", str(bus), "config", "user.name", "Task10 Test"])

    spec = {
        "schema": v3.V3_TRAINING_SCHEMA,
        "run_id": "VERA_SUCCESSOR_V3_DEV_TEST",
        "task9_source_commit": TASK9,
        "task9_ready_receipt_path": str(ready),
        "task9_ready_receipt_sha256": file_sha(ready),
        "task9_repo_root": str(repo),
        "task9_readiness_dir": str(private),
        "task9_blind_dir": str(private / "blind"),
        "task9_training_lane_key": "training-test",
        "task9_external_verifier_source_commit": VERIFIER,
        "task9_source_review_receipt_path": str(source_review),
        "task9_source_review_receipt_sha256": file_sha(source_review),
        "task9_behavior_review_receipt_path": str(behavior_review),
        "task9_behavior_review_receipt_sha256": file_sha(behavior_review),
        "task9_radical_registration_receipt_path": str(radical_registration),
        "task9_radical_registration_receipt_sha256": file_sha(radical_registration),
        "task9_pragmatic_registration_receipt_path": str(pragmatic_registration),
        "task9_pragmatic_registration_receipt_sha256": file_sha(pragmatic_registration),
        "training_authorization_receipt_path": str(private / "training_authorization_receipt.json"),
        "training_authority_bus_repo_path": str(bus),
        "training_authority_bus_branch_ref": v3._CANONICAL_AUTHORITY_BUS_BRANCH_REF,
        "base_repo_id": base_manifest()["repo_id"],
        "base_revision": base_manifest()["revision"],
        "base_tree_sha256": BASE_TREE,
        "base_inventory_sha256": BASE_INVENTORY,
        "parent_adapter_path": str(parent),
        "parent_adapter_sha256": parent_sha,
        "parent_adapter_config_sha256": parent_config_sha,
        "parent_candidate_subject_digest": parent_subject,
        "parent_selection_basis": "development parent only",
        "train_path": str(train),
        "train_sha256": file_sha(train),
        "validation_path": str(validation),
        "validation_sha256": file_sha(validation),
        "corpus_manifest_path": str(corpus),
        "corpus_manifest_sha256": file_sha(corpus),
        "output_dir": str(private / "output"),
        "seed": 20260919,
        "learning_rate": 5e-6,
        "gradient_accumulation": 8,
        "assistant_only_loss": True,
        "neutral_system_override": True,
        "general_rehearsal_fraction": 0.5,
    }
    spec_path = repo / "successor" / "v3_training_config.json"
    write_json(spec_path, spec)
    return repo, private, spec_path, spec


def fake_git(monkeypatch, *, trust_external_bus=True):
    monkeypatch.setattr(v3, "_git_head", lambda repo: HEAD)
    monkeypatch.setattr(v3, "_is_ancestor", lambda repo, ancestor, descendant: True)
    monkeypatch.setattr(
        v3,
        "_read_committed_v3_spec",
        lambda repo, commit: json.loads(
            (Path(repo) / v3._CANONICAL_V3_SPEC_REL).read_text(encoding="utf-8")
        ),
    )
    monkeypatch.setattr(
        v3,
        "_verify_task9_external",
        lambda spec, receipt, reasons: receipt["external_verification_digest"],
    )
    if trust_external_bus:
        monkeypatch.setattr(v3, "_authority_bus_remote_is_canonical", lambda repo: True)
        monkeypatch.setattr(v3, "_refresh_authority_bus", lambda repo: None)


def authorize(private: Path, spec: dict):
    ready_path = Path(spec["task9_ready_receipt_path"])
    ready_sha = file_sha(ready_path)
    bus = Path(spec["training_authority_bus_repo_path"])
    bus_path = "messages/task10-authority.md"
    target = bus / bus_path
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Task 10 external Patrick authority test receipt",
        f"AUTHORITY_FIELD authority_kind=PATRICK_EXPLICIT",
        f"AUTHORITY_FIELD effect=WEIGHT_CHANGING_TRAINING",
        f"AUTHORITY_FIELD training_code_commit={HEAD}",
        f"AUTHORITY_FIELD run_id={spec['run_id']}",
        f"AUTHORITY_FIELD task9_source_commit={spec['task9_source_commit']}",
        f"AUTHORITY_FIELD task9_ready_receipt_sha256={ready_sha}",
        f"AUTHORITY_FIELD parent_adapter_sha256={spec['parent_adapter_sha256']}",
        f"AUTHORITY_FIELD parent_adapter_config_sha256={spec['parent_adapter_config_sha256']}",
        f"AUTHORITY_FIELD parent_candidate_subject_digest={spec['parent_candidate_subject_digest']}",
        f"AUTHORITY_FIELD corpus_manifest_sha256={spec['corpus_manifest_sha256']}",
        f"AUTHORITY_FIELD train_sha256={spec['train_sha256']}",
        f"AUTHORITY_FIELD validation_sha256={spec['validation_sha256']}",
        f"AUTHORITY_FIELD user_instruction_sha256={USER_INSTRUCTION_SHA}",
        "AUTHORITY_FIELD resource_execution=LOCAL_ONLY",
        "AUTHORITY_FIELD paid_compute=NOT_AUTHORIZED",
        "AUTHORITY_FIELD replay_policy=EXACT_SUBJECT_NO_OUTPUT_NAMESPACE_REUSE",
        "AUTHORITY_FIELD excluded_effects=MERGE|DEPLOY|RUNTIME_ACTIVATION|SD1_INSTALL|PROVIDER_CREDENTIAL_MUTATION|CANDIDATE_PROMOTION",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.check_call(["git", "-C", str(bus), "add", bus_path])
    subprocess.check_call(
        ["git", "-C", str(bus), "commit", "-q", "-m", "test authority"],
    )
    bus_commit = subprocess.check_output(
        ["git", "-C", str(bus), "rev-parse", "HEAD"], text=True
    ).strip()
    bus_payload = subprocess.check_output(
        ["git", "-C", str(bus), "show", f"{bus_commit}:{bus_path}"]
    )

    write_json(private / "training_authorization_receipt.json", {
        "schema": v3.TRAINING_AUTH_SCHEMA,
        "authorized": True,
        "effect": "WEIGHT_CHANGING_TRAINING",
        "authority_kind": "PATRICK_EXPLICIT",
        "authority_ref": f"bus:{bus_commit}:{bus_path}",
        "authority_bus_commit": bus_commit,
        "authority_bus_path": bus_path,
        "authority_bus_file_sha256": bytes_sha(bus_payload),
        "user_instruction_sha256": USER_INSTRUCTION_SHA,
        "training_code_commit": HEAD,
        "task9_ready_receipt_sha256": ready_sha,
        "task9_source_commit": spec["task9_source_commit"],
        "run_id": spec["run_id"],
        "parent_adapter_sha256": spec["parent_adapter_sha256"],
        "parent_adapter_config_sha256": spec["parent_adapter_config_sha256"],
        "parent_candidate_subject_digest": spec["parent_candidate_subject_digest"],
        "corpus_manifest_sha256": spec["corpus_manifest_sha256"],
        "train_sha256": spec["train_sha256"],
        "validation_sha256": spec["validation_sha256"],
        "resource_execution": "LOCAL_ONLY",
        "paid_compute": "NOT_AUTHORIZED",
        "replay_policy": "EXACT_SUBJECT_NO_OUTPUT_NAMESPACE_REUSE",
        "excluded_effects": "MERGE|DEPLOY|RUNTIME_ACTIVATION|SD1_INSTALL|PROVIDER_CREDENTIAL_MUTATION|CANDIDATE_PROMOTION",
    })


def test_validate_v3_config_enforces_initial_general_rehearsal(tmp_path):
    _repo, _private, _spec_path, spec = make_subject(tmp_path)
    spec["general_rehearsal_fraction"] = 0.49
    with pytest.raises(ValueError, match="general_rehearsal_fraction"):
        validate_v3_training_config(spec, base_manifest())


def test_static_subject_holds_only_for_missing_training_authority(tmp_path, monkeypatch):
    repo, private, spec_path, spec = make_subject(tmp_path)
    fake_git(monkeypatch)
    decision = preflight_v3_training(spec_path, repo_root=repo)
    assert decision.runnable is False
    assert decision.status == "HOLD"
    assert decision.reasons == ("training_authorization_receipt_missing",)
    assert decision.parent_adapter_sha256 == spec["parent_adapter_sha256"]
    assert decision.train_sha256 == spec["train_sha256"]
    assert decision.validation_sha256 == spec["validation_sha256"]


def test_missing_task9_and_training_authority_are_hold_not_runnable(tmp_path, monkeypatch):
    repo, private, spec_path, spec = make_subject(tmp_path)
    Path(spec["task9_ready_receipt_path"]).unlink()
    fake_git(monkeypatch)
    decision = preflight_v3_training(spec_path, repo_root=repo)
    assert decision.status == "HOLD"
    assert decision.reasons == (
        "task9_ready_receipt_missing",
        "training_authorization_receipt_missing",
    )


def test_ready_plus_exact_authorization_is_runnable(tmp_path, monkeypatch):
    repo, private, spec_path, spec = make_subject(tmp_path)
    authorize(private, spec)
    fake_git(monkeypatch)
    decision = preflight_v3_training(spec_path, repo_root=repo)
    assert decision.runnable is True
    assert decision.status == "RUNNABLE"
    assert decision.reasons == ()
    assert len(decision.task9_ready_receipt_sha256) == 64
    assert len(decision.training_authorization_receipt_sha256) == 64


def test_tampered_parent_or_dataset_blocks(tmp_path, monkeypatch):
    repo, private, spec_path, spec = make_subject(tmp_path)
    authorize(private, spec)
    (Path(spec["parent_adapter_path"]) / "adapter_model.safetensors").write_bytes(b"tamper")
    Path(spec["train_path"]).write_text('{"prompt":"tamper","response":"x"}\n', encoding="utf-8")
    fake_git(monkeypatch)
    decision = preflight_v3_training(spec_path, repo_root=repo)
    assert decision.status == "BLOCKED"
    assert "parent_adapter_sha256_mismatch" in decision.reasons
    assert "train_sha256_mismatch" in decision.reasons
    assert "corpus_train_file_binding_mismatch" in decision.reasons


def test_corpus_cannot_smuggle_blind_plaintext_or_overlap(tmp_path, monkeypatch):
    repo, private, spec_path, spec = make_subject(tmp_path)
    authorize(private, spec)
    corpus_path = Path(spec["corpus_manifest_path"])
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["train_validation_overlap"] = 1
    corpus["blind_plaintext_included"] = True
    write_json(corpus_path, corpus)
    spec["corpus_manifest_sha256"] = file_sha(corpus_path)
    write_json(spec_path, spec)
    fake_git(monkeypatch)
    decision = preflight_v3_training(spec_path, repo_root=repo)
    assert "corpus_train_validation_overlap_nonzero" in decision.reasons
    assert "blind_plaintext_included_in_training_corpus" in decision.reasons


def test_stale_task9_source_blocks_even_with_receipts(tmp_path, monkeypatch):
    repo, private, spec_path, spec = make_subject(tmp_path)
    authorize(private, spec)
    monkeypatch.setattr(v3, "_git_head", lambda repo: HEAD)
    monkeypatch.setattr(v3, "_is_ancestor", lambda repo, ancestor, descendant: False)
    decision = preflight_v3_training(spec_path, repo_root=repo)
    assert "task9_source_not_ancestor_of_training_code" in decision.reasons


def test_authorization_is_bound_to_exact_run_material(tmp_path, monkeypatch):
    repo, private, spec_path, spec = make_subject(tmp_path)
    authorize(private, spec)
    auth_path = Path(spec["training_authorization_receipt_path"])
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    auth["parent_adapter_sha256"] = "f" * 64
    write_json(auth_path, auth)
    fake_git(monkeypatch)
    decision = preflight_v3_training(spec_path, repo_root=repo)
    assert "training_authorization_parent_mismatch" in decision.reasons


def test_committed_task10_config_is_structurally_valid():
    repo = Path(__file__).resolve().parents[1]
    spec = json.loads(
        (repo / "successor" / "v3_training_config.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (repo / "successor" / "base_model_manifest.json").read_text(encoding="utf-8")
    )
    validate_v3_training_config(spec, manifest)
    assert spec["general_rehearsal_fraction"] >= 0.50
    assert spec["assistant_only_loss"] is True
    assert spec["task9_source_commit"] == "065035bb75d73467b8f9beb5480f5dd56bc0697d"
    assert spec["task9_ready_receipt_sha256"] == "a67b01897ad03c56be4d6defa5686807615f5ce9095b6df165d3069003f7d9bf"
    assert spec["corpus_manifest_sha256"] == "838a166bd968976d12bca474adb105e70660a56cd1e29dacbce0ce58262f4de1"
    assert spec["task9_source_review_receipt_sha256"] == "d2332b9a0739068e7fb992fba257bd7c807da9fa9bbe602898f1980c5364b8fd"
    assert spec["task9_behavior_review_receipt_sha256"] == "bf4171865a8d894023862913e3b8922ad5bc4967f2b9bc34fb5fc9b20afe9b71"
    assert spec["task9_radical_registration_receipt_sha256"] == "4a7c90c8075e46c725e1a29b14ec83fecdfa7d87458720f870037f517aba5f2c"
    assert spec["task9_pragmatic_registration_receipt_sha256"] == "4e6631f343f443cdec9c928430be7b3a97280f60103d71f047d6ade818aea691"
    assert spec["parent_adapter_sha256"] == "94c6af3b16fd8e21305aa69e095e480721d1e31b33a2a40e1e504d59c73cfa7a"


def test_v3_dry_run_returns_preflight_without_legacy_prepare(tmp_path, monkeypatch):
    import successor.train_pilot_sft as runner

    spec_path = tmp_path / "spec.json"
    write_json(spec_path, {"schema": v3.V3_TRAINING_SCHEMA, "run_id": "VERA_SUCCESSOR_V3_DEV_X"})
    decision = v3.V3TrainingPreflight(
        runnable=False,
        status="HOLD",
        code_commit=HEAD,
        parent_adapter_sha256=None,
        parent_adapter_config_sha256=None,
        parent_candidate_subject_digest=None,
        base_tree_sha256=None,
        base_inventory_sha256=None,
        train_sha256=None,
        validation_sha256=None,
        corpus_manifest_sha256=None,
        task9_ready_receipt_sha256=None,
        training_authorization_receipt_sha256=None,
        reasons=("task9_ready_receipt_missing",),
    )
    monkeypatch.setattr(runner, "preflight_v3_training", lambda *args, **kwargs: decision)
    monkeypatch.setattr(
        runner,
        "prepare",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy prepare called")),
    )
    result = runner.dry_run(spec_path)
    assert result["run_id"] == "VERA_SUCCESSOR_V3_DEV_X"
    assert result["runnable"] is False
    assert result["status"] == "HOLD"
    assert result["reasons"] == ["task9_ready_receipt_missing"]


def test_v3_prepare_blocks_before_private_file_loading(tmp_path, monkeypatch):
    import successor.train_pilot_sft as runner

    spec_path = tmp_path / "spec.json"
    write_json(spec_path, {"schema": v3.V3_TRAINING_SCHEMA})
    decision = v3.V3TrainingPreflight(
        runnable=False,
        status="HOLD",
        code_commit=HEAD,
        parent_adapter_sha256=None,
        parent_adapter_config_sha256=None,
        parent_candidate_subject_digest=None,
        base_tree_sha256=None,
        base_inventory_sha256=None,
        train_sha256=None,
        validation_sha256=None,
        corpus_manifest_sha256=None,
        task9_ready_receipt_sha256=None,
        training_authorization_receipt_sha256=None,
        reasons=("training_authorization_receipt_missing",),
    )
    monkeypatch.setattr(runner, "preflight_v3_training", lambda *args, **kwargs: decision)
    with pytest.raises(RuntimeError, match="V3 training preflight blocked"):
        runner.prepare(spec_path)


def test_parent_candidate_subject_is_recomputed_from_config_weights_and_base(tmp_path, monkeypatch):
    repo, private, spec_path, spec = make_subject(tmp_path)
    authorize(private, spec)
    fake_git(monkeypatch)

    clean = preflight_v3_training(spec_path, repo_root=repo)
    assert clean.runnable is True
    assert clean.parent_adapter_config_sha256 == spec["parent_adapter_config_sha256"]
    assert clean.parent_candidate_subject_digest == spec["parent_candidate_subject_digest"]

    config_path = Path(spec["parent_adapter_path"]) / "adapter_config.json"
    write_json(config_path, {"peft_type": "LORA", "r": 16})
    tampered_config_sha = file_sha(config_path)
    spec["parent_adapter_config_sha256"] = tampered_config_sha
    # Deliberately do not update the bound candidate subject.
    write_json(spec_path, spec)
    authorize(private, spec)

    changed = preflight_v3_training(spec_path, repo_root=repo)
    assert changed.runnable is False
    assert "parent_candidate_subject_digest_mismatch" in changed.reasons


def test_training_authorization_requires_patrick_explicit_authority_and_full_subject(tmp_path, monkeypatch):
    repo, private, spec_path, spec = make_subject(tmp_path)
    authorize(private, spec)
    auth_path = Path(spec["training_authorization_receipt_path"])
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    auth["authority_kind"] = "AUTOMATED"
    auth["authority_ref"] = ""
    auth["parent_candidate_subject_digest"] = "f" * 64
    auth["corpus_manifest_sha256"] = "e" * 64
    write_json(auth_path, auth)
    fake_git(monkeypatch)

    decision = preflight_v3_training(spec_path, repo_root=repo)
    assert decision.runnable is False
    assert "training_authority_not_explicit" in decision.reasons
    assert "training_authority_ref_missing" in decision.reasons
    assert "training_authorization_parent_subject_mismatch" in decision.reasons
    assert "training_authorization_corpus_mismatch" in decision.reasons


def test_fabricated_local_authority_without_external_bus_receipt_is_blocked(
    tmp_path, monkeypatch
):
    repo, private, spec_path, spec = make_subject(tmp_path)
    ready_path = Path(spec["task9_ready_receipt_path"])
    write_json(private / "training_authorization_receipt.json", {
        "schema": v3.TRAINING_AUTH_SCHEMA,
        "authorized": True,
        "effect": "WEIGHT_CHANGING_TRAINING",
        "authority_kind": "PATRICK_EXPLICIT",
        "authority_ref": "fabricated-local-string",
        "task9_ready_receipt_sha256": file_sha(ready_path),
        "task9_source_commit": spec["task9_source_commit"],
        "run_id": spec["run_id"],
        "parent_adapter_sha256": spec["parent_adapter_sha256"],
        "parent_adapter_config_sha256": spec["parent_adapter_config_sha256"],
        "parent_candidate_subject_digest": spec["parent_candidate_subject_digest"],
        "corpus_manifest_sha256": spec["corpus_manifest_sha256"],
        "train_sha256": spec["train_sha256"],
        "validation_sha256": spec["validation_sha256"],
    })
    fake_git(monkeypatch)

    decision = preflight_v3_training(spec_path, repo_root=repo)
    assert decision.runnable is False
    assert decision.status == "BLOCKED"
    assert "training_authorization_code_commit_mismatch" in decision.reasons
    assert "training_authority_bus_commit_invalid" in decision.reasons


def test_task9_ready_receipt_bytes_are_source_pinned(tmp_path, monkeypatch):
    repo, private, spec_path, spec = make_subject(tmp_path)
    authorize(private, spec)
    ready_path = Path(spec["task9_ready_receipt_path"])
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    ready["subject_digest"] = "f" * 64
    write_json(ready_path, ready)
    fake_git(monkeypatch)

    decision = preflight_v3_training(spec_path, repo_root=repo)
    assert decision.runnable is False
    assert "task9_ready_receipt_sha256_mismatch" in decision.reasons
    assert "training_authorization_task9_receipt_mismatch" in decision.reasons


def test_existing_output_directory_blocks_rerun(tmp_path, monkeypatch):
    repo, private, spec_path, spec = make_subject(tmp_path)
    authorize(private, spec)
    Path(spec["output_dir"]).mkdir(parents=True)
    fake_git(monkeypatch)

    decision = preflight_v3_training(spec_path, repo_root=repo)
    assert decision.runnable is False
    assert decision.status == "BLOCKED"
    assert "output_dir_already_exists" in decision.reasons


def test_bus_authority_must_bind_final_code_head(tmp_path, monkeypatch):
    repo, private, spec_path, spec = make_subject(tmp_path)
    authorize(private, spec)
    auth_path = Path(spec["training_authorization_receipt_path"])
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    auth["training_code_commit"] = "d" * 40
    write_json(auth_path, auth)
    fake_git(monkeypatch)

    decision = preflight_v3_training(spec_path, repo_root=repo)
    assert decision.runnable is False
    assert "training_authorization_code_commit_mismatch" in decision.reasons


def test_caller_created_local_bus_cannot_mint_patrick_authority(tmp_path, monkeypatch):
    repo, private, spec_path, spec = make_subject(tmp_path)
    authorize(private, spec)
    fake_git(monkeypatch, trust_external_bus=False)

    decision = preflight_v3_training(spec_path, repo_root=repo)

    assert decision.runnable is False
    assert decision.status == "BLOCKED"
    assert "training_authority_bus_not_canonical" in decision.reasons


def test_alternate_v3_spec_path_cannot_repin_authority_root(tmp_path, monkeypatch):
    repo, private, spec_path, spec = make_subject(tmp_path)
    authorize(private, spec)
    fake_git(monkeypatch)
    alternate = tmp_path / "attacker-v3-spec.json"
    alternate.write_text(spec_path.read_text(encoding="utf-8"), encoding="utf-8")

    decision = preflight_v3_training(alternate, repo_root=repo)

    assert decision.runnable is False
    assert decision.status == "BLOCKED"
    assert "training_spec_not_canonical" in decision.reasons


def test_dirty_canonical_spec_cannot_repin_authority_root(tmp_path, monkeypatch):
    repo, private, spec_path, spec = make_subject(tmp_path)
    authorize(private, spec)
    committed = json.loads(spec_path.read_text(encoding="utf-8"))
    fake_git(monkeypatch)
    monkeypatch.setattr(v3, "_read_committed_v3_spec", lambda repo, commit: committed)

    changed = dict(committed)
    changed["run_id"] = "VERA_SUCCESSOR_V3_DEV_ATTACKER"
    write_json(spec_path, changed)

    decision = preflight_v3_training(spec_path, repo_root=repo)

    assert decision.runnable is False
    assert decision.status == "BLOCKED"
    assert "training_spec_not_committed" in decision.reasons


def test_training_authority_scope_fields_fail_closed(tmp_path, monkeypatch):
    repo, private, spec_path, spec = make_subject(tmp_path)
    authorize(private, spec)
    auth_path = Path(spec["training_authorization_receipt_path"])
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    auth["resource_execution"] = "HOSTED"
    auth["paid_compute"] = "AUTHORIZED"
    auth["replay_policy"] = "REUSABLE"
    auth["excluded_effects"] = ""
    write_json(auth_path, auth)
    fake_git(monkeypatch)

    decision = preflight_v3_training(spec_path, repo_root=repo)

    assert decision.runnable is False
    assert decision.status == "BLOCKED"
    assert "training_authorization_resource_scope_mismatch" in decision.reasons
    assert "training_authorization_paid_compute_mismatch" in decision.reasons
    assert "training_authorization_replay_policy_mismatch" in decision.reasons
    assert "training_authorization_excluded_effects_mismatch" in decision.reasons
