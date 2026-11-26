import pytest

from successor.pilot_sft import build_run_manifest, validate_run_spec


BASE = {
    "repo_id": "HuggingFaceTB/SmolLM3-3B",
    "revision": "a07cc9a04f16550a088caea529712d1d335b0ac1",
    "local_path": r"C:\Vera\models\base\SmolLM3-3B",
    "mutable_revision_allowed": False,
}


def _spec():
    return {
        "run_id": "BV_SMOLLM3_SUCCESSOR_V2_DEV_20260914",
        "base_repo_id": BASE["repo_id"],
        "base_revision": BASE["revision"],
        "parent_adapter_path": r"C:\private\parent",
        "train_path": r"C:\private\train.jsonl",
        "validation_path": r"C:\private\validation.jsonl",
        "output_dir": r"C:\private\candidate",
        "seed": 20260914,
        "learning_rate": 1e-5,
        "gradient_accumulation": 8,
        "assistant_only_loss": True,
        "neutral_system_override": True,
    }


def test_validate_run_spec_binds_exact_pilot_base_and_training_guards():
    spec = _spec()
    assert validate_run_spec(spec, BASE) == spec


def test_validate_run_spec_rejects_revision_drift():
    spec = _spec()
    spec["base_revision"] = "main"
    with pytest.raises(ValueError, match="base revision"):
        validate_run_spec(spec, BASE)


def test_validate_run_spec_requires_assistant_only_and_neutral_system():
    spec = _spec()
    spec["assistant_only_loss"] = False
    with pytest.raises(ValueError, match="assistant_only_loss"):
        validate_run_spec(spec, BASE)
    spec = _spec()
    spec["neutral_system_override"] = False
    with pytest.raises(ValueError, match="neutral_system_override"):
        validate_run_spec(spec, BASE)


def test_build_run_manifest_records_hashes_without_private_text():
    manifest = build_run_manifest(
        _spec(),
        code_commit="abc123",
        train_sha256="trainhash",
        validation_sha256="valhash",
        parent_adapter_sha256="parenthash",
        output_adapter_sha256="childhash",
        gpu_type="RTX 3050 Laptop GPU",
        package_versions={"torch": "2.11.0+cu128"},
    )
    assert manifest["run_id"] == _spec()["run_id"]
    assert manifest["code_commit"] == "abc123"
    assert manifest["dataset_digests"] == {
        "train_sha256": "trainhash",
        "validation_sha256": "valhash",
    }
    assert manifest["parent_adapter_sha256"] == "parenthash"
    assert manifest["output_adapter_sha256"] == "childhash"
    blob = repr(manifest)
    assert "train.jsonl" not in blob
    assert "validation.jsonl" not in blob


def test_sha256_file_is_content_bound(tmp_path):
    from successor.pilot_sft import sha256_file

    p = tmp_path / "x.bin"
    p.write_bytes(b"vera")
    assert sha256_file(p) == "c7f6d322bc205f26de153999e8d923b63b9261ce34bcb0ef1b9c711f843e05d5"


def test_cosine_floor_scale_has_expected_endpoints():
    from successor.pilot_sft import cosine_floor_scale

    assert cosine_floor_scale(0, 10, warmup_steps=2, floor=0.2) == 0.5
    assert cosine_floor_scale(1, 10, warmup_steps=2, floor=0.2) == 1.0
    assert cosine_floor_scale(9, 10, warmup_steps=2, floor=0.2) == pytest.approx(0.2)
