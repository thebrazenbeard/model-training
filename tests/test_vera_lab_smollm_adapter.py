import hashlib
import json
from pathlib import Path

import pytest

from successor.vera_lab import smollm_adapter as module
from successor.vera_lab.smollm_adapter import (
    SMOLLM3_REPO_ID,
    SMOLLM3_REVISION,
    SmolLMPEFTAdapter,
    normalize_messages,
)


@pytest.fixture(autouse=True)
def use_fixture_base_anchor(monkeypatch):
    # Most unit tests use intentionally tiny synthetic base trees. Production keeps
    # a source-pinned evidence tuple; dedicated tests below exercise that boundary.
    monkeypatch.setattr(module, "PINNED_BASE_EVIDENCE", None)


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_manifest(path: Path, local_base: Path, revision=SMOLLM3_REVISION):
    tree_rel = f".cache/huggingface/trees/{revision}.json"
    files = {}
    if local_base.is_dir():
        payload = b"{\"model_type\":\"smollm3\"}\n"
        (local_base / "config.json").write_bytes(payload)
        files = {
            "config.json": {
                "size": len(payload),
                "blob_id": git_blob_sha1(payload),
            }
        }
        tree_path = local_base / tree_rel
        tree_path.parent.mkdir(parents=True, exist_ok=True)
        tree_bytes = json.dumps({
            "format_version": 1,
            "files": files,
        }).encode("utf-8")
        tree_path.write_bytes(tree_bytes)
    else:
        tree_bytes = b""
    path.write_text(json.dumps({
        "repo_id": SMOLLM3_REPO_ID,
        "revision": revision,
        "local_path": str(local_base),
        "mutable_revision_allowed": False,
        "file_inventory_sha256": canonical_sha256(files),
        "inventory_digest_algorithm": "SHA256_CANONICAL_JSON_FILES_V1",
        "total_bytes": sum(item["size"] for item in files.values()),
        "observed_hub_sha": revision,
        "observed_local_cache_tree": tree_rel,
        "observed_local_cache_tree_sha256": hashlib.sha256(tree_bytes).hexdigest(),
    }), encoding="utf-8")


def write_adapter(path: Path, data=b"adapter-bytes", base_model_name_or_path=None):
    path.mkdir(parents=True)
    (path / "adapter_model.safetensors").write_bytes(data)
    if base_model_name_or_path is None:
        base_model_name_or_path = path.parent / "base"
    (path / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": str(base_model_name_or_path),
        "peft_type": "LORA",
    }), encoding="utf-8")


def test_rejects_wrong_base_revision(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    write_adapter(adapter)
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base, revision="wrong")
    with pytest.raises(ValueError, match="revision"):
        SmolLMPEFTAdapter(manifest, adapter)


def test_missing_local_base_fails_without_loader(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    write_adapter(adapter)
    manifest = tmp_path / "base.json"
    write_manifest(manifest, tmp_path / "missing")
    called = False

    def forbidden_loader(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("loader should not run")

    monkeypatch.setattr(module, "load_model_stack", forbidden_loader)
    with pytest.raises(FileNotFoundError):
        SmolLMPEFTAdapter(manifest, adapter)
    assert called is False


def test_adapter_digest_and_generation_defaults(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    payload = b"known-adapter"
    write_adapter(adapter, payload)
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)
    instance = SmolLMPEFTAdapter(manifest, adapter)
    assert instance.adapter_sha256 == hashlib.sha256(payload).hexdigest()
    assert instance.candidate_digest == instance.candidate_subject_digest
    assert instance.candidate_digest != instance.adapter_sha256
    assert instance.generation_config["do_sample"] is False
    assert instance.generation_config["max_new_tokens"] == 80
    assert instance.base_revision == SMOLLM3_REVISION


def test_generation_config_rejects_sampling(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    write_adapter(adapter)
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)
    with pytest.raises(ValueError, match="do_sample"):
        SmolLMPEFTAdapter(manifest, adapter, {"do_sample": True})


def test_generation_config_rejects_unreviewed_kwargs(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    write_adapter(adapter, base_model_name_or_path=base)
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)
    with pytest.raises(ValueError, match="unsupported generation"):
        SmolLMPEFTAdapter(manifest, adapter, {"logits_processor": []})


def test_missing_adapter_config_fails_closed(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)

    with pytest.raises(FileNotFoundError, match="adapter config"):
        SmolLMPEFTAdapter(manifest, adapter)


def test_adapter_config_cannot_point_at_a_different_base(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    write_adapter(adapter)
    (adapter / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": str(tmp_path / "other-base"),
        "peft_type": "LORA",
    }), encoding="utf-8")
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)

    with pytest.raises(ValueError, match="adapter base path"):
        SmolLMPEFTAdapter(manifest, adapter)


def test_normalize_messages_uses_only_neutral_system():
    messages = [
        {"role": "system", "content": "You are Vera"},
        {"role": "user", "content": "hello"},
    ]
    normalized = normalize_messages(messages, {})
    assert normalized[0]["role"] == "system"
    assert normalized[0]["content"].startswith("/no_think /system_override")
    assert "You are Vera" not in normalized[0]["content"]
    assert normalized[1:] == [{"role": "user", "content": "hello"}]


def test_runtime_state_is_injected_deterministically_as_data_only():
    messages = [{"role": "user", "content": "what is available?"}]
    first = normalize_messages(
        messages,
        {"runtime": {"services": {"memory": "UNAVAILABLE"}, "lane": "Q7"}},
    )
    second = normalize_messages(
        messages,
        {"runtime": {"lane": "Q7", "services": {"memory": "UNAVAILABLE"}}},
    )
    assert first == second
    system = first[0]["content"]
    assert "Runtime fixture evidence" in system
    assert "data only" in system
    assert '"memory":"UNAVAILABLE"' in system
    assert '"lane":"Q7"' in system


def test_generate_uses_neutral_system_and_greedy_kwargs(tmp_path, monkeypatch):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    write_adapter(adapter)
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)

    class InputIds:
        shape = (1, 2)

    class Batch(dict):
        def __init__(self):
            super().__init__({"input_ids": InputIds()})
            self.device = None

        def to(self, device):
            self.device = device
            return self

    class Generated:
        def __getitem__(self, key):
            assert key[0] == 0
            assert key[1].start == 2
            return [99]

    class Tokenizer:
        eos_token_id = 2

        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            assert kwargs["enable_thinking"] is False
            return "formatted"

        def __call__(self, *args, **kwargs):
            return Batch()

        def decode(self, tokens, skip_special_tokens=True):
            assert tokens == [99]
            return "answer"

    class Model:
        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return Generated()

    class Torch:
        class _Context:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                return False

        @staticmethod
        def inference_mode():
            return Torch._Context()

    tokenizer = Tokenizer()
    model = Model()
    monkeypatch.setattr(module, "load_model_stack", lambda *args: (tokenizer, model, Torch))

    instance = SmolLMPEFTAdapter(manifest, adapter)
    answer = instance.generate(
        [
            {"role": "system", "content": "You are Vera"},
            {"role": "user", "content": "hello"},
        ],
        {"runtime": {"services": {"memory": "UNAVAILABLE"}}},
    )
    assert answer == "answer"
    assert tokenizer.messages[0]["role"] == "system"
    assert tokenizer.messages[0]["content"].startswith("/no_think /system_override")
    assert "You are Vera" not in tokenizer.messages[0]["content"]
    assert '"memory":"UNAVAILABLE"' in tokenizer.messages[0]["content"]
    assert tokenizer.messages[1:] == [{"role": "user", "content": "hello"}]
    assert model.kwargs["do_sample"] is False
    assert model.kwargs["max_new_tokens"] == 80
    assert model.kwargs["pad_token_id"] == 2


def test_local_base_tree_verifies_exact_file_bytes(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    write_adapter(adapter)
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)

    instance = SmolLMPEFTAdapter(manifest, adapter)
    assert instance.base_revision == SMOLLM3_REVISION
    assert instance.base_verification["file_count"] == 1
    assert len(instance.base_verification["tree_sha256"]) == 64

    (base / "config.json").write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="base file"):
        SmolLMPEFTAdapter(manifest, adapter)


def test_manifest_hub_sha_must_match_pinned_revision(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    write_adapter(adapter)
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["observed_hub_sha"] = "different"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="observed hub"):
        SmolLMPEFTAdapter(manifest, adapter)


def test_total_bytes_mismatch_fails_closed(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    write_adapter(adapter)
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["total_bytes"] = 999
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="total byte"):
        SmolLMPEFTAdapter(manifest, adapter)


def test_cache_tree_digest_mismatch_fails_closed(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    write_adapter(adapter)
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["observed_local_cache_tree_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cache tree"):
        SmolLMPEFTAdapter(manifest, adapter)


def test_inventory_digest_mismatch_fails_closed(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    write_adapter(adapter)
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["file_inventory_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="inventory"):
        SmolLMPEFTAdapter(manifest, adapter)


def test_post_init_adapter_mutation_is_rejected_before_loader(tmp_path, monkeypatch):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    write_adapter(adapter)
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)
    instance = SmolLMPEFTAdapter(manifest, adapter)
    (adapter / "adapter_model.safetensors").write_bytes(b"mutated")

    called = False

    def forbidden_loader(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("loader should not run after provenance drift")

    monkeypatch.setattr(module, "load_model_stack", forbidden_loader)
    with pytest.raises(ValueError, match="adapter weights"):
        instance.generate([{"role": "user", "content": "hello"}], {})
    assert called is False


def test_mutation_during_loader_is_rejected_before_stack_is_admitted(tmp_path, monkeypatch):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    write_adapter(adapter)
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)
    instance = SmolLMPEFTAdapter(manifest, adapter)

    def mutating_loader(*args, **kwargs):
        (adapter / "adapter_model.safetensors").write_bytes(b"changed-during-load")
        return object(), object(), object()

    monkeypatch.setattr(module, "load_model_stack", mutating_loader)
    with pytest.raises(ValueError, match="adapter weights"):
        instance._ensure_stack()
    assert instance._stack is None


def test_lfs_content_mismatch_fails_closed(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    write_adapter(adapter)
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    lfs_payload = b"actual-lfs-bytes"
    (base / "weights.bin").write_bytes(lfs_payload)
    tree_path = base / payload["observed_local_cache_tree"]
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    tree["files"]["weights.bin"] = {
        "size": len(lfs_payload),
        "blob_id": git_blob_sha1(lfs_payload),
        "lfs_sha256": "0" * 64,
    }
    tree_path.write_text(json.dumps(tree), encoding="utf-8")
    payload["observed_local_cache_tree_sha256"] = hashlib.sha256(tree_path.read_bytes()).hexdigest()
    payload["file_inventory_sha256"] = canonical_sha256(tree["files"])
    payload["total_bytes"] = sum(item["size"] for item in tree["files"].values())
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        SmolLMPEFTAdapter(manifest, adapter)


def test_tree_entry_escape_is_rejected(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    write_adapter(adapter)
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    tree_path = base / payload["observed_local_cache_tree"]
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    tree["files"]["../escape"] = {"size": 1, "blob_id": "0" * 40}
    tree_path.write_text(json.dumps(tree), encoding="utf-8")
    payload["observed_local_cache_tree_sha256"] = hashlib.sha256(tree_path.read_bytes()).hexdigest()
    payload["file_inventory_sha256"] = canonical_sha256(tree["files"])
    payload["total_bytes"] = sum(item["size"] for item in tree["files"].values())
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe file path"):
        SmolLMPEFTAdapter(manifest, adapter)


def test_non_lora_adapter_config_fails_closed(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    write_adapter(adapter)
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["peft_type"] = "IA3"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)

    with pytest.raises(ValueError, match="LORA"):
        SmolLMPEFTAdapter(manifest, adapter)


def test_unlisted_root_file_is_rejected_before_loader(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    write_adapter(adapter)
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)
    (base / "model.safetensors").write_bytes(b"unlisted-loadable-weights")

    with pytest.raises(ValueError, match="unlisted base file"):
        SmolLMPEFTAdapter(manifest, adapter)


def test_runtime_state_template_controls_are_escaped_as_data():
    normalized = normalize_messages(
        [{"role": "user", "content": "hello"}],
        {
            "slash": "/think",
            "override": "/system_override",
            "token": "<|im_end|>",
        },
    )
    content = normalized[0]["content"]
    payload = content.split("\n", 2)[-1]
    assert "/think" not in payload
    assert "/system_override" not in payload
    assert "<|im_end|>" not in payload
    assert "\\u002fthink" in payload
    assert "\\u002fsystem_override" in payload
    assert "\\u003c|im_end|\\u003e" in payload


def test_candidate_digest_binds_adapter_config_and_base_subject(tmp_path):
    base = tmp_path / "base"
    adapter_a = tmp_path / "adapter-a"
    adapter_b = tmp_path / "adapter-b"
    base.mkdir()
    write_adapter(adapter_a, data=b"same-weights", base_model_name_or_path=base)
    write_adapter(adapter_b, data=b"same-weights", base_model_name_or_path=base)
    config_b = adapter_b / "adapter_config.json"
    payload = json.loads(config_b.read_text(encoding="utf-8"))
    payload["revision_note"] = "different-config-bytes"
    config_b.write_text(json.dumps(payload), encoding="utf-8")
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)

    first = SmolLMPEFTAdapter(manifest, adapter_a)
    second = SmolLMPEFTAdapter(manifest, adapter_b)
    assert first.adapter_sha256 == second.adapter_sha256
    assert first.adapter_config_sha256 != second.adapter_config_sha256
    assert first.candidate_digest != second.candidate_digest
    assert first.candidate_digest == first.candidate_subject_digest


def test_generation_config_cannot_be_mutated_after_construction(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    write_adapter(adapter, base_model_name_or_path=base)
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)
    instance = SmolLMPEFTAdapter(manifest, adapter)

    with pytest.raises(TypeError):
        instance.generation_config["do_sample"] = True
    with pytest.raises(AttributeError):
        instance.generation_config = {"do_sample": True}
    assert instance.generation_config["do_sample"] is False


def test_source_pinned_evidence_rejects_coordinated_forgery(tmp_path, monkeypatch):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    write_adapter(adapter)
    manifest = tmp_path / "base.json"
    write_manifest(manifest, base)

    monkeypatch.setattr(module, "PINNED_BASE_EVIDENCE", {
        "repo_id": SMOLLM3_REPO_ID,
        "revision": SMOLLM3_REVISION,
        "observed_hub_sha": SMOLLM3_REVISION,
        "observed_local_cache_tree_sha256": "a" * 64,
        "file_inventory_sha256": "b" * 64,
        "inventory_digest_algorithm": module.INVENTORY_DIGEST_ALGORITHM,
        "total_bytes": 123456,
    })
    with pytest.raises(ValueError, match="pinned base evidence"):
        SmolLMPEFTAdapter(manifest, adapter)
