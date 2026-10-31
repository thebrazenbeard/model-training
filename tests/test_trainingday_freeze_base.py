from pathlib import Path
from types import SimpleNamespace

import pytest

from trainingday.admission import AdmissionError, _hash_file
from trainingday.freeze_base import freeze
from trainingday.train_neural import _required_base


class FakeHub:
    def __init__(self, sha, file, *, override=None):
        self.sha = sha
        self.file = file
        self.override = override

    def model_info(self, model_id, revision, files_metadata):
        digest = self.override or _hash_file(self.file)
        return SimpleNamespace(sha=self.sha, siblings=[SimpleNamespace(
            rfilename=self.file.name, size=self.file.stat().st_size,
            lfs=SimpleNamespace(sha256=digest))])


def _state(tmp_path):
    root = tmp_path / "base"
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "tokenizer_config.json").write_text("{}")
    weights = root / "model.safetensors"
    weights.write_bytes(b"fixture bytes only, not a real model")
    return root, weights, tmp_path / "binding.json", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def test_freeze_exact_lfs_and_reload_custody(tmp_path):
    root, weights, binding, sha = _state(tmp_path)
    freeze(root, sha, binding, api=FakeHub(sha, weights))
    data, path = _required_base(binding)
    assert data["hub_lfs_verified"] is True and path == root
    assert len(data["files"]) == 3


def test_mismatched_hub_lfs_refused_without_output(tmp_path):
    root, weights, binding, sha = _state(tmp_path)
    with pytest.raises(AdmissionError, match="LFS"):
        freeze(root, sha, binding, api=FakeHub(sha, weights, override="0" * 64))
    assert not binding.exists()


def test_wrong_hub_revision_refused(tmp_path):
    root, weights, binding, sha = _state(tmp_path)
    with pytest.raises(AdmissionError, match="unexpected"):
        freeze(root, sha, binding, api=FakeHub("d" * 40, weights))


def test_extra_unbound_tensor_rejected(tmp_path):
    root, weights, binding, sha = _state(tmp_path)
    (root / "extra.safetensors").write_bytes(b"extra")
    with pytest.raises(AdmissionError, match="missing/extra"):
        freeze(root, sha, binding, api=FakeHub(sha, weights))


def test_no_overwrite_existing_binding(tmp_path):
    root, weights, binding, sha = _state(tmp_path)
    binding.write_text("user evidence")
    with pytest.raises(AdmissionError, match="already exists"):
        freeze(root, sha, binding, api=FakeHub(sha, weights))
    assert binding.read_text() == "user evidence"
