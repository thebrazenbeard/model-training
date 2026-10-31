"""Adversarial pinned-base inventory tests; public synthetic bytes only."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from trainingday.admission import AdmissionError
from trainingday.train_neural import _required_base


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class R37RecursiveBaseClosure(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        self.contents = {
            "config.json": b"{}",
            "tokenizer_config.json": b"{}",
            "model.safetensors": b"synthetic-weight-bytes",
        }
        self.binding = self.root / "binding.json"

    def write_binding(self):
        for name, contents in self.contents.items():
            file = self.snapshot.joinpath(*name.split("/"))
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(contents)
        self.binding.write_text(json.dumps({
            "schema": "TRAININGDAY_PINNED_BASE_V1", "model_id": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "loader_class": "Qwen3_5ForCausalLM",
            "quantization": "NF4_DOUBLE_BF16", "snapshot_path": str(self.snapshot),
            "files": {name: _sha(data) for name, data in self.contents.items()},
        }), encoding="utf-8")

    def test_unbound_nested_template_does_not_escape_inventory(self):
        self.write_binding()
        templates = self.snapshot / "additional_chat_templates"
        templates.mkdir()
        (templates / "default.jinja").write_text("{{ hostile }}", encoding="utf-8")
        with self.assertRaisesRegex(AdmissionError, "unbound"):
            _required_base(self.binding)

    def test_symlinked_nested_template_directory_is_rejected(self):
        self.write_binding()
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "default.jinja").write_text("{{ external }}", encoding="utf-8")
        try:
            (self.snapshot / "additional_chat_templates").symlink_to(
                outside, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        with self.assertRaisesRegex(AdmissionError, "symlink"):
            _required_base(self.binding)

    def test_unbound_versioned_fast_tokenizer_file_is_rejected(self):
        self.write_binding()
        (self.snapshot / "tokenizer.4.20.0.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(AdmissionError, "unbound"):
            _required_base(self.binding)

    def test_bound_nested_template_and_versioned_tokenizer_are_allowed(self):
        self.contents["additional_chat_templates/default.jinja"] = b"{{ safe }}"
        self.contents["tokenizer_config.json"] = (
            b'{"fast_tokenizer_files":["tokenizer.4.20.0.json"]}')
        self.contents["tokenizer.4.20.0.json"] = b"{}"
        self.write_binding()
        _, path = _required_base(self.binding)
        self.assertEqual(path, self.snapshot)
        (self.snapshot / "additional_chat_templates" / "default.jinja").write_bytes(b"tampered")
        with self.assertRaisesRegex(AdmissionError, "custody mismatch"):
            _required_base(self.binding)

    def test_config_selected_fast_tokenizer_must_be_in_pinned_files(self):
        self.contents["tokenizer_config.json"] = (
            b'{"fast_tokenizer_files":["tokenizer.4.20.0.json"]}')
        self.write_binding()
        with self.assertRaisesRegex(AdmissionError, "fast tokenizer candidate unpinned"):
            _required_base(self.binding)

    def test_parent_traversal_cannot_be_pinned(self):
        self.write_binding()
        data = json.loads(self.binding.read_text(encoding="utf-8"))
        data["files"]["../outside.json"] = _sha(b"unknown")
        self.binding.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(AdmissionError, "malformed"):
            _required_base(self.binding)


if __name__ == "__main__":
    unittest.main()
