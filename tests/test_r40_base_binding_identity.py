"""Source-only pinned R15 base binding authority and path-custody checks."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from trainingday.admission import AdmissionError
from trainingday.train_neural import _required_base


OFFICIAL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def _fixture(root: Path, revision: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "snapshot"
    snapshot.mkdir()
    contents = {"config.json": b"{}", "tokenizer_config.json": b"{}",
                "model.safetensors": b"synthetic weights"}
    for name, data in contents.items():
        (snapshot / name).write_bytes(data)
    binding = root / "binding.json"
    binding.write_text(json.dumps({
        "schema": "TRAININGDAY_PINNED_BASE_V1",
        "model_id": "Qwen/Qwen3.5-4B", "revision": revision,
        "loader_class": "Qwen3_5ForCausalLM", "quantization": "NF4_DOUBLE_BF16",
        "snapshot_path": str(snapshot),
        "files": {name: hashlib.sha256(data).hexdigest()
                  for name, data in contents.items()},
    }), encoding="utf-8")
    return binding


class R40BaseBindingIdentityTests(unittest.TestCase):
    def test_exact_source_pinned_revision_is_admitted(self):
        with tempfile.TemporaryDirectory() as temp:
            binding = _fixture(Path(temp) / "real", OFFICIAL_REVISION)
            data, _ = _required_base(binding)
            self.assertEqual(data["revision"], OFFICIAL_REVISION)

    def test_unrelated_valid_40_hex_revision_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            binding = _fixture(Path(temp) / "real", "c" * 40)
            with self.assertRaisesRegex(AdmissionError, "revision"):
                _required_base(binding)

    def test_binding_parent_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _fixture(root / "real", OFFICIAL_REVISION)
            link = root / "link-parent"
            try:
                link.symlink_to(root / "real", target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            with self.assertRaisesRegex(AdmissionError, "symlink|junction"):
                _required_base(link / "binding.json")


if __name__ == "__main__":
    unittest.main()
