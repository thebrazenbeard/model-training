"""Offline, source-bound Qwen3.5-4B loader diagnostics; not a runtime test."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from trainingday.r75_qwen_loader_preflight import LoaderPreflightError, audit


PIN = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


class LoaderPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        self.sources = self.root / "source"
        (self.sources / "trainingday").mkdir(parents=True)
        for module in ("r18_mezo", "r18_qlora"):
            (self.sources / "trainingday" / f"{module}.py").write_text(
                "from transformers import Qwen3_5ForCausalLM\n"
                "def load(path):\n"
                "    return Qwen3_5ForCausalLM.from_pretrained(path, local_files_only=True)\n",
                encoding="utf-8",
            )

    def _valid(self) -> tuple[Path, str]:
        config = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {"model_type": "qwen3_5_text"},
            "vision_config": {"model_type": "qwen3_5"},
        }
        raw = json.dumps(config).encode()
        (self.snapshot / "config.json").write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        binding = {
            "schema": "TRAININGDAY_PINNED_BASE_V1",
            "model_id": "Qwen/Qwen3.5-4B",
            "revision": PIN,
            "loader_class": "Qwen3_5ForCausalLM",
            "snapshot_path": str(self.snapshot),
            "files": {"config.json": digest},
        }
        path = self.root / "binding.json"
        path.write_text(json.dumps(binding), encoding="utf-8")
        return path, digest

    def test_official_declared_layout_is_hold_even_with_matching_external_digest(self) -> None:
        path, digest = self._valid()
        result = audit(path, expected_config_sha256=digest, source_root=self.sources)
        self.assertEqual(result["status"], "DECLARED_MISMATCH_UNKNOWN_RUNTIME")
        self.assertEqual(result["declared_architecture"], "Qwen3_5ForConditionalGeneration")
        self.assertEqual(result["runner_loader_classes"], {
            "r18_mezo": "Qwen3_5ForCausalLM",
            "r18_qlora": "Qwen3_5ForCausalLM",
        })
        self.assertEqual(result["operator_config_sha256"], digest)
        self.assertEqual(len(result["source_sha256"]), 2)
        self.assertIs(result["runtime_loader_verified"], False)
        self.assertIs(result["weights_loaded"], False)
        self.assertIs(result["training_authorized"], False)
        self.assertIs(result["external_hash_authority_verified"], False)

    def test_current_repository_r18_source_loaders_are_bound_without_importing_torch(self) -> None:
        path, digest = self._valid()
        result = audit(path, expected_config_sha256=digest,
                       source_root=Path(__file__).resolve().parents[1])
        self.assertEqual(result["status"], "DECLARED_MISMATCH_UNKNOWN_RUNTIME")
        self.assertEqual(set(result["runner_loader_classes"].values()),
                         {"Qwen3_5ForCausalLM"})
        self.assertTrue(all(len(value) == 64 for value in result["source_sha256"].values()))

    def test_wrong_external_digest_is_rejected_even_if_self_binding_matches(self) -> None:
        path, _ = self._valid()
        with self.assertRaisesRegex(LoaderPreflightError, "independent config digest"):
            audit(path, expected_config_sha256="f" * 64, source_root=self.sources)

    def test_malformed_hash_missing_binding_and_wrong_revision_are_rejected(self) -> None:
        path, digest = self._valid()
        for sha in ("F" * 64, "a" * 63, "", None, True):
            with self.subTest(sha=sha), self.assertRaises(LoaderPreflightError):
                audit(path, expected_config_sha256=sha, source_root=self.sources)
        binding = json.loads(path.read_text())
        binding["revision"] = "f" * 40
        path.write_text(json.dumps(binding))
        with self.assertRaises(LoaderPreflightError):
            audit(path, expected_config_sha256=digest, source_root=self.sources)

    def test_bad_actual_config_structure_rejected_despite_matching_digest(self) -> None:
        path, _ = self._valid()
        for config in (
            '{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text"}}',
            '{"architectures":["Qwen3_5ForConditionalGeneration"],"architectures":[],"model_type":"qwen3_5"}',
            '{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text"},"vision_config":{"model_type":NaN}}',
        ):
            raw = config.encode()
            digest = hashlib.sha256(raw).hexdigest()
            (self.snapshot / "config.json").write_bytes(raw)
            binding = json.loads(path.read_text())
            binding["files"]["config.json"] = digest
            path.write_text(json.dumps(binding))
            with self.subTest(config=config), self.assertRaises(LoaderPreflightError):
                audit(path, expected_config_sha256=digest, source_root=self.sources)

    def test_unexpected_loader_or_network_enabled_load_is_rejected(self) -> None:
        path, digest = self._valid()
        source = self.sources / "trainingday" / "r18_mezo.py"
        source.write_text(
            "from transformers import Qwen3_5ForConditionalGeneration\n"
            "x = Qwen3_5ForConditionalGeneration.from_pretrained('base', local_files_only=True)\n")
        with self.assertRaisesRegex(LoaderPreflightError, "source loader"):
            audit(path, expected_config_sha256=digest, source_root=self.sources)
        source.write_text(
            "from transformers import Qwen3_5ForCausalLM\n"
            "x = Qwen3_5ForCausalLM.from_pretrained('base', local_files_only=False)\n")
        with self.assertRaisesRegex(LoaderPreflightError, "local_files_only"):
            audit(path, expected_config_sha256=digest, source_root=self.sources)

    def test_loader_symbol_cannot_be_disguised_by_alias_or_shadowing(self) -> None:
        path, digest = self._valid()
        source = self.sources / "trainingday" / "r18_mezo.py"
        hostile_sources = (
            "from transformers import Qwen3_5ForCausalLM, AutoTokenizer\n"
            "base = Qwen3_5ForCausalLM.from_pretrained('base', local_files_only=True)\n"
            "tok = AutoTokenizer.from_pretrained('remote', local_files_only=False)\n",
            "from transformers import Qwen3_5ForCausalLM\n"
            "import transformers as t\n"
            "base = Qwen3_5ForCausalLM.from_pretrained('base', local_files_only=True)\n"
            "other = t.models.some.Model.from_pretrained('remote', local_files_only=False)\n",
            "from transformers import Qwen3_5ForCausalLM\n"
            "from otherlib import ForeignModel as Model\n"
            "base = Qwen3_5ForCausalLM.from_pretrained('base', local_files_only=True)\n"
            "other = Model.from_pretrained('remote', local_files_only=False)\n",
            "from transformers import Qwen3_5ForCausalLM\n"
            "import transformers as t\n"
            "decoy = Qwen3_5ForCausalLM.from_pretrained('base', local_files_only=True)\n"
            "actual = t.Qwen3_5ForConditionalGeneration.from_pretrained('base', local_files_only=False)\n",
            "from transformers import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration as Model\n"
            "decoy = Qwen3_5ForCausalLM.from_pretrained('base', local_files_only=True)\n"
            "actual = Model.from_pretrained('base', local_files_only=False)\n",
            "from transformers import Qwen3_5ForCausalLM, AutoModelForCausalLM as Model\n"
            "decoy = Qwen3_5ForCausalLM.from_pretrained('base', local_files_only=True)\n"
            "actual = Model.from_pretrained('base', local_files_only=False)\n",
            "from .transformers import Qwen3_5ForCausalLM\n"
            "model = Qwen3_5ForCausalLM.from_pretrained('base', local_files_only=True)\n",
            "from ..transformers import Qwen3_5ForCausalLM\n"
            "model = Qwen3_5ForCausalLM.from_pretrained('base', local_files_only=True)\n",
            "from transformers import Qwen3_5ForConditionalGeneration as Qwen3_5ForCausalLM\n"
            "model = Qwen3_5ForCausalLM.from_pretrained('base', local_files_only=True)\n",
            "from transformers import AutoModel as Qwen3_5ForCausalLM\n"
            "model = Qwen3_5ForCausalLM.from_pretrained('base', local_files_only=True)\n",
            "if False:\n"
            "    from transformers import Qwen3_5ForCausalLM\n"
            "class Qwen3_5ForCausalLM:\n"
            "    @classmethod\n"
            "    def from_pretrained(cls, *a, **kw): return object()\n"
            "model = Qwen3_5ForCausalLM.from_pretrained('base', local_files_only=True)\n",
            "from transformers import Qwen3_5ForCausalLM\n"
            "Qwen3_5ForCausalLM = object()\n"
            "model = Qwen3_5ForCausalLM.from_pretrained('base', local_files_only=True)\n",
            "if False:\n"
            "    from transformers import Qwen3_5ForCausalLM\n"
            "model = Qwen3_5ForCausalLM.from_pretrained('base', local_files_only=True)\n",
            "from transformers import Qwen3_5ForCausalLM\n"
            "def load(path, Qwen3_5ForCausalLM):\n"
            "    return Qwen3_5ForCausalLM.from_pretrained(path, local_files_only=True)\n",
            "from transformers import Qwen3_5ForCausalLM\n"
            "if False:\n"
            "    model = Qwen3_5ForCausalLM.from_pretrained('base', local_files_only=True)\n",
            "def unrelated():\n"
            "    from transformers import Qwen3_5ForCausalLM\n"
            "def load(path):\n"
            "    return Qwen3_5ForCausalLM.from_pretrained(path, local_files_only=True)\n",
            "from transformers import Qwen3_5ForCausalLM\n"
            "import Qwen3_5ForCausalLM.submodule\n"
            "model = Qwen3_5ForCausalLM.from_pretrained('base', local_files_only=True)\n",
            "from transformers import Qwen3_5ForCausalLM\n"
            "from something_unknown import *\n"
            "model = Qwen3_5ForCausalLM.from_pretrained('base', local_files_only=True)\n",
        )
        for hostile in hostile_sources:
            with self.subTest(hostile=hostile.splitlines()[0]):
                source.write_text(hostile, encoding="utf-8")
                with self.assertRaises(LoaderPreflightError):
                    audit(path, expected_config_sha256=digest, source_root=self.sources)

    def test_symlinked_config_is_rejected(self) -> None:
        path, digest = self._valid()
        config = self.snapshot / "config.json"
        original = self.root / "original-config.json"
        config.rename(original)
        config.symlink_to(original)
        with self.assertRaises(LoaderPreflightError):
            audit(path, expected_config_sha256=digest, source_root=self.sources)

    def test_cli_failure_does_not_print_snapshot_path(self) -> None:
        path, _ = self._valid()
        process = subprocess.run(
            [sys.executable, "-m", "trainingday.r75_qwen_loader_preflight",
             "--base-binding", str(path), "--expected-config-sha256", "f" * 64,
             "--source-root", str(self.sources)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 2)
        self.assertNotIn(str(self.snapshot), process.stderr + process.stdout)


if __name__ == "__main__":
    unittest.main()
