"""Synthetic R18 preload-custody race test; never accesses a private bank/GPU."""

import hashlib
import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from trainingday.admission import AdmissionError
from trainingday.eval_structural_r16 import CORPUS_SHA256
from trainingday.r18_eval import BASE_MODEL, BASE_REVISION, _InferenceSession, evaluate
from trainingday.train_neural import _required_base


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class R35PreunsealBaseCustodyTests(unittest.TestCase):
    def test_unbound_tokenizer_json_is_rejected_as_loader_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            contents = {
                "config.json": b"{}",
                "tokenizer_config.json": b"{}",
                "model.safetensors": b"synthetic-weights",
            }
            for name, value in contents.items():
                (snapshot / name).write_bytes(value)
            binding = root / "binding.json"
            binding.write_text(json.dumps({
                "schema": "TRAININGDAY_PINNED_BASE_V1", "model_id": BASE_MODEL,
                "revision": BASE_REVISION, "loader_class": "Qwen3_5ForCausalLM",
                "quantization": "NF4_DOUBLE_BF16", "snapshot_path": str(snapshot),
                "files": {name: _sha(value) for name, value in contents.items()},
            }), encoding="utf-8")
            _required_base(binding)  # Existing hashes pass.
            (snapshot / "tokenizer.json").write_text('{"malicious":true}', encoding="utf-8")
            with self.assertRaisesRegex(AdmissionError, "unbound"):
                _required_base(binding)

    def test_session_cleanup_cannot_mask_original_custody_failure(self):
        session = _InferenceSession(Mock(), Mock(), Mock())
        with patch("trainingday.r18_eval.gc.collect", side_effect=RuntimeError("GC broken")):
            with self.assertRaisesRegex(AdmissionError, "custody failure"):
                with closing(session):
                    raise AdmissionError("custody failure")
        self.assertIsNone(session.model)
        self.assertIsNone(session.tokenizer)
        self.assertIsNone(session.torch)

    def test_persistent_base_swap_during_model_load_fails_before_bank_open(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            contents = {
                "config.json": b"{}",
                "tokenizer_config.json": b'{"eos_token_id":1}',
                "model.safetensors": b"synthetic-bound-weights",
            }
            for filename, data in contents.items():
                (snapshot / filename).write_bytes(data)
            binding = root / "base-binding.json"
            binding.write_text(json.dumps({
                "schema": "TRAININGDAY_PINNED_BASE_V1",
                "model_id": BASE_MODEL,
                "revision": BASE_REVISION,
                "loader_class": "Qwen3_5ForCausalLM",
                "quantization": "NF4_DOUBLE_BF16",
                "snapshot_path": str(snapshot),
                "files": {name: _sha(data) for name, data in contents.items()},
            }), encoding="utf-8")

            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            args = SimpleNamespace(
                output=root / "public.json", private_output=root / "private.json",
                manifest=manifest, base_binding=binding, plan=root / "plan.json",
                old_base_eval=root / "old.json", exclusions=root / "exclusions.json",
                sealed_bank=root / "sealed.json",
                eval_source_manifest=root / "source.json", generator_source=root / "generator.py",
                adapter_receipt=None, peer_private=None, preflight_only=False,
                source="frozen", control="frozen",
            )
            session = Mock()

            def swap_during_fake_load(*_args):
                # A persistent writer-supplied replacement after the original
                # _required_base hash check, before the sealed bank is opened.
                (snapshot / "model.safetensors").write_bytes(b"different fake weights")
                return session

            with patch("trainingday.r18_eval._hash_file", return_value=CORPUS_SHA256), \
                 patch("trainingday.r18_eval._source_manifest", return_value=({}, "2" * 64)), \
                 patch("trainingday.r18_eval._plan", return_value=({}, "3" * 64)), \
                 patch("trainingday.r18_eval._adapter", return_value=(None, None, "")), \
                 patch("trainingday.r18_eval._preunseal_inventory", return_value=("4" * 64, {}, {})), \
                 patch("trainingday.r18_eval._require_independent_preunseal_attestation"), \
                 patch("trainingday.r18_eval._cuda_preflight"), \
                 patch("trainingday.r18_eval._load_model", side_effect=swap_during_fake_load), \
                 patch("trainingday.r18_eval._bank", side_effect=AssertionError("PRIVATE BANK OPENED")) as bank:
                with self.assertRaisesRegex(AdmissionError, "base file custody mismatch"):
                    evaluate(args)
            bank.assert_not_called()
            session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
