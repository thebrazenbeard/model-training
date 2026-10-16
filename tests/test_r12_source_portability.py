"""Transport normalization may not alter frozen training-content provenance."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from successor.experiments import build_r12_voice_corrective as subject

FROZEN_GIT_SHA = "419e01f529bd857da2e6dc62ffea73ccece22ced53b8e24175401f126e7b1f81"


class SourceTransportTests(unittest.TestCase):
    def test_canonical_sha_accepts_git_lf_and_windows_crlf(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "voice.jsonl"
            payload = b'{"id":"one"}\n{"id":"two"}\n'
            expected = hashlib.sha256(payload).hexdigest()
            p.write_bytes(payload)
            self.assertEqual(subject.canonical_lf_sha256(p), expected)
            p.write_bytes(payload.replace(b"\n", b"\r\n"))
            self.assertEqual(subject.canonical_lf_sha256(p), expected)
            self.assertNotEqual(hashlib.sha256(p.read_bytes()).hexdigest(), expected)

    def test_refuse_content_change_or_unsupported_newline(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "voice.jsonl"
            p.write_bytes(b'a\rb\n')
            with self.assertRaisesRegex(ValueError,"unsupported"):
                subject.canonical_lf_sha256(p)
            p.write_bytes(b'a\nb\n')
            with self.assertRaisesRegex(ValueError,"source SHA"):
                subject.verify_curated_source(p, FROZEN_GIT_SHA)

    def test_current_checkout_normalizes_to_exact_git_binding(self):
        self.assertEqual(subject.canonical_lf_sha256(subject.VOICE), FROZEN_GIT_SHA)
        subject.verify_curated_source(subject.VOICE, FROZEN_GIT_SHA)
        binding = json.loads(subject.BINDING_PATH.read_text(encoding="utf8"))
        self.assertEqual(binding["r12_source_voice_sha256"], FROZEN_GIT_SHA)
        self.assertEqual(subject.VOICE_SHA, FROZEN_GIT_SHA)

    def test_existing_r12_train_corpus_still_exact(self):
        from successor.experiments.train_v10_qwen35_authorized import sha256_file
        self.assertEqual(sha256_file(subject.OUT_DIR / "train.jsonl"),
                         "ca46a0340c370e2284cafcf2f76c439a5ff0826c4c8b93960f26db9573bf09cf")


if __name__ == "__main__":
    unittest.main()
