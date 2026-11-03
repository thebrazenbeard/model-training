"""R10 source hash is transport-stable without weakening parent custody."""
from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from successor.experiments import build_v10r3r11_vera_voice as r11


ROOT = Path(__file__).resolve().parents[1]


class R10SpecTransportTests(unittest.TestCase):
    def test_exact_checked_in_git_source_matches_original_windows_digest(self):
        checked_in = ROOT / r11.R10_SPEC_REL
        data = checked_in.read_bytes()
        self.assertEqual(hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest(),
                         "1d099d6329e0b051cc8a70043594036dfcc8795a407a22f79f1c69d739ab73eb")
        self.assertEqual(r11.r10_spec_transport_sha256(checked_in),
                         r11.R10_SPEC_SHA)

    def test_exact_crlf_is_equivalent_but_mutated_bytes_do_not_pass(self):
        data = (ROOT / r11.R10_SPEC_REL).read_bytes().replace(b"\r\n", b"\n")
        with TemporaryDirectory() as temp:
            candidate = Path(temp) / "r10.json"
            candidate.write_bytes(data.replace(b"\n", b"\r\n"))
            self.assertEqual(r11.r10_spec_transport_sha256(candidate),
                             r11.R10_SPEC_SHA)
            candidate.write_bytes(data.replace(b"QWEN", b"OTHER"))
            with self.assertRaisesRegex(ValueError, "Git source hash"):
                r11.r10_spec_transport_sha256(candidate)

    def test_reject_mixed_or_unsupported_carriage_returns(self):
        with TemporaryDirectory() as temp:
            candidate = Path(temp) / "r10.json"
            for raw in (b"a\r\nb\n", b"a\rb\n", b"a\nb\r\n"):
                candidate.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, "transport"):
                    r11.r10_spec_transport_sha256(candidate)


if __name__ == "__main__":
    unittest.main()
