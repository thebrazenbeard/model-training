"""Synthetic GGUF v3 header tests; no model bytes or external downloads."""

import struct
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from trainingday.gguf_header_triage import inspect, MAX_KV_PAIRS, MAX_ITEMS


def st(text):
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def kv(key, typ, encoded_value):
    return st(key) + struct.pack("<I", typ) + encoded_value


def gguf(entries, *, version=3, tensor_count=7, count=None):
    return (struct.pack("<4sIQQ", b"GGUF", version, tensor_count,
                        len(entries) if count is None else count) + b"".join(entries))


class HeaderTriageTests(unittest.TestCase):
    def inspect_bytes(self, payload):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "not-a-model.gguf"
            path.write_bytes(payload)
            return inspect(path)

    def test_selected_metadata_is_read_without_tensors(self):
        entries = [
            kv("general.architecture", 8, st("qwen35")),
            kv("general.name", 8, st("Untrusted arbitrary title")),
            kv("general.source.url", 8, st("https://example.invalid/forged")),
            kv("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 3) +
               st("foo") + st("bar") + st("baz")),
            kv("general.file_type", 4, struct.pack("<I", 17)),
        ]
        result = self.inspect_bytes(gguf(entries) + b"X" * 10_000)
        self.assertEqual(result["selected_metadata"]["general.architecture"], "qwen35")
        self.assertEqual(result["selected_metadata"]["general.file_type"], 17)
        self.assertNotIn("tokenizer.ggml.tokens", result["selected_metadata"])
        self.assertEqual(result["adapter_compatibility"], "NOT_ESTABLISHED")
        self.assertEqual(result["upstream_tensor_ancestry"], "UNKNOWN_NOT_MEASURED")
        self.assertLess(result["actual_header_read_bytes"], 1024)
        self.assertLess(result["metadata_scan_bytes"], result["file_bytes"])

    def test_unknown_or_malformed_magic_version_and_key_count_rejected(self):
        for payload, message in ((b"NOPE" + gguf([])[4:], "not GGUF"),
                                 (gguf([], version=2), "version 3"),
                                 (gguf([], count=MAX_KV_PAIRS + 1), "too many")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.inspect_bytes(payload)

    def test_duplicate_and_invalid_type_fail_closed(self):
        item = kv("general.name", 8, st("A"))
        with self.assertRaisesRegex(ValueError, "duplicate GGUF metadata key"):
            self.inspect_bytes(gguf([item, item]))
        with self.assertRaisesRegex(ValueError, "unknown scalar type"):
            self.inspect_bytes(gguf([kv("general.name", 127, b"")]))

    def test_selected_string_and_header_truncation_do_not_allocate_unbounded(self):
        huge = kv("general.name", 8, struct.pack("<Q", 2**63))
        with self.assertRaisesRegex(ValueError, "selected metadata string too long"):
            self.inspect_bytes(gguf([huge]))
        with self.assertRaisesRegex(ValueError, "past file bounds"):
            self.inspect_bytes(gguf([kv("tokenizer.ggml.tokens", 8,
                                         struct.pack("<Q", 9999999))]))

    def test_huge_array_count_rejected_before_element_loop(self):
        giant = kv("tokenizer.ggml.tokens", 9,
                   struct.pack("<IQ", 8, MAX_ITEMS + 1))
        with self.assertRaisesRegex(ValueError, "array element budget exceeded"):
            self.inspect_bytes(gguf([giant]))

    def test_unknown_array_type_and_nested_array_budget(self):
        invalid = kv("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 234, 1))
        with self.assertRaisesRegex(ValueError, "unknown element type"):
            self.inspect_bytes(gguf([invalid]))
        nested = struct.pack("<IQ", 9, 1)
        deeply_nested = kv("unused", 9, nested + nested + nested + b"")
        with self.assertRaisesRegex(ValueError, "excessive array nesting"):
            self.inspect_bytes(gguf([deeply_nested]))

    def test_boolean_metadata_is_strict(self):
        with self.assertRaisesRegex(ValueError, "boolean is not 0 or 1"):
            # Selected keys have no fixed schema; a malicious field can be a bool.
            self.inspect_bytes(gguf([kv("general.name", 7, b"\x02")]))

    def test_missing_advertised_payload_fails_in_header_scan(self):
        broken = gguf([kv("tokenizer.ggml.tokens", 9,
                          struct.pack("<IQ", 8, 3) + st("A"))])
        with self.assertRaisesRegex(ValueError, "past file bounds"):
            self.inspect_bytes(broken)

    def test_hard_read_and_scan_caps_are_enforced(self):
        payload = gguf([kv("extra.large", 8, st("Y" * 300))])
        with patch("trainingday.gguf_header_triage.MAX_SCAN_BYTES", 100):
            with self.assertRaisesRegex(ValueError, "bounded scan window"):
                self.inspect_bytes(payload)
        with patch("trainingday.gguf_header_triage.MAX_READ_BYTES", 16):
            with self.assertRaisesRegex(ValueError, "byte-read budget"):
                self.inspect_bytes(payload)


if __name__ == "__main__":
    unittest.main()
