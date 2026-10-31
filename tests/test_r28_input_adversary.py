"""Deterministic public-toy input abuse cases; no external code execution."""

import hashlib
import io
import json
import unittest
import zipfile

from trainingday.r28_input_adversary import (
    MAX_MANIFEST_BYTES,
    MAX_SHARD_BYTES,
    assess_public_admission_fixture,
    generate_adversarial_public_cases,
    public_fixture,
)


def _encode(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _mutate_manifest(raw: bytes, mutation) -> bytes:
    manifest = json.loads(raw)
    mutation(manifest)
    return _encode(manifest)


def _mutate_row(manifest: bytes, shards: dict[str, bytes], split: str, mutation):
    name = f"shards/{split}.jsonl"
    rows = [json.loads(line) for line in shards[name].splitlines()]
    mutation(rows)
    changed = dict(shards)
    changed[name] = b"".join(_encode(row) + b"\n" for row in rows)
    adjusted = json.loads(manifest)
    for spec in adjusted["shards"]:
        if spec["path"] == name:
            spec["rows"] = len(rows)
            spec["sha256"] = hashlib.sha256(changed[name]).hexdigest()
    return _encode(adjusted), changed


def _zip(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in entries:
            z.writestr(name, content)
    return buffer.getvalue()


class BoundedAdversarialAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.shards = public_fixture()

    def assess(self, manifest=None, shards=None, archive=None):
        return assess_public_admission_fixture(
            self.manifest if manifest is None else manifest,
            self.shards if shards is None else shards,
            archive=archive,
        )

    def test_valid_public_toy_fixture_is_source_admitted_not_training_authorized(self):
        receipt = self.assess()
        self.assertEqual(receipt.reason, "PUBLIC_TOY_ADMISSION_ONLY")
        self.assertEqual(receipt.status, "SOURCE_ADMITTED")
        self.assertEqual(receipt.selected_count, 2)
        self.assertFalse(receipt.rights_independently_authenticated)
        self.assertFalse(receipt.gpu_training_authorized)
        self.assertFalse(receipt.model_release_authorized)

    def test_deterministic_malicious_cases_reject(self):
        cases = generate_adversarial_public_cases()
        labels = [case.label for case in cases]
        self.assertEqual(labels, [case.label for case in generate_adversarial_public_cases()])
        self.assertEqual(len(set(labels)), len(labels))
        self.assertGreaterEqual(len(cases), 14)
        for case in cases:
            with self.subTest(case=case.label):
                receipt = assess_public_admission_fixture(
                    case.manifest, case.shards, archive=case.archive,
                )
                self.assertEqual(receipt.status, "REJECTED")
                self.assertNotEqual(receipt.reason, "PUBLIC_TOY_ADMISSION_ONLY")
                self.assertFalse(receipt.gpu_training_authorized)

    def test_manifest_path_traversal_refused_without_writing_outside_scratch(self):
        raw = _mutate_manifest(
            self.manifest,
            lambda d: d["shards"][0].update(path="shards/../../escape.jsonl"),
        )
        self.assertEqual(self.assess(manifest=raw).reason, "UNSAFE_SHARD_PATH")

    def test_oversized_payloads_rejected_before_json_parsing(self):
        self.assertEqual(self.assess(manifest=b" " * (MAX_MANIFEST_BYTES + 1)).reason,
                         "MANIFEST_SIZE_LIMIT")
        shards = dict(self.shards)
        shards["shards/train.jsonl"] = b"x" * (MAX_SHARD_BYTES + 1)
        self.assertEqual(self.assess(shards=shards).reason, "SHARD_SIZE_LIMIT")

    def test_malformed_unicode_and_duplicate_keys_rejected(self):
        for payload in (
            b'\xff',
            b'{"schema":1,"schema":2}',
            b'{"schema":NaN}',
            b'{"schema":1e999}',
        ):
            with self.subTest(payload=payload[:20]):
                self.assertEqual(self.assess(manifest=payload).reason, "MANIFEST_JSON_INVALID")
        changed = dict(self.shards)
        changed["shards/train.jsonl"] = b'{"id":"x","id":"y"}\n'
        self.assertEqual(self.assess(shards=changed).reason, "JSONL_ROW_INVALID")
        changed = dict(self.shards)
        changed["shards/train.jsonl"] = b'{"id":"\\ud800"}\n'
        self.assertEqual(self.assess(shards=changed).reason, "JSONL_ROW_INVALID")

    def test_forged_split_and_rights_fail_before_selection(self):
        bad, shards = _mutate_row(
            self.manifest, self.shards, "dev", lambda rows: rows[0].update(split="train"),
        )
        self.assertEqual(self.assess(manifest=bad, shards=shards).reason,
                         "UPSTREAM_ADMISSION_REJECTED")
        bad, shards = _mutate_row(
            self.manifest, self.shards, "train",
            lambda rows: rows[0]["provenance"].update(rights_review="UNKNOWN"),
        )
        self.assertEqual(self.assess(manifest=bad, shards=shards).reason,
                         "UPSTREAM_ADMISSION_REJECTED")

    def test_code_snippet_is_inert_data_not_executed(self):
        snippet = '__import__("os").system("touch /never-execute-public-toy")'
        manifest, shards = _mutate_row(
            self.manifest, self.shards, "train",
            lambda rows: rows[0].update(prompt=snippet),
        )
        result = self.assess(manifest=manifest, shards=shards)
        self.assertEqual(result.reason, "PUBLIC_TOY_ADMISSION_ONLY")
        self.assertFalse(result.gpu_training_authorized)

    def test_archive_metadata_only_accepts_safe_and_refuses_bomb_or_traversal(self):
        good = _zip([("public/model.bin", b"toy")])
        result = self.assess(archive=good)
        self.assertEqual(result.reason, "PUBLIC_TOY_ADMISSION_ONLY")
        self.assertFalse(result.model_release_authorized)
        for payload in (
            _zip([("../escape.bin", b"x")]),
            _zip([("bomb.bin", b"A" * 200_000)]),
            _zip([("nested.tar.gz", b"pretend")]),
        ):
            with self.subTest(digest=hashlib.sha256(payload).hexdigest()):
                self.assertEqual(self.assess(archive=payload).reason,
                                 "ARCHIVE_STATIC_METADATA_REJECTED")


if __name__ == "__main__":
    unittest.main()
