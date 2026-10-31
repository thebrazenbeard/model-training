"""Direct production admission abuse regressions using public toy corpus only."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trainingday import admission as admission_module
from trainingday.admission import AdmissionError, admit
from trainingday.r28_input_adversary import public_fixture
from trainingday.r18_benchmark import selected_id_digest as r18_legacy_selected_id_digest


class ProductionStrictJsonTests(unittest.TestCase):
    def _admit(self, *, manifest=None, shards=None):
        public_manifest, public_shards = public_fixture()
        manifest = public_manifest if manifest is None else manifest
        shards = public_shards if shards is None else shards
        with tempfile.TemporaryDirectory(prefix="r29-production-admission-") as temp:
            root = Path(temp)
            (root / "shards").mkdir()
            (root / "manifest.json").write_bytes(manifest)
            for rel, content in shards.items():
                (root / rel).write_bytes(content)
            (root / "scratch").mkdir()
            return admit(root / "manifest.json", root / "scratch", sample_size=2)

    def _rebuild_shard(self, split, transform):
        raw, shards = public_fixture()
        shards = dict(shards)
        rel = f"shards/{split}.jsonl"
        lines = shards[rel].splitlines()
        lines[0] = transform(lines[0])
        payload = b"\n".join(lines) + b"\n"
        shards[rel] = payload
        manifest = json.loads(raw)
        spec = next(item for item in manifest["shards"] if item["split"] == split)
        spec["sha256"] = hashlib.sha256(payload).hexdigest()
        return json.dumps(manifest).encode(), shards

    def _admit_selected_ids(self, ids):
        raw, shards = public_fixture()
        document = json.loads(raw)
        document["corpus_id"] = "r31-toy-12"
        rows = [json.loads(line) for line in shards["shards/train.jsonl"].splitlines()]
        for row, case_id in zip(rows, ids, strict=True):
            row["id"] = case_id
            row["prompt"] = f"Public distinct prompt for {case_id!r}?"
        shards = dict(shards)
        payload = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)
        shards["shards/train.jsonl"] = payload
        next(spec for spec in document["shards"] if spec["split"] == "train")["sha256"] = (
            hashlib.sha256(payload).hexdigest()
        )
        return self._admit(manifest=json.dumps(document).encode(), shards=shards)

    def test_public_toy_is_still_source_admitted(self):
        receipt, rows = self._admit()
        self.assertEqual(receipt["status"], "ADMITTED_SOURCE_ONLY_NOT_TRAINED")
        self.assertEqual(receipt["selected_count"], 2)
        self.assertEqual(len(rows), 2)

    def test_conflicting_duplicate_source_commit_rejected_at_real_entrypoint(self):
        raw, _ = public_fixture()
        attack = raw.replace(b'"source_commit":"' + b"b" * 40 + b'"',
                             b'"source_commit":"' + b"a" * 40 + b'",'
                             b'"source_commit":"' + b"c" * 40 + b'"')
        with self.assertRaisesRegex(AdmissionError, "JSON|duplicate"):
            self._admit(manifest=attack)

    def test_duplicate_nested_rights_review_rejected_at_real_entrypoint(self):
        def replace(line):
            return line.replace(b'"rights_review":"PASS"',
                                b'"rights_review":"DENY","rights_review":"PASS"')

        manifest, shards = self._rebuild_shard("train", replace)
        with self.assertRaisesRegex(AdmissionError, "JSON|duplicate"):
            self._admit(manifest=manifest, shards=shards)

    def test_case_variant_rights_key_rejected_at_real_entrypoint(self):
        def replace(line):
            return line.replace(b'"rights_review":"PASS"',
                                b'"RIGHTS_REVIEW":"DENY","rights_review":"PASS"')

        manifest, shards = self._rebuild_shard("train", replace)
        with self.assertRaisesRegex(AdmissionError, "JSON|ambiguous"):
            self._admit(manifest=manifest, shards=shards)

    def test_nfkc_rights_key_collision_rejected_at_real_entrypoint(self):
        def replace(line):
            return line.replace(b'"rights_review":"PASS"',
                                ('"ｒｉｇｈｔｓ＿ｒｅｖｉｅｗ":"DENY",'
                                 '"rights_review":"PASS"').encode("utf-8"))

        manifest, shards = self._rebuild_shard("train", replace)
        with self.assertRaisesRegex(AdmissionError, "JSON|ambiguous"):
            self._admit(manifest=manifest, shards=shards)

    def test_nonfinite_constant_rejected(self):
        raw, _ = public_fixture()
        attack = raw.replace(b'"corpus_id":"r28-public-toy-input-1"',
                             b'"corpus_id":"r28-public-toy-input-1","other":NaN')
        with self.assertRaisesRegex(AdmissionError, "JSON|nonfinite"):
            self._admit(manifest=attack)

    def test_surrogate_escape_rejected(self):
        raw, _ = public_fixture()
        attack = raw.replace(b'"corpus_id":"r28-public-toy-input-1"',
                             b'"corpus_id":"broken\\ud800"')
        with self.assertRaisesRegex(AdmissionError, "JSON|surrogate"):
            self._admit(manifest=attack)

    def test_row_surrogate_escape_rejected(self):
        def replace(line):
            return line.replace(b'"answer":"Toy answer train0"',
                                b'"answer":"bad\\ud800"')

        manifest, shards = self._rebuild_shard("train", replace)
        with self.assertRaisesRegex(AdmissionError, "JSON|surrogate"):
            self._admit(manifest=manifest, shards=shards)

    def test_nonfinite_exponent_rejected(self):
        raw, _ = public_fixture()
        attack = raw.replace(b'"corpus_id":"r28-public-toy-input-1"',
                             b'"corpus_id":"r28-public-toy-input-1","other":1e999')
        with self.assertRaisesRegex(AdmissionError, "JSON|nonfinite"):
            self._admit(manifest=attack)

    def test_invalid_utf8_manifest_rejected_as_admission_error(self):
        raw, _ = public_fixture()
        with self.assertRaisesRegex(AdmissionError, "JSON"):
            self._admit(manifest=raw + b"\xff")

    def test_shard_path_traversal_rejected(self):
        raw, _ = public_fixture()
        document = json.loads(raw)
        document["shards"][0]["path"] = "shards/../../escape.jsonl"
        with self.assertRaisesRegex(AdmissionError, "traversal"):
            self._admit(manifest=json.dumps(document).encode())

    def test_portable_shard_case_collision_rejected(self):
        raw, shards = public_fixture()
        document = json.loads(raw)
        # Two distinct names on case-sensitive filesystems may collide on Windows.
        document["shards"].append(dict(document["shards"][0], path="shards/TRAIN.jsonl"))
        with self.assertRaisesRegex(AdmissionError, "ambiguous shard path"):
            self._admit(manifest=json.dumps(document).encode(), shards=shards)

    def test_declared_denied_rights_remains_denied(self):
        def replace(line):
            return line.replace(b'"rights_review":"PASS"', b'"rights_review":"DENY"')

        manifest, shards = self._rebuild_shard("train", replace)
        with self.assertRaisesRegex(AdmissionError, "review"):
            self._admit(manifest=manifest, shards=shards)

    def test_evaluation_holdout_integer_zero_is_not_boolean_false(self):
        def replace(line):
            return line.replace(b'"evaluation_holdout":false', b'"evaluation_holdout":0')

        manifest, shards = self._rebuild_shard("train", replace)
        with self.assertRaisesRegex(AdmissionError, "evaluation_holdout|protected"):
            self._admit(manifest=manifest, shards=shards)

    def test_evaluation_holdout_missing_is_not_implicit_false(self):
        def replace(line):
            row = json.loads(line)
            del row["evaluation_holdout"]
            return json.dumps(row, sort_keys=True).encode()

        manifest, shards = self._rebuild_shard("train", replace)
        with self.assertRaisesRegex(AdmissionError, "evaluation_holdout|protected"):
            self._admit(manifest=manifest, shards=shards)

    def test_evaluation_holdout_boolean_true_remains_protected(self):
        def replace(line):
            return line.replace(b'"evaluation_holdout":false', b'"evaluation_holdout":true')

        manifest, shards = self._rebuild_shard("train", replace)
        with self.assertRaisesRegex(AdmissionError, "evaluation_holdout|protected"):
            self._admit(manifest=manifest, shards=shards)

    def test_manifest_receipt_hash_identifies_bytes_actually_parsed(self):
        initial, _ = public_fixture()
        expected = hashlib.sha256(initial).hexdigest()
        modified = initial.replace(b'"source_commit":"' + b"b" * 40 + b'"',
                                   b'"source_commit":"' + b"c" * 40 + b'"')
        self.assertNotEqual(initial, modified)
        original_hash_file = admission_module._hash_file
        mutation_count = 0

        def mutate_manifest_during_shard_hash(path):
            nonlocal mutation_count
            if path.name == "train.jsonl" and mutation_count == 0:
                manifest_path = path.parent.parent / "manifest.json"
                manifest_path.write_bytes(modified)
                mutation_count += 1
            return original_hash_file(path)

        with patch("trainingday.admission._hash_file", side_effect=mutate_manifest_during_shard_hash):
            summary, _ = self._admit(manifest=initial)
        self.assertEqual(mutation_count, 1)
        self.assertEqual(summary["source_commit"], "b" * 40)
        self.assertEqual(summary["manifest_sha256"], expected)

    def test_legacy_selected_id_collision_is_separated_by_v2_commitment(self):
        a, a_rows = self._admit_selected_ids(["a\nb", "c"])
        b, b_rows = self._admit_selected_ids(["a", "b\nc"])
        self.assertEqual([row["id"] for row in a_rows], ["a\nb", "c"])
        self.assertEqual([row["id"] for row in b_rows], ["a", "b\nc"])
        self.assertNotEqual(a["manifest_sha256"], b["manifest_sha256"])
        self.assertEqual(a["selected_id_sha256"], b["selected_id_sha256"])
        self.assertEqual(a["selected_id_sha256"], hashlib.sha256(b"a\nb\nc").hexdigest())
        self.assertEqual(a["selected_id_sha256"], r18_legacy_selected_id_digest(a_rows))
        self.assertEqual(b["selected_id_sha256"], r18_legacy_selected_id_digest(b_rows))

        # Fixed domain marker and canonical JSON encode distinct ordered lists.
        marker = b"VERA_TRAININGDAY_SELECTED_ID_DIGEST_V2\x00"
        encode = lambda ids: json.dumps(ids, ensure_ascii=False,
                                       separators=(",", ":")).encode("utf-8")
        expected_a = hashlib.sha256(marker + encode(["a\nb", "c"])).hexdigest()
        expected_b = hashlib.sha256(marker + encode(["a", "b\nc"])).hexdigest()
        self.assertNotEqual(expected_a, expected_b)
        self.assertEqual(a["selected_id_v2_sha256"], expected_a)
        self.assertEqual(b["selected_id_v2_sha256"], expected_b)
        for receipt in (a, b):
            self.assertEqual(receipt["selected_id_v2_schema"],
                             "VERA_TRAININGDAY_SELECTED_ID_DIGEST_V2")
            self.assertEqual(receipt["selected_id_sha256_status"],
                             "LEGACY_DELIMITER_AMBIGUOUS_DO_NOT_USE_ALONE")

    def test_manifest_depth_limit(self):
        raw, _ = public_fixture()
        deep = b'"nest":' + b"[" * 80 + b"0" + b"]" * 80 + b","
        attack = raw.replace(b"{", b"{" + deep, 1)
        with self.assertRaisesRegex(AdmissionError, "JSON|depth"):
            self._admit(manifest=attack)

    def test_manifest_size_limit(self):
        raw, _ = public_fixture()
        with patch("trainingday.admission.MAX_MANIFEST_BYTES", len(raw) - 1):
            with self.assertRaisesRegex(AdmissionError, "size"):
                self._admit()

    def test_row_size_limit(self):
        raw, shards = public_fixture()
        with patch("trainingday.admission.MAX_JSONL_ROW_BYTES", 32):
            with self.assertRaisesRegex(AdmissionError, "size"):
                self._admit(manifest=raw, shards=shards)


if __name__ == "__main__":
    unittest.main()
