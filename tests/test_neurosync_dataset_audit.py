"""Offline tests using temporary synthetic JSONL, never real private data."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if Path(__file__).resolve().parent.name == "tests":
    # The published repository test must import its production sibling.
    from trainingday import neurosync_dataset_audit as auditor
else:
    # This isolated scratch workspace stores the implementation beside tests.
    import neurosync_dataset_audit as auditor

DatasetAuditHold = auditor.DatasetAuditHold
audit_jsonl_overlap = auditor.audit_jsonl_overlap
main = auditor.main
MAX_FILE_BYTES = auditor.MAX_FILE_BYTES
MAX_LINE_BYTES = auditor.MAX_LINE_BYTES
MAX_RECORDS = auditor.MAX_RECORDS


class DatasetOverlapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ns-data-audit-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.train = self.root / "train.jsonl"
        self.heldout = self.root / "heldout.jsonl"

    def write(self, train, heldout):
        def encode(rows):
            return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        self.train.write_text(encode(train), encoding="utf-8")
        self.heldout.write_text(encode(heldout), encoding="utf-8")

    @staticmethod
    def row(rid, text, role="user"):
        return {"id": rid, "role": role, "text": text}

    def audit(self):
        return audit_jsonl_overlap(
            allowed_root=self.root,
            train_path=self.train,
            heldout_path=self.heldout,
        )

    def test_clean_lexical_split_never_promotes_semantic_disjointness(self):
        self.write(
            [self.row("train-a", "A lone glacier sleeps beneath the moon.")],
            [self.row("heldout-b", "Questionable caterpillars dream of plum jam.")],
        )
        result = self.audit()
        self.assertEqual(result.disposition, "NO_MATCH_OBSERVED_UNVERIFIED_HOLD")
        self.assertEqual(result.exact_text_any_role_pair_count, 0)
        self.assertEqual(result.near_text_pair_count, 0)
        self.assertFalse(result.semantic_disjointness_verified)
        self.assertFalse(result.rights_verified)
        self.assertFalse(result.physical_launch_authorized)
        self.assertFalse(result.persistent_state_commit_authorized)
        self.assertFalse(result.independent_heldout_custody_verified)
        self.assertEqual(len(result.train_file_sha256), 64)

    def test_exact_match_different_ids(self):
        phrase = "A narrow orange bridge crosses the canyon."
        self.write([self.row("train-a", phrase)], [self.row("heldout-b", phrase)])
        result = self.audit()
        self.assertEqual(result.exact_role_text_pair_count, 1)
        self.assertEqual(result.exact_text_any_role_pair_count, 1)
        self.assertEqual(result.near_text_pair_count, 0)
        self.assertEqual(result.disposition, "CONTAMINATION_CANDIDATE_HOLD")

    def test_same_text_different_roles_still_detected(self):
        phrase = "The anchor point is located underneath the station."
        self.write([self.row("train-a", phrase, "assistant")],
                   [self.row("heldout-b", phrase, "user")])
        result = self.audit()
        self.assertEqual(result.exact_role_text_pair_count, 0)
        self.assertEqual(result.exact_text_any_role_pair_count, 1)

    def test_unicode_nfkc_casefold_zero_width_punctuation(self):
        self.write(
            [self.row("train-a", "ＦＵＬＬＷＩＤＴＨ Te\u200bst, and    Coffee!")],
            [self.row("heldout-b", "fullwidth test and coffee")],
        )
        self.assertEqual(self.audit().exact_role_text_pair_count, 1)

    def test_lexical_near_duplicate_flagged_as_heuristic(self):
        self.write(
            [self.row("train-a", "The distant violet mountains sparkle quietly over the frozen northern valley during winter evenings.")],
            [self.row("heldout-b", "The distant violet mountains sparkle brightly over the frozen northern valley during winter evenings.")],
        )
        result = self.audit()
        self.assertEqual(result.exact_text_any_role_pair_count, 0)
        self.assertEqual(result.near_text_pair_count, 1)
        self.assertIn("HEURISTIC", " ".join(result.warnings))

    def test_semantic_paraphrase_beyond_lexical_heuristic_remains_unknown(self):
        self.write(
            [self.row("train-a", "An elderly person purchases a motor vehicle.")],
            [self.row("heldout-b", "A senior citizen buys a car.")],
        )
        result = self.audit()
        self.assertEqual(result.exact_text_any_role_pair_count, 0)
        self.assertFalse(result.semantic_disjointness_verified)
        self.assertIn("SEMANTIC_PARAPHRASES_AND_CONFUSABLE_UNICODE_MAY_ESCAPE",
                      result.warnings)

    def test_same_user_prompt_with_different_answer_detected(self):
        question = "Can you confirm the lighthouse beacon coordinates?"
        self.write(
            [{"id": "turn-a", "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "No; those data are missing."},
            ]}],
            [{"id": "turn-b", "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "The coordinates are east of the harbor."},
            ]}],
        )
        result = self.audit()
        self.assertEqual(result.shared_user_prompt_count, 1)
        self.assertEqual(result.shared_dialogue_prefix_count, 1)
        self.assertEqual(result.disposition, "CONTAMINATION_CANDIDATE_HOLD")

    def test_multi_turn_prefix_reuse_with_new_final_turn(self):
        prefix = [
            {"role": "system", "content": "Avoid inventing remembered events."},
            {"role": "user", "content": "What did we talk about yesterday?"},
        ]
        self.write(
            [{"id": "turn-a", "messages": prefix + [
                {"role": "assistant", "content": "I cannot verify that history."},
            ]}],
            [{"id": "turn-b", "messages": prefix + [
                {"role": "assistant", "content": "We discussed the evening rainfall."},
            ]}],
        )
        self.assertEqual(self.audit().shared_dialogue_prefix_count, 1)

    def test_duplicate_text_and_id_signals(self):
        phrase = "A small copper compass points toward the shoreline."
        self.write(
            [self.row("same-id", phrase), self.row("other-id", phrase)],
            [self.row("same-id", "A totally different statement.")],
        )
        result = self.audit()
        self.assertEqual(result.train_duplicate_text_count, 1)
        self.assertEqual(result.shared_id_count, 1)

    def test_identical_distinct_files_have_matching_file_digest(self):
        rows = [self.row("a", "The lantern light goes out at midnight.")]
        self.write(rows, rows)
        result = self.audit()
        self.assertTrue(result.matching_file_digest)
        self.assertEqual(result.disposition, "CONTAMINATION_CANDIDATE_HOLD")

    def test_blank_line_rejected(self):
        self.train.write_text("\n", encoding="utf-8")
        self.heldout.write_text(json.dumps(self.row("b", "hello")) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(DatasetAuditHold, "blank line"):
            self.audit()

    def test_json_duplicate_keys_rejected_without_text_disclosure(self):
        sentinel = "top-secret-sentinel-value"
        self.train.write_text('{"id":"a","role":"user","text":"' + sentinel +
                              '","text":"alternate"}\n', encoding="utf-8")
        self.heldout.write_text(json.dumps(self.row("b", "hello")) + "\n", encoding="utf-8")
        with self.assertRaises(DatasetAuditHold) as case:
            self.audit()
        self.assertNotIn(sentinel, str(case.exception))

    def test_malformed_json_and_invalid_utf8_rejected(self):
        self.train.write_bytes(b'{"id":"a", "role":"user", "text":\xff}\n')
        self.heldout.write_text(json.dumps(self.row("b", "hello")) + "\n", encoding="utf-8")
        with self.assertRaises(DatasetAuditHold):
            self.audit()

    def test_deep_nested_json_parser_overflow_fails_closed(self):
        payload = ('{"id":"a","messages":' + "[" * 1200 + "0" +
                   "]" * 1200 + "}\n")
        self.train.write_text(payload, encoding="utf-8")
        self.heldout.write_text(json.dumps(self.row("b", "normal")) + "\n",
                                encoding="utf-8")
        with self.assertRaises(DatasetAuditHold):
            self.audit()
        self.train.write_text('{"id":"a","role":"user",}\n', encoding="utf-8")
        with self.assertRaises(DatasetAuditHold):
            self.audit()

    def test_unknown_answer_bearing_field_rejected(self):
        self.write(
            [{"id": "train-a", "role": "user", "text": "Question", "answer": "SECRET"}],
            [self.row("heldout-b", "Other")],
        )
        with self.assertRaisesRegex(DatasetAuditHold, "unknown or missing"):
            self.audit()

    def test_empty_or_zero_width_only_content_rejected(self):
        self.write([self.row("train-a", "\u200b")], [self.row("heldout-b", "normal")])
        with self.assertRaises(DatasetAuditHold):
            self.audit()

    def test_nonfinite_json_constant_rejected(self):
        self.train.write_text('{"id":"a","role":"user","text":NaN}\n', encoding="utf-8")
        self.heldout.write_text(json.dumps(self.row("b", "normal")) + "\n", encoding="utf-8")
        with self.assertRaises(DatasetAuditHold):
            self.audit()

    def test_oversize_line_rejected(self):
        too_long = "x" * MAX_LINE_BYTES
        self.write([self.row("train-a", too_long)], [self.row("heldout-b", "other")])
        with self.assertRaises(DatasetAuditHold):
            self.audit()

    def test_file_byte_cap_rejects_sparse_oversize_file(self):
        self.train.write_bytes(b"")
        with self.train.open("r+b") as stream:
            stream.truncate(MAX_FILE_BYTES + 1)
        self.heldout.write_text(json.dumps(self.row("b", "normal")) + "\n",
                                encoding="utf-8")
        with self.assertRaisesRegex(DatasetAuditHold, "size limit"):
            self.audit()

    def test_empty_file_rejected(self):
        self.train.write_bytes(b"")
        self.heldout.write_text(json.dumps(self.row("b", "normal")) + "\n",
                                encoding="utf-8")
        with self.assertRaises(DatasetAuditHold):
            self.audit()

    def test_record_count_cap_enforced(self):
        self.write(
            [self.row(f"t-{index}", "bounded") for index in range(MAX_RECORDS + 1)],
            [self.row("h-1", "other")],
        )
        with self.assertRaisesRegex(DatasetAuditHold, "record budget"):
            self.audit()

    def test_overlong_multiturn_and_bad_roles_rejected(self):
        self.write([{"id": "t1", "messages": [
            {"role": "user", "content": "word"}
            for _ in range(17)
        ]}], [self.row("h1", "different")])
        with self.assertRaises(DatasetAuditHold):
            self.audit()
        self.write([self.row("t1", "word", role="root")],
                   [self.row("h1", "different")])
        with self.assertRaises(DatasetAuditHold):
            self.audit()

    def test_same_path_and_path_traversal_rejected(self):
        self.write([self.row("a", "one")], [self.row("b", "two")])
        with self.assertRaises(DatasetAuditHold):
            audit_jsonl_overlap(
                allowed_root=self.root, train_path=self.train,
                heldout_path=self.train,
            )
        with self.assertRaisesRegex(DatasetAuditHold, "escapes allowed root"):
            audit_jsonl_overlap(
                allowed_root=self.root / "nested",
                train_path=self.train, heldout_path=self.heldout,
            )
        if os.name != "nt":
            with self.assertRaisesRegex(DatasetAuditHold, "too broad"):
                audit_jsonl_overlap(
                    allowed_root="/", train_path=self.train,
                    heldout_path=self.heldout,
                )
        with self.assertRaisesRegex(DatasetAuditHold, "non-absolute"):
            audit_jsonl_overlap(
                allowed_root=self.root, train_path="train.jsonl",
                heldout_path=self.heldout,
            )
        with self.assertRaises(DatasetAuditHold):
            audit_jsonl_overlap(
                allowed_root=self.root, train_path=str(self.root) + "/bad\x00name",
                heldout_path=self.heldout,
            )

    @unittest.skipIf(not hasattr(os, "symlink"), "symlink unsupported")
    def test_symlink_denied(self):
        self.write([self.row("a", "one")], [self.row("b", "two")])
        symlink = self.root / "linked.jsonl"
        try:
            symlink.symlink_to(self.train)
        except (OSError, NotImplementedError):
            self.skipTest("cannot create symlink in this environment")
        with self.assertRaisesRegex(DatasetAuditHold, "symbolic-link"):
            audit_jsonl_overlap(
                allowed_root=self.root, train_path=symlink,
                heldout_path=self.heldout,
            )

    @unittest.skipIf(not hasattr(os, "symlink"), "symlink unsupported")
    def test_symlinked_parent_denied(self):
        self.write([self.row("a", "one")], [self.row("b", "two")])
        link = self.root / "alias"
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("cannot create symlink in this environment")
        with self.assertRaisesRegex(DatasetAuditHold, "symbolic-link"):
            audit_jsonl_overlap(
                allowed_root=self.root, train_path=link / self.train.name,
                heldout_path=self.heldout,
            )

    @unittest.skipIf(not hasattr(os, "symlink"), "symlink unsupported")
    def test_symlinked_ancestor_of_allowed_root_denied(self):
        self.write([self.row("a", "one")], [self.row("b", "two")])
        link = self.root / "alias"
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("cannot create symlink in this environment")
        with self.assertRaisesRegex(DatasetAuditHold, "symbolic-link"):
            audit_jsonl_overlap(
                allowed_root=link, train_path=link / self.train.name,
                heldout_path=link / self.heldout.name,
            )

    def test_report_contains_no_raw_examples_or_ids_and_no_authority_forgery(self):
        sensitive = "private-evaluation-data-that-should-never-appear"
        self.write([self.row("secret-id", sensitive)],
                   [self.row("heldout-id", sensitive)])
        result = self.audit()
        serialized = json.dumps(result.as_dict())
        self.assertNotIn(sensitive, serialized)
        self.assertNotIn("secret-id", serialized)
        self.assertNotIn("heldout-id", serialized)
        with self.assertRaises((AttributeError, FrozenInstanceError, TypeError)):
            result.rights_verified = True
        with self.assertRaises(TypeError):
            type(result)(**{
                **{field.name: getattr(result, field.name) for field in fields(result)},
                "physical_launch_authorized": True,
            })

    def test_cli_never_returns_positive_launch_status_or_raw_examples(self):
        secret = "private-train-eval-passphrase"
        self.write([self.row("a", secret)], [self.row("b", secret)])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = main([
                "--allowed-root", str(self.root),
                "--train", str(self.train),
                "--heldout", str(self.heldout),
            ])
        self.assertEqual(rc, 1)
        self.assertNotIn(secret, output.getvalue())
        self.assertFalse(json.loads(output.getvalue())["physical_launch_authorized"])

    def test_cli_failure_is_bounded_redacted_and_nonzero(self):
        secret = "do-not-echo-my-data"
        self.train.write_text('{"id":"a","role":"user","text":' + secret + "}\n",
                              encoding="utf-8")
        self.heldout.write_text(json.dumps(self.row("b", "other")) + "\n",
                                encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = main([
                "--allowed-root", str(self.root),
                "--train", str(self.train),
                "--heldout", str(self.heldout),
            ])
        self.assertEqual(rc, 2)
        data = json.loads(output.getvalue())
        self.assertEqual(data["disposition"], "INSPECTION_ERROR_HOLD")
        self.assertNotIn(secret, output.getvalue())
        self.assertFalse(data["physical_launch_authorized"])

    def test_weak_windows_style_path_open_is_advertised(self):
        self.write([self.row("train-a", "A lengthy harmless training passage.")],
                   [self.row("heldout-b", "Some unrelated heldout text.")])
        with patch.object(auditor, "_secure_descriptor_walk_supported",
                          return_value=False):
            result = self.audit()
        self.assertIn("PLATFORM_PATH_CONTAINMENT_NOT_ATOMIC_NO_DIRFD_WALK",
                      result.warnings)


if __name__ == "__main__":
    unittest.main()
