"""Contract tests for the rights-pending, original synthetic curriculum."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from trainingday import original_curriculum
from trainingday.original_curriculum import DOMAINS, build_rows, validate, write_candidate


class OriginalCurriculumTests(unittest.TestCase):
    def test_domain_counts_and_split_isolation(self):
        rows = build_rows()
        self.assertEqual(len(rows), 576)
        for domain in DOMAINS:
            self.assertEqual(sum(r["domain"] == domain and r["split"] == "train" for r in rows), 72)
            self.assertEqual(sum(r["domain"] == domain and r["split"] == "dev" for r in rows), 24)
        for key in ("family_id", "template_id", "source_id", "prompt"):
            groups = {}
            for row in rows:
                groups.setdefault(row[key], set()).add(row["split"])
            self.assertTrue(all(len(s) == 1 for s in groups.values()), key)
        self.assertEqual({r["archetype_id"] for r in rows if r["split"] == "train"},
                         {r["archetype_id"] for r in rows if r["split"] == "dev"})

    def test_rights_are_not_implied(self):
        for row in build_rows():
            self.assertEqual(row["rights_status"], "NOT_ADMITTED")
            self.assertIs(row["training_authorized"], False)
            self.assertIs(row["provenance"]["external_corpus"], False)

    def test_reject_family_leakage(self):
        rows = build_rows()
        rows[0] = dict(rows[0], split="dev")
        with self.assertRaisesRegex(ValueError, "family_id leaked"):
            validate(rows)

    def test_reject_template_and_source_leakage(self):
        baseline = build_rows()
        for key in ("template_id", "source_id"):
            rows = [dict(r) for r in baseline]
            dev_index = next(i for i, row in enumerate(rows) if row["split"] == "dev")
            rows[dev_index][key] = rows[0][key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate(rows)

    def test_reject_duplicate_prompt_and_rights_change(self):
        baseline = build_rows()
        for key, value, message in (
            ("prompt", baseline[0]["prompt"], "duplicate prompt"),
            ("rights_status", "ADMITTED", "rights status"),
            ("training_authorized", True, "rights status"),
        ):
            rows = [dict(r) for r in baseline]
            rows[1][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                validate(rows)

    def test_reject_noncanonical_answer_identity_and_generator(self):
        baseline = build_rows()
        changes = (
            ("answer", "incorrect"),
            ("case_id", "forged-case"),
            ("id", "forged-id"),
            ("variant", 99),
            ("provenance", {"kind": "source_authored", "generator": "forged",
                            "external_corpus": False}),
        )
        for key, value in changes:
            rows = [dict(row) for row in baseline]
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(rows)

    def test_existing_dangling_or_live_symlink_cannot_escape(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            output = root / "output"
            output.mkdir()
            outside = root / "external-private.jsonl"
            link = output / "train.jsonl"
            for existing in (False, True):
                if existing:
                    outside.write_bytes(b"private sentinel")
                link.symlink_to(outside)
                with self.subTest(existing=existing), self.assertRaises(FileExistsError):
                    write_candidate(output)
                self.assertEqual(outside.exists(), existing)
                if existing:
                    self.assertEqual(outside.read_bytes(), b"private sentinel")
                self.assertEqual(sorted(p.name for p in output.iterdir()), ["train.jsonl"])
                link.unlink()

    def test_refuse_symlink_as_output_directory(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            target = root / "elsewhere"
            target.mkdir()
            alias = root / "output"
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaises(OSError):
                write_candidate(alias)
            self.assertEqual(list(target.iterdir()), [])

    def test_staging_failure_leaves_no_candidate_or_temp_files(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "candidate"
            real_fsync = original_curriculum.os.fsync
            calls = 0

            def fail_second_fsync(fd):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected dev staging failure")
                return real_fsync(fd)

            with mock.patch.object(original_curriculum.os, "fsync", side_effect=fail_second_fsync):
                with self.assertRaisesRegex(OSError, "injected dev staging failure"):
                    write_candidate(output)
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(write_candidate(output)["files"]["train.jsonl"]["rows"], 432)

    def test_second_publication_link_failure_rolls_back(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "candidate"
            real_link = original_curriculum.os.link
            calls = 0

            def fail_second_link(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected dev publication failure")
                return real_link(*args, **kwargs)

            with mock.patch.object(original_curriculum.os, "link", side_effect=fail_second_link):
                with self.assertRaisesRegex(OSError, "injected dev publication failure"):
                    write_candidate(output)
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(write_candidate(output)["files"]["dev.jsonl"]["rows"], 144)

    def test_deterministic_files_and_manifest_hashes(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            one, two = Path(first), Path(second)
            self.assertEqual(write_candidate(one), write_candidate(two))
            for name in ("train.jsonl", "dev.jsonl", "manifest.json"):
                self.assertEqual((one / name).read_bytes(), (two / name).read_bytes())
            manifest = json.loads((one / "manifest.json").read_text())
            self.assertEqual(manifest["generator_source_sha256"],
                             hashlib.sha256(Path(original_curriculum.__file__).read_bytes()).hexdigest())
            self.assertTrue(manifest["archetype_overlap"])
            self.assertIn("NOT unseen-archetype transfer", manifest["evaluation_scope"])
            for name in ("train.jsonl", "dev.jsonl"):
                payload = (one / name).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), manifest["files"][name]["sha256"])
                self.assertEqual(len(payload.splitlines()), manifest["files"][name]["rows"])
            with self.assertRaises(FileExistsError):
                write_candidate(one)


if __name__ == "__main__":
    unittest.main()
