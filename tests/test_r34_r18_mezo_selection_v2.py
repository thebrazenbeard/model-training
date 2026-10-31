"""Public CPU regressions: frozen V2 selected IDs must gate R18 MeZO preflight."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from trainingday.admission import SCHEMA as PRODUCTION_ADMISSION_SCHEMA, SELECTED_ID_V2_SCHEMA
from trainingday.r34_r18_mezo_selection_v2 import (
    FrozenSelectionV2Error, make_frozen_selection_v2,
    verify_frozen_selection_v2,
)


def _rows(ids):
    return [{"id": case_id, "prompt": f"Public toy prompt {index}", "answer": str(index)}
            for index, case_id in enumerate(ids)]


class _ImpostorString:
    """A non-string that passes ordinary string equality and set membership."""

    def __init__(self, expected):
        self.expected = expected

    def __hash__(self):
        return hash(self.expected)

    def __eq__(self, other):
        return other == self.expected


class FrozenSelectionV2Tests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)
        self.path = self.directory / "frozen-v2.json"
        self.plan_sha256 = "1" * 64
        self.rows = _rows(["a\nb", "c"])
        self.summary = {
            "status": "ADMITTED_LABYRINTH_TRAIN_ONLY_NO_WEIGHT_UPDATE",
            "manifest_sha256": "2" * 64,
            "selected_id_sha256": hashlib.sha256(b"a\nb\nc").hexdigest(),
            "selected_count": 2,
        }
        binding = make_frozen_selection_v2(
            plan_sha256=self.plan_sha256, summary=self.summary,
            selected_rows=self.rows)
        self.path.write_text(json.dumps(binding), encoding="utf-8")

    def verify(self, rows=None, summary=None, plan_sha=None, path=None):
        return verify_frozen_selection_v2(
            binding_path=path or self.path, plan_sha256=plan_sha or self.plan_sha256,
            summary=self.summary if summary is None else summary,
            selected_rows=self.rows if rows is None else rows,
        )

    def test_positive_source_binding_only_does_not_authorize_training(self):
        evidence = self.verify()
        self.assertEqual(evidence["selected_id_v2_sha256"],
                         json.loads(self.path.read_text())["selected_id_v2_sha256"])
        self.assertEqual(evidence["status"], "SOURCE_ONLY_V2_SELECTION_MATCH")
        self.assertFalse(evidence["independent_custody_verified"])
        self.assertFalse(evidence["neural_training_authorized"])

    def test_legacy_collision_has_distinct_typed_v2_and_is_rejected(self):
        changed = _rows(["a", "b\nc"])
        self.assertEqual(self.summary["selected_id_sha256"],
                         hashlib.sha256("\n".join(r["id"] for r in changed).encode()).hexdigest())
        with self.assertRaisesRegex(FrozenSelectionV2Error, "V2 selected IDs"):
            self.verify(rows=changed)

    def test_legacy_only_missing_sidecar_fails_closed(self):
        with self.assertRaisesRegex(FrozenSelectionV2Error, "frozen V2"):
            self.verify(path=self.directory / "absent.json")

    def test_changed_plan_manifest_count_and_status_fail(self):
        with self.assertRaisesRegex(FrozenSelectionV2Error, "plan"):
            self.verify(plan_sha="3" * 64)
        for field, replacement in (("manifest_sha256", "f" * 64),
                                   ("selected_count", 3),
                                   ("status", "UNVERIFIED")):
            changed = {**self.summary, field: replacement}
            with self.subTest(field=field):
                with self.assertRaises(FrozenSelectionV2Error):
                    self.verify(summary=changed)

    def test_duplicate_ids_and_holdout_shaped_inputs_rejected(self):
        for rows in (_rows(["a", "a"]),
                     [{"id": "a", "evaluation_holdout": True, "prompt": "x", "answer": "0"},
                      {"id": "c", "prompt": "y", "answer": "1"}]):
            with self.subTest(rows=rows):
                with self.assertRaises(FrozenSelectionV2Error):
                    self.verify(rows=rows)

    def test_labyrinth_rows_reject_every_present_holdout_marker(self):
        # Labyrinth replay rows do not carry production admission split fields.
        # Even False and integer 1 are foreign, untrusted shape extensions.
        for marker in (False, 0, 1, None, "false"):
            with self.subTest(marker=marker):
                rows = [dict(row) for row in self.rows]
                rows[0]["evaluation_holdout"] = marker
                with self.assertRaisesRegex(FrozenSelectionV2Error, "holdout"):
                    self.verify(rows=rows)

    def test_duplicate_json_keys_and_symlink_fail_closed(self):
        self.path.write_text('{"schema":"X","schema":"Y"}', encoding="utf-8")
        with self.assertRaises(FrozenSelectionV2Error):
            self.verify()
        link = self.directory / "link.json"
        try:
            link.symlink_to(self.path)
        except (NotImplementedError, OSError):
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(FrozenSelectionV2Error, "frozen V2"):
            self.verify(path=link)

    def test_boolean_selected_count_in_frozen_sidecar_is_not_one(self):
        one = _rows(["single"])
        summary = {**self.summary, "selected_count": 1,
                   "selected_id_sha256": hashlib.sha256(b"single").hexdigest()}
        declaration = make_frozen_selection_v2(
            plan_sha256=self.plan_sha256, summary=summary, selected_rows=one)
        declaration["selected_count"] = True  # True == 1 must not pass as an integer
        self.path.write_text(json.dumps(declaration), encoding="utf-8")
        with self.assertRaises(FrozenSelectionV2Error):
            self.verify(rows=one, summary=summary)

    def test_production_receipt_v2_must_match_selected_row_bytes(self):
        rows = [dict(row, split="train", training_use="ADMITTED",
                     evaluation_holdout=False) for row in self.rows]
        summary = {**self.summary,
                   "schema": PRODUCTION_ADMISSION_SCHEMA,
                   "status": "ADMITTED_SOURCE_ONLY_NOT_TRAINED",
                   "selected_id_v2_schema": SELECTED_ID_V2_SCHEMA,
                   "selected_id_v2_sha256": "f" * 64,
                   "selected_id_sha256_status": "LEGACY_DELIMITER_AMBIGUOUS_DO_NOT_USE_ALONE"}
        with self.assertRaisesRegex(FrozenSelectionV2Error, "production V2"):
            make_frozen_selection_v2(plan_sha256=self.plan_sha256,
                                     summary=summary, selected_rows=rows)

    def test_exact_production_v2_receipt_matches_same_admitted_corpus(self):
        rows = [dict(row, split="train", training_use="ADMITTED",
                     evaluation_holdout=False) for row in self.rows]
        summary = {**self.summary,
                   "schema": PRODUCTION_ADMISSION_SCHEMA,
                   "status": "ADMITTED_SOURCE_ONLY_NOT_TRAINED",
                   "selected_id_v2_schema": SELECTED_ID_V2_SCHEMA,
                   "selected_id_v2_sha256": hashlib.sha256(
                       SELECTED_ID_V2_SCHEMA.encode("ascii") + b"\0" +
                       json.dumps([row["id"] for row in rows], ensure_ascii=False,
                                  separators=(",", ":")).encode("utf-8")).hexdigest(),
                   "selected_id_sha256_status": "LEGACY_DELIMITER_AMBIGUOUS_DO_NOT_USE_ALONE"}
        binding = make_frozen_selection_v2(
            plan_sha256=self.plan_sha256, summary=summary, selected_rows=rows)
        self.path.write_text(json.dumps(binding), encoding="utf-8")
        self.assertEqual(self.verify(rows=rows, summary=summary)["status"],
                         "SOURCE_ONLY_V2_SELECTION_MATCH")

        # The same V2 sidecar and IDs must not sanitize malformed production
        # rows: an absent marker, numeric false/true, and truthy string all
        # change the admitted source shape without changing either ID digest.
        for bad_marker in (None, 0, 1, "false", True):
            with self.subTest(production_marker=bad_marker):
                malformed = [dict(row) for row in rows]
                if bad_marker is None:
                    del malformed[0]["evaluation_holdout"]
                else:
                    malformed[0]["evaluation_holdout"] = bad_marker
                with self.assertRaisesRegex(FrozenSelectionV2Error, "holdout"):
                    self.verify(rows=malformed, summary=summary)

    def test_impostor_status_cannot_pass_frozen_verifier(self):
        # With an unchanged sidecar and identical selected IDs, a crafted
        # hash/eq object must not stand in for an authenticated source status.
        changed = {**self.summary,
                   "status": _ImpostorString(self.summary["status"])}
        with self.assertRaises(FrozenSelectionV2Error):
            self.verify(summary=changed)

        # A Python dict with an attacker-defined key can also impersonate the
        # string key that the verifier looks up. The receipt is JSON-shaped:
        # no custom Python keys should be accepted before field lookup.
        keyed = dict(self.summary)
        value = keyed.pop("status")
        keyed[_ImpostorString("status")] = value
        with self.assertRaises(FrozenSelectionV2Error):
            self.verify(summary=keyed)

    def test_impostor_production_row_and_schema_fields_cannot_pass(self):
        rows = [dict(row, split="train", training_use="ADMITTED",
                     evaluation_holdout=False) for row in self.rows]
        ids = [row["id"] for row in rows]
        summary = {**self.summary,
                   "schema": PRODUCTION_ADMISSION_SCHEMA,
                   "status": "ADMITTED_SOURCE_ONLY_NOT_TRAINED",
                   "selected_id_v2_schema": SELECTED_ID_V2_SCHEMA,
                   "selected_id_v2_sha256": hashlib.sha256(
                       SELECTED_ID_V2_SCHEMA.encode("ascii") + b"\0" +
                       json.dumps(ids, ensure_ascii=False,
                                  separators=(",", ":")).encode("utf-8")).hexdigest(),
                   "selected_id_sha256_status": "LEGACY_DELIMITER_AMBIGUOUS_DO_NOT_USE_ALONE"}
        self.path.write_text(json.dumps(make_frozen_selection_v2(
            plan_sha256=self.plan_sha256, summary=summary,
            selected_rows=rows)), encoding="utf-8")
        self.assertEqual(self.verify(rows=rows, summary=summary)["status"],
                         "SOURCE_ONLY_V2_SELECTION_MATCH")

        for field, expected in (("split", "train"),
                                ("training_use", "ADMITTED")):
            with self.subTest(row_field=field):
                malformed = [dict(row) for row in rows]
                malformed[0][field] = _ImpostorString(expected)
                with self.assertRaises(FrozenSelectionV2Error):
                    self.verify(rows=malformed, summary=summary)

                forged_key = [dict(row) for row in rows]
                value = forged_key[0].pop(field)
                forged_key[0][_ImpostorString(field)] = value
                with self.assertRaises(FrozenSelectionV2Error):
                    self.verify(rows=forged_key, summary=summary)

        for field in ("schema", "selected_id_v2_schema",
                      "selected_id_v2_sha256", "selected_id_sha256_status"):
            with self.subTest(summary_field=field):
                malformed = {**summary, field: _ImpostorString(summary[field])}
                with self.assertRaises(FrozenSelectionV2Error):
                    self.verify(rows=rows, summary=malformed)


if __name__ == "__main__":
    unittest.main()
