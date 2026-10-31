"""Public CPU regressions for a nonauthorizing R18 selection-V2 bridge."""

import hashlib
import json
import unittest
from unittest.mock import patch

from trainingday.admission import SCHEMA as ADMISSION_SCHEMA, SELECTED_ID_V2_SCHEMA
from trainingday.r18_benchmark import make_plan, selected_id_digest
import trainingday.r32_selection_v2_bridge as bridge
from trainingday.r32_selection_v2_bridge import (
    SelectionV2BridgeError, assess_v2_selection_bridge,
)


def _rows(ids):
    return [
        {"id": case_id, "split": "train", "training_use": "ADMITTED",
         "evaluation_holdout": False, "prompt": f"Public toy: compute {index} plus one",
         "answer": str(index + 1)}
        for index, case_id in enumerate(ids)
    ]


def _v2(ids):
    return hashlib.sha256(
        SELECTED_ID_V2_SCHEMA.encode("ascii") + b"\0"
        + json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _inputs(ids=("a\nb", "c")):
    rows = _rows(ids)
    plan = make_plan(
        selected_rows=rows, admission_selected_id_sha256=selected_id_digest(rows),
        source_admission_status="ADMITTED_SOURCE_ONLY_NOT_TRAINED",
        eval_index=[{"id": f"public-eval-{index}",
                     "prompt_sha256": hashlib.sha256(f"toy-eval-{index}".encode()).hexdigest()}
                    for index in range(64)],
        r16_holdout_ids=[f"historical-exclusion-{index}" for index in range(64)],
        base_revision="1" * 40, base_binding_sha256="a" * 64,
        tokenizer_binding_sha256="b" * 64,
        corpus_manifest_sha256="c" * 64,
        eval_custody_sha256="d" * 64,
        eval_source_manifest_sha256="e" * 64,
        r16_eval_custody_sha256="f" * 64,
        seed=41, training_budget_seconds=100.0, probe_budget_seconds=5.0,
    )
    summary = {
        "schema": ADMISSION_SCHEMA,
        "status": "ADMITTED_SOURCE_ONLY_NOT_TRAINED",
        "manifest_sha256": "c" * 64,
        "selected_count": len(rows),
        "selected_id_sha256": selected_id_digest(rows),
        "selected_id_v2_schema": SELECTED_ID_V2_SCHEMA,
        "selected_id_v2_sha256": _v2(list(ids)),
        "selected_id_sha256_status": "LEGACY_DELIMITER_AMBIGUOUS_DO_NOT_USE_ALONE",
    }
    return rows, summary, plan


def _assess(rows, summary, plan, pin):
    return assess_v2_selection_bridge(selected_rows=rows, admission_receipt=summary,
                                      frozen_plan=plan, frozen_v2_selection_sha256=pin)


class SelectionV2BridgeTests(unittest.TestCase):
    def test_exact_public_v2_receipt_and_frozen_pin_are_consistent_but_never_authorize(self):
        rows, summary, plan = _inputs()
        report = _assess(rows, summary, plan, summary["selected_id_v2_sha256"])
        self.assertTrue(report["synthetic_v2_source_consistent"])
        self.assertEqual(report["status"], "HOLD")
        self.assertEqual(report["blockers"], [])
        for field in ("r18_runners_v2_migrated", "protected_eval_unseal_authorized",
                      "gpu_start_authorized", "neural_training_authorized",
                      "model_promotion_authorized", "independent_custody_verified"):
            self.assertFalse(report[field], field)

    def test_legacy_digest_collision_requires_distinct_independent_frozen_v2_pin(self):
        first = _inputs(("a\nb", "c"))
        second = _inputs(("a", "b\nc"))
        self.assertEqual(first[1]["selected_id_sha256"], second[1]["selected_id_sha256"])
        self.assertNotEqual(first[1]["selected_id_v2_sha256"], second[1]["selected_id_v2_sha256"])
        result = _assess(*second, pin=first[1]["selected_id_v2_sha256"])
        self.assertIn("FROZEN_V2_SELECTION_MISMATCH", result["blockers"])
        self.assertFalse(result["synthetic_v2_source_consistent"])

    def test_legacy_only_or_missing_frozen_pin_cannot_be_upgraded(self):
        rows, summary, plan = _inputs()
        legacy = dict(summary)
        legacy.pop("selected_id_v2_schema")
        legacy.pop("selected_id_v2_sha256")
        report = _assess(rows, legacy, plan, summary["selected_id_v2_sha256"])
        self.assertIn("V2_SELECTION_RECEIPT_MISSING", report["blockers"])
        self.assertFalse(report["synthetic_v2_source_consistent"])
        no_pin = _assess(rows, summary, plan, None)
        self.assertIn("FROZEN_V2_SELECTION_PIN_MISSING", no_pin["blockers"])

    def test_wrong_manifest_legacy_hash_or_content_is_rejected(self):
        rows, summary, plan = _inputs()
        for key, value, expected in (
            ("manifest_sha256", "0" * 64, "MANIFEST_PLAN_MISMATCH"),
            ("selected_id_sha256", "0" * 64, "LEGACY_SELECTED_IDS_MISMATCH"),
            ("selected_id_v2_sha256", "0" * 64, "V2_RECEIPT_SELECTED_IDS_MISMATCH"),
        ):
            modified = dict(summary, **{key: value})
            self.assertIn(expected, _assess(rows, modified, plan,
                                            summary["selected_id_v2_sha256"])["blockers"])
        altered = [dict(row) for row in rows]
        altered[0]["answer"] = "wrong"
        self.assertIn("TRAIN_CONTENT_PLAN_MISMATCH",
                      _assess(altered, summary, plan, summary["selected_id_v2_sha256"])["blockers"])

    def test_untrusted_python_objects_cannot_run_copy_hooks_or_spoof_ids(self):
        rows, summary, plan = _inputs()
        invoked = []

        class Spoof:
            def __deepcopy__(self, memo):
                invoked.append("called")
                return "a\nb"

        rows[0]["id"] = Spoof()
        with self.assertRaises(SelectionV2BridgeError):
            _assess(rows, summary, plan, summary["selected_id_v2_sha256"])
        self.assertEqual(invoked, [])

    def test_invalid_v2_schema_does_not_silently_fallback_to_legacy(self):
        rows, summary, plan = _inputs()
        changed = dict(summary, selected_id_v2_schema="MALFORMED_V2")
        report = _assess(rows, changed, plan, summary["selected_id_v2_sha256"])
        self.assertIn("V2_SELECTION_SCHEMA_MISMATCH", report["blockers"])
        self.assertFalse(report["synthetic_v2_source_consistent"])

    def test_wrong_or_absent_production_receipt_schema_cannot_claim_v2_consistency(self):
        rows, summary, plan = _inputs()
        bad = [dict(summary, schema="NOT_PRODUCTION"), dict(summary)]
        bad[-1].pop("schema")
        for receipt in bad:
            with self.subTest(schema=receipt.get("schema")):
                result = _assess(rows, receipt, plan,
                                 summary["selected_id_v2_sha256"])
                self.assertIn("PRODUCTION_ADMISSION_SCHEMA_MISMATCH", result["blockers"])
                self.assertFalse(result["synthetic_v2_source_consistent"])
                self.assertFalse(result["gpu_start_authorized"])

    def test_mutation_of_original_bad_receipt_after_plan_validation_does_not_clean_it(self):
        rows, summary, plan = _inputs()
        original_good_v2 = summary["selected_id_v2_sha256"]
        summary["selected_id_v2_sha256"] = "0" * 64
        from trainingday.r18_benchmark import validate_plan as real_validate

        def mutate_original_after_plan(snapshot):
            summary["selected_id_v2_sha256"] = original_good_v2
            return real_validate(snapshot)

        with patch.object(bridge, "validate_plan", side_effect=mutate_original_after_plan):
            report = _assess(rows, summary, plan, original_good_v2)
        self.assertIn("V2_RECEIPT_SELECTED_IDS_MISMATCH", report["blockers"])
        self.assertFalse(report["synthetic_v2_source_consistent"])


if __name__ == "__main__":
    unittest.main()
