"""Synthetic-only checks for the offline historical owner-evidence verifier."""

import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from successor.experiments import owner_historical_evidence_preflight as evidence


ROOT = Path(__file__).resolve().parents[1]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jsonl(*rows: dict) -> bytes:
    return b"".join(json.dumps(row).encode("utf8") + b"\n" for row in rows)


class OwnerHistoricalEvidencePreflightTests(unittest.TestCase):
    def test_empty_inventory_cannot_become_historical_pass(self):
        with self.assertRaises(ValueError):
            evidence.evaluate(())

    def test_missing_owner_file_is_blocked_and_path_is_not_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            secret_path = Path(directory) / "private-protected-eval-DO_NOT_EXPORT.jsonl"
            pin = evidence.EvidencePin("R9_CORPUS", secret_path, "0" * 64, rows=50000)
            report = evidence.evaluate((pin,))
            self.assertEqual(report["status"], "HISTORICAL_BLOCKED_MISSING_EVIDENCE")
            self.assertEqual(report["checks"], [
                {"id": "R9_CORPUS", "status": "BLOCKED_MISSING_OWNER_EVIDENCE"}
            ])
            self.assertEqual(evidence.exit_code(report), 42)
            self.assertNotIn(str(secret_path), json.dumps(report))
            self.assertFalse(report["training_qualified"])

    def test_hash_and_row_count_are_both_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.jsonl"
            contents = jsonl({"prompt": "synthetic-one"}, {"prompt": "synthetic-two"})
            path.write_bytes(contents)
            pin = evidence.EvidencePin("R10_CORPUS", path, sha(contents), rows=2)
            self.assertEqual(evidence.evaluate((pin,))["status"], "HISTORICAL_PASS")
            wrong_count = evidence.EvidencePin("R10_CORPUS", path, sha(contents), rows=3)
            report = evidence.evaluate((wrong_count,))
            self.assertEqual(report["checks"][0]["status"], "FAIL_ROW_COUNT")
            path.write_bytes(contents + b"mutation\n")
            report = evidence.evaluate((pin,))
            self.assertEqual(report["checks"][0]["status"], "FAIL_DIGEST")
            self.assertEqual(evidence.exit_code(report), 1)

    def test_mutation_between_digest_and_row_check_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutable.jsonl"
            contents = jsonl({"prompt": "synthetic"})
            path.write_bytes(contents)
            pin = evidence.EvidencePin("R9_CORPUS", path, sha(contents), rows=1)
            original = evidence._count_and_validate_rows

            def mutate_after_count(candidate, binding):
                result = original(candidate, binding)
                candidate.write_bytes(contents + b"other\n")
                return result

            with patch.object(evidence, "_count_and_validate_rows",
                              side_effect=mutate_after_count):
                report = evidence.evaluate((pin,))
            self.assertEqual(report["checks"][0]["status"],
                             "FAIL_MUTABLE_EVIDENCE")

    def test_canonical_receipt_signature_and_fields_are_not_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "DEV_TRAINING_COMPLETE.json"
            unsigned = {"status": "DEVELOPMENT_OPTIMIZER_STEP_COMPLETE",
                        "cumulative_optimizer_steps": 36,
                        "weight_digest_changed": True}
            digest = sha(json.dumps(unsigned, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode("utf8"))
            receipt = dict(unsigned, receipt_sha256=digest)
            path.write_text(json.dumps(receipt), encoding="utf8")
            pin = evidence.EvidencePin(
                "R10_RECEIPT", path, receipt_self_sha256=digest,
                fields=(("status", "DEVELOPMENT_OPTIMIZER_STEP_COMPLETE"),
                        ("cumulative_optimizer_steps", 36),
                        ("weight_digest_changed", True)),
            )
            self.assertEqual(evidence.evaluate((pin,))["status"], "HISTORICAL_PASS")
            receipt["cumulative_optimizer_steps"] = 35
            path.write_text(json.dumps(receipt), encoding="utf8")
            report = evidence.evaluate((pin,))
            self.assertEqual(report["checks"][0]["status"], "FAIL_RECEIPT")
            self.assertNotIn("DEV_TRAINING_COMPLETE.json", json.dumps(report))

    def test_r10_git_lf_is_not_historical_byte_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "r10.json"
            original = b"{\r\n\"answer\":42\r\n}\r\n"
            pin = evidence.EvidencePin(
                "R10_SPEC", path, sha(original), crlf_historical=True,
            )
            path.write_bytes(original.replace(b"\r\n", b"\n"))
            report = evidence.evaluate((pin,))
            self.assertEqual(report["checks"][0]["status"],
                             "BLOCKED_CRLF_BYTE_PROVENANCE")
            path.write_bytes(original)
            self.assertEqual(evidence.evaluate((pin,))["status"], "HISTORICAL_PASS")

    def test_protected_heldout_is_never_json_parsed_or_semantically_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            heldout = Path(directory) / "heldout.jsonl"
            # Deliberately malformed, synthetic non-JSON: exact owner custody
            # is bytes/size/count ONLY, no data-rights permit to parse eval.
            opaque = b"not-json-protected-row-A\nnot-json-protected-row-B\n"
            heldout.write_bytes(opaque)
            pin = evidence.EvidencePin(
                "HELDOUT", heldout, sha(opaque), rows=2, kind="heldout"
            )
            with patch.object(evidence.json, "loads",
                              side_effect=AssertionError("protected decode forbidden")):
                report = evidence.evaluate((pin,))
            self.assertEqual(report["status"], "HISTORICAL_PASS")
            self.assertEqual(report["checks"], [{"id": "HELDOUT", "status": "PASS"}])
            self.assertIs(report["protected_plaintext_opened"], False)
            self.assertEqual(report["protected_semantic_qualification"],
                             "NOT_AUTHORIZED_NOT_RUN")
            self.assertNotIn("not-json-protected", json.dumps(report))

            wrong_rows = evidence.EvidencePin(
                "HELDOUT", heldout, sha(opaque), rows=3, kind="heldout"
            )
            row_report = evidence.evaluate((wrong_rows,))
            self.assertEqual(row_report["checks"][0]["status"], "FAIL_ROW_COUNT")
            self.assertIs(row_report["protected_plaintext_opened"], False)

            heldout.write_bytes(opaque + b"altered")
            mismatch = evidence.evaluate((pin,))
            self.assertEqual(mismatch["checks"][0]["status"], "FAIL_DIGEST")
            self.assertIs(mismatch["protected_plaintext_opened"], False)

    def test_heldout_byte_pin_does_not_claim_prompt_overlap_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            heldout = Path(directory) / "heldout.jsonl"
            source = Path(directory) / "source.jsonl"
            held = jsonl({"user": "same synthetic prompt", "training_use": "UNKNOWN"})
            src = jsonl({"id": "s1", "user": "same synthetic prompt",
                         "privacy": "NON_PERSONAL", "training_use": "targeted_behavioral_sft",
                         "source_class": "NEW_ORIGINAL_BEHAVIOR_NOT_HELDOUT", "axis": "identity"})
            heldout.write_bytes(held)
            source.write_bytes(src)
            report = evidence.evaluate((
                evidence.EvidencePin("HELDOUT", heldout, sha(held), rows=1,
                                     kind="heldout"),
                evidence.EvidencePin("R13_SOURCE", source, sha(src), rows=1,
                                     kind="r13_source"),
            ))
            self.assertEqual(report["status"], "HISTORICAL_PASS")
            self.assertIs(report["protected_plaintext_opened"], False)
            self.assertEqual(report["protected_semantic_qualification"],
                             "NOT_AUTHORIZED_NOT_RUN")

    def test_frozen_production_inventory_excludes_training_execution(self):
        pins = evidence.historical_pins(ROOT, r9_train=None)
        labels = {pin.id for pin in pins}
        self.assertEqual(labels, {
            "R9_CORPUS", "R9_SPEC", "R9_ADAPTER", "R9_RECEIPT",
            "R10_CORPUS", "R10_SPEC", "R10_ADAPTER", "R10_RECEIPT",
            "R11_CORPUS", "R11_SPEC", "R11_ADAPTER", "R11_RECEIPT",
            "R12_CORPUS", "R12_SPEC", "R12_ADAPTER", "R12_RECEIPT",
            "HELDOUT", "R13_SOURCE", "R13_SPEC", "R13_CORPUS",
        })
        self.assertEqual(len(labels), len(pins))
        self.assertEqual(next(p.expected_sha256 for p in pins if p.id == "R10_SPEC"),
                         "8098a6f8c7102cacb956815e083f144d481cbc5bd718447b622138508076f2f6")
        self.assertEqual(len(pins), 20)

    def test_real_public_specs_and_r13_source_keep_their_exact_pins(self):
        pinned = evidence.historical_pins(ROOT, r9_train=None)
        selected = tuple(pin for pin in pinned if pin.id in {
            "R9_SPEC", "R10_SPEC", "R11_SPEC", "R12_SPEC", "R13_SPEC", "R13_SOURCE"
        })
        report = evidence.evaluate(selected)
        statuses = {entry["id"]: entry["status"] for entry in report["checks"]}
        self.assertEqual({key: status for key, status in statuses.items()
                          if key != "R10_SPEC"},
                         dict.fromkeys(("R9_SPEC", "R11_SPEC", "R12_SPEC",
                                        "R13_SPEC", "R13_SOURCE"), "PASS"))
        # Git LF is proven transport-equivalent to frozen Windows CRLF bytes,
        # but cannot count as physical byte-identical historical evidence.
        self.assertEqual(statuses["R10_SPEC"], "BLOCKED_CRLF_BYTE_PROVENANCE")
        self.assertEqual(report["status"], "HISTORICAL_BLOCKED_MISSING_EVIDENCE")

    @unittest.skipIf(os.name == "nt", "Linux-only unsupported-host assertion")
    def test_cli_refuses_to_claim_windows_owner_evidence_on_linux(self):
        with redirect_stdout(io.StringIO()) as output:
            result = evidence.main([])
        self.assertEqual(result, 42)
        self.assertIn("BLOCKED_UNSUPPORTED_HOST", output.getvalue())
        self.assertNotIn("D:\\VERA", output.getvalue())


if __name__ == "__main__":
    unittest.main()
