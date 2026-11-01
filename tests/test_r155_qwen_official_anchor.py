"""R155 exact upstream digest anchors: syntactic claims never grant host trust."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from trainingday.r155_qwen_official_anchor import (
    OFFICIAL_MODEL_ID, OFFICIAL_REVISION, OFFICIAL_SHA256,
    OfficialAnchorError, assess_r140_report, audit_offline_against_official,
)


def _claimed_r140() -> dict:
    return {
        "schema": "VERA_TRAININGDAY_R140_QWEN_OFFLINE_SNAPSHOT_ATTESTATION_V1",
        "status": "SOURCE_SNAPSHOT_INVENTORY_VALID_RUNTIME_HOLD",
        "evidence_class": "LOCAL_SOURCE_AND_FILE_INVENTORY_ONLY",
        "model_id": OFFICIAL_MODEL_ID,
        "revision": OFFICIAL_REVISION,
        "observed_file_sha256": dict(OFFICIAL_SHA256),
        "safetensors_shard_count": 2,
        "safetensors_tensor_count": 426,
        "physical_tokenizer_verified": False,
        "runtime_loader_verified": False,
        "actual_model_tensors_materialized": False,
        "finite_forward_loss_verified": False,
        "external_pin_provenance_verified": False,
        "host_immutable_custody_verified": False,
        "exclusive_cuda_lease_verified": False,
        "authorizes_training": False,
    }


class OfficialQwenAnchorTests(unittest.TestCase):
    def test_matching_report_is_still_only_a_non_authorizing_claim(self) -> None:
        result = assess_r140_report(_claimed_r140())
        self.assertEqual(result["status"], "UPSTREAM_DIGEST_CLAIM_MATCH_HOST_HOLD")
        self.assertEqual(result["model_id"], OFFICIAL_MODEL_ID)
        self.assertEqual(result["revision"], OFFICIAL_REVISION)
        self.assertEqual(set(result["public_reference_sha256"]), set(OFFICIAL_SHA256))
        for name in ("authenticates_host_snapshot", "authenticates_weight_loader",
                     "authenticates_tokenizer_runtime", "authorizes_training",
                     "authorizes_gguf_release", "authorizes_host_execution"):
            self.assertIs(result[name], False, name)

    def test_corrupt_or_rehashed_self_consistent_weight_claim_rejected(self) -> None:
        for filename in OFFICIAL_SHA256:
            with self.subTest(filename=filename):
                report = _claimed_r140()
                report["observed_file_sha256"][filename] = "f" * 64
                with self.assertRaisesRegex(OfficialAnchorError, "published upstream"):
                    assess_r140_report(report)

    def test_missing_or_mutated_public_pin_rejected(self) -> None:
        for filename in OFFICIAL_SHA256:
            with self.subTest(filename=filename):
                report = _claimed_r140()
                del report["observed_file_sha256"][filename]
                with self.assertRaises(OfficialAnchorError):
                    assess_r140_report(report)

    def test_wrong_model_revision_schema_and_evidence_class_rejected(self) -> None:
        for field, altered in (("model_id", "Qwen/Qwen3.5-4B-GGUF"),
                               ("revision", "a" * 40),
                               ("schema", "R140_FAKED"),
                               ("status", "QUALIFIED"),
                               ("evidence_class", "HOST_ATTESTED"),
                               ("safetensors_shard_count", True),
                               ("safetensors_tensor_count", 0)):
            with self.subTest(field=field):
                report = _claimed_r140()
                report[field] = altered
                with self.assertRaises(OfficialAnchorError):
                    assess_r140_report(report)

    def test_claimed_host_custody_or_training_yes_is_refused(self) -> None:
        for field in ("physical_tokenizer_verified", "runtime_loader_verified",
                      "actual_model_tensors_materialized", "finite_forward_loss_verified",
                      "external_pin_provenance_verified", "host_immutable_custody_verified",
                      "exclusive_cuda_lease_verified", "authorizes_training"):
            with self.subTest(field=field):
                report = _claimed_r140()
                report[field] = True
                with self.assertRaisesRegex(OfficialAnchorError, "unauthenticated authority"):
                    assess_r140_report(report)

    def test_wrapper_requires_real_r140_audit_then_stays_hold(self) -> None:
        fake = _claimed_r140()
        with patch("trainingday.r155_qwen_official_anchor.audit_snapshot", return_value=fake) as audit:
            result = audit_offline_against_official(
                Path("/nonexistent/base.json"), expected_binding_sha256="a" * 64,
                expected_config_sha256="b" * 64,
                expected_tokenizer_sha256="c" * 64,
                source_root=Path("/nonexistent"),
            )
        audit.assert_called_once()
        self.assertEqual(result["status"], "UPSTREAM_DIGEST_CLAIM_MATCH_HOST_HOLD")
        self.assertIs(result["authorizes_training"], False)

    def test_non_mapping_or_malformed_inventory_rejected(self) -> None:
        for bad in (None, [], True, "x", {}):
            with self.subTest(value=str(bad)):
                with self.assertRaises(OfficialAnchorError):
                    assess_r140_report(bad)
        report = _claimed_r140()
        report["observed_file_sha256"] = {"tokenizer.json": "a" * 64}
        with self.assertRaises(OfficialAnchorError):
            assess_r140_report(report)


if __name__ == "__main__":
    unittest.main()
