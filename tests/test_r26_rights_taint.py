"""Public/synthetic R26 revocation-to-derived-artifact HOLD regressions."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import unittest
from unittest.mock import patch

from trainingday.r26_rights_taint import RightsTaintError, assess_revocation


def digest(text: str) -> str:
    return sha256(text.encode("ascii")).hexdigest()


def synthetic_graph() -> list[dict]:
    return [
        {"id": "shard-a", "kind": "TRAIN_SHARD", "sha256": digest("a"),
         "parents": [], "license_sha256": digest("license-a"), "rights": "DECLARED_ADMITTED"},
        {"id": "shard-b", "kind": "TRAIN_SHARD", "sha256": digest("b"),
         "parents": [], "license_sha256": digest("license-b"), "rights": "DECLARED_ADMITTED"},
        {"id": "adapter-a", "kind": "ADAPTER", "sha256": digest("adapter"),
         "parents": ["shard-a"]},
        {"id": "checkpoint-1", "kind": "CHECKPOINT", "sha256": digest("checkpoint"),
         "parents": ["adapter-a", "shard-b"]},
        {"id": "eval-1", "kind": "EVAL_RECEIPT", "sha256": digest("eval"),
         "parents": ["checkpoint-1"]},
        {"id": "candidate-1", "kind": "PROMOTION_CANDIDATE", "sha256": digest("candidate"),
         "parents": ["checkpoint-1", "eval-1"]},
    ]


def event(action: str, source: str = "shard-a") -> dict:
    node = next(n for n in synthetic_graph() if n["id"] == source)
    out = {"id": "event-1", "source_id": source, "source_sha256": node["sha256"],
           "action": action, "evidence_sha256": digest("authority-unverified")}
    if action == "LICENSE_CHANGED":
        out["prior_license_sha256"] = node["license_sha256"]
        out["replacement_license_sha256"] = digest("new-license")
    return out


class RightsTaintTests(unittest.TestCase):
    def test_clean_declared_lineage_never_authorizes_release(self):
        assessment = assess_revocation(synthetic_graph(), [])
        self.assertEqual(assessment["release_gate"], "HOLD")
        self.assertFalse(assessment["promotion_allowed"])
        self.assertFalse(assessment["unlearning_verified"])
        self.assertEqual(assessment["impacted_artifacts"], [])
        self.assertTrue(all(v == "DECLARED_CLEAN_ONLY" for v in assessment["status_by_artifact"].values()))

    def test_revoked_training_shard_taints_all_descendants(self):
        assessment = assess_revocation(synthetic_graph(), [event("REVOKED")])
        self.assertEqual(set(assessment["impacted_artifacts"]),
                         {"shard-a", "adapter-a", "checkpoint-1", "eval-1", "candidate-1"})
        self.assertEqual(assessment["status_by_artifact"]["shard-b"], "DECLARED_CLEAN_ONLY")
        self.assertTrue(all(assessment["status_by_artifact"][x] == "REVOKED_LINEAGE"
                            for x in assessment["impacted_artifacts"]))
        self.assertFalse(assessment["promotion_allowed"])

    def test_license_change_requires_review_and_propagates(self):
        assessment = assess_revocation(synthetic_graph(), [event("LICENSE_CHANGED")])
        self.assertEqual(assessment["status_by_artifact"]["candidate-1"], "LICENSE_REVIEW_REQUIRED")

    def test_privacy_flag_propagates_and_never_claims_unlearning(self):
        assessment = assess_revocation(synthetic_graph(), [event("PRIVACY_TAINT")])
        self.assertEqual(assessment["status_by_artifact"]["candidate-1"], "PRIVACY_TAINTED_LINEAGE")
        self.assertFalse(assessment["unlearning_verified"])
        self.assertEqual(assessment["evidence_retention_action"], "NO_DELETION_ATTEMPTED")

    def test_ambiguous_rights_fail_closed(self):
        nodes = synthetic_graph()
        nodes[0]["rights"] = "UNKNOWN"
        assessment = assess_revocation(nodes, [])
        self.assertEqual(assessment["status_by_artifact"]["candidate-1"], "RIGHTS_REVIEW_REQUIRED")
        self.assertFalse(assessment["promotion_allowed"])

    def test_revocation_takes_precedence_over_license_and_privacy_declaration(self):
        events = [event("REVOKED"), event("PRIVACY_TAINT"), event("LICENSE_CHANGED")]
        for i, item in enumerate(events):
            item["id"] = f"event-{i}"
        self.assertEqual(assess_revocation(synthetic_graph(), events)["status_by_artifact"]["candidate-1"],
                         "REVOKED_LINEAGE")

    def test_reordered_graph_events_produce_same_commitments(self):
        a = synthetic_graph()
        e = [event("REVOKED"), event("PRIVACY_TAINT")]
        e[1]["id"] = "event-2"
        x = assess_revocation(a, e)
        y = assess_revocation(list(reversed(a)), list(reversed(e)))
        self.assertEqual(x["assessment_sha256"], y["assessment_sha256"])
        self.assertEqual(x["status_by_artifact"], y["status_by_artifact"])

    def test_tampered_event_source_hash_rejected(self):
        e = event("REVOKED")
        e["source_sha256"] = digest("other")
        with self.assertRaises(RightsTaintError):
            assess_revocation(synthetic_graph(), [e])

    def test_forged_license_baseline_rejected(self):
        e = event("LICENSE_CHANGED")
        e["prior_license_sha256"] = digest("fake")
        with self.assertRaises(RightsTaintError):
            assess_revocation(synthetic_graph(), [e])

    def test_no_change_license_event_rejected(self):
        e = event("LICENSE_CHANGED")
        e["replacement_license_sha256"] = e["prior_license_sha256"]
        with self.assertRaises(RightsTaintError):
            assess_revocation(synthetic_graph(), [e])

    def test_unknown_source_rejected(self):
        e = event("REVOKED")
        e["source_id"] = "missing-shard"
        with self.assertRaises(RightsTaintError):
            assess_revocation(synthetic_graph(), [e])

    def test_duplicate_source_or_event_rejected(self):
        nodes = synthetic_graph()
        with self.assertRaises(RightsTaintError):
            assess_revocation(nodes + [deepcopy(nodes[0])], [])
        e = event("REVOKED")
        with self.assertRaises(RightsTaintError):
            assess_revocation(nodes, [e, deepcopy(e)])

    def test_dangling_dependency_and_cycle_rejected(self):
        nodes = synthetic_graph()
        nodes[-1]["parents"].append("nonexistent")
        with self.assertRaises(RightsTaintError):
            assess_revocation(nodes, [])
        nodes = synthetic_graph()
        nodes[2]["parents"] = ["checkpoint-1"]
        with self.assertRaises(RightsTaintError):
            assess_revocation(nodes, [])

    def test_orphan_derived_checkpoint_rejected(self):
        nodes = synthetic_graph()
        nodes[3]["parents"] = []
        with self.assertRaises(RightsTaintError):
            assess_revocation(nodes, [])

    def test_promotion_cannot_reuse_evaluation_from_different_model(self):
        nodes = synthetic_graph()
        nodes.append({"id": "adapter-b", "kind": "ADAPTER", "sha256": digest("adapter-b"),
                      "parents": ["shard-b"]})
        nodes[-2]["parents"] = ["adapter-b", "eval-1"]
        with self.assertRaises(RightsTaintError):
            assess_revocation(nodes, [])

    def test_promotion_cannot_substitute_ancestor_adapter_for_scored_checkpoint(self):
        nodes = synthetic_graph()
        # eval-1 directly scored checkpoint-1. adapter-a is its ancestor, not
        # the exact evaluated model artifact; a hash-consistent ancestry
        # declaration must not be interpreted as an evaluation of adapter-a.
        nodes[-1]["parents"] = ["adapter-a", "eval-1"]
        with self.assertRaisesRegex(RightsTaintError, "exact model identity"):
            assess_revocation(nodes, [])

    def test_promotion_cannot_add_unscored_ancestor_as_second_model(self):
        nodes = synthetic_graph()
        nodes[-1]["parents"] = ["adapter-a", "checkpoint-1", "eval-1"]
        with self.assertRaisesRegex(RightsTaintError, "exact model identity"):
            assess_revocation(nodes, [])

    def test_derived_artifact_cannot_pose_as_training_source(self):
        e = event("REVOKED")
        e["source_id"] = "adapter-a"
        e["source_sha256"] = digest("adapter")
        with self.assertRaises(RightsTaintError):
            assess_revocation(synthetic_graph(), [e])

    def test_unknown_action_and_extra_authority_field_rejected(self):
        e = event("REVOKED")
        e["action"] = "UNLEARNED"
        with self.assertRaises(RightsTaintError):
            assess_revocation(synthetic_graph(), [e])
        e = event("REVOKED")
        e["release_authorized"] = True
        with self.assertRaises(RightsTaintError):
            assess_revocation(synthetic_graph(), [e])

    def test_unknown_sha_or_reused_node_digest_rejected(self):
        nodes = synthetic_graph()
        nodes[0]["sha256"] = "z" * 64
        with self.assertRaises(RightsTaintError):
            assess_revocation(nodes, [])
        nodes = synthetic_graph()
        nodes[1]["sha256"] = nodes[0]["sha256"]
        with self.assertRaises(RightsTaintError):
            assess_revocation(nodes, [])

    def test_unhashable_invalid_fields_fail_closed_with_domain_error(self):
        nodes = synthetic_graph()
        nodes[0]["kind"] = []
        with self.assertRaises(RightsTaintError):
            assess_revocation(nodes, [])
        nodes = synthetic_graph()
        nodes[0]["rights"] = {"untrusted": "claim"}
        with self.assertRaises(RightsTaintError):
            assess_revocation(nodes, [])
        e = event("REVOKED")
        e["action"] = []
        with self.assertRaises(RightsTaintError):
            assess_revocation(synthetic_graph(), [e])

    def test_bounded_input_rejects_before_canonicalization(self):
        with patch("trainingday.r26_rights_taint._canonical", side_effect=AssertionError("serialized too soon")):
            with self.assertRaises(RightsTaintError):
                assess_revocation(synthetic_graph() * 50, [])

    def test_output_is_detached_from_mutated_input(self):
        nodes, events = synthetic_graph(), [event("REVOKED")]
        x = assess_revocation(nodes, events)
        nodes[0]["rights"] = "UNKNOWN"
        events[0]["action"] = "LICENSE_CHANGED"
        self.assertEqual(x["status_by_artifact"]["candidate-1"], "REVOKED_LINEAGE")


if __name__ == "__main__":
    unittest.main()
