"""Public synthetic historical/current-corpus provenance checks; never train."""

from __future__ import annotations

from hashlib import sha256
import json
import unittest
from unittest.mock import patch

from tests.test_r27_continuous_learning_policy import sha
from tests.test_r28_rights_cycle_integration import bundle
import trainingday.r33_historical_provenance_v2 as bridge_module
from trainingday.r33_historical_provenance_v2 import (
    HistoricalProvenanceError, assess_historical_bridge_v2,
    freeze_historical_graph,
)


def fixture():
    current, events, baseline, request, consent, evidence, cycle_events = bundle()
    historical = [
        {"id": "base-1", "kind": "BASE_MODEL", "sha256": baseline.base_sha256,
         "parents": [], "license_sha256": sha("6"), "rights": "DECLARED_ADMITTED"},
        {"id": "older-corpus", "kind": "HISTORICAL_SHARD", "sha256": sha("7"),
         "parents": [], "license_sha256": sha("8"), "rights": "DECLARED_ADMITTED"},
        {"id": "previous-adapter", "kind": "HISTORICAL_ADAPTER",
         "sha256": baseline.adapter_sha256, "parents": ["base-1", "older-corpus"]},
        {"id": "previous-checkpoint", "kind": "HISTORICAL_CHECKPOINT",
         "sha256": baseline.checkpoint_sha256, "parents": ["previous-adapter"]},
        {"id": "previous-evidence", "kind": "HISTORICAL_EVIDENCE",
         "sha256": baseline.evidence_sha256, "parents": ["previous-checkpoint"]},
    ]
    frozen_history = freeze_historical_graph(historical)
    bridge = {
        "historical_graph_sha256": frozen_history,
        "previous_checkpoint_sha256": baseline.checkpoint_sha256,
        "current_shard_sha256": request.source_sha256,
        "candidate_adapter_sha256": request.proposed_adapter_sha256,
    }
    return dict(historical_nodes=historical, historical_events=[],
                frozen_historical_sha256=frozen_history, bridge_claim=bridge,
                current_nodes=current, current_rights_events=events,
                baseline=baseline, request=request, consent=consent,
                evidence=evidence, cycle_events=cycle_events, now_unix=130)


def assess(**overrides):
    kwargs = fixture()
    kwargs.update(overrides)
    return assess_historical_bridge_v2(**kwargs)


def attacker_recommit_invalid_graph(nodes):
    """Re-pin arbitrary public claims without calling the policy validator."""
    canonical_nodes = [dict(n, parents=sorted(n["parents"]))
                       for n in sorted(nodes, key=lambda n: n["id"])]
    body = {"schema": bridge_module.FROZEN_SCHEMA, "nodes": canonical_nodes}
    return sha256(json.dumps(body, sort_keys=True, ensure_ascii=True,
                             separators=(",", ":")).encode("ascii")).hexdigest()


class HistoricalBridgeTests(unittest.TestCase):
    def test_distinct_historical_and_current_corpora_bind_synthetically_without_authority(self):
        f = fixture()
        self.assertNotEqual(f["historical_nodes"][1]["sha256"],
                            f["request"].source_sha256)
        result = assess_historical_bridge_v2(**f)
        self.assertEqual(result["blockers"], [])
        self.assertTrue(result["synthetic_cross_generation_lineage_consistent"])
        self.assertFalse(result["previous_baseline_lineage_verified"])
        self.assertFalse(result["gpu_start_authorized"])
        self.assertFalse(result["protected_eval_unseal_authorized"])
        self.assertFalse(result["model_promotion_authorized"])

    def test_wrong_frozen_baseline_model_and_artifact_hashes_are_detected(self):
        for index in (0, 2, 3, 4):
            with self.subTest(index=index):
                f = fixture()
                f["historical_nodes"][index]["sha256"] = sha("9")
                f["frozen_historical_sha256"] = freeze_historical_graph(f["historical_nodes"])
                f["bridge_claim"]["historical_graph_sha256"] = f["frozen_historical_sha256"]
                result = assess_historical_bridge_v2(**f)
                self.assertIn("HISTORICAL_BASELINE_IDENTITY_MISMATCH", result["blockers"])
                self.assertFalse(result["synthetic_cross_generation_lineage_consistent"])

    def test_cross_graph_edge_cannot_swap_current_source_or_previous_checkpoint(self):
        for key in ("previous_checkpoint_sha256", "current_shard_sha256",
                    "candidate_adapter_sha256", "historical_graph_sha256"):
            with self.subTest(key=key):
                f = fixture()
                f["bridge_claim"][key] = sha("9")
                self.assertIn("CROSS_GRAPH_CLAIM_MISMATCH",
                              assess_historical_bridge_v2(**f)["blockers"])

    def test_unconnected_second_historical_shard_does_not_disappear(self):
        f = fixture()
        f["historical_nodes"].append(
            {"id": "unbound-history", "kind": "HISTORICAL_SHARD",
             "sha256": sha("9"), "parents": [], "license_sha256": sha("0"),
             "rights": "UNKNOWN"})
        f["frozen_historical_sha256"] = freeze_historical_graph(f["historical_nodes"])
        f["bridge_claim"]["historical_graph_sha256"] = f["frozen_historical_sha256"]
        result = assess_historical_bridge_v2(**f)
        self.assertIn("HISTORICAL_SHARD_UNBOUND", result["blockers"])
        self.assertIn("HISTORICAL_RIGHTS_TAINTED", result["blockers"])

    def test_checkpoint_rejects_second_sibling_adapter_parent_even_if_recommitted(self):
        f = fixture()
        f["historical_nodes"].append(
            {"id": "sibling-adapter", "kind": "HISTORICAL_ADAPTER",
             "sha256": sha("9"), "parents": ["base-1", "older-corpus"]})
        f["historical_nodes"][3]["parents"].append("sibling-adapter")
        frozen = attacker_recommit_invalid_graph(f["historical_nodes"])
        f["frozen_historical_sha256"] = frozen
        f["bridge_claim"]["historical_graph_sha256"] = frozen
        with self.assertRaises(HistoricalProvenanceError):
            assess_historical_bridge_v2(**f)

    def test_evidence_rejects_second_sibling_checkpoint_parent_even_if_recommitted(self):
        f = fixture()
        f["historical_nodes"].append(
            {"id": "sibling-checkpoint", "kind": "HISTORICAL_CHECKPOINT",
             "sha256": sha("9"), "parents": ["previous-adapter"]})
        f["historical_nodes"][4]["parents"].append("sibling-checkpoint")
        frozen = attacker_recommit_invalid_graph(f["historical_nodes"])
        f["frozen_historical_sha256"] = frozen
        f["bridge_claim"]["historical_graph_sha256"] = frozen
        with self.assertRaises(HistoricalProvenanceError):
            assess_historical_bridge_v2(**f)

    def test_previous_evidence_cannot_bind_checkpoint_only_as_indirect_ancestor(self):
        f = fixture()
        f["historical_nodes"].extend([
            {"id": "subsequent-adapter", "kind": "HISTORICAL_ADAPTER",
             "sha256": sha("9"), "parents": ["previous-checkpoint"]},
            {"id": "subsequent-checkpoint", "kind": "HISTORICAL_CHECKPOINT",
             "sha256": sha("0"), "parents": ["subsequent-adapter"]},
        ])
        f["historical_nodes"][4]["parents"] = ["subsequent-checkpoint"]
        frozen = freeze_historical_graph(f["historical_nodes"])
        f["frozen_historical_sha256"] = frozen
        f["bridge_claim"]["historical_graph_sha256"] = frozen
        result = assess_historical_bridge_v2(**f)
        self.assertIn("HISTORICAL_DIRECT_PARENT_MISMATCH", result["blockers"])
        self.assertFalse(result["synthetic_cross_generation_lineage_consistent"])

    def test_clock_is_bounded_exact_builtin_int(self):
        class IntSubclass(int):
            pass

        for bad in (IntSubclass(130), 1 << 90, True):
            with self.subTest(type=type(bad).__name__):
                with self.assertRaises(HistoricalProvenanceError):
                    assess(now_unix=bad)

    def test_historical_revocation_and_privacy_taint_propagate_to_candidate(self):
        for action in ("REVOKED", "PRIVACY_TAINT", "LICENSE_CHANGED"):
            with self.subTest(action=action):
                f = fixture()
                e = {"id": "event-one", "source_id": "older-corpus",
                     "source_sha256": sha("7"), "action": action,
                     "evidence_sha256": sha("9")}
                if action == "LICENSE_CHANGED":
                    e.update(prior_license_sha256=sha("8"),
                             replacement_license_sha256=sha("0"))
                f["historical_events"] = [e]
                result = assess_historical_bridge_v2(**f)
                self.assertIn("HISTORICAL_RIGHTS_TAINTED", result["blockers"])
                self.assertFalse(result["synthetic_cross_generation_lineage_consistent"])
                self.assertEqual(result["status"], "HOLD")

    def test_current_revocation_blocks_even_if_historical_source_clean(self):
        f = fixture()
        f["current_rights_events"] = [
            {"id": "new-revocation", "source_id": "source-1",
             "source_sha256": f["request"].source_sha256, "action": "REVOKED",
             "evidence_sha256": sha("9")},
        ]
        self.assertIn("CURRENT_CYCLE_SOURCE_BLOCKED",
                      assess_historical_bridge_v2(**f)["blockers"])

    def test_history_cycle_and_missing_parent_rejected_before_scoring(self):
        f = fixture()
        f["historical_nodes"][2]["parents"].append("previous-checkpoint")
        with self.assertRaises(HistoricalProvenanceError):
            freeze_historical_graph(f["historical_nodes"])

        f = fixture()
        f["historical_nodes"][3]["parents"] = [{"unhashable": "parent"}]
        with self.assertRaises(HistoricalProvenanceError):
            freeze_historical_graph(f["historical_nodes"])

        f = fixture()
        f["historical_nodes"][3]["parents"] = ["absent"]
        with self.assertRaises(HistoricalProvenanceError):
            freeze_historical_graph(f["historical_nodes"])

    def test_history_current_role_collision_and_base_model_rights_fail_closed(self):
        f = fixture()
        f["historical_nodes"][1]["sha256"] = f["request"].source_sha256
        f["frozen_historical_sha256"] = freeze_historical_graph(f["historical_nodes"])
        f["bridge_claim"]["historical_graph_sha256"] = f["frozen_historical_sha256"]
        self.assertIn("CURRENT_HISTORICAL_ROLE_COLLISION",
                      assess_historical_bridge_v2(**f)["blockers"])

        f = fixture()
        f["historical_nodes"][0]["rights"] = "UNKNOWN"
        f["frozen_historical_sha256"] = freeze_historical_graph(f["historical_nodes"])
        f["bridge_claim"]["historical_graph_sha256"] = f["frozen_historical_sha256"]
        self.assertIn("HISTORICAL_RIGHTS_TAINTED",
                      assess_historical_bridge_v2(**f)["blockers"])

    def test_frozen_history_hash_ignores_node_and_parent_order_only(self):
        f = fixture()
        graph = list(reversed(f["historical_nodes"]))
        for node in graph:
            node["parents"].reverse()
        self.assertEqual(freeze_historical_graph(graph),
                         f["frozen_historical_sha256"])

    def test_cross_assessor_mutation_uses_prior_snapshots(self):
        f = fixture()
        original = bridge_module.assess_rights_bound_cycle

        def mutate_original_after_history_checked(**kwargs):
            f["historical_nodes"][0]["rights"] = "REVOKED"
            f["current_nodes"][0]["rights"] = "REVOKED"
            f["bridge_claim"]["current_shard_sha256"] = sha("0")
            object.__setattr__(f["request"], "source_sha256", sha("0"))
            return original(**kwargs)

        with patch.object(bridge_module, "assess_rights_bound_cycle",
                          side_effect=mutate_original_after_history_checked):
            result = assess_historical_bridge_v2(**f)
        self.assertEqual(result["blockers"], [])
        self.assertTrue(result["synthetic_cross_generation_lineage_consistent"])
        self.assertFalse(result["gpu_start_authorized"])

    def test_nested_custom_copy_hooks_and_dataclass_hidden_payloads_rejected(self):
        calls = []

        class Spoof(list):
            def __iter__(self):
                calls.append("iter")
                return iter([])

        f = fixture()
        f["historical_nodes"][2]["parents"] = Spoof(["base-1", "older-corpus"])
        with self.assertRaises(HistoricalProvenanceError):
            assess_historical_bridge_v2(**f)
        self.assertEqual(calls, [])

        f = fixture()
        object.__setattr__(f["baseline"], "_rogue", Spoof([]))
        with self.assertRaises(HistoricalProvenanceError):
            assess_historical_bridge_v2(**f)
        self.assertEqual(calls, [])

    def test_oversized_exact_builtin_cycle_dataclass_string_rejected(self):
        f = fixture()
        object.__setattr__(f["consent"], "issuer_role", "X" * 70000)
        with self.assertRaises(HistoricalProvenanceError):
            assess_historical_bridge_v2(**f)

    def test_invalid_event_schema_and_history_commitment_rejected(self):
        f = fixture()
        f["frozen_historical_sha256"] = sha("9")
        self.assertIn("FROZEN_HISTORICAL_GRAPH_MISMATCH",
                      assess_historical_bridge_v2(**f)["blockers"])
        f = fixture()
        f["historical_events"] = [{"id": "bad", "action": "UNKNOWN"}]
        with self.assertRaises(HistoricalProvenanceError):
            assess_historical_bridge_v2(**f)


if __name__ == "__main__":
    unittest.main()
