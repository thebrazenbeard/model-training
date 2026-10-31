"""Public toy integration: a rights-tainted artifact cannot look cycle-ready."""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from tests.test_r27_continuous_learning_policy import fixture, sha
from trainingday.r26_rights_taint import assess_revocation
from trainingday.r27_continuous_learning_policy import baseline_digest, assess_cycle
import trainingday.r28_rights_cycle_integration as binding
from trainingday.r28_rights_cycle_integration import assess_rights_bound_cycle


def bundle():
    baseline, request, consent, evidence, cycle_events = fixture()
    graph = [
        {"id": "source-1", "kind": "TRAIN_SHARD", "sha256": request.source_sha256,
         "parents": [], "license_sha256": sha("4"), "rights": "DECLARED_ADMITTED"},
        {"id": "adapter-1", "kind": "ADAPTER", "sha256": request.proposed_adapter_sha256,
         "parents": ["source-1"]},
        {"id": "checkpoint-1", "kind": "CHECKPOINT",
         "sha256": request.proposed_checkpoint_sha256, "parents": ["adapter-1"]},
    ]
    baseline = replace(baseline, rights_sha256=assess_revocation(graph, [])["assessment_sha256"])
    request = replace(request, baseline_sha256=baseline_digest(baseline))
    consent = replace(consent, baseline_sha256=request.baseline_sha256)
    evidence = replace(evidence, rights_sha256=baseline.rights_sha256)
    return graph, [], baseline, request, consent, evidence, cycle_events


def assess(*, graph=None, rights_events=None, baseline=None, request=None,
           consent=None, evidence=None, cycle_events=None):
    g, re, b, r, c, e, ce = bundle()
    return assess_rights_bound_cycle(
        nodes=g if graph is None else graph,
        rights_events=re if rights_events is None else rights_events,
        baseline=b if baseline is None else baseline,
        request=r if request is None else request,
        consent=c if consent is None else consent,
        evidence=e if evidence is None else evidence,
        cycle_events=ce if cycle_events is None else cycle_events,
        now_unix=130,
    )


class RightsCycleIntegrationTests(unittest.TestCase):
    def test_source_clean_never_authorizes_training(self):
        result = assess()
        self.assertTrue(result["synthetic_source_conditions_satisfied"])
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(result["blockers"], [])
        # The R26 single-current-source graph cannot prove historical
        # baseline model lineage across earlier distinct training corpora.
        self.assertFalse(result["previous_baseline_lineage_verified"])
        for key in ("gpu_start_authorized", "continuous_weight_update_authorized",
                    "model_promotion_authorized", "consent_independently_authenticated",
                    "rights_independently_attested"):
            self.assertFalse(result[key], key)

    def test_revocation_after_claimed_freeze_blocks_clean_consent(self):
        graph, _, _, request, _, _, _ = bundle()
        event = {"id": "revocation-1", "source_id": "source-1",
                 "source_sha256": request.source_sha256, "action": "REVOKED",
                 "evidence_sha256": sha("5")}
        result = assess(rights_events=[event])
        self.assertIn("RIGHTS_TAINTED_LINEAGE", result["blockers"])
        self.assertFalse(result["synthetic_source_conditions_satisfied"])
        self.assertEqual(result["rights_status_by_artifact"]["checkpoint-1"], "REVOKED_LINEAGE")

    def test_unknown_and_privacy_tainted_source_block(self):
        graph, _, _, request, _, _, _ = bundle()
        graph[0]["rights"] = "UNKNOWN"
        self.assertIn("RIGHTS_TAINTED_LINEAGE", assess(graph=graph)["blockers"])
        privacy = {"id": "privacy-1", "source_id": "source-1",
                   "source_sha256": request.source_sha256, "action": "PRIVACY_TAINT",
                   "evidence_sha256": sha("6")}
        self.assertIn("RIGHTS_TAINTED_LINEAGE", assess(rights_events=[privacy])["blockers"])

    def test_forged_candidate_sha_and_additional_source_fail_closed(self):
        graph, _, _, _, _, _, _ = bundle()
        graph[1]["sha256"] = sha("7")
        self.assertIn("CANDIDATE_ARTIFACT_UNBOUND", assess(graph=graph)["blockers"])

        graph, _, _, _, _, _, _ = bundle()
        graph[0]["sha256"] = sha("7")
        self.assertIn("TRAINING_SOURCE_UNBOUND", assess(graph=graph)["blockers"])

        graph, _, _, _, _, _, _ = bundle()
        graph.append({"id": "source-2", "kind": "TRAIN_SHARD", "sha256": sha("8"),
                      "parents": [], "license_sha256": sha("9"),
                      "rights": "DECLARED_ADMITTED"})
        graph[2]["parents"].append("source-2")
        result = assess(graph=graph)
        self.assertIn("MULTI_SOURCE_LINEAGE_UNSUPPORTED", result["blockers"])

    def test_checkpoint_cannot_claim_unrelated_adapter(self):
        graph, _, _, _, _, _, _ = bundle()
        graph[2]["parents"] = ["source-1"]
        result = assess(graph=graph)
        self.assertIn("CANDIDATE_CHECKPOINT_NOT_DESCENDED_FROM_ADAPTER", result["blockers"])

    def test_disconnected_unresolved_second_source_still_fails_single_shard_contract(self):
        g, re, b, r, c, e, ce = bundle()
        g.append({"id": "unrelated-source", "kind": "TRAIN_SHARD", "sha256": sha("8"),
                  "parents": [], "license_sha256": sha("9"), "rights": "UNKNOWN"})
        # Self-consistent repinning of a caller-controlled frozen graph must
        # not conceal ambiguity by excluding a disconnected unresolved source.
        b = replace(b, rights_sha256=assess_revocation(g, [])["assessment_sha256"])
        r = replace(r, baseline_sha256=baseline_digest(b))
        c = replace(c, baseline_sha256=r.baseline_sha256)
        e = replace(e, rights_sha256=b.rights_sha256)
        result = assess_rights_bound_cycle(nodes=g, rights_events=re, baseline=b,
                                           request=r, consent=c, evidence=e,
                                           cycle_events=ce, now_unix=130)
        self.assertIn("MULTI_SOURCE_LINEAGE_UNSUPPORTED", result["blockers"])
        self.assertFalse(result["synthetic_source_conditions_satisfied"])

    def test_forged_clean_graph_is_never_authenticated(self):
        # Even if the caller repins every mutable input to a fabricated graph,
        # source consistency cannot become an authenticated rights decision.
        result = assess()
        self.assertFalse(result["rights_independently_attested"])
        self.assertFalse(result["protected_eval_unseal_authorized"])
        self.assertFalse(result["gpu_start_authorized"])

    def test_mutation_after_rights_assessment_cannot_change_artifact_binding(self):
        g, re, b, r, c, e, ce = bundle()

        def mutate_original_when_cycle_assessed(**kwargs):
            # The input belonged to a caller; it may change concurrently with
            # a downstream callback. The right result must use the snapshot.
            g[0]["rights"] = "REVOKED"
            g[1]["sha256"] = sha("7")
            return assess_cycle(**kwargs)

        with patch.object(binding, "assess_cycle",
                          side_effect=mutate_original_when_cycle_assessed):
            result = binding.assess_rights_bound_cycle(
                nodes=g, rights_events=re, baseline=b, request=r,
                consent=c, evidence=e, cycle_events=ce, now_unix=130)

        self.assertTrue(result["synthetic_source_conditions_satisfied"])
        self.assertEqual(result["blockers"], [])
        self.assertEqual(result["rights_status_by_artifact"]["source-1"],
                         "DECLARED_CLEAN_ONLY")
        self.assertFalse(result["gpu_start_authorized"])

    def test_request_mutation_after_cycle_validation_cannot_rebind_consent(self):
        g, re, b, request, consent, evidence, ce = bundle()
        legitimate_source = request.source_sha256
        # An internally consistent but wrong source request is initially not
        # bound to the rights graph. A frozen dataclass may still be mutated by
        # a holder using object.__setattr__ after R27's assessor returns.
        request = replace(request, source_sha256=sha("0"))
        consent = replace(consent, source_sha256=sha("0"))

        def change_original_request_after_validation(**kwargs):
            result = assess_cycle(**kwargs)
            object.__setattr__(request, "source_sha256", legitimate_source)
            return result

        with patch.object(binding, "assess_cycle",
                          side_effect=change_original_request_after_validation):
            result = binding.assess_rights_bound_cycle(
                nodes=g, rights_events=re, baseline=b, request=request,
                consent=consent, evidence=evidence, cycle_events=ce,
                now_unix=130)

        self.assertIn("TRAINING_SOURCE_UNBOUND", result["blockers"])
        self.assertFalse(result["synthetic_source_conditions_satisfied"])
        self.assertFalse(result["gpu_start_authorized"])

    def test_frozen_rights_graph_commitment_mismatch_blocks(self):
        graph, _, _, _, _, _, _ = bundle()
        graph[0]["license_sha256"] = sha("a")
        self.assertIn("FROZEN_RIGHTS_COMMITMENT_MISMATCH", assess(graph=graph)["blockers"])


if __name__ == "__main__":
    unittest.main()
