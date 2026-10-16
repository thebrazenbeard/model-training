"""Synthetic hostile reviewer false-challenge counterexamples; no actual model."""
from dataclasses import replace
from pathlib import Path
import copy
import json
import unittest

from independent_challenge_eval_v1 import (
    AttributedClaim,ChallengeError,measure_false_challenge_rate,review_pair,canonical
)

FIXTURE=Path(__file__).resolve().parents[1]/"training"/"attributed_dissent_devset_v1.json"

def trusted_evidence(c):
    return c.evidence_digest in {"a"*64,"b"*64}

def independent_sources(a,b):
    return (a.node_id!=b.node_id and a.agent_id!=b.agent_id
            and a.evidence_group!=b.evidence_group
            and a.evidence_group in {"sensor:A","sensor:B"}
            and b.evidence_group in {"sensor:A","sensor:B"})

class DissentFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data=json.loads(FIXTURE.read_text(encoding="utf8"))
        assert data["schema"]=="VERA_ATTRIBUTED_DISSENT_DEVSET_V1"
        cls.cases=data["cases"]

    def test_dissent_remains_two_source_attributed_records(self):
        c=self.cases[0]
        a=AttributedClaim(**c["a"]); b=AttributedClaim(**c["b"])
        d=review_pair(a,b,verify_evidence=trusted_evidence,
                      verify_independence=independent_sources)
        self.assertEqual(d.classification,"EVIDENCE_CONFLICT_FOR_REVIEW")
        self.assertNotEqual(d.source_digests[0],d.source_digests[1])
        self.assertEqual(d.retained_claims[0]["value"],True)
        self.assertEqual(d.retained_claims[1]["value"],False)
        self.assertNotEqual(d.retained_claims[0]["node_id"],d.retained_claims[1]["node_id"])
        self.assertEqual(d.retained_claims[0]["shared_identity"],"Vera")
        self.assertEqual(d.retained_claims[1]["shared_identity"],"Vera")
        self.assertFalse(d.model_independence_proven)
        self.assertFalse(d.effect_authorized)
        self.assertFalse(d.accepted_as_fact)

    def test_synthetic_false_challenge_measurement(self):
        r=measure_false_challenge_rate(self.cases,
            verify_evidence=trusted_evidence,verify_independence=independent_sources)
        self.assertEqual(r["total"],15)
        self.assertEqual(r["totals"]["positive"],2)
        self.assertEqual(r["totals"]["negative"],13)
        self.assertEqual(r["totals"]["candidate_tp"],2)
        self.assertEqual(r["totals"]["candidate_fp"],0)
        self.assertEqual(r["totals"]["candidate_fn"],0)
        self.assertGreater(r["totals"]["naive_fp"],0)
        self.assertGreater(r["naive_false_challenge_rate"],r["candidate_false_challenge_rate"])
        self.assertEqual(r["qualification"],"NONE")
        self.assertEqual(r["scope"],"SYNTHETIC_DEVELOPMENT_NOT_HELD_OUT")

    def test_no_receiver_verifier_is_unverified_dissent_not_auto_conflict(self):
        c=self.cases[0]
        a=AttributedClaim(**c["a"]); b=AttributedClaim(**c["b"])
        d=review_pair(a,b)
        self.assertEqual(d.classification,"UNVERIFIED_DISSENT")
        self.assertFalse(d.challenge_for_review)
        r=measure_false_challenge_rate(self.cases)
        self.assertEqual(r["totals"]["candidate_tp"],0)
        self.assertEqual(r["totals"]["candidate_fn"],2)

    def test_changed_subject_scope_revision_and_claim_type_not_mismerged(self):
        expected={
            "different-referent":"DIFFERENT_PROPOSITION",
            "different-predicate":"DIFFERENT_PROPOSITION",
            "different-scope":"DIFFERENT_SCOPE",
            "stale-revision":"DIFFERENT_REVISION",
            "same-value":"SAME_CLAIM",
            "shared-evidence-group":"SELF_OR_CORRELATED_CLAIMS",
            "mixed-hypothesis":"MIXED_CLAIM_TYPES",
            "inference-is-not-observation":"MIXED_CLAIM_TYPES",
            "unverified-source":"UNVERIFIED_DISSENT",
            "same-agent-and-node":"SELF_OR_CORRELATED_CLAIMS",
            "different-shared-identity":"DIFFERENT_PROPOSITION",
            "unknown-claim-type":"MIXED_CLAIM_TYPES",
            "different-model-same-evidence":"SELF_OR_CORRELATED_CLAIMS"
        }
        actual={}
        for c in self.cases:
            finding=review_pair(AttributedClaim(**c["a"]),AttributedClaim(**c["b"]),
                                verify_evidence=trusted_evidence,
                                verify_independence=independent_sources)
            actual[c["id"]]=finding.classification
        for key,v in expected.items():
            with self.subTest(case=key):
                self.assertEqual(actual[key],v)

    def test_neither_same_model_nor_different_model_confers_verification(self):
        c=self.cases[0]
        a=AttributedClaim(**c["a"])
        b=AttributedClaim(**c["b"])
        self.assertEqual(review_pair(a,b).classification,"UNVERIFIED_DISSENT")
        b=replace(b,model_digest="d"*64)
        self.assertEqual(review_pair(a,b).classification,"UNVERIFIED_DISSENT")
        b=replace(b,model_digest=None)
        self.assertEqual(review_pair(a,b).classification,"UNVERIFIED_DISSENT")

    def test_external_validator_errors_hold_dissent(self):
        c=self.cases[0]
        a=AttributedClaim(**c["a"]);b=AttributedClaim(**c["b"])
        def raises(*args):
            raise OSError("Evidence store unavailable")
        self.assertEqual(review_pair(a,b,verify_evidence=raises,
                        verify_independence=independent_sources).classification,
                         "UNVERIFIED_DISSENT")
        self.assertEqual(review_pair(a,b,verify_evidence=trusted_evidence,
                        verify_independence=raises).classification,
                         "UNVERIFIED_DISSENT")

    def test_rejects_noncanonical_values_and_bad_role_referents(self):
        c=self.cases[0]
        with self.assertRaises(ChallengeError):
            AttributedClaim(**dict(c["a"],value=float("nan")))
        with self.assertRaises(ChallengeError):
            AttributedClaim(**dict(c["a"],claim_type="TRUTH"))
        with self.assertRaises(ChallengeError):
            AttributedClaim(**dict(c["a"],agent_id="new\nspoof"))
        with self.assertRaises(ChallengeError):
            AttributedClaim(**dict(c["a"],model_digest="model:foo"))
        with self.assertRaises(ChallengeError):
            AttributedClaim(**dict(c["a"],revision=True))
        with self.assertRaises(ChallengeError):
            canonical({"unsafe_float":1.5})

    def test_nested_original_value_and_returned_copy_cannot_rewrite_digest(self):
        c=copy.deepcopy(self.cases[0])
        c["a"]["value"]={"edges":[[17,42]]}
        a=AttributedClaim(**c["a"])
        initial=a.digest
        c["a"]["value"]["edges"][0][1]=99
        self.assertEqual(a.digest,initial)
        self.assertEqual(a.mapping()["value"],{"edges":[[17,42]]})
        out=a.mapping()
        out["value"]["edges"][0][1]=100
        self.assertEqual(a.digest,initial)
        self.assertEqual(a.mapping()["value"],{"edges":[[17,42]]})

    def test_duplicate_gold_ids_fail_before_measurement(self):
        cases=copy.deepcopy(self.cases)
        cases[1]["id"]=cases[0]["id"]
        with self.assertRaisesRegex(ChallengeError,"Duplicate"):
            measure_false_challenge_rate(cases,
                verify_evidence=trusted_evidence,
                verify_independence=independent_sources)

    def test_untrusted_gold_label_is_not_silently_cast(self):
        cases=copy.deepcopy(self.cases)
        cases[0]["expected_challenge"]="YES"
        with self.assertRaises(ChallengeError):
            measure_false_challenge_rate(cases,
                verify_evidence=trusted_evidence,
                verify_independence=independent_sources)

if __name__=="__main__":
    unittest.main()
