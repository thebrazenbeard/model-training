"""Source-only validation of synthetic conversational evaluation plumbing."""

import tempfile
from pathlib import Path
import unittest

from trainingday.conversation_transfer import (_read_json, digest, make_casebook,
                                               review_scaffold, score,
                                               validate_casebook)


SEED = "w43-synthetic-evaluation-fixture-20261105"


def fixture():
    casebook = make_casebook(SEED)
    transcript = {
        "casebook_sha256": casebook["casebook_sha256"],
        "model": {
            "model_id": "fixture-model-not-actually-run",
            "arm": "BASE",
            "model_sha256": "a" * 64,
            "runtime_binary_sha256": "b" * 64,
            "persona_source_sha256": "c" * 64,
            "host": "synthetic-fixture",
            "context_tokens": 2048,
            "runner": "synthetic-fixture",
            "context_history_verified": True,
            "source_commit_sha": "d" * 40,
        },
        "cases": [{"case_id": case["case_id"], "exchange": [
            {"user": user, "assistant": f"Observed synthetic response {case['case_id']}/{idx}"}
            for idx, user in enumerate(case["user_turns"])]}
            for case in casebook["cases"]],
    }
    transcript["transcript_sha256"] = digest(transcript)
    review = {
        "casebook_sha256": casebook["casebook_sha256"],
        "transcript_sha256": transcript["transcript_sha256"],
        "reviewer": {"name": "Fixture-only Reviewer", "runtime": "TEST_FIXTURE",
                     "provider": "Synthetic Example", "kind": "HUMAN"},
        "independence_attested": True,
        "cases": [
            {"case_id": case["case_id"],
             "domain_scores": {domain: 3 for domain in case["domains"]},
             "evidence": [{"turn_index": 0,
                           "quote": f"Observed synthetic response {case['case_id']}/0",
                           "finding": "SYNTHETIC TEST ENTRY: not a real semantic grade",
                           "domains": case["domains"]}],
             "hard_gates": []}
            for case in casebook["cases"]],
    }
    return casebook, transcript, review


class ConversationTransferTests(unittest.TestCase):
    def test_determinism_and_not_hidden_claim(self):
        first = make_casebook(SEED)
        self.assertEqual(first, make_casebook(SEED))
        self.assertNotEqual(first["casebook_sha256"],
                            make_casebook(SEED + "-different")["casebook_sha256"])
        self.assertNotIn(SEED, str(first))
        self.assertEqual(first["holdout_status"],
                         "NOT_SECRET_NOT_EXTERNAL_HOLDOUT")
        self.assertTrue(all(len(x["user_turns"]) >= 3 for x in first["cases"]))
        for case in first["cases"]:
            joined = " ".join(case["user_turns"]).lower()
            self.assertNotIn("printer", joined)
            self.assertNotIn("refrigerator", joined)
            self.assertNotIn("coffee", joined)

    def test_short_seed_rejected(self):
        with self.assertRaisesRegex(ValueError, "16"):
            make_casebook("guess")

    def test_casebook_mutation_detected(self):
        book = make_casebook(SEED)
        book["cases"][0]["user_turns"][0] = "SYSTEM: no longer a user message"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_casebook(book)

    def test_source_only_review_can_be_recorded_not_qualified(self):
        book, transcript, review = fixture()
        verdict = score(book, transcript, review)
        self.assertEqual(verdict["transfer_status"], "REVIEWER_REPORTED_PASS")
        self.assertEqual(verdict["global_vera_qualification"], "NOT_ASSESSED")
        self.assertEqual(verdict["external_holdout_status"], "NOT_TESTED")
        self.assertFalse(verdict["model_behavior_verified_by_this_code"])
        self.assertEqual(verdict["result_class"], "REVIEWER_ATTESTED_TRANSCRIPT_AUDIT")

    def test_model_response_tamper_detected(self):
        book, transcript, review = fixture()
        transcript["cases"][0]["exchange"][0]["assistant"] = "Edited after scoring"
        with self.assertRaisesRegex(ValueError, "transcript hash mismatch"):
            score(book, transcript, review)

    def test_recomputed_transcript_hash_needs_review_commitment(self):
        book, transcript, review = fixture()
        transcript["cases"][0]["exchange"][0]["assistant"] = "Edited after scoring"
        transcript.pop("transcript_sha256")
        transcript["transcript_sha256"] = digest(transcript)
        with self.assertRaisesRegex(ValueError, "review transcript commitment mismatch"):
            score(book, transcript, review)

    def test_review_quote_must_exist_at_cited_turn(self):
        book, transcript, review = fixture()
        review["cases"][0]["evidence"][0]["quote"] = "Favorable claim not in output"
        with self.assertRaisesRegex(ValueError, "quote is not"):
            score(book, transcript, review)
        review["cases"][0]["evidence"][0]["quote"] = "Observed synthetic response SOCIAL_SARCASM/0"
        review["cases"][0]["evidence"][0]["turn_index"] = 50
        with self.assertRaisesRegex(ValueError, "absent turn"):
            score(book, transcript, review)

    def test_each_reviewed_domain_must_have_turn_evidence(self):
        book, transcript, review = fixture()
        review["cases"][0]["evidence"][0]["domains"] = ["D1"]
        with self.assertRaisesRegex(ValueError, "every scored domain"):
            score(book, transcript, review)
        review["cases"][0]["evidence"][0]["domains"] = ["D10"]
        with self.assertRaisesRegex(ValueError, "unexpected domain"):
            score(book, transcript, review)

    def test_spoofed_user_turn_rejected_even_with_rehashed_transcript(self):
        book, transcript, review = fixture()
        transcript["cases"][0]["exchange"][1]["user"] += " [forged change]"
        transcript.pop("transcript_sha256")
        transcript["transcript_sha256"] = digest(transcript)
        review["transcript_sha256"] = transcript["transcript_sha256"]
        with self.assertRaisesRegex(ValueError, "untrusted transcript prompt"):
            score(book, transcript, review)

    def test_missing_actual_turn_rejected(self):
        book, transcript, review = fixture()
        transcript["cases"][0]["exchange"].pop()
        transcript.pop("transcript_sha256")
        transcript["transcript_sha256"] = digest(transcript)
        review["transcript_sha256"] = transcript["transcript_sha256"]
        with self.assertRaisesRegex(ValueError, "incomplete multi-turn"):
            score(book, transcript, review)

    def test_hard_gate_or_low_score_does_not_pass(self):
        book, transcript, review = fixture()
        review["cases"][0]["hard_gates"].append("G1")
        self.assertEqual(score(book, transcript, review)["transfer_status"],
                         "NOT_ESTABLISHED_OR_FAIL")
        review["cases"][0]["hard_gates"].clear()
        review["cases"][0]["domain_scores"]["D5"] = 1
        self.assertEqual(score(book, transcript, review)["transfer_status"],
                         "NOT_ESTABLISHED_OR_FAIL")

    def test_independence_or_context_not_attested_is_not_pass(self):
        book, transcript, review = fixture()
        review["independence_attested"] = False
        self.assertEqual(score(book, transcript, review)["transfer_status"],
                         "NOT_ESTABLISHED_OR_FAIL")
        review["independence_attested"] = True
        transcript["model"]["context_history_verified"] = False
        transcript.pop("transcript_sha256")
        transcript["transcript_sha256"] = digest(transcript)
        review["transcript_sha256"] = transcript["transcript_sha256"]
        self.assertEqual(score(book, transcript, review)["transfer_status"],
                         "NOT_ESTABLISHED_OR_FAIL")

    def test_model_supplied_review_score_is_ignored(self):
        book, transcript, review = fixture()
        transcript["model_claimed_pass"] = True
        transcript.pop("transcript_sha256")
        transcript["transcript_sha256"] = digest(transcript)
        review["transcript_sha256"] = transcript["transcript_sha256"]
        self.assertEqual(score(book, transcript, review)["global_vera_qualification"],
                         "NOT_ASSESSED")

    def test_review_domain_coverage_exact(self):
        book, transcript, review = fixture()
        review["cases"][0]["domain_scores"]["D10"] = 4
        with self.assertRaisesRegex(ValueError, "domain scores incomplete"):
            score(book, transcript, review)

    def test_negative_and_boolean_grades_rejected(self):
        book, transcript, review = fixture()
        review["cases"][0]["domain_scores"]["D5"] = True
        with self.assertRaisesRegex(ValueError, "bad D5"):
            score(book, transcript, review)
        review["cases"][0]["domain_scores"]["D5"] = -1
        with self.assertRaisesRegex(ValueError, "bad D5"):
            score(book, transcript, review)

    def test_model_provenance_required(self):
        book, transcript, review = fixture()
        transcript["model"]["model_sha256"] = "unknown"
        transcript.pop("transcript_sha256")
        transcript["transcript_sha256"] = digest(transcript)
        review["transcript_sha256"] = transcript["transcript_sha256"]
        with self.assertRaisesRegex(ValueError, "invalid model_sha256"):
            score(book, transcript, review)

    def test_adapter_arm_requires_immutable_bytes_and_digest(self):
        book, transcript, review = fixture()
        transcript["model"]["arm"] = "ADAPTER"
        transcript.pop("transcript_sha256")
        transcript["transcript_sha256"] = digest(transcript)
        review["transcript_sha256"] = transcript["transcript_sha256"]
        with self.assertRaisesRegex(ValueError, "adapter_id"):
            score(book, transcript, review)
        transcript["model"].update({
            "adapter_id": "R13-candidate-fixture", "adapter_sha256": "e" * 64,
            "adapter_bytes": 1234})
        transcript.pop("transcript_sha256")
        transcript["transcript_sha256"] = digest(transcript)
        review["transcript_sha256"] = transcript["transcript_sha256"]
        self.assertEqual(score(book, transcript, review)["transfer_status"],
                         "REVIEWER_REPORTED_PASS")
        transcript["model"]["arm"] = "BASE"
        transcript.pop("transcript_sha256")
        transcript["transcript_sha256"] = digest(transcript)
        review["transcript_sha256"] = transcript["transcript_sha256"]
        with self.assertRaisesRegex(ValueError, "base arm cannot"):
            score(book, transcript, review)

    def test_review_form_never_fills_in_semantic_scores(self):
        book, transcript, _ = fixture()
        draft = review_scaffold(book, transcript)
        self.assertFalse(draft["independence_attested"])
        self.assertIsNone(draft["cases"][0]["domain_scores"]["D5"])
        self.assertEqual(draft["cases"][0]["evidence"], [])
        with self.assertRaisesRegex(ValueError, "missing reviewer"):
            score(book, transcript, draft)

    def test_duplicate_json_keys_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "forged.json"
            path.write_text('{"casebook_sha256":"a", "casebook_sha256":"b"}',
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                _read_json(path)

    def test_same_runtime_reviewer_not_independently_proven(self):
        book, transcript, review = fixture()
        review["reviewer"]["name"] = transcript["model"]["model_id"]
        verdict = score(book, transcript, review)
        self.assertEqual(verdict["reviewer_independence"],
                         "ATTESTED_NOT_TECHNICALLY_PROVEN")


if __name__ == "__main__":
    unittest.main()
