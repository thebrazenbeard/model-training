"""R18 protected-exclusion public-shape checks, never private-bank access."""

import hashlib
import unittest

from trainingday.r18_benchmark import make_plan, selected_id_digest
from trainingday.protected_eval_custody_v1 import (
    EXCLUSION_SCHEMA, ProtectedEvalCustodyError, assess_public_exclusion_claim,
)


def _plan():
    rows = [
        {"id": f"train-{i}", "split": "train", "training_use": "ADMITTED",
         "evaluation_holdout": False, "prompt": f"Calculate {i} plus one.",
         "answer": str(i + 1)} for i in range(8)
    ]
    index = [
        {"id": f"r18-eval-{i}", "prompt_sha256": hashlib.sha256(
            f"eval-prompt-{i}".encode()).hexdigest()} for i in range(64)
    ]
    return make_plan(
        selected_rows=rows, admission_selected_id_sha256=selected_id_digest(rows),
        source_admission_status="ADMITTED_SOURCE_ONLY_NOT_TRAINED",
        eval_index=index, r16_holdout_ids=[f"r16-eval-{i}" for i in range(64)],
        base_revision="1" * 40, base_binding_sha256="a" * 64,
        tokenizer_binding_sha256="b" * 64, corpus_manifest_sha256="c" * 64,
        eval_custody_sha256="d" * 64, eval_source_manifest_sha256="e" * 64,
        r16_eval_custody_sha256="f" * 64, seed=41,
        training_budget_seconds=120.0, probe_budget_seconds=5.0,
    )


def _claim(plan):
    from trainingday.r18_benchmark import validate_plan
    return {
        "schema": EXCLUSION_SCHEMA,
        "plan_sha256": validate_plan(plan),
        "r18_eval_custody_sha256": plan["eval_custody_sha256"],
        "r18_eval_index_sha256": plan["eval_index_sha256"],
        "r18_eval_source_manifest_sha256": plan["eval_source_manifest_sha256"],
        "r16_exclusion_custody_sha256": plan["r16_eval_custody_sha256"],
        "r16_exclusion_id_sha256": plan["r16_holdout_id_sha256"],
        "train_content_sha256": plan["train_content_sha256"],
        "protected_dev_exclusion_sha256": "2" * 64,
        "protected_eval_exclusion_sha256": "3" * 64,
        "complete_private_roster_sha256": "4" * 64,
        "independent_steward_id": "steward-independent",
        "independent_evaluator_id": "evaluator-independent",
        "trainer_id": "trainer-host",
        "claimed_complete_private_roster": True,
    }


class ProtectedEvalCustodySourceTests(unittest.TestCase):
    def test_matching_claim_can_never_authorize_protected_evaluation(self):
        result = assess_public_exclusion_claim(_plan(), _claim(_plan()))
        self.assertTrue(result.source_bindings_match)
        self.assertFalse(result.independent_custody_verified)
        self.assertFalse(result.protected_evaluation_authorized)
        self.assertEqual(result.status, "SOURCE_ONLY_EXCLUSION_SHAPE_HOLD")

    def test_plan_hash_and_each_prior_exclusion_commitment_must_match(self):
        plan = _plan()
        for key in (
            "plan_sha256", "r18_eval_custody_sha256", "r18_eval_index_sha256",
            "r18_eval_source_manifest_sha256", "r16_exclusion_custody_sha256",
            "r16_exclusion_id_sha256", "train_content_sha256",
        ):
            claim = _claim(plan)
            claim[key] = "0" * 64
            with self.subTest(key=key), self.assertRaises(ProtectedEvalCustodyError):
                assess_public_exclusion_claim(plan, claim)

    def test_missing_complete_dev_eval_exclusions_or_claimed_completeness_rejected(self):
        plan = _plan()
        for key in ("protected_dev_exclusion_sha256", "protected_eval_exclusion_sha256",
                    "complete_private_roster_sha256"):
            claim = _claim(plan)
            del claim[key]
            with self.subTest(key=key), self.assertRaises(ProtectedEvalCustodyError):
                assess_public_exclusion_claim(plan, claim)
        claim = _claim(plan)
        claim["claimed_complete_private_roster"] = False
        with self.assertRaises(ProtectedEvalCustodyError):
            assess_public_exclusion_claim(plan, claim)

    def test_caller_fields_and_collapsed_principals_fail_closed(self):
        plan = _plan()
        for key, value in (
            ("evaluator_signature", "self-signed"), ("raw_eval_ids", ["secret"]),
            ("private_bank_path", "D:/VERA/eval"), ("authorized", True),
        ):
            claim = _claim(plan)
            claim[key] = value
            with self.subTest(key=key), self.assertRaises(ProtectedEvalCustodyError):
                assess_public_exclusion_claim(plan, claim)
        claim = _claim(plan)
        claim["independent_evaluator_id"] = claim["trainer_id"]
        with self.assertRaises(ProtectedEvalCustodyError):
            assess_public_exclusion_claim(plan, claim)

    def test_malformed_hash_bool_or_custom_string_identity_rejected(self):
        plan = _plan()

        class ForgedStr(str):
            def __eq__(self, other):
                return True

        for key, value in (
            ("protected_dev_exclusion_sha256", "A" * 64),
            ("claimed_complete_private_roster", 1),
            ("independent_steward_id", ForgedStr("bad")),
            ("r18_eval_index_sha256", ForgedStr("wrong")),
        ):
            claim = _claim(plan)
            claim[key] = value
            with self.subTest(key=key), self.assertRaises(ProtectedEvalCustodyError):
                assess_public_exclusion_claim(plan, claim)


if __name__ == "__main__":
    unittest.main()
