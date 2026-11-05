"""No-model tests for the bounded public conversation receipt validator."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from q4_conversation_eval.baseline import (
    CASES,
    PUBLISHER_Q4_SHA256,
    ReceiptError,
    build_plan,
    evaluate_record,
)


def valid_fixture():
    plan = build_plan()
    return {
        "protocol": plan["protocol"],
        "source": "offline_fixture",
        "prompts_sha256": plan["prompts_sha256"],
        "model_sha256": PUBLISHER_Q4_SHA256,
        "model_size_bytes": 2783446880,
        "runner_sha256": "f" * 64,
        "owner_lease_receipt": "TEST-NOT-A-REAL-LEASE",
        "decode": {
            "context_size": 1024,
            "max_new_tokens": 64,
            "n_gpu_layers": 0,
            "temperature": 0,
            "seed": 0,
        },
        "cases": [
            {
                "id": case["id"],
                "turns": [
                    {
                        "prompt": prompt,
                        "assistant": "A simulated answer for unit testing only.",
                        "finish_reason": "stop",
                        "output_tokens": 9,
                        "latency_ms": 20,
                    }
                    for prompt in case["turns"]
                ],
            }
            for case in CASES
        ],
    }


class ConversationBaselineTests(unittest.TestCase):
    def test_plan_exact_public_eight_turn_four_case_binding(self):
        plan = build_plan()
        self.assertEqual(plan["evidence_class"], "SOURCE_ONLY_PLAN")
        self.assertFalse(plan["physical_model_execution"])
        self.assertFalse(plan["model_identity_attested"])
        self.assertEqual(sum(len(c["turns"]) for c in plan["cases"]), 8)
        self.assertEqual([c["id"] for c in plan["cases"]], [
            "self_identification", "corrected_name", "dry_humor",
            "memory_uncertainty",
        ])
        self.assertEqual(plan["arm"], "twoody_publisher_q4_base_only_UNVERIFIED")
        self.assertEqual(plan["decode"]["context_size"], 1024)
        self.assertEqual(plan["decode"]["max_new_tokens"], 64)
        self.assertEqual(plan["decode"]["n_gpu_layers"], 0)
        self.assertEqual(build_plan(), plan)

    def test_fixture_only_gets_structure_ack_never_semantic_pass(self):
        report = evaluate_record(valid_fixture())
        self.assertEqual(report["status"], "REVIEW_REQUIRED")
        self.assertEqual(report["receipt_class"], "OFFLINE_FIXTURE_ONLY")
        self.assertFalse(report["model_identity_attested"])
        self.assertFalse(report["physical_qualification_proved"])
        self.assertEqual(report["completed_turns"], 8)
        self.assertEqual(report["truncated_turns"], 0)
        self.assertEqual(len(report["human_review_queue"]), 4)
        self.assertNotIn("simulated answer", json.dumps(report).lower())
        self.assertNotIn("TEST-NOT-A-REAL-LEASE", json.dumps(report))
        self.assertNotIn("pass", json.dumps(report).lower())

    def test_operator_record_still_cannot_prove_physical_qualification(self):
        record = valid_fixture()
        record["source"] = "operator_transcript"
        report = evaluate_record(record)
        self.assertEqual(report["receipt_class"], "UNATTESTED_OPERATOR_TRANSCRIPT")
        self.assertFalse(report["physical_qualification_proved"])
        self.assertEqual(report["status"], "REVIEW_REQUIRED")

    def test_allows_length_but_marks_not_evaluable_without_semantic_score(self):
        record = valid_fixture()
        record["cases"][0]["turns"][1]["finish_reason"] = "length"
        record["cases"][0]["turns"][1]["output_tokens"] = 64
        report = evaluate_record(record)
        self.assertEqual(report["status"], "NOT_EVALUABLE")
        self.assertEqual(report["truncated_turns"], 1)
        self.assertEqual(report["human_review_queue"][0]["status"], "NOT_EVALUABLE")

    def test_rejects_tampered_case_order_and_prompt(self):
        record = valid_fixture()
        record["cases"][0]["id"] = "memory_uncertainty"
        with self.assertRaises(ReceiptError):
            evaluate_record(record)
        record = valid_fixture()
        record["cases"][1]["turns"][2]["prompt"] = "Different prompt"
        with self.assertRaises(ReceiptError):
            evaluate_record(record)

    def test_rejects_unbound_runner_model_and_changed_caps(self):
        for mutation in (
            ("model_sha256", "0" * 64),
            ("runner_sha256", "unverified"),
            ("model_size_bytes", 1234),
            ("prompts_sha256", "0" * 64),
        ):
            record = valid_fixture()
            record[mutation[0]] = mutation[1]
            with self.subTest(field=mutation[0]), self.assertRaises(ReceiptError):
                evaluate_record(record)
        record = valid_fixture()
        record["decode"]["max_new_tokens"] = 96
        with self.assertRaises(ReceiptError):
            evaluate_record(record)
        record = valid_fixture()
        record["decode"]["n_gpu_layers"] = 1
        with self.assertRaises(ReceiptError):
            evaluate_record(record)
        record = valid_fixture()
        record["decode"]["n_gpu_layers"] = False
        with self.assertRaises(ReceiptError):
            evaluate_record(record)
        record = valid_fixture()
        record["decode"]["context_size"] = 1024.0
        with self.assertRaises(ReceiptError):
            evaluate_record(record)

    def test_rejects_missing_or_unbounded_or_nonassistant_responses(self):
        for key, value in (
            ("assistant", ""),
            ("assistant", "a" * 16001),
            ("finish_reason", "tool_calls"),
            ("output_tokens", 65),
            ("latency_ms", -1),
        ):
            record = valid_fixture()
            record["cases"][0]["turns"][0][key] = value
            with self.subTest(field=key), self.assertRaises(ReceiptError):
                evaluate_record(record)
        record = valid_fixture()
        record["cases"][0]["turns"][0]["tool_calls"] = [{"function": "delete"}]
        with self.assertRaises(ReceiptError):
            evaluate_record(record)

    def test_cli_no_model_no_raw_response_and_finite_input_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            payload = valid_fixture()
            payload["cases"][0]["turns"][0]["assistant"] = "DO_NOT_ECHO_PRIVATE_RESPONSE"
            path.write_text(json.dumps(payload), encoding="utf-8")
            command = [
                sys.executable, "-m", "q4_conversation_eval.baseline",
                "--evaluate", str(path),
            ]
            run = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)["status"], "REVIEW_REQUIRED")
            self.assertNotIn("DO_NOT_ECHO_PRIVATE_RESPONSE", run.stdout + run.stderr)
            path.write_bytes(b"x" * 131073)
            run = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(run.returncode, 0)
            self.assertNotIn("x" * 1000, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
