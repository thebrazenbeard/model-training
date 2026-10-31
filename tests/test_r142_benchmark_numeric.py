"""R142 CPU-only adversarial bounds for R18 public benchmark admission."""

from __future__ import annotations

import hashlib
import math
import unittest

from trainingday.r18_benchmark import (
    BenchmarkError, _number, make_plan, selected_id_digest, validate_plan,
)


def valid_plan() -> dict:
    rows = [
        {"id": f"train-{i}", "split": "train", "training_use": "ADMITTED",
         "prompt": f"What is {i} plus one?", "answer": str(i + 1)}
        for i in range(8)
    ]
    index = [
        {"id": f"r18-holdout-{i}",
         "prompt_sha256": hashlib.sha256(f"r18-prompt-{i}".encode()).hexdigest()}
        for i in range(64)
    ]
    return make_plan(
        selected_rows=rows,
        admission_selected_id_sha256=selected_id_digest(rows),
        source_admission_status="ADMITTED_SOURCE_ONLY_NOT_TRAINED",
        eval_index=index,
        r16_holdout_ids=[f"r16-fresh64-{i}" for i in range(64)],
        base_revision="1" * 40,
        base_binding_sha256="a" * 64,
        tokenizer_binding_sha256="b" * 64,
        corpus_manifest_sha256="c" * 64,
        eval_custody_sha256="d" * 64,
        eval_source_manifest_sha256="f" * 64,
        r16_eval_custody_sha256="e" * 64,
        seed=41,
        training_budget_seconds=100.0,
        probe_budget_seconds=5.0,
    )


class BenchmarkNumericAdmissionTests(unittest.TestCase):
    def test_direct_number_guard_rejects_hostile_bounds_without_exception(self):
        for value in (10**1000, 10**5000, math.inf, math.nan, -1, True,
                      "100", object()):
            with self.subTest(type=type(value).__name__):
                self.assertIs(_number(value), False)
        self.assertTrue(_number(0))
        self.assertTrue(_number(100.0, zero=False))
        self.assertFalse(_number(0, zero=False))

    def test_validate_plan_oversized_budget_returns_benchmark_error(self):
        plan = valid_plan()
        plan["training_budget_seconds"] = 10**1000
        with self.assertRaisesRegex(BenchmarkError, "invalid plan budgets/seed"):
            validate_plan(plan)

    def test_validate_plan_oversized_probe_budget_returns_benchmark_error(self):
        plan = valid_plan()
        plan["probe_budget_seconds"] = 10**5000
        with self.assertRaisesRegex(BenchmarkError, "invalid plan budgets/seed"):
            validate_plan(plan)

    def test_validate_plan_oversized_seed_refused_before_json_digest(self):
        plan = valid_plan()
        plan["seed"] = 10**5000
        with self.assertRaisesRegex(BenchmarkError, "invalid plan budgets/seed"):
            validate_plan(plan)

    def test_make_plan_oversized_budget_fails_before_publication(self):
        plan = valid_plan()
        # Keep the known-good envelope, but ensure the public plan validator
        # refuses hostile numeric values rather than leaking OverflowError.
        plan["training_budget_seconds"] = 1e309
        with self.assertRaisesRegex(BenchmarkError, "invalid plan budgets/seed"):
            validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
